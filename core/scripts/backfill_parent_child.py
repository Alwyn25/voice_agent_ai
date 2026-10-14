#!/usr/bin/env python3
"""R5.03 backfill: give EXISTING knowledge chunks their ~200-char children, in place.

The original document text is not stored, and is not needed: every existing chunk is already
exactly what a parent is (a chunk_text() output), so each one large enough to benefit is marked
'parent' and its children are cut from its own text with the same split_children() ingestion
uses. The parent keeps its embedding, so nothing about "flat" retrieval changes — only child
embeddings are paid for (a few hundred short strings for a typical tenant: well under a cent).

SAFE BY DEFAULT: without --apply nothing is written; the plan, the child count and the embedding
cost estimate are printed. Each document is committed on its own, so an interruption leaves
every document either fully converted or untouched, and a re-run skips finished documents.

    docker compose exec core python scripts/backfill_parent_child.py                 # dry run
    docker compose exec core python scripts/backfill_parent_child.py --apply         # do it
    docker compose exec core python scripts/backfill_parent_child.py --apply --document <uuid>
    docker compose exec core python scripts/backfill_parent_child.py --apply --force # rebuild
    docker compose exec core python scripts/backfill_parent_child.py --verify        # integrity
    docker compose exec core python scripts/backfill_parent_child.py --revert --apply # undo

--tenant limits any mode to one tenant (uuid). --show prints every child that would be created.
Children are only SEARCHED once RETRIEVAL_GRANULARITY=parent_child; until then this is invisible.

Exit codes: 0 ok, 1 --verify found a problem / setup failure.
"""
import argparse
import asyncio
import sys
import uuid
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sqlalchemy import delete, select, update  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db.base import get_session_factory  # noqa: E402
from app.db.models import KnowledgeChunk, KnowledgeDocument  # noqa: E402
from app.engine.rag.chunkers import Chunk, split_children, uncovered_text  # noqa: E402
from app.engine.rag.parent_child import (ROLE_CHILD, ROLE_PARENT,  # noqa: E402
                                         ROLE_STANDALONE, child_split_kwargs)
from app.engine.rag.retrieval import invalidate_tenant_knowledge_cache  # noqa: E402
from app.engine.rag.ingestion import embed_resilient  # noqa: E402
from app.providers.embeddings.router import get_embeddings  # noqa: E402

# USD per 1M input tokens — for the dry-run estimate only.
_PRICE_PER_M = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}


def body_of(chunk: KnowledgeChunk) -> tuple[str, str | None]:
    """(body, heading) of a stored chunk. Stored content is embed_text = heading + "\\n" + body
    (see Chunk.embed_text), so the breadcrumb is stripped before re-splitting — children
    re-add it themselves."""
    heading = (chunk.meta or {}).get("heading")
    content = chunk.content or ""
    if heading and content.startswith(heading + "\n"):
        return content[len(heading) + 1:], heading
    return content, heading


def children_for(chunk: KnowledgeChunk, s, title: str | None) -> list[Chunk]:
    body, heading = body_of(chunk)
    parent = Chunk(chunk.chunk_index, body, heading, {"kind": (chunk.meta or {}).get("kind")})
    # Same settings -> same children as ingestion would cut (one shared source of kwargs).
    return split_children(parent, **child_split_kwargs(s, title=title))


async def embed_all(texts: list[str]) -> list[list[float] | None]:
    """Same resilient batching as ingestion (a failed batch is retried text by text; one slow
    response never loses a whole document). None for any text that could not be embedded."""
    res = await embed_resilient(get_embeddings(), texts)
    if res.retried_batches:
        print(f"    {res.retried_batches} embedding batch(es) retried one at a time; "
              f"{res.failed} text(s) could not be embedded")
    return res.vectors


async def _documents(db, tenant: uuid.UUID | None, document: uuid.UUID | None):
    q = select(KnowledgeDocument)
    if tenant:
        q = q.where(KnowledgeDocument.tenant_id == tenant)
    if document:
        q = q.where(KnowledgeDocument.id == document)
    return (await db.execute(q.order_by(KnowledgeDocument.tenant_id,
                                        KnowledgeDocument.title))).scalars().all()


async def _chunks(db, doc: KnowledgeDocument) -> list[KnowledgeChunk]:
    return (await db.execute(select(KnowledgeChunk).where(
        KnowledgeChunk.tenant_id == doc.tenant_id, KnowledgeChunk.document_id == doc.id)
        .order_by(KnowledgeChunk.chunk_index))).scalars().all()


async def _reset_document(db, doc: KnowledgeDocument) -> int:
    """Delete a document's children and turn its parents back into standalone chunks."""
    removed = (await db.execute(delete(KnowledgeChunk).where(
        KnowledgeChunk.tenant_id == doc.tenant_id, KnowledgeChunk.document_id == doc.id,
        KnowledgeChunk.chunk_role == ROLE_CHILD))).rowcount or 0
    await db.execute(update(KnowledgeChunk).where(
        KnowledgeChunk.tenant_id == doc.tenant_id, KnowledgeChunk.document_id == doc.id,
        KnowledgeChunk.chunk_role == ROLE_PARENT).values(chunk_role=ROLE_STANDALONE))
    return removed


async def backfill(args) -> int:
    s = get_settings()
    price = _PRICE_PER_M.get(s.embedding_model)
    total_children = total_chars = converted = skipped = failed_embeds = 0
    touched_tenants: set[uuid.UUID] = set()

    async with get_session_factory()() as db:
        docs = await _documents(db, args.tenant, args.document)
        print(f"\n  {len(docs)} document(s); child target={s.chunk_child_target_chars} chars, "
              f"list group={s.chunk_child_list_group_size}, overlap={s.chunk_child_overlap_chars}, "
              f"split>{s.chunk_child_split_min_chars}, title={s.chunk_child_title_prefix}, "
              f"records={s.chunk_child_record_split}"
              f"  [{'APPLY' if args.apply else 'DRY RUN - nothing is written'}]\n")
        for doc in docs:
            chunks = await _chunks(db, doc)
            roles = defaultdict(int)
            for c in chunks:
                roles[c.chunk_role] += 1
            if (roles[ROLE_PARENT] or roles[ROLE_CHILD]) and not args.force:
                print(f"  skip  {doc.title[:50]:52} already has {roles[ROLE_PARENT]} parents / "
                      f"{roles[ROLE_CHILD]} children (use --force to rebuild)")
                skipped += 1
                continue

            plan = []                                   # (standalone chunk, its children)
            for c in chunks:
                if c.chunk_role == ROLE_CHILD:
                    continue                            # --force: rebuilt from the parent below
                kids = children_for(c, s, doc.title)
                if kids:
                    plan.append((c, kids))
            n_kids = sum(len(k) for _, k in plan)
            chars = sum(len(k.embed_text) for _, kids in plan for k in kids)
            total_children += n_kids
            total_chars += chars
            print(f"  {'split' if plan else 'keep '} {doc.title[:50]:52} "
                  f"{len([c for c in chunks if c.chunk_role != ROLE_CHILD]):>3} chunks -> "
                  f"{len(plan):>3} parents, {n_kids:>3} children")
            if args.show:
                for c, kids in plan:
                    print(f"        parent #{c.chunk_index} ({len(c.content)} chars)")
                    for k in kids:
                        print(f"          child {k.index}: {k.content[:110]!r}")
            if not args.apply or not plan:
                continue

            try:
                if args.force:
                    await _reset_document(db, doc)
                vectors = await embed_all([k.embed_text for _, kids in plan for k in kids])
                it = iter(vectors)
                for parent, kids in plan:
                    kept = [(k, v) for k, v in ((k, next(it)) for k in kids) if v is not None]
                    failed_embeds += len(kids) - len(kept)
                    if not kept:
                        continue                        # stays standalone; still fully searchable
                    parent.chunk_role = ROLE_PARENT
                    parent.meta = {**(parent.meta or {}), "children": len(kept)}
                    for k, v in kept:
                        db.add(KnowledgeChunk(
                            id=uuid.uuid4(), tenant_id=parent.tenant_id,
                            document_id=parent.document_id, project_id=parent.project_id,
                            chunk_index=parent.chunk_index, content=k.embed_text, embedding=v,
                            token_count=len(k.content) // 4, chunk_role=ROLE_CHILD,
                            parent_id=parent.id,
                            meta={**{key: val for key, val in (parent.meta or {}).items()
                                     if key != "children"},
                                  "child_index": k.metadata.get("child_index")}))
                await db.commit()                       # one document = one transaction
                converted += 1
                touched_tenants.add(doc.tenant_id)
            except Exception as e:  # noqa: BLE001 -- one bad document must not stop the run
                await db.rollback()
                print(f"        FAILED, rolled back: {e.__class__.__name__}: {e}")

    for t in touched_tenants:
        await invalidate_tenant_knowledge_cache(t)
    tokens = total_chars // 4
    cost = f"~${tokens * price / 1_000_000:.5f}" if price is not None else "(unknown model price)"
    print(f"\n  children: {total_children}  (~{tokens:,} tokens, {cost} with {s.embedding_model})")
    if args.apply:
        print(f"  converted {converted} document(s), skipped {skipped}, "
              f"child embeddings failed {failed_embeds}")
        print("  Next: set RETRIEVAL_GRANULARITY=parent_child (or A/B it with "
              "scripts/eval_parent_child.py) — children are not searched until then.\n")
    else:
        print("  Dry run. Re-run with --apply to write.\n")
    return 0


async def revert(args) -> int:
    touched: set[uuid.UUID] = set()
    async with get_session_factory()() as db:
        docs = await _documents(db, args.tenant, args.document)
        total = 0
        for doc in docs:
            n = (await db.execute(select(KnowledgeChunk.id).where(
                KnowledgeChunk.tenant_id == doc.tenant_id, KnowledgeChunk.document_id == doc.id,
                KnowledgeChunk.chunk_role == ROLE_CHILD))).scalars().all()
            if not n:
                continue
            total += len(n)
            print(f"  {'revert' if args.apply else 'would revert'} {doc.title[:50]:52} "
                  f"{len(n)} children")
            if args.apply:
                await _reset_document(db, doc)
                await db.commit()
                touched.add(doc.tenant_id)
    for t in touched:
        await invalidate_tenant_knowledge_cache(t)
    print(f"\n  {total} children {'removed' if args.apply else 'would be removed'}; "
          f"parents are standalone again.\n")
    return 0


async def verify(args) -> int:
    """Integrity: every parent has children, every child's parent is a parent, and COVERAGE —
    every sentence / line of a parent's text is present in at least one of its children (in
    parent_child mode text no child carries cannot be found)."""
    problems = 0
    async with get_session_factory()() as db:
        docs = await _documents(db, args.tenant, args.document)
        for doc in docs:
            chunks = await _chunks(db, doc)
            by_id = {c.id: c for c in chunks}
            kids: dict[uuid.UUID, list[KnowledgeChunk]] = defaultdict(list)
            for c in chunks:
                if c.chunk_role == ROLE_CHILD:
                    parent = by_id.get(c.parent_id)
                    if parent is None or parent.chunk_role != ROLE_PARENT:
                        problems += 1
                        print(f"  BAD  {doc.title[:40]}: child {c.id} -> parent "
                              f"{c.parent_id} is {'missing' if parent is None else parent.chunk_role}")
                    else:
                        kids[parent.id].append(c)
                    if c.embedding is None:
                        problems += 1
                        print(f"  BAD  {doc.title[:40]}: child {c.id} has no embedding")
            for c in chunks:
                if c.chunk_role != ROLE_PARENT:
                    continue
                if not kids[c.id]:
                    problems += 1
                    print(f"  BAD  {doc.title[:40]}: parent {c.id} has no children")
                    continue
                body, heading = body_of(c)
                parent = Chunk(c.chunk_index, body, heading, {"kind": (c.meta or {}).get("kind")})
                missing = uncovered_text(parent, [k.content for k in kids[c.id]])
                if missing:
                    problems += 1
                    print(f"  BAD  {doc.title[:40]}: parent #{c.chunk_index} text not in any "
                          f"child: {missing[0][:80]!r}")
    print(f"\n  verify: {'OK' if not problems else f'{problems} problem(s)'} "
          f"across {len(docs)} document(s)\n")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild children for documents that already have them")
    ap.add_argument("--revert", action="store_true", help="remove children, parents -> standalone")
    ap.add_argument("--verify", action="store_true", help="integrity + coverage check only")
    ap.add_argument("--tenant", type=uuid.UUID)
    ap.add_argument("--document", type=uuid.UUID)
    ap.add_argument("--show", action="store_true", help="print every planned child")
    args = ap.parse_args()
    if args.verify:
        return asyncio.run(verify(args))
    if args.revert:
        return asyncio.run(revert(args))
    return asyncio.run(backfill(args))


if __name__ == "__main__":
    sys.exit(main())
