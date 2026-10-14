#!/usr/bin/env python3
"""What the knowledge base actually contains.

    docker compose exec core python scripts/show_knowledge.py            # summary
    docker compose exec core python scripts/show_knowledge.py --full     # every chunk
    docker compose exec core python scripts/show_knowledge.py --grep bhk # find a phrase

Reads the database directly. No model calls, so it costs nothing to run.

Worth doing when the agent says something the documents do not support. Asked what sizes
were available, it answered "2, 3 and 4 BHK" — there is no 4 BHK. Either the chunk holding
the configurations was never ingested, or it was and retrieval did not reach it. Those are
different problems with different fixes, and this tells you which.
"""
import argparse
import asyncio
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import KnowledgeChunk, KnowledgeDocument, Tenant  # noqa: E402


async def main(args) -> int:
    async with get_session_factory()() as db:
        tenant = (await db.execute(select(Tenant))).scalars().first()
        if tenant is None:
            print("No tenant. Run seed.py first.")
            return 1

        docs = (await db.execute(select(KnowledgeDocument).where(
            KnowledgeDocument.tenant_id == tenant.id
        ).order_by(KnowledgeDocument.title))).scalars().all()

        if not docs:
            print("\n  Nothing ingested. The agent has no documents to answer from —\n"
                  "  every knowledge question will be a guess.\n")
            return 0

        print(f"\n  {len(docs)} document(s) for {tenant.name}\n")
        print(f"  {'title':38} {'type':10} {'chunks':>7} {'embedded':>9}  effective")
        print("  " + "-" * 86)

        total_chunks = total_embedded = 0
        for d in docs:
            n = (await db.execute(select(func.count()).select_from(KnowledgeChunk).where(
                KnowledgeChunk.document_id == d.id))).scalar() or 0
            # A chunk with no vector is invisible to semantic search. It can still be found
            # by keyword, which is why partial ingestion shows up as "sometimes it knows".
            embedded = (await db.execute(select(func.count()).select_from(KnowledgeChunk).where(
                KnowledgeChunk.document_id == d.id,
                KnowledgeChunk.embedding.isnot(None)))).scalar() or 0
            total_chunks += n
            total_embedded += embedded
            window = "-".join(str(x) if x else "…" for x in (d.effective_from, d.effective_to))
            # Zero chunks is not "not all embedded" — it is a document the agent cannot
            # see at all, which is the difference between a weak answer and no answer.
            flag = ("  <-- NO CHUNKS, invisible to the agent" if n == 0
                    else "" if embedded == n else "  <-- not all embedded")
            print(f"  {d.title[:37]:38} {d.doc_type:10} {n:>7} {embedded:>9}  "
                  f"{window}{flag}")

        print(f"\n  {total_chunks} chunks, {total_embedded} embedded\n")
        empty = [d.title for d in docs if (await db.execute(
            select(func.count()).select_from(KnowledgeChunk).where(
                KnowledgeChunk.document_id == d.id))).scalar() == 0]
        if empty:
            print("  These have NO chunks — the agent cannot answer from them at all:")
            for t in empty:
                print(f"    {t}")
            print("  Re-run scripts/seed_knowledge_dsr.py once credits are available.\n")

        if total_embedded < total_chunks:
            print("  Chunks without a vector cannot be found by meaning, only by exact\n"
                  "  words. Re-run ingestion for those documents.\n")

        if args.grep:
            needle = args.grep.lower()
            hits = (await db.execute(select(KnowledgeChunk, KnowledgeDocument).join(
                KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id).where(
                KnowledgeChunk.tenant_id == tenant.id,
                func.lower(KnowledgeChunk.content).contains(needle)))).all()
            print(f"  {len(hits)} chunk(s) containing {args.grep!r}\n")
            for chunk, doc in hits[:args.limit]:
                print(f"  ── {doc.title} · chunk {chunk.chunk_index}")
                for line in textwrap.wrap(chunk.content, 84):
                    print(f"     {line}")
                print()
            if not hits:
                print("  Nothing. If the agent has been saying it, it is inventing it.\n")

        elif args.full:
            for d in docs:
                chunks = (await db.execute(select(KnowledgeChunk).where(
                    KnowledgeChunk.document_id == d.id
                ).order_by(KnowledgeChunk.chunk_index))).scalars().all()
                print(f"\n{'=' * 88}\n  {d.title}  ({d.doc_type})\n{'=' * 88}")
                for c in chunks[:args.limit]:
                    print(f"\n  ── chunk {c.chunk_index}  "
                          f"({c.token_count or '?'} tokens"
                          f"{', NO VECTOR' if c.embedding is None else ''})")
                    for line in textwrap.wrap(c.content, 84):
                        print(f"     {line}")
            print()
        else:
            print("  --full to print every chunk, --grep <word> to search them.\n")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="print every chunk")
    ap.add_argument("--grep", default="", help="find chunks containing this text")
    ap.add_argument("--limit", type=int, default=40, help="max chunks to print")
    sys.exit(asyncio.run(main(ap.parse_args())))