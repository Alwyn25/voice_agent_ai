#!/usr/bin/env python3
"""R5.08 — tag existing knowledge chunks with price / rate / area / BHK facts.

Chunks ingested before R5.08 have no ``metadata.facts``; retrieval treats them as neutral, so
the metadata boost does nothing for them until they are tagged. This script tags them from the
chunk's STORED text — the same text ingestion extracts from — so the result is identical to a
fresh ingest, with NO re-embedding, NO model calls and NO cost.

Safe by construction:
  * Dry run by default — prints what it WOULD change. Nothing is written without --apply.
  * Idempotent — only chunks with no facts, or facts from an older extractor version
    (enrichment.FACTS_VERSION), are touched. Re-running is a no-op. --force re-tags all.
  * Only the ``facts`` key is written (jsonb merge); heading, doc_type, title and kind are
    left exactly as they are. One transaction per tenant: a failure leaves that tenant as it was.

    docker compose exec core python scripts/backfill_chunk_metadata.py                 # dry run
    docker compose exec core python scripts/backfill_chunk_metadata.py --apply
    docker compose exec core python scripts/backfill_chunk_metadata.py --show-chunks   # per chunk
    docker compose exec core python scripts/backfill_chunk_metadata.py --tenant <uuid> --apply
    docker compose exec core python scripts/backfill_chunk_metadata.py --force --apply  # re-tag all

Also a coverage report: per document, how many chunks state a price / area / BHK. A pricing
document showing 0 priced chunks means the extractor does not understand its format — the
first thing to check when the metadata boost does not help a tenant.
"""
import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text as sql_text  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.engine.rag.enrichment import FACTS_VERSION, enrich_chunk_meta  # noqa: E402

SELECT_SQL = sql_text("""
    SELECT c.id, c.tenant_id, c.document_id, c.chunk_index, c.content,
           c.metadata->'facts' AS facts, d.title, d.doc_type
    FROM knowledge_chunks c
    JOIN knowledge_documents d ON d.id = c.document_id
    WHERE (CAST(:tenant_id AS uuid) IS NULL OR c.tenant_id = CAST(:tenant_id AS uuid))
    ORDER BY c.tenant_id, d.title, c.chunk_index
""")

# Merge ONLY the facts key into the existing metadata object.
UPDATE_SQL = sql_text("""
    UPDATE knowledge_chunks
    SET metadata = COALESCE(metadata, '{}'::jsonb)
                   || jsonb_build_object('facts', CAST(:facts AS jsonb))
    WHERE id = CAST(:id AS uuid)
""")


def _as_dict(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def _needs_tagging(existing: dict | None, force: bool) -> bool:
    if force or not existing:
        return True
    try:
        return int(existing.get("v", 0)) < FACTS_VERSION
    except (TypeError, ValueError):
        return True


def _fmt(facts: dict) -> str:
    parts = []
    if facts["bhk"]:
        parts.append(f"bhk={facts['bhk']}")
    if facts["prices"]:
        parts.append(f"price={facts.get('price_min')}..{facts.get('price_max')}"
                     f" ({len(facts['prices'])})")
    if facts["rates"]:
        parts.append(f"rate/sqft={facts['rates']}")
    if facts["areas"]:
        parts.append(f"area={facts.get('area_min')}..{facts.get('area_max')}"
                     f" ({len(facts['areas'])})")
    return " ".join(parts) or "-"


async def main(args) -> int:
    async with get_session_factory()() as db:
        rows = (await db.execute(SELECT_SQL, {"tenant_id": args.tenant})).mappings().all()
    if not rows:
        print("No knowledge chunks found" + (f" for tenant {args.tenant}" if args.tenant else ""))
        return 0

    by_tenant: dict[str, list] = defaultdict(list)
    for r in rows:
        by_tenant[str(r["tenant_id"])].append(r)

    grand = {"chunks": 0, "to_tag": 0, "written": 0}
    for tenant_id, trows in by_tenant.items():
        updates: list[dict] = []
        # doc title -> [chunks, price, area, bhk, untagged_before]
        per_doc: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
        doc_types: dict[str, str] = {}
        for r in trows:
            existing = _as_dict(r["facts"])
            facts = enrich_chunk_meta(r["content"] or "")
            stat = per_doc[r["title"]]
            doc_types[r["title"]] = r["doc_type"] or ""
            stat[0] += 1
            stat[1] += facts["has_price"]
            stat[2] += facts["has_area"]
            stat[3] += bool(facts["bhk"])
            stat[4] += existing is None
            if _needs_tagging(existing, args.force) and facts != existing:
                updates.append({"id": str(r["id"]), "facts": json.dumps(facts)})
            if args.show_chunks:
                mark = "*" if updates and updates[-1]["id"] == str(r["id"]) else " "
                print(f"  {mark} {r['title'][:40]:40} #{r['chunk_index']:<3} {_fmt(facts)}")

        print(f"\nTenant {tenant_id}: {len(trows)} chunks, {len(updates)} to tag "
              f"(extractor v{FACTS_VERSION})")
        print(f"  {'document':44} {'type':10} {'chunks':>6} {'price':>6} {'area':>6} "
              f"{'bhk':>6} {'untagged':>9}")
        for title, (n, price, area, bhk, untagged) in sorted(per_doc.items()):
            warn = "  <-- pricing doc with no priced chunk" \
                if doc_types[title] == "pricing" and price == 0 else ""
            print(f"  {title[:44]:44} {doc_types[title][:10]:10} {n:>6} {price:>6} {area:>6} "
                  f"{bhk:>6} {untagged:>9}{warn}")

        grand["chunks"] += len(trows)
        grand["to_tag"] += len(updates)
        if args.apply and updates:
            async with get_session_factory()() as db:
                try:
                    for u in updates:
                        await db.execute(UPDATE_SQL, u)
                    await db.commit()
                    grand["written"] += len(updates)
                    print(f"  wrote facts to {len(updates)} chunks")
                except Exception as e:  # noqa: BLE001 — report and leave the tenant untouched
                    await db.rollback()
                    print(f"  FAILED ({e.__class__.__name__}: {e}) — tenant rolled back")
                    return 1

    print(f"\nTotal: {grand['chunks']} chunks, {grand['to_tag']} needing tags, "
          f"{grand['written']} written.")
    if not args.apply and grand["to_tag"]:
        print("Dry run — nothing written. Re-run with --apply to tag.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write the facts (default: dry run)")
    ap.add_argument("--force", action="store_true",
                    help="re-tag every chunk, not only untagged / older-version ones")
    ap.add_argument("--tenant", help="limit to one tenant id (default: all tenants)")
    ap.add_argument("--show-chunks", action="store_true",
                    help="print the facts of every chunk (* = would be / was written)")
    sys.exit(asyncio.run(main(ap.parse_args())))
