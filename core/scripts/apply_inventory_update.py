#!/usr/bin/env python3
"""One-shot, idempotent deploy step for the structured-inventory feature.

    docker compose exec core python scripts/apply_inventory_update.py
    docker compose exec core python scripts/apply_inventory_update.py --dry-run

Run this ONCE on a deployed VM after pulling the code. It brings the database in line with the
inventory_query feature, doing every DB change the feature needs and nothing else. Safe to
re-run: every step checks state first and only changes what is out of date.

WHAT IT DOES, in order:
  1. alembic upgrade head        — creates the `units` table (migration 0009) if missing.
  2. tool definitions            — upserts the built-in tool contracts for EVERY existing
                                   tenant, so `inventory_query` reaches the planner. Never
                                   creates a tenant (that is the footgun in seed.py: a wrong
                                   --tenant slug spawns a stray tenant + duplicate admin user).
  3. units                       — (re)seeds the DSR cost-sheet rows + the unpriced 1 BHK/1.5
                                   BHK and Lemon Tree configs, for every tenant that has the
                                   'dsr-highland-greenz' project. Replace-all per tenant.
  4. pricing-doc de-fang         — surgically rewrites the "cheapest published unit" paragraph
                                   of the DSR pricing document (the retrieval magnet that used
                                   to be parroted as "nothing under one crore") and re-embeds
                                   THAT ONE document. Does not touch any other document, so
                                   Lemon Tree knowledge is left intact and nothing else is
                                   re-embedded. Skipped when already de-fanged.

It does NOT run the full knowledge seeders (seed_knowledge_dsr.py is replace-all and would
wipe Lemon Tree; this script avoids that entirely) and it does NOT touch demo leads, visits,
rules or personas.

REQUIREMENTS on the VM: the DB is reachable (DATABASE_URL) and the embeddings provider is
configured (step 4 re-embeds one document). Run it from inside the core container so alembic
and the app package are importable.
"""
import argparse
import asyncio
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete, select  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import (KnowledgeChunk, KnowledgeDocument, Project,  # noqa: E402
                           Tenant, ToolDefinition, Unit)
from app.engine.rag.ingestion import ingest_document  # noqa: E402
from app.engine.tools.definitions import BUILTIN_TOOLS  # noqa: E402
from scripts.seed_knowledge_dsr import DOCUMENTS as DSR_DOCUMENTS  # noqa: E402
from scripts.seed_units import (DSR_TOWER, DSR_UNITS, LEMON_UNITS, _rows)  # noqa: E402

DSR_SLUG = "dsr-highland-greenz"
LEMON_SLUG = "lemon-tree"
PRICING_TITLE = "Which Prices We Can Quote"

# The canonical, de-fanged pricing document text is the single source of truth in
# seed_knowledge_dsr.py (a KnowledgeDocument row stores NO body — only its chunks do — so we
# re-ingest from the authoritative script content rather than trying to reconstruct it).
_PRICING_DOC = next((d for d in DSR_DOCUMENTS if d["title"] == PRICING_TITLE), None)
PRICING_CONTENT = (_PRICING_DOC or {}).get("content", "")

# The tell-tale of the OLD, un-de-fanged copy, checked against the stored CHUNKS.
_TRIGGER_RE = re.compile(r"nothing published comes in under|under one crore", re.IGNORECASE)


def _run_migration(dry: bool) -> None:
    print("\n[1/4] alembic upgrade head")
    if dry:
        print("      (dry-run) would run: alembic upgrade head")
        return
    # WORKDIR in the container is /app/core, where alembic.ini lives.
    res = subprocess.run(["alembic", "upgrade", "head"],
                         cwd=str(Path(__file__).resolve().parent.parent),
                         capture_output=True, text=True)
    tail = (res.stdout + res.stderr).strip().splitlines()[-3:]
    for line in tail:
        print("      " + line)
    if res.returncode != 0:
        print("      ! alembic failed — aborting")
        sys.exit(1)


async def _upsert_tools(db, dry: bool) -> None:
    print("\n[2/4] tool definitions (upsert built-ins for every existing tenant)")
    tenants = (await db.execute(select(Tenant))).scalars().all()
    if not tenants:
        print("      no tenants found — nothing to do")
        return
    for tenant in tenants:
        existing = {t.name: t for t in (await db.execute(select(ToolDefinition).where(
            ToolDefinition.tenant_id == tenant.id))).scalars().all()}
        added = updated = 0
        for spec in BUILTIN_TOOLS:
            row = existing.get(spec["name"])
            if row is None:
                added += 1
                if not dry:
                    db.add(ToolDefinition(tenant_id=tenant.id, is_enabled=True, **spec))
            else:
                # Keep every field in step with code (esp. inventory_query's contract).
                changed = any(getattr(row, k) != v for k, v in spec.items())
                if changed:
                    updated += 1
                    if not dry:
                        for k, v in spec.items():
                            setattr(row, k, v)
        has_inv = "inventory_query" in existing or any(
            s["name"] == "inventory_query" for s in BUILTIN_TOOLS)
        print(f"      {tenant.slug:22} +{added} new, ~{updated} updated"
              f"  (inventory_query present: {has_inv})")


async def _seed_units(db, dry: bool) -> None:
    print("\n[3/4] units (per tenant that has the DSR project)")
    projects = (await db.execute(select(Project).where(
        Project.slug == DSR_SLUG))).scalars().all()
    if not projects:
        print(f"      no '{DSR_SLUG}' project on any tenant — skipping (run seed_projects.py "
              f"first if this tenant should have inventory)")
        return
    for dsr in projects:
        lemon = (await db.execute(select(Project).where(
            Project.tenant_id == dsr.tenant_id, Project.slug == LEMON_SLUG))).scalar_one_or_none()
        rows = _rows(DSR_UNITS, dsr.id, dsr.tenant_id, "DSR cost sheets", DSR_TOWER)
        if lemon is not None:
            rows += _rows(LEMON_UNITS, lemon.id, dsr.tenant_id, "Lemon Tree brochure (no price)")
        priced = sum(1 for u in rows if u.all_in_total is not None)
        if dry:
            print(f"      tenant {dsr.tenant_id}: would replace with {len(rows)} unit(s) "
                  f"({priced} priced)")
            continue
        # Replace-all for THIS tenant only — units are derived from the cost sheets.
        await db.execute(delete(Unit).where(Unit.tenant_id == dsr.tenant_id))
        for u in rows:
            db.add(u)
        print(f"      tenant {dsr.tenant_id}: seeded {len(rows)} unit(s) ({priced} priced, "
              f"{len(rows) - priced} unpriced)")


async def _defang_pricing_doc(db, dry: bool) -> None:
    print("\n[4/4] pricing-doc de-fang (surgical, re-embeds one document)")
    if not PRICING_CONTENT or _TRIGGER_RE.search(PRICING_CONTENT):
        print("      ! canonical pricing content missing or still contains the trigger phrase "
              "in seed_knowledge_dsr.py — aborting this step (code is out of date)")
        return
    docs = (await db.execute(select(KnowledgeDocument).where(
        KnowledgeDocument.title == PRICING_TITLE))).scalars().all()
    if not docs:
        print(f"      no document titled '{PRICING_TITLE}' — skipping")
        return
    for doc in docs:
        # A KnowledgeDocument stores no body; decide by whether any of its CHUNKS still carry
        # the old trigger line.
        chunk_text = "\n".join((await db.execute(select(KnowledgeChunk.content).where(
            KnowledgeChunk.tenant_id == doc.tenant_id,
            KnowledgeChunk.document_id == doc.id))).scalars().all())
        if not _TRIGGER_RE.search(chunk_text):
            print(f"      tenant {doc.tenant_id}: already de-fanged — skipping")
            continue
        if dry:
            print(f"      tenant {doc.tenant_id}: would re-embed document {doc.id} from the "
                  f"canonical de-fanged text")
            continue
        # Re-ingest ONLY this document from the authoritative de-fanged content. project_id is
        # carried through by ingest_document, so isolation is preserved and nothing else moves.
        report = await ingest_document(db, tenant_id=doc.tenant_id, document_id=doc.id,
                                       text=PRICING_CONTENT)
        print(f"      tenant {doc.tenant_id}: re-embedded from canonical text ({report.status})")


async def main(dry: bool) -> None:
    _run_migration(dry)
    async with get_session_factory()() as db:
        await _upsert_tools(db, dry)
        await _seed_units(db, dry)
        await _defang_pricing_doc(db, dry)
        if dry:
            print("\n(dry-run) no changes committed.")
        else:
            await db.commit()
            print("\nDone. inventory_query is live; units seeded; pricing prose de-fanged.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="show what each step would do, change nothing")
    args = ap.parse_args()
    asyncio.run(main(args.dry_run))
