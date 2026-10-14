#!/usr/bin/env python3
"""Stand up multi-project-per-tenant on the DSR test tenant, and backfill existing data.

    docker compose exec core python scripts/seed_projects.py               # set up + backfill
    docker compose exec core python scripts/seed_projects.py --default dsr # DSR is the fallback
    docker compose exec core python scripts/seed_projects.py --list        # show projects + counts

WHAT IT DOES (idempotent, DSR-safe):
  1. Creates two projects in the tenant — "DSR Highland Greenz" and "Lemon Tree" — each with
     its own facts, synonyms and name aliases. DSR's facts/synonyms are copied from
     tenant.settings (where the single-project build kept them).
  2. TAGS existing knowledge to a project so retrieval isolates them: every document whose
     title starts with the Lemon Tree prefix goes to the Lemon Tree project, everything else
     to the DSR project. Chunks are re-tagged from their document in one UPDATE (no re-embed).
  3. BACKFILLS existing appointments to the DSR project (every visit so far was DSR's).

WHY NO DEFAULT BY DEFAULT. Without ``--default`` neither project is the fallback, so a caller
who neither dials a project-specific number, nor has a project on their history, nor names one,
is ASKED which project they mean — instead of silently getting one. That is the behaviour the
multi-project work exists to produce. Pass ``--default dsr`` to restore "when in doubt, DSR".

This is a direct-DB script (like seed_dsr_settings.py): tenant settings and cross-table
backfills have no HTTP endpoint, and writing the rows directly mirrors the existing
idempotent seeding pattern.
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, text  # noqa: E402
from sqlalchemy.orm.attributes import flag_modified  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import (Appointment, KnowledgeDocument, Location, Project,  # noqa: E402
                           Tenant, User)

LEMON_PREFIX = "Lemon Tree — "   # matches seed_knowledge_lemontree.py

# Lemon Tree facts — brochure-traceable only, NO marketing/ROI promises (the agent's grounding
# guards forbid asserting those). Pricing is explicitly "not published" so nothing is invented.
LEMON_FACTS = [
    {"label": "Project", "value": "Lemon Tree by Legend, a residential and commercial "
     "development by Legend Properties"},
    {"label": "Location", "value": "Channasandra Junction, Whitefield, on the Whitefield - "
     "Channasandra Main Road, Bengaluru"},
    {"label": "Configurations", "value": "2 BHK, 2.5 BHK and 3 BHK apartments"},
    {"label": "Commercial", "value": "Commercial retail space on the lower floors "
     "(about 13,058 sq ft on the ground floor)"},
    {"label": "Pricing", "value": "No price list is published in the brochure — the sales "
     "team confirms current pricing"},
    {"label": "Contact", "value": "+91 8884441863 / 8884441836; lemontree.legendproperties.co.in"},
]

# Name aliases a caller might say for each project (lowercased; matched on word boundaries).
DSR_ALIASES = ["highland", "highland greenz", "greenz", "dsr highland", "dsr highland greenz"]
LEMON_ALIASES = ["lemon tree", "lemontree", "lemon", "legend"]


async def _get_tenant(db, tenant_slug, email):
    if tenant_slug:
        return (await db.execute(select(Tenant).where(
            Tenant.slug == tenant_slug))).scalar_one_or_none()
    user = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if user is None:
        return None
    return (await db.execute(select(Tenant).where(
        Tenant.id == user.tenant_id))).scalar_one_or_none()


async def _upsert_project(db, tenant, *, slug, name, facts, synonyms, aliases,
                          is_default, default_centre_id):
    proj = (await db.execute(select(Project).where(
        Project.tenant_id == tenant.id, Project.slug == slug))).scalar_one_or_none()
    if proj is None:
        proj = Project(tenant_id=tenant.id, slug=slug, name=name)
        db.add(proj)
    proj.name = name
    proj.status = "active"
    proj.facts = facts
    proj.synonyms = synonyms
    proj.aliases = aliases
    proj.is_default = is_default
    proj.default_centre_id = default_centre_id
    flag_modified(proj, "facts")
    flag_modified(proj, "synonyms")
    await db.flush()
    return proj


async def _list(db, tenant) -> None:
    projects = (await db.execute(select(Project).where(
        Project.tenant_id == tenant.id).order_by(Project.created_at))).scalars().all()
    print(f"\n  tenant {tenant.name} ({tenant.id}) — {len(projects)} project(s)\n")
    for p in projects:
        docs = (await db.execute(text(
            "SELECT count(*) FROM knowledge_documents WHERE project_id = :pid"),
            {"pid": str(p.id)})).scalar()
        appts = (await db.execute(text(
            "SELECT count(*) FROM appointments WHERE project_id = :pid"),
            {"pid": str(p.id)})).scalar()
        flag = "  [default]" if p.is_default else ""
        print(f"  {p.name:24} slug={p.slug:18} docs={docs:<3} appts={appts:<3}"
              f" facts={len(p.facts or [])}{flag}")
    untagged = (await db.execute(text(
        "SELECT count(*) FROM knowledge_documents WHERE tenant_id = :tid AND project_id IS NULL"),
        {"tid": str(tenant.id)})).scalar()
    print(f"\n  {untagged} document(s) still tenant-wide (project_id NULL)\n")


async def main(tenant_slug, email, default, do_list) -> None:
    async with get_session_factory()() as db:
        tenant = await _get_tenant(db, tenant_slug, email)
        if tenant is None:
            who = f"slug '{tenant_slug}'" if tenant_slug else f"user '{email}'"
            print(f"! no tenant found for {who} — run seed.py / seed_dsr.py first")
            sys.exit(1)

        if do_list:
            await _list(db, tenant)
            return

        tsettings = dict(tenant.settings or {})
        dsr_facts = [f for f in (tsettings.get("project_facts") or [])
                     if isinstance(f, dict) and f.get("label") and f.get("value")]
        base_synonyms = tsettings.get("retrieval_synonyms") or []

        # DSR's default site office, if the tenant has exactly one active centre — so a DSR
        # booking can take the centre from the project without asking.
        centres = (await db.execute(select(Location).where(
            Location.tenant_id == tenant.id, Location.is_active.is_(True)))).scalars().all()
        dsr_centre = centres[0].id if len(centres) == 1 else None

        dsr = await _upsert_project(
            db, tenant, slug="dsr-highland-greenz", name="DSR Highland Greenz",
            facts=dsr_facts, synonyms=base_synonyms, aliases=DSR_ALIASES,
            is_default=(default == "dsr"), default_centre_id=dsr_centre)
        lemon = await _upsert_project(
            db, tenant, slug="lemon-tree", name="Lemon Tree",
            facts=LEMON_FACTS, synonyms=base_synonyms, aliases=LEMON_ALIASES,
            is_default=(default == "lemon"), default_centre_id=None)

        # Tag documents: Lemon Tree by its title prefix, everything else to DSR.
        docs = (await db.execute(select(KnowledgeDocument).where(
            KnowledgeDocument.tenant_id == tenant.id))).scalars().all()
        dsr_docs = lemon_docs = 0
        for d in docs:
            d.project_id = lemon.id if d.title.startswith(LEMON_PREFIX) else dsr.id
            if d.project_id == lemon.id:
                lemon_docs += 1
            else:
                dsr_docs += 1

        # Re-tag chunks from their document in one pass — no re-embedding needed.
        await db.execute(text(
            "UPDATE knowledge_chunks c SET project_id = d.project_id "
            "FROM knowledge_documents d "
            "WHERE c.document_id = d.id AND c.tenant_id = :tid"),
            {"tid": str(tenant.id)})

        # Backfill existing appointments to DSR — every visit so far was DSR's.
        appt_rows = (await db.execute(select(Appointment).where(
            Appointment.tenant_id == tenant.id,
            Appointment.project_id.is_(None)))).scalars().all()
        for a in appt_rows:
            a.project_id = dsr.id

        await db.commit()
        print(f"\n  projects: {dsr.name} (default={dsr.is_default}), {lemon.name} "
              f"(default={lemon.is_default})")
        print(f"  documents tagged: DSR {dsr_docs}, Lemon Tree {lemon_docs}")
        print(f"  appointments backfilled to DSR: {len(appt_rows)}")
        if default is None:
            print("\n  No default project set — an unresolved caller will be ASKED which "
                  "project.\n  Pass --default dsr to fall back to DSR instead.\n")
        else:
            print(f"\n  Default project: {default}. Unresolved callers fall back to it.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", help="tenant slug (takes precedence over --email)")
    ap.add_argument("--email", default="rahul@advora.ai",
                    help="resolve the tenant via this user (default: the DSR admin)")
    ap.add_argument("--default", choices=["dsr", "lemon"], default=None,
                    help="which project is the fallback when resolution is ambiguous "
                         "(default: none — the agent asks)")
    ap.add_argument("--list", action="store_true", help="show projects and counts, then exit")
    args = ap.parse_args()
    asyncio.run(main(args.tenant, args.email, args.default, args.list))
