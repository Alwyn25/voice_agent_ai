#!/usr/bin/env python3
"""Seed the shareable-document catalogue the send_document tool delivers from.

    docker compose exec core python scripts/seed_documents.py            # replace all
    docker compose exec core python scripts/seed_documents.py --list      # show what is seeded

These are the files a caller can be sent on WhatsApp (floor plans, brochure, price sheet, …),
one row per document with the URL the provider delivers. Direct-DB, replace-all, idempotent —
like seed_units.py: this file is the source of truth for the catalogue, so re-running rebuilds it.

THE URLs ARE PLACEHOLDERS. Point file_url at real, publicly reachable files (the WhatsApp Cloud
API fetches them itself) before going live — or override the base with --base-url. Until a real
WhatsApp token is configured the send is stubbed and the URL is only logged, so these are safe
to seed as-is for a demo.
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete, select  # noqa: E402

from app.db.models import Project, ShareableDocument, Tenant, User  # noqa: E402

DEFAULT_BASE_URL = "https://assets.example.com/dsr"

# The one document type here that does NOT need a real uploaded file to be genuinely
# useful: unlike a floor plan or brochure (which needs an actual PDF somewhere), a location
# is just a Google Maps search URL, and that can be built from the centre's own address —
# already real, already seeded (see seed_centres.py) — right now rather than left as a
# placeholder like the rest of this catalogue until real files exist.
_DSR_SITE_OFFICE_ADDRESS = "Chikkanayakanahalli, Off Sarjapur Road, Bengaluru 562125"


def _maps_url(address: str) -> str:
    from urllib.parse import quote
    return f"https://www.google.com/maps/search/?api=1&query={quote(address)}"


# (doc_type, title, description, file_slug_or_full_url). For every type except
# location_map, this is a slug joined onto base_url in _rows(); for location_map it is
# already a complete, real, working URL and is used as-is.
DSR_DOCS = [
    ("floor_plan", "DSR Highland Greenz — 2 BHK Floor Plan",
     "2 BHK unit layouts and dimensions", "floor-plans/2bhk.pdf"),
    ("floor_plan", "DSR Highland Greenz — 2.5 BHK Floor Plan",
     "2.5 BHK unit layouts and dimensions", "floor-plans/2-5bhk.pdf"),
    ("floor_plan", "DSR Highland Greenz — 3 BHK Floor Plan",
     "3 BHK unit layouts and dimensions", "floor-plans/3bhk.pdf"),
    ("brochure", "DSR Highland Greenz — Project Brochure",
     "the full project brochure: amenities, specifications, gallery", "brochure.pdf"),
    ("price_sheet", "DSR Highland Greenz — Price Sheet",
     "current cost sheet and per-sqft rates", "price-sheet.pdf"),
    ("payment_plan", "DSR Highland Greenz — Payment Plan",
     "construction-linked payment schedule", "payment-plan.pdf"),
    ("location_map", "DSR Highland Greenz — Location Map",
     "Google Maps pin for the site office, for turn-by-turn navigation",
     _maps_url(_DSR_SITE_OFFICE_ADDRESS)),
    ("master_plan", "DSR Highland Greenz — Master Plan",
     "the layout of towers, amenities and open space", "master-plan.pdf"),
]

# Lemon Tree — a brochure exists; no price sheet is published (mirrors its unpriced units).
LEMON_DOCS = [
    ("brochure", "Lemon Tree — Project Brochure",
     "the Lemon Tree project brochure", "lemon-tree/brochure.pdf"),
    ("floor_plan", "Lemon Tree — Floor Plans",
     "unit layouts for all configurations", "lemon-tree/floor-plans.pdf"),
]


async def _get_tenant(db, tenant_slug, email):
    if tenant_slug:
        return (await db.execute(select(Tenant).where(
            Tenant.slug == tenant_slug))).scalar_one_or_none()
    user = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if user is None:
        return None
    return (await db.execute(select(Tenant).where(
        Tenant.id == user.tenant_id))).scalar_one_or_none()


async def _project(db, tenant, slug):
    return (await db.execute(select(Project).where(
        Project.tenant_id == tenant.id, Project.slug == slug))).scalar_one_or_none()


def _rows(specs, project_id, tenant_id, base_url):
    out = []
    for doc_type, title, desc, slug in specs:
        # location_map carries a real, complete Google Maps URL already (see _maps_url
        # above) rather than a placeholder slug to join onto base_url like the rest.
        is_full_url = slug.startswith("http://") or slug.startswith("https://")
        out.append(ShareableDocument(
            tenant_id=tenant_id, project_id=project_id, doc_type=doc_type,
            title=title, description=desc,
            file_url=slug if is_full_url else f"{base_url.rstrip('/')}/{slug}",
            file_name="location.html" if is_full_url else slug.rsplit("/", 1)[-1],
            mime_type="text/html" if is_full_url else "application/pdf",
            is_active=True))
    return out


async def _list(db, tenant):
    rows = (await db.execute(select(ShareableDocument).where(
        ShareableDocument.tenant_id == tenant.id).order_by(
        ShareableDocument.project_id, ShareableDocument.doc_type,
        ShareableDocument.title))).scalars().all()
    print(f"\n  tenant {tenant.name} — {len(rows)} document(s)\n")
    for d in rows:
        active = "" if d.is_active else "  (retired)"
        print(f"  {d.doc_type:14} {d.title}{active}\n                 -> {d.file_url}")
    print()


async def main(tenant_slug, email, base_url, do_list):
    from app.db.base import get_session_factory
    async with get_session_factory()() as db:
        tenant = await _get_tenant(db, tenant_slug, email)
        if tenant is None:
            who = f"slug '{tenant_slug}'" if tenant_slug else f"user '{email}'"
            print(f"! no tenant found for {who} — run seed_dsr.py / seed_projects.py first")
            sys.exit(1)

        if do_list:
            await _list(db, tenant)
            return

        dsr = await _project(db, tenant, "dsr-highland-greenz")
        lemon = await _project(db, tenant, "lemon-tree")
        if dsr is None:
            print("! project 'dsr-highland-greenz' not found — run seed_projects.py first")
            sys.exit(1)

        # Replace-all: the catalogue is derived from this file, so rebuild rather than merge.
        await db.execute(delete(ShareableDocument).where(
            ShareableDocument.tenant_id == tenant.id))

        rows = _rows(DSR_DOCS, dsr.id, tenant.id, base_url)
        if lemon is not None:
            rows += _rows(LEMON_DOCS, lemon.id, tenant.id, base_url)
        else:
            print("  (project 'lemon-tree' not found — skipping Lemon Tree documents)")
        for d in rows:
            db.add(d)
        await db.commit()

        print(f"\n  seeded {len(rows)} document(s)  (DSR: {len(DSR_DOCS)}, "
              f"Lemon Tree: {len(rows) - len(DSR_DOCS)})")
        print(f"  base url: {base_url}")
        if base_url == DEFAULT_BASE_URL:
            print("  NOTE: placeholder URLs — point file_url at real files (or pass "
                  "--base-url) before real WhatsApp sends.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", help="tenant slug (takes precedence over --email)")
    ap.add_argument("--email", default="rahul@advora.ai",
                    help="resolve the tenant via this user (default: the DSR admin)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL,
                    help="base URL the document file paths hang off")
    ap.add_argument("--list", action="store_true", help="show seeded documents, then exit")
    args = ap.parse_args()
    asyncio.run(main(args.tenant, args.email, args.base_url, args.list))
