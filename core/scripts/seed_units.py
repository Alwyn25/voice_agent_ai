#!/usr/bin/env python3
"""Seed the structured unit inventory from the cost sheets.

    docker compose exec core python scripts/seed_units.py           # replace all units
    docker compose exec core python scripts/seed_units.py --list     # show what is seeded

Every DSR row here is traceable to the same two source documents scripts/seed_knowledge_dsr.py
was built from (the June brochure and the 11,499 cost sheet) — the SAME figures, now as
queryable rows so inventory_query can compare a caller's budget against them instead of a
model guessing over prose. The un-priced configurations (DSR's 1 BHK / 1.5 BHK, every Lemon
Tree config, whose brochure publishes no price) are seeded too, with price_published=False, so
"what does a 1 BHK cost?" is answered honestly rather than by reaching for the 2 BHK rate.

Direct-DB, replace-all, idempotent — like seed_projects.py. Units are derived data; re-running
rebuilds them from this file, so this file is the source of truth for the inventory.
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete, select  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import Project, Tenant, Unit, User  # noqa: E402

# (bhk_type, sqft, facing, carpet, rate_psf, flat_cost, all_in_total). rate/flat/total None =>
# exists but unpriced (price_published False).
DSR_UNITS = [
    # 2 BHK — 11,000 psf cost sheet
    ("2 BHK", 875, "East and West", 592, 11000, 9625000, 10393750),
    ("2 BHK", 880, "North", 595, 11000, 9680000, 10450000),
    ("2 BHK", 1099, "North", 728, 11000, 12089000, 12913750),
    ("2 BHK", 1100, "North", 728, 11000, 12100000, 12925000),
    ("2 BHK", 1103, "West", 726, 11000, 12133000, 12958750),
    ("2 BHK", 1104, "East", 716, 11000, 12144000, 12970000),
    ("2 BHK", 1105, "East and West", 726, 11000, 12155000, 12981250),
    ("2 BHK", 1175, "East and West", 768, 11000, 12925000, 13768750),
    ("2 BHK", 1210, "East and West", 796, 11000, 13310000, 14162500),
    # 2.5 BHK — 11,000 psf cost sheet
    ("2.5 BHK", 1325, "South", 859, 11000, 14575000, 15531250),
    ("2.5 BHK", 1330, "North", 850, 11000, 14630000, 15587500),
    # 3 BHK — 11,499 psf cost sheet
    ("3 BHK", 1239, "West", 808, 11499, 14247261, 15432011),
    ("3 BHK", 1390, "East and West", 923, 11499, 15983610, 17206110),
    ("3 BHK", 1560, "East and West", 1032, 11499, 17938440, 19253440),
    # Exists, no cost sheet — Tower D small formats. price_published False.
    ("1 BHK", 619, None, None, None, None, None),
    ("1 BHK", 625, None, None, None, None, None),
    ("1 BHK", 643, None, None, None, None, None),
    ("1 BHK", 646, None, None, None, None, None),
    ("1.5 BHK", 780, None, None, None, None, None),
]
# Tower D holds the small formats; the priced 2/2.5/3 BHK sit across several towers, so tower
# is left unset for them rather than picking one.
DSR_TOWER = {"1 BHK": "D", "1.5 BHK": "D"}

# Lemon Tree — configurations exist, brochure publishes NO price. Config-only rows.
LEMON_UNITS = [
    ("2 BHK", None, None, None, None, None, None),
    ("2.5 BHK", None, None, None, None, None, None),
    ("3 BHK", None, None, None, None, None, None),
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


def _rows(specs, project_id, tenant_id, source, tower_map=None):
    out = []
    for bhk, sqft, facing, carpet, rate, flat, total in specs:
        out.append(Unit(
            tenant_id=tenant_id, project_id=project_id, bhk_type=bhk,
            super_builtup_sqft=sqft, carpet_sqft=carpet, facing=facing,
            tower=(tower_map or {}).get(bhk),
            rate_psf=rate, flat_cost=flat, all_in_total=total,
            price_published=total is not None, availability="available",
            source_note=source))
    return out


async def _list(db, tenant):
    rows = (await db.execute(select(Unit).where(
        Unit.tenant_id == tenant.id).order_by(Unit.project_id, Unit.bhk_type,
                                              Unit.super_builtup_sqft))).scalars().all()
    print(f"\n  tenant {tenant.name} — {len(rows)} unit(s)\n")
    for u in rows:
        price = f"{u.all_in_total:>10,}" if u.all_in_total else "  (unpriced)"
        print(f"  {u.bhk_type:8} {str(u.super_builtup_sqft or '-'):>5} sqft  {price}  "
              f"{u.facing or ''}")
    print()


async def main(tenant_slug, email, do_list):
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

        # Replace-all: units are derived from this file, so rebuild rather than merge.
        await db.execute(delete(Unit).where(Unit.tenant_id == tenant.id))

        rows = _rows(DSR_UNITS, dsr.id, tenant.id, "DSR cost sheets", DSR_TOWER)
        if lemon is not None:
            rows += _rows(LEMON_UNITS, lemon.id, tenant.id, "Lemon Tree brochure (no price)")
        else:
            print("  (project 'lemon-tree' not found — skipping Lemon Tree units)")
        for u in rows:
            db.add(u)
        await db.commit()

        priced = sum(1 for u in rows if u.all_in_total is not None)
        print(f"\n  seeded {len(rows)} unit(s): {priced} priced, {len(rows) - priced} unpriced")
        print(f"  DSR: {len(DSR_UNITS)}  Lemon Tree: {len(rows) - len(DSR_UNITS)}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", help="tenant slug (takes precedence over --email)")
    ap.add_argument("--email", default="rahul@advora.ai",
                    help="resolve the tenant via this user (default: the DSR admin)")
    ap.add_argument("--list", action="store_true", help="show seeded units, then exit")
    args = ap.parse_args()
    asyncio.run(main(args.tenant, args.email, args.list))
