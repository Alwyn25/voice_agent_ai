"""Clear all demo prospect data (docs/gtm-demo-links.md §12).

Erases every company record with its people, link, visitors, calls, transcripts, leads and
whole demo tenant (app/demo/erase.py), plus all imports and batches. Keeps the industry
definitions and their dummy-data templates, so a workbook can be imported again straight
away. Customer tenants are never touched (erase refuses anything that isn't a demo tenant).

    python scripts/demo_clear.py            # shows what would be erased
    python scripts/demo_clear.py --yes      # erases it
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, func, select  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import DemoBatch, DemoImport, DemoLead, DemoProspect, Tenant  # noqa: E402
from app.db.models.platform import TENANT_KIND_DEMO  # noqa: E402
from app.demo.erase import erase_records  # noqa: E402


async def main(yes: bool) -> None:
    async with get_session_factory()() as db:
        count = lambda model: select(func.count()).select_from(model)  # noqa: E731
        before = {"records": (await db.execute(count(DemoProspect))).scalar_one(),
                  "leads": (await db.execute(count(DemoLead))).scalar_one(),
                  "imports": (await db.execute(count(DemoImport))).scalar_one(),
                  "batches": (await db.execute(count(DemoBatch))).scalar_one(),
                  "demo tenants": (await db.execute(count(Tenant).where(
                      Tenant.kind == TENANT_KIND_DEMO))).scalar_one()}
        print("Demo data now:", ", ".join(f"{v} {k}" for k, v in before.items()))
        if not yes:
            print("Nothing erased. Run again with --yes to erase all of it.")
            return
        prospects = (await db.execute(select(DemoProspect))).scalars().all()
        counts = await erase_records(db, list(prospects))
        await db.execute(delete(DemoLead))                    # leads sent without a link
        await db.execute(delete(DemoImport))
        await db.execute(delete(DemoBatch))
        orphans = (await db.execute(count(Tenant).where(Tenant.kind == TENANT_KIND_DEMO))).scalar_one()
        await db.commit()
        print(f"Erased {counts['records']} records, {counts['tenants']} demo tenants "
              f"({counts['rows']} tenant rows), all leads, imports and batches.")
        if orphans:
            print(f"Note: {orphans} demo tenant(s) without a record remain; they were not created "
                  "by an import and were left alone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="really erase")
    asyncio.run(main(ap.parse_args().yes))
