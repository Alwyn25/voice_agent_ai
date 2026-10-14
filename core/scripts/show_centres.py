#!/usr/bin/env python3
"""Every centre, and whether the id the engine is holding still resolves.

    docker compose exec core python scripts/show_centres.py

A booking fails with "Centre not found or inactive" for three quite different reasons —
the centre is inactive, it belongs to another tenant, or the id in working memory points
at a row that no longer exists because the demo data was reseeded. The message is the same
for all three, and guessing between them has cost several rounds.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import Conversation, Location, Tenant  # noqa: E402
from app.engine.memory import working  # noqa: E402


async def main() -> int:
    async with get_session_factory()() as db:
        tenants = (await db.execute(select(Tenant))).scalars().all()
        print(f"\n  {len(tenants)} tenant(s)")
        for t in tenants:
            print(f"    {t.id}  {t.name}")

        centres = (await db.execute(select(Location))).scalars().all()
        print(f"\n  {len(centres)} centre(s) across all tenants\n")
        print(f"  {'id':38} {'active':7} {'tenant':38} name")
        print("  " + "-" * 100)
        for c in centres:
            print(f"  {str(c.id):38} {str(c.is_active):7} {str(c.tenant_id):38} {c.name}")

        # What the most recent conversation is actually holding.
        conv = (await db.execute(select(Conversation)
                                 .order_by(Conversation.started_at.desc())
                                 .limit(1))).scalars().first()
        if conv is None:
            print()
            return 0

        session = await working.load(conv.id)
        held = session.get("centre_id")
        print(f"\n  Last conversation ({conv.started_at:%d %b %H:%M}) is holding:")
        print(f"    centre_id = {held}")
        if not held:
            print("    (nothing — the engine would have to look one up)")
            return 0

        match = next((c for c in centres if str(c.id) == str(held)), None)
        if match is None:
            print("\n    NO CENTRE HAS THAT ID. The session is holding an id from before the")
            print("    demo data was reseeded — seed_dsr deletes and recreates centres, and")
            print("    working memory outlives that by an hour. Every booking on this call")
            print("    will fail until the session expires or a fresh locate_centre runs.")
        elif not match.is_active:
            print(f"\n    That centre exists but is INACTIVE: {match.name}")
        elif str(match.tenant_id) != str(conv.tenant_id):
            print(f"\n    That centre belongs to a DIFFERENT TENANT: {match.tenant_id}")
        else:
            print(f"\n    Resolves correctly to {match.name}. If a booking still failed,")
            print("    the id reaching the check is not this one.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))