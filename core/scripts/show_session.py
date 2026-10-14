#!/usr/bin/env python3
"""What the engine actually had in memory during the last call.

    docker compose exec core python scripts/show_session.py

No arguments — it finds the most recent conversation itself. Copying a UUID out of a
browser and into a shell is exactly the sort of step that goes wrong, and twice now the
placeholder was pasted literally and returned an empty dict for a session that never
existed.

The transcript shows what the agent SAID. This shows what it HAD. Five rounds of fixes
have been written from the first when only the second could have settled the question —
if caller_phone is missing here, the bug is in the console; if it is present, the bug is
in the engine, and they need completely different repairs.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import Conversation, Customer, Message  # noqa: E402
from app.engine.memory import working  # noqa: E402

INTERESTING = ("active_entities", "registration", "caller_phone", "wanted_tool",
               "pending_action", "identity_verified", "centre_id", "offered_slots",
               "appointment_id")


async def main(n: int) -> int:
    async with get_session_factory()() as db:
        convs = (await db.execute(select(Conversation)
                                  .order_by(Conversation.started_at.desc())
                                  .limit(n))).scalars().all()
        if not convs:
            print("\n  No conversations yet.\n")
            return 0

        for conv in convs:
            who = "unknown caller"
            if conv.customer_id:
                c = (await db.execute(select(Customer).where(
                    Customer.id == conv.customer_id))).scalar_one_or_none()
                who = c.full_name if c else str(conv.customer_id)

            msgs = (await db.execute(select(Message).where(
                Message.conversation_id == conv.id
            ).order_by(Message.created_at))).scalars().all()
            turns = [m for m in msgs if m.role == "customer"]

            print(f"\n{'=' * 78}")
            print(f"  {conv.started_at:%d %b %H:%M}  ·  {who}  ·  {conv.direction}  "
                  f"·  {len(turns)} customer turns")
            print(f"  session id: {conv.id}")
            print("=" * 78)

            session = await working.load(conv.id)
            if not session:
                print("\n  Working memory is empty — it expires an hour after the call.\n")
                continue

            print("\n  WORKING MEMORY\n")
            for key in INTERESTING:
                if key in session:
                    val = session[key]
                    if key == "offered_slots" and isinstance(val, list):
                        val = f"{len(val)} slot(s)" + (
                            f", first {val[0].get('when')}" if val else "")
                    print(f"    {key:20} {val}")
            other = {k: v for k, v in session.items() if k not in INTERESTING}
            if other:
                print("\n    everything else:")
                for k, v in other.items():
                    print(f"      {k:18} {v}")

            # The one question this script exists to answer.
            print()
            reg = session.get("registration")
            phone = session.get("caller_phone")
            if phone:
                print(f"    caller_phone IS set ({phone}). If create_customer still failed,")
                print(f"    the fault is in the engine, not the console.")
            else:
                print("    caller_phone is NOT set. The console did not send a phone when it")
                print("    opened this session, so create_customer had nothing to register")
                print("    with — the fix belongs in the UI, not the engine.")
            if reg:
                print(f"    registration state: {reg}")

            # What the call is about — the thing that steers retrieval on follow-up turns.
            ents = session.get("active_entities")
            if ents:
                print(f"\n    active_entities: {ents}")
                print("    A follow-up like \"tell me the cost\" will be searched with these")
                print("    values prepended, so it finds the variant they were discussing.")
            else:
                print("\n    active_entities is EMPTY. Follow-up turns are searched on their")
                print("    own words alone, so \"tell me the cost\" has no subject and the")
                print("    densest cost sheet wins. Either the planner emitted no entities")
                print("    or the merge is not running.")
    print()
    return 0


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sys.exit(asyncio.run(main(count)))