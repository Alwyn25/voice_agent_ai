#!/usr/bin/env python3
"""Offline smoke test for P1 (understanding routing) + P4-0 (stage profiler).

Runs the FULL process_turn pipeline against the live DB with LLM_PROVIDER=mock, so it needs
no API credits. Confirms the turn completes, retrieval routing behaves per the classifier,
and debug.timing.stages is populated with the per-stage breakdown.

    docker compose exec -T -e LLM_PROVIDER=mock core python scripts/smoke_p1_p4.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import Conversation, Customer  # noqa: E402
from app.engine.memory import working  # noqa: E402
from app.engine.planner.turn_runner import process_turn  # noqa: E402

UTTERANCES = [
    "i am looking for a flat to buy",       # product question -> should retrieve
    "i want something on 19th in the afternoon",  # scheduling -> should NOT retrieve
    "is my visit already scheduled",        # status -> should NOT retrieve
]


async def main() -> int:
    async with get_session_factory()() as db:
        cust = (await db.execute(select(Customer).limit(1))).scalar_one_or_none()
        if cust is None:
            print("No customers seeded — run scripts/seed_dsr.py first.")
            return 1

        conv = Conversation(tenant_id=cust.tenant_id, customer_id=cust.id,
                            channel="chat", direction="inbound", status="active")
        db.add(conv)
        await db.flush()
        await working.save(conv.id, {})
        print(f"lead: {cust.full_name}  ·  session: {conv.id}\n")

        for utt in UTTERANCES:
            result = await process_turn(db, tenant_id=cust.tenant_id, conversation=conv,
                                        user_input=utt, actor="smoke")
            dbg = result.debug or {}
            routing = dbg.get("routing") or {}
            stages = (dbg.get("timing") or {}).get("stages") or {}
            retrieved = len(dbg.get("knowledge_used") or [])
            print(f"> {utt}")
            print(f"  routing:   source={routing.get('source')} "
                  f"do_retrieve={routing.get('do_retrieve')} "
                  f"is_scheduling={routing.get('is_scheduling')} "
                  f"is_status={routing.get('is_status_question')}")
            print(f"  retrieved: {retrieved} chunk(s)")
            print(f"  stages_ms: {stages}")
            print(f"  reply:     {(result.response_text or '')[:90]}\n")

        # Leave no test conversation behind.
        conv.status = "ended"
        await db.commit()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
