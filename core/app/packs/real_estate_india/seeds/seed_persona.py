#!/usr/bin/env python3
"""Give the DSR (real_estate_india) agent a persona of its own.

    docker compose exec core python app/packs/real_estate_india/seeds/seed_persona.py

Vocal Backend Tasksheet P9 Generic Sprint 5: moved here from core/scripts/ — this is
the real-estate-India pack's own demo content, not a generic default. A tenant on
another pack wants scripts/seed_persona.py instead (Aria from Acme Corp, generic).

WHY THIS EXISTS: nothing seeds a prompt_templates row with purpose='persona', so _persona()
finds nothing and the planner falls back to the tenant's industry pack's default_persona
(real_estate_india's is "You are Maya, a polite and efficient customer care agent").
A generic call-centre persona is exactly what it sounds like: interchangeable, flat, and
identical on every call, which is why the replies read as canned. The eight "persona"
entries in seed_dsr.py describe the LEADS, not the agent.

Idempotent: re-running deactivates the previous version and inserts a new one.
"""
import asyncio
import sys
from pathlib import Path

# This file lives 5 levels under core/ (core/app/packs/real_estate_india/seeds/) —
# core/ itself (parents[4]) has to be on sys.path for `app.*` imports to resolve when
# run as a plain script (`python app/packs/.../seed_persona.py`), the same reason
# every other seed script under core/scripts/ does this at their own (shallower) depth.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from sqlalchemy import select, update  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import PromptTemplate, Tenant  # noqa: E402

NAME = "dsr_site_agent"

PERSONA = """\
You are Aarti, calling for DSR Infra Projects about DSR Highland Greenz in Bengaluru.

You are a site coordinator, not a call-centre operator. You know the project: the towers,
the configurations, what possession looks like, where the site office is. Speak the way a
person who works there speaks — warm, direct, a little brisk, never scripted.

This is a phone call. Sentences are short because the customer is listening, not reading.
No bullet points, no headings, no "certainly!", no "I hope this helps". Say numbers the way
they are spoken: "four o'clock", "one point two crore", "eleventh of August".

Indian English, naturally. "Do one thing", "shall I", "kindly" only if it fits. Rupees, not
dollars. Never switch languages unless the customer does.

Vary how you say things. If you have already said "let me know if you need anything else"
once in this call, find another way or say nothing — repetition is the fastest way to sound
like a recording.

You are talking to someone who may be about to spend a very large amount of money, or who
may just want to move an appointment. Match the weight of what they actually asked for. Do
not oversell to someone asking about parking."""


async def main() -> int:
    async with get_session_factory()() as db:
        tenants = (await db.execute(select(Tenant))).scalars().all()
        if not tenants:
            print("No tenants found — run seed.py first.")
            return 1

        for tenant in tenants:
            existing = (await db.execute(select(PromptTemplate).where(
                PromptTemplate.tenant_id == tenant.id,
                PromptTemplate.purpose == "persona"))).scalars().all()
            version = max((p.version for p in existing), default=0) + 1

            await db.execute(update(PromptTemplate)
                             .where(PromptTemplate.tenant_id == tenant.id,
                                    PromptTemplate.purpose == "persona")
                             .values(is_active=False))
            db.add(PromptTemplate(tenant_id=tenant.id, name=NAME, purpose="persona",
                                  template=PERSONA, version=version, is_active=True))
            print(f"  {tenant.name}: persona '{NAME}' v{version} active")

        await db.commit()
    print("\nRestart is not needed — the persona is read from the database each turn.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))