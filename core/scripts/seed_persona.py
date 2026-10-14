#!/usr/bin/env python3
"""Give a generic-pack tenant's agent a persona of its own.

    docker compose exec core python scripts/seed_persona.py

Vocal Backend Tasksheet P9 Generic Sprint 5: this is now the GENERIC default (Aria
from Acme Corp) — the original real-estate content (Aarti/DSR Infra Projects/
Bengaluru) moved to app/packs/real_estate_india/seeds/seed_persona.py, since it was
never actually generic, just the only demo tenant this project has ever seeded.

WHY THIS EXISTS: nothing seeds a prompt_templates row with purpose='persona', so
_persona() finds nothing and the planner falls back to the tenant's industry pack's
own default_persona (app.packs.base.GENERIC_DEFAULT_PERSONA for a generic-pack
tenant) — flat and interchangeable by design, since it has to work for any business.
A named, specific persona seeded here is what makes a demo call sound like it is
coming from an actual person at an actual company instead.

Idempotent: re-running deactivates the previous version and inserts a new one. Only
touches tenants currently on the generic pack (see app.packs.PackRegistry) — running
this against a real_estate_india tenant would overwrite Aarti with a persona that
does not know anything about that tenant's actual business, so it deliberately
skips them rather than guessing.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, update  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import PromptTemplate, Tenant  # noqa: E402
from app.packs import PackRegistry  # noqa: E402

NAME = "generic_agent"

PERSONA = """\
You are Aria, calling on behalf of Acme Corp.

You are a helpful, capable representative of the business — not a call-centre script
reader. Speak the way a real person who works there speaks: warm, direct, and never
scripted.

This is a phone call. Sentences are short because the customer is listening, not
reading. No bullet points, no headings, no "certainly!", no "I hope this helps". Say
numbers the way they are spoken aloud.

Plain, natural English. Match the customer's own register rather than imposing a
house style. Never switch languages unless the customer does.

Vary how you say things. If you have already said "let me know if you need anything
else" once in this call, find another way or say nothing — repetition is the fastest
way to sound like a recording.

Match the weight of what the customer actually asked for. Do not oversell someone
asking a simple logistics question."""


async def main() -> int:
    async with get_session_factory()() as db:
        tenants = (await db.execute(select(Tenant))).scalars().all()
        if not tenants:
            print("No tenants found — run seed.py first.")
            return 1

        seeded = 0
        for tenant in tenants:
            pack = await PackRegistry.for_tenant(db, tenant.id)
            if pack.slug != "generic":
                print(f"  {tenant.name}: skipped (pack={pack.slug}, not generic)")
                continue

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
            seeded += 1

        await db.commit()
    if seeded == 0:
        print("\nNo generic-pack tenants found — nothing seeded. "
             "(A real_estate_india tenant wants app/packs/real_estate_india/seeds/seed_persona.py instead.)")
    else:
        print("\nRestart is not needed — the persona is read from the database each turn.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
