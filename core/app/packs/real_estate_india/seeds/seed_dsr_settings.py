"""Seed DSR's retrieval synonyms and curated project facts into ``tenant.settings``.

WHY THIS IS A SEPARATE, DIRECT-DB SCRIPT. Synonyms and project facts are tenant DATA, not
engine behaviour — the retrieval engine and the prompt builder read them generically from
``tenant.settings`` and carry no real-estate vocabulary of their own (see
``app.engine.rag.synonyms`` and the PROJECT FACTS block in ``prompts.build_system_prompt``).
The rest of the DSR fixture (``seed_dsr.py``) loads through the HTTP API, which has no
endpoint for tenant settings, so this one writes the settings row directly, mirroring the
idempotent direct-DB pattern of ``seed.py``.

WHAT IT FIXES. The values below close the Voice Agent Feedback vocabulary/retrieval gaps:
  * #3  daycare  → the brochure says "creche"
  * #5  generator → the brochure says "power backup" / "DG"
  * #6  bricks   → the spec says "RCC framed structure"
  * #7  location → answered vaguely though the brochure states it exactly
The facts are copied from the seeded DSR documents, so nothing here asserts anything the
knowledge base does not already contain — the block just guarantees it is always in front of
the model instead of depending on a keyword search to resurface it.

Usage (inside core/, DB up):
    python app/packs/real_estate_india/seeds/seed_dsr_settings.py                       # tenant of rahul@advora.ai
    python app/packs/real_estate_india/seeds/seed_dsr_settings.py --tenant advora       # by tenant slug
    python app/packs/real_estate_india/seeds/seed_dsr_settings.py --email rahul@advora.ai
Idempotent: re-running overwrites only these keys and leaves the rest of settings alone.
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from sqlalchemy import select  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import Tenant, User  # noqa: E402

# Each group lists surface phrases that mean the same thing. The customer may say any of
# them; the document uses one. Matching is word-level (see app.engine.rag.synonyms), so a
# multi-word phrase like "power backup" contributes both its words to the group.
RETRIEVAL_SYNONYMS = [
    # Feedback #5 — "is electricity generator present?" vs the brochure's "power backup"/"DG".
    ["generator", "dg", "dg set", "diesel generator", "power backup", "backup power",
     "power back-up", "genset"],
    # Feedback #3 — "daycare" vs the clubhouse list's "creche".
    ["creche", "crèche", "daycare", "day care", "childcare", "child care", "kids care"],
    # Feedback #6 — "are bricks used?" vs the spec's "RCC framed structure".
    ["bricks", "brick", "brickwork", "masonry", "rcc", "rcc structure", "rcc framed",
     "framed structure", "construction", "structure", "building material"],
    # Feedback #4 — "how many dog parks" vs amenity naming.
    ["dog park", "dog parks", "pet park", "pet area", "dog run", "pet zone"],
    # Feedback #2 — "park facing flat" vs floor-plan / view naming.
    ["park facing", "park-facing", "park view", "garden facing", "garden view",
     "green view"],
    # Common clubhouse/amenity vocabulary drift.
    ["gym", "gymnasium", "fitness centre", "fitness center"],
    ["swimming pool", "pool", "swimming"],
    ["clubhouse", "club house", "club"],
]

# Always-answerable, high-frequency facts, copied verbatim from the seeded DSR documents.
# Kept short and unambiguous: variant-specific pricing is deliberately left to retrieval,
# which has careful published/unpublished rules the pricing document enforces.
PROJECT_FACTS = [
    {"label": "Project", "value": "DSR Highland Greenz, a gated luxury-apartment community "
     "by DSR Infra Projects"},
    # Scope of what we offer — promoted to a curated fact so it is in context on EVERY turn, not
    # only when retrieval happens to pull the policy chunk. Without this, a "what's the rent?"
    # question was answered "we only sell" when the policy chunk was retrieved and "I don't have
    # that" when it was not — the same question, two answers, decided by retrieval luck. Stated as
    # a fact, the sale-only scope is grounded every time, so the answer is consistent (and the
    # scope_claim guard sees it and correctly stands down). Copied from the seeded policy document.
    {"label": "What we offer", "value": "For sale only — apartments are sold outright for "
     "ownership. We do not offer rentals, leases, paying-guest or short stays, resale of "
     "sold units, or commercial/office space"},
    {"label": "Location", "value": "Chikkanayakanahalli, off Sarjapur Road, Bengaluru "
     "562125"},
    {"label": "Structure", "value": "RCC framed structure designed for wind and seismic "
     "loads (not brick load-bearing)"},
    {"label": "Scale", "value": "10.10 acres; 4 towers across 8 wings; 900 flats; "
     "1 basement + ground + 12 floors; 3 clubhouses totalling 23,156 sq ft"},
    {"label": "Configurations", "value": "1, 1.5, 2, 2.5 and 3 BHK, 619 to 1,560 sq ft. "
     "No 4 BHK, villas, plots or penthouses"},
    {"label": "RERA", "value": "PRM/KA/RERA/1251/446/PR/110522/004876 (Karnataka RERA)"},
]

# Spoken in every DSR voice call's opening line, before the caller says anything (see
# app/api/sessions.with_recording_disclosure). Replaces the dsr_turn_policy rule
# recording_disclosure_first, which could never fire and never spoke the disclosure.
RECORDING_DISCLOSURE = "This call may be recorded for quality and training purposes."


async def main(tenant_slug: str | None, email: str | None) -> None:
    async with get_session_factory()() as db:
        tenant = None
        if tenant_slug:
            tenant = (await db.execute(select(Tenant).where(
                Tenant.slug == tenant_slug))).scalar_one_or_none()
        elif email:
            user = (await db.execute(select(User).where(
                User.email == email))).scalar_one_or_none()
            if user is not None:
                tenant = (await db.execute(select(Tenant).where(
                    Tenant.id == user.tenant_id))).scalar_one_or_none()

        if tenant is None:
            who = f"slug '{tenant_slug}'" if tenant_slug else f"user '{email}'"
            print(f"! no tenant found for {who} — run seed.py / seed_dsr.py first")
            sys.exit(1)

        # Merge, do not replace: settings may already hold unrelated keys, and clobbering
        # the whole JSONB blob to write these keys would silently drop them.
        settings = dict(tenant.settings or {})
        settings["retrieval_synonyms"] = RETRIEVAL_SYNONYMS
        settings["project_facts"] = PROJECT_FACTS
        settings["recording_disclosure"] = RECORDING_DISCLOSURE
        tenant.settings = settings
        # JSONB is a mutable column; reassigning the attribute is what flags it dirty, but
        # be explicit so a future refactor to in-place mutation still persists.
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(tenant, "settings")

        await db.commit()
        print(f"updated tenant {tenant.name} ({tenant.id})")
        print(f"  retrieval_synonyms: {len(RETRIEVAL_SYNONYMS)} groups")
        print(f"  project_facts:      {len(PROJECT_FACTS)} facts")
        print(f"  recording_disclosure: {RECORDING_DISCLOSURE!r}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tenant", help="tenant slug (takes precedence over --email)")
    p.add_argument("--email", default="rahul@advora.ai",
                   help="resolve the tenant via this user's membership (default: the DSR admin)")
    args = p.parse_args()
    asyncio.run(main(args.tenant, args.email))
