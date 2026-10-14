"""The caller's chosen unit — as DURABLE memory, not a transient retrieval hint.

WHY THIS EXISTS. The engine already carries what the call is "about" in `active_entities`
(focus.py) — but that is a ranking nudge for the next search, deliberately soft and prone to
drift: it is capped, it is overwritten as the conversation moves through a booking, and it is
never written down. That is the right tool for "give me both cost breakdowns" and the wrong
tool for "show me the details of the flat I selected". A caller who chose the 3 BHK and booked
a visit for it made a DECISION; answering it from a hint that has since drifted to whatever
ranked highest is how they got quoted the 2.5 BHK cost sheet.

WHAT IT DOES. Three things, all deterministic (no model call):

  - `desired_unit(active_entities)` — pull a canonical configuration ("3 BHK", "2.5 BHK",
    "Studio") out of whatever slot names the planner happened to emit. Domain-light: it keys
    off the SHAPE of the value ("3 bhk"), not a fixed slot name, the same way focus.py blocks
    non-subjects rather than allow-listing subjects.

  - `remember_product_interest(...)` — persist that choice to BOTH stores the schema already
    provides: `CustomerProductInterest` (the structured, cross-call "wants the 3 BHK" row) and
    a single active `CustomerFact(product_interest / desired_unit)` (so it rides into the
    planner's fact list). Idempotent — re-stating the same unit writes nothing new; a genuinely
    different unit supersedes the old fact rather than piling up.

  - `load_active_preference(...)` — read the standing choice back for a caller who has NOT yet
    booked (once a visit is booked the unit is bound to the appointment itself, which is the
    stronger record and is surfaced from there).

The unit is bound to the appointment at book time from THIS value, plumbed through the tool
context — not asked of the model, which loses such things exactly as reliably as it loses a
centre id.
"""
import re
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CustomerFact, CustomerProductInterest
from app.packs.base import Vocabulary

# Vocal Backend Tasksheet P6 Generic Sprint 3: the fallback used when no per-tenant
# vocabulary is threaded through (e.g. a not-yet-updated caller) — reproduces the original
# hardcoded real-estate-India behaviour exactly (see RealEstateIndiaPack.vocabulary for the
# canonical, documented version of these same values).
_FALLBACK_VOCABULARY = Vocabulary(
    technical_terms=["BHK"],
    unit_patterns=[r"\b(\d(?:\.\d)?)\s*(?:bhk|bedroom(?:s)?|bed)\b",
                  r"\b(?:studio|penthouse|duplex|villa|plot|showflat)\b"])

# Slot NAMES that tend to hold a unit/configuration, so a value from one is preferred over a
# stray look-alike elsewhere in the entity bag. A hint list, not a gate — anything shaped like
# a unit is still accepted from any slot below. Not part of Vocabulary: this is an internal
# matching heuristic over SESSION KEY NAMES, not customer-facing vocabulary, and the terms
# here are broad enough (unit/config/variant/layout/typology as well as the RE-specific
# bhk/flat/apartment/property) to stay harmless noise for a non-real-estate tenant rather than
# something that needs to vary per pack.
_UNIT_SLOT_HINTS = ("bhk", "unit", "config", "flat", "property_type", "property",
                    "variant", "layout", "typology", "apartment", "home_type",
                    "bedroom", "configuration")

# The fact that holds the standing choice. One key, one active row per customer.
FACT_CATEGORY = "product_interest"
FACT_KEY = "desired_unit"


def _canonical_unit(text: str | None, vocabulary: Vocabulary | None = None) -> str | None:
    """Normalise a raw value to a canonical unit string per `vocabulary.unit_patterns`
    (see Vocabulary's own docstring for the numeric-vs-named-pattern convention), or None
    if it matches none of them — including for a pack with no unit concept at all
    (empty `unit_patterns`), which honestly finds nothing rather than guessing."""
    if not text:
        return None
    voc = vocabulary or _FALLBACK_VOCABULARY
    for pattern in voc.unit_patterns:
        m = re.search(pattern, text, re.I)
        if not m:
            continue
        if m.lastindex:  # has a capture group -> numeric configuration
            num = m.group(1)
            if num.endswith(".0"):  # "3.0" -> "3"; keep "2.5"
                num = num[:-2]
            suffix = voc.technical_terms[0] if voc.technical_terms else ""
            return f"{num} {suffix}".strip()
        return m.group(0).strip().capitalize()  # whole-match named unit
    return None


def desired_unit(active_entities: dict | None, vocabulary: Vocabulary | None = None) -> str | None:
    """The unit configuration the call is about, canonicalised — or None."""
    entities = active_entities or {}
    # A value from a slot whose NAME suggests a unit wins over one that merely looks like a
    # unit somewhere else in the bag.
    for slot, raw in entities.items():
        if any(h in str(slot).lower() for h in _UNIT_SLOT_HINTS):
            unit = _canonical_unit(str(raw), vocabulary)
            if unit:
                return unit
    for raw in entities.values():
        unit = _canonical_unit(str(raw), vocabulary)
        if unit:
            return unit
    return None


async def remember_product_interest(db: AsyncSession, *, tenant_id: uuid.UUID,
                                    customer_id: uuid.UUID,
                                    conversation_id: uuid.UUID | None, unit: str,
                                    project_id: uuid.UUID | None = None) -> None:
    """Persist a stated unit preference to both the structured interest table and the
    customer's active fact set. Idempotent for a repeated unit; supersedes a changed one.

    Writes only — the caller owns the transaction and commits. Safe to call every turn the
    unit is present; it does not accumulate duplicates.

    ``project_id`` scopes the interest (multi-project-per-tenant): the same unit in two
    projects is two distinct interests, so the dedup key includes it.
    """
    now = datetime.now(timezone.utc)

    # 1) CustomerProductInterest — one row per (customer, project, product_ref). Re-stating
    #    just refreshes noted_at; it never downgrades a stronger signal (purchased/rejected).
    row = (await db.execute(select(CustomerProductInterest).where(
        CustomerProductInterest.tenant_id == tenant_id,
        CustomerProductInterest.customer_id == customer_id,
        CustomerProductInterest.project_id == project_id,
        CustomerProductInterest.product_ref == unit))).scalar_one_or_none()
    if row is None:
        db.add(CustomerProductInterest(
            tenant_id=tenant_id, customer_id=customer_id, project_id=project_id,
            product_ref=unit, interest_level="interested", noted_at=now,
            source_conversation_id=conversation_id))
    else:
        row.noted_at = now

    # 2) CustomerFact — a single active desired_unit. Already recording this unit? Nothing to
    #    do. A different unit means they changed their mind: retire the old, record the new.
    prior = (await db.execute(select(CustomerFact).where(
        CustomerFact.tenant_id == tenant_id, CustomerFact.customer_id == customer_id,
        CustomerFact.category == FACT_CATEGORY, CustomerFact.key == FACT_KEY,
        CustomerFact.status == "active"))).scalars().all()
    if any((p.value or {}).get("text") == unit for p in prior):
        return
    for p in prior:
        p.status = "superseded"
    db.add(CustomerFact(
        tenant_id=tenant_id, customer_id=customer_id, category=FACT_CATEGORY, key=FACT_KEY,
        value={"text": unit}, confidence=0.9, source_conversation_id=conversation_id,
        status="active"))


async def load_active_preference(db: AsyncSession, *, tenant_id: uuid.UUID,
                                 customer_id: uuid.UUID) -> str | None:
    """The caller's standing unit choice, from the active desired_unit fact — or None."""
    row = (await db.execute(select(CustomerFact).where(
        CustomerFact.tenant_id == tenant_id, CustomerFact.customer_id == customer_id,
        CustomerFact.category == FACT_CATEGORY, CustomerFact.key == FACT_KEY,
        CustomerFact.status == "active").order_by(
        CustomerFact.created_at.desc()))).scalars().first()
    if row and (row.value or {}).get("text"):
        return str(row.value["text"])
    return None
