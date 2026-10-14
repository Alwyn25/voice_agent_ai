"""Visit slots for demo tenants (docs/gtm-demo-links.md §6.5).

A demo runs on the real-estate engine pack unchanged: search_slots, the slot_available /
slot_was_offered checks, book_appointment and reschedule all ask the scheduling service's
compute_slots() what is free. For a tenant of kind 'demo' that function asks here instead
of computing from working hours, and gets the record's dummy-data slots resolved at this
moment ("tomorrow 11 am", "day after 11 am"). Customer tenants never reach this code: the
first call per tenant caches its kind and returns None for them.

Every caller sees every slot. Bookings made on other demo calls (other people at the same
company trying the same link) are deliberately ignored, so each person's demo starts from
the same fresh calendar; the engine's no_conflicting_appointment check still stops one
caller double-booking themselves.
"""
import logging
import uuid
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DemoProspect, Tenant
from app.db.models.platform import TENANT_KIND_DEMO

log = logging.getLogger(__name__)

# A tenant's kind never changes after creation, so it is cached for the process lifetime.
_KIND: dict[uuid.UUID, str] = {}


async def tenant_kind(db: AsyncSession, tenant_id: uuid.UUID) -> str:
    kind = _KIND.get(tenant_id)
    if kind is None:
        kind = (await db.execute(select(Tenant.kind).where(Tenant.id == tenant_id))).scalar() or "customer"
        _KIND[tenant_id] = kind
    return kind


async def demo_slot_windows(db: AsyncSession, tenant_id: uuid.UUID, date_from: date, date_to: date,
                            *, now: datetime | None = None) -> list[tuple[datetime, datetime]] | None:
    """(start, end) of each free demo slot whose local date is within [date_from, date_to],
    or None when this is not a demo tenant (the caller then computes slots as usual)."""
    if await tenant_kind(db, tenant_id) != TENANT_KIND_DEMO:
        return None
    from app.demo import records as svc     # local: records imports the engine lazily too
    prospect = (await db.execute(select(DemoProspect).where(
        DemoProspect.tenant_id == tenant_id))).scalar_one_or_none()
    if prospect is None:
        log.warning("demo tenant %s has no record; offering no slots", tenant_id)
        return []
    definition = await svc.load_definition(db, prospect.industry_key, prospect.industry_version)
    engine = definition.engine or {}
    collection = engine.get("slot_collection")
    if not collection:
        return []
    try:
        ctx, _, _ = await svc.resolve_for_call(db, prospect, now or datetime.now(timezone.utc))
    except svc.DemoError as e:
        log.warning("demo tenant %s: dummy data unusable (%s); offering no slots", tenant_id, e.message)
        return []
    minutes = int(engine.get("slot_minutes", 60))
    out = []
    for slot in ctx.data.get(collection) or []:
        start = datetime.fromisoformat(slot["at"])
        if date_from <= start.date() <= date_to:
            out.append((start, start + timedelta(minutes=minutes)))
    return out
