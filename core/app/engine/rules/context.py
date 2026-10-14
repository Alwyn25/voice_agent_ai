"""Builds the frozen RuleContext namespace that conditions address.

Namespaces:
  context.*   event, evaluation_point
  customer.*  id, current_state, tags, language, region, dnc flag, contact window, counters
  now.*       local_time (HH:MM), weekday (0=Mon), date, is_weekend
  proposal.*  tool, params.<name>   (when evaluating a proposed tool call)
  session.*   identity_verified, channel, direction, turn_index
  turn.*      sentiment, intent, stt_confidence
  counters.*  attempts_today, consecutive_no_answer, failed_interactions, total_calls
  policy.*    tenant-configured constants (settings.rule_policy)
"""
import uuid
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Call, Customer, DncEntry, Tenant

NO_ANSWER_STATUSES = ("busy", "no_answer", "failed")


async def build_context(
    db: AsyncSession,
    tenant_id: uuid.UUID,
    *,
    evaluation_point: str,
    customer: Customer | None = None,
    proposal: dict | None = None,
    session: dict | None = None,
    turn: dict | None = None,
) -> dict[str, Any]:
    tenant = (await db.execute(select(Tenant).where(Tenant.id == tenant_id))).scalar_one_or_none()
    tz = ZoneInfo((tenant.timezone if tenant else None) or "Asia/Kolkata")
    local = datetime.now(timezone.utc).astimezone(tz)

    ctx: dict[str, Any] = {
        "context": {"event": evaluation_point.split(":")[0],
                    "evaluation_point": evaluation_point},
        "now": {"local_time": local.strftime("%H:%M"),
                "weekday": local.weekday(),
                "date": local.date().isoformat(),
                "datetime": local.isoformat(),
                "is_weekend": local.weekday() >= 5},
        "proposal": {"tool": (proposal or {}).get("tool"),
                     "params": (proposal or {}).get("params", {})},
        "session": {"identity_verified": (session or {}).get("identity_verified", False),
                    "channel": (session or {}).get("channel"),
                    "direction": (session or {}).get("direction"),
                    # outbound.py passes purpose; without it every purpose-scoped rule
                    # resolves to None and silently evaluates false.
                    "purpose": (session or {}).get("purpose"),
                    "campaign_id": (session or {}).get("campaign_id"),
                    "turn_index": (session or {}).get("turn_index", 0)},
        "turn": {"sentiment": (turn or {}).get("sentiment"),
                 "intent": (turn or {}).get("intent"),
                 "stt_confidence": (turn or {}).get("stt_confidence")},
        # Captured conversation information (entities), for slot-based `require` gates. Passed
        # through the session dict so no caller signature changes; empty for evaluations that
        # do not supply it. `require_asks` is the per-condition ask counter that lets a slot
        # requirement stand aside after a couple of asks (anti-trap), also carried on session.
        "slots": (session or {}).get("__slots__") or {},
        "require_asks": (session or {}).get("__require_asks__") or {},
        "policy": ((tenant.settings or {}).get("rule_policy", {}) if tenant else {}),
        "customer": {},
        "counters": {"attempts_today": 0, "consecutive_no_answer": 0,
                     "failed_interactions": 0, "total_calls": 0},
    }

    if customer is None:
        return ctx

    on_dnc_all = await _dnc_hit(db, tenant_id, customer.phone_normalized, "all")
    on_dnc_marketing = await _dnc_hit(db, tenant_id, customer.phone_normalized, "marketing")

    ctx["customer"] = {
        "id": str(customer.id),
        "current_state": customer.current_state,
        "phone": customer.phone_normalized,
        "tags": list(customer.tags or []),
        "preferred_language": customer.preferred_language,
        "region_code": customer.region_code,
        "priority_score": float(customer.priority_score or 0),
        "lead_source": customer.lead_source,
        "on_dnc_list": bool(on_dnc_all),
        "on_dnc_marketing": bool(on_dnc_all or on_dnc_marketing),
        "preferred_contact_start": customer.preferred_contact_start.strftime("%H:%M")
        if customer.preferred_contact_start else None,
        "preferred_contact_end": customer.preferred_contact_end.strftime("%H:%M")
        if customer.preferred_contact_end else None,
        "has_contact_window": bool(customer.preferred_contact_start and customer.preferred_contact_end),
    }
    ctx["counters"] = await _counters(db, tenant_id, customer.id, local)
    return ctx


async def _dnc_hit(db: AsyncSession, tenant_id: uuid.UUID, phone: str, scope: str) -> bool:
    now = datetime.now(timezone.utc)
    row = (await db.execute(select(DncEntry.id).where(
        DncEntry.tenant_id == tenant_id,
        DncEntry.phone_normalized == phone,
        DncEntry.scope == scope))).first()
    if row is None:
        return False
    entry = (await db.execute(select(DncEntry).where(
        DncEntry.tenant_id == tenant_id, DncEntry.phone_normalized == phone,
        DncEntry.scope == scope))).scalars().first()
    if entry and entry.expires_at and entry.expires_at < now:
        return False
    return True


async def _counters(db: AsyncSession, tenant_id: uuid.UUID, customer_id: uuid.UUID,
                    local: datetime) -> dict[str, int]:
    day_start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    attempts_today = (await db.execute(select(func.count()).select_from(Call).where(
        Call.tenant_id == tenant_id, Call.customer_id == customer_id,
        Call.created_at >= day_start))).scalar_one()
    total = (await db.execute(select(func.count()).select_from(Call).where(
        Call.tenant_id == tenant_id, Call.customer_id == customer_id))).scalar_one()
    recent = (await db.execute(select(Call.status).where(
        Call.tenant_id == tenant_id, Call.customer_id == customer_id)
        .order_by(Call.created_at.desc()).limit(10))).scalars().all()
    consecutive = 0
    for s in recent:
        if s in NO_ANSWER_STATUSES:
            consecutive += 1
        else:
            break
    failed = sum(1 for s in recent if s in NO_ANSWER_STATUSES)
    return {"attempts_today": int(attempts_today), "total_calls": int(total),
            "consecutive_no_answer": consecutive, "failed_interactions": failed}
