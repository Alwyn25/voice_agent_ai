"""The dialer worker — what turns a queued Call into an actual (here, simulated) dial.

One tick does three things, per tenant:

  1. **Enqueue** work that is due: running campaigns' targets (paced by calls_per_hour with a
     token bucket, so a 30/hour campaign does not fire all 30 in the first second) and
     "call me later" callbacks that have come due. Both go through the same place_outbound_call
     gate everything else does.
  2. **Originate** queued calls that are due (scheduled_for null or past), oldest first, up to
     the per-tenant concurrency cap and the batch size — re-checking the calling policy at THIS
     moment, so a customer who opted out after the call was queued is still protected (PRD §7).
  3. Retries are already queued by apply_call_event when a call is not answered; they flow
     through step 2 once their back-off has elapsed.

The tick is safe to run from a background loop (app/main.py) or on demand (POST /telephony/tick).
Everything it does is idempotent enough to run every few seconds without double-dialling: a
call leaves 'queued' the instant it is originated, and campaign pacing credit is persisted.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.base import get_session_factory
from app.db.models import Call, CallbackRequest, Campaign, CampaignTarget, Customer, Tenant
from app.db.models.platform import customer_tenant_ids
from app.services.outbound import check_can_call, place_outbound_call
from app.services.telephony.base import (ACTIVE_STATUSES, DialRequest, STATUS_FAILED,
                                         STATUS_QUEUED, TERMINAL_STATUSES)
from app.services.telephony.events import call_is_active  # noqa: F401 (kept for callers)
from app.services.telephony.provider import get_telephony_provider

logger = logging.getLogger(__name__)


@dataclass
class DialerReport:
    tenant_id: uuid.UUID
    enqueued_campaign: int = 0
    enqueued_callbacks: int = 0
    originated: int = 0
    blocked_at_dial: int = 0
    failed: int = 0
    active_before: int = 0
    capacity: int = 0
    details: list[dict] = field(default_factory=list)

    def as_json(self) -> dict:
        return {"tenant_id": str(self.tenant_id), "enqueued_campaign": self.enqueued_campaign,
                "enqueued_callbacks": self.enqueued_callbacks, "originated": self.originated,
                "blocked_at_dial": self.blocked_at_dial, "failed": self.failed,
                "active_before": self.active_before, "capacity": self.capacity,
                "details": self.details}


async def run_dialer_tick() -> list[DialerReport]:
    """Process every tenant once. Used by the background loop; opens its own sessions."""
    reports: list[DialerReport] = []
    async with get_session_factory()() as db:
        # Demo tenants never dial: their calls are browser-only (GTM demo links).
        tenant_ids = (await db.execute(customer_tenant_ids())).scalars().all()
    for tid in tenant_ids:
        async with get_session_factory()() as db:
            try:
                reports.append(await dial_pending(db, tenant_id=tid))
                await db.commit()
            except Exception:
                await db.rollback()
                logger.exception("dialer tick failed for tenant %s", tid)
    return reports


async def dial_pending(db: AsyncSession, *, tenant_id: uuid.UUID) -> DialerReport:
    """One tenant's tick. Caller owns the transaction (commit outside)."""
    settings = get_settings()
    report = DialerReport(tenant_id=tenant_id)

    report.enqueued_campaign = await _enqueue_campaign_targets(db, tenant_id, settings)
    report.enqueued_callbacks = await _enqueue_due_callbacks(db, tenant_id)

    active = (await db.execute(select(func.count()).select_from(Call).where(
        Call.tenant_id == tenant_id, Call.status.in_(ACTIVE_STATUSES)))).scalar_one()
    report.active_before = int(active)
    capacity = max(0, settings.dialer_max_concurrent_calls - int(active))
    report.capacity = capacity
    if capacity == 0:
        return report

    now = datetime.now(timezone.utc)
    due = (await db.execute(select(Call).where(
        Call.tenant_id == tenant_id, Call.status == STATUS_QUEUED,
        or_(Call.scheduled_for.is_(None), Call.scheduled_for <= now))
        .order_by(Call.scheduled_for.nullsfirst(), Call.created_at)
        .limit(min(capacity, settings.dialer_batch_size)))).scalars().all()

    provider = get_telephony_provider(settings)
    for call in due:
        outcome = await _originate_one(db, call, provider, settings)
        report.details.append(outcome)
        if outcome["result"] == "originated":
            report.originated += 1
        elif outcome["result"] == "blocked":
            report.blocked_at_dial += 1
        else:
            report.failed += 1
    return report


async def _originate_one(db: AsyncSession, call: Call, provider, settings) -> dict:
    customer = None
    if call.customer_id is not None:
        customer = (await db.execute(select(Customer).where(
            Customer.id == call.customer_id))).scalar_one_or_none()

    # Re-check the calling policy at dial time. A queued call is a promise to *consider*
    # dialling, not a licence to dial regardless of what changed since (PRD §7).
    if customer is not None:
        decision = await check_can_call(db, tenant_id=call.tenant_id, customer=customer,
                                        purpose=call.purpose)
        if not decision.allowed:
            call.status = STATUS_FAILED
            call.ended_at = datetime.now(timezone.utc)
            call.hangup_cause = "blocked_at_dial"
            call.retry_reason = decision.blocked_by
            logger.info("dial blocked at dial time (call=%s rule=%s)", call.id, decision.blocked_by)
            return {"call_id": str(call.id), "result": "blocked", "rule": decision.blocked_by}

    from_number = await _caller_id(db, call.tenant_id, settings)
    media_url = (f"{settings.telephony_media_stream_base_url.rstrip('/')}/{call.id}"
                 if settings.telephony_media_stream_base_url else None)
    req = DialRequest(call_id=call.id, tenant_id=call.tenant_id, to_number=call.to_number or "",
                      from_number=from_number, purpose=call.purpose, media_stream_url=media_url)

    result = await provider.originate(req)
    call.provider = result.provider
    call.from_number = from_number
    if not result.ok:
        call.status = STATUS_FAILED
        call.ended_at = datetime.now(timezone.utc)
        call.hangup_cause = result.error or "originate_failed"
        logger.warning("originate failed (call=%s error=%s)", call.id, result.error)
        return {"call_id": str(call.id), "result": "failed", "error": result.error}

    call.provider_call_id = result.provider_call_id
    call.status = result.status
    logger.info("originated call %s via %s (%s) -> %s",
                call.id, result.provider, result.provider_call_id, result.status)
    return {"call_id": str(call.id), "result": "originated",
            "provider_call_id": result.provider_call_id, "status": result.status}


async def _caller_id(db: AsyncSession, tenant_id: uuid.UUID, settings) -> str:
    """The number outbound presents as caller-ID: the tenant's own setting if it has one,
    else the platform default."""
    tenant = (await db.execute(select(Tenant).where(Tenant.id == tenant_id))).scalar_one_or_none()
    if tenant is not None and isinstance(tenant.settings, dict):
        cid = tenant.settings.get("caller_id")
        if cid:
            return str(cid)
    return settings.telephony_caller_id


# --------------------------------------------------------------------------- enqueue

async def _enqueue_campaign_targets(db: AsyncSession, tenant_id: uuid.UUID, settings) -> int:
    """Turn pending campaign targets into queued Calls, paced per campaign by calls_per_hour.

    A token bucket persisted in campaign.stats['pacing_credit'] accrues allowance each tick
    (calls_per_hour * tick_seconds / 3600) so a campaign dials at its configured rate rather
    than all at once, and pacing survives a process restart.

    Vocal Backend Tasksheet P5's concurrency_limit is a SEPARATE cap layered on top
    of that pacing, not a replacement for it: pacing controls the DIAL RATE (how
    many per hour), concurrency_limit controls how many of THIS campaign's calls
    may be simultaneously in-flight (queued/ringing/answered) regardless of rate —
    "25 lines in use" is a concurrency fact, not a rate one. A campaign with no
    concurrency_limit set behaves exactly as before this column existed.

    In-flight is deliberately NOT the same set as ACTIVE_STATUSES (ringing/answered only):
    a just-enqueued Call sits 'queued' for however long it takes dial_pending's own
    origination step (a separate, tenant-wide pass with its own batch size) to reach it, and
    that queued call has already claimed one of this campaign's concurrency slots even though
    no line is ringing yet. Counting only ringing/answered would let a burst of enqueues in
    consecutive ticks pile up queued calls with nothing to stop them, all of which would then
    originate together once the tenant-wide origination step caught up — an overshoot the
    load-test DoD (scripts/verify_campaigns.py) is specifically checking for."""
    running = (await db.execute(select(Campaign).where(
        Campaign.tenant_id == tenant_id, Campaign.status == "running"))).scalars().all()
    total = 0
    for c in running:
        sched = c.schedule or {}
        per_hour = float(sched.get("calls_per_hour", 30) or 30)
        purpose = sched.get("purpose", "marketing")
        credit = float((c.stats or {}).get("pacing_credit", 0.0))
        credit += per_hour * (settings.dialer_interval_seconds / 3600.0)
        allowance = int(credit)
        if allowance < 1:
            c.stats = {**(c.stats or {}), "pacing_credit": credit}
            continue

        if c.concurrency_limit is not None:
            in_flight_for_campaign = (await db.execute(select(func.count()).select_from(Call).where(
                Call.tenant_id == tenant_id, Call.campaign_id == c.id,
                Call.status.notin_(TERMINAL_STATUSES)))).scalar_one()
            allowance = min(allowance, max(0, c.concurrency_limit - int(in_flight_for_campaign)))
            if allowance < 1:
                c.stats = {**(c.stats or {}), "pacing_credit": credit}
                continue

        targets = (await db.execute(select(CampaignTarget).where(
            CampaignTarget.tenant_id == tenant_id, CampaignTarget.campaign_id == c.id,
            CampaignTarget.status == "pending").limit(allowance))).scalars().all()

        used = 0
        for t in targets:
            cust = (await db.execute(select(Customer).where(
                Customer.id == t.customer_id))).scalar_one_or_none()
            if cust is None:
                t.status = "failed"
                continue
            decision = await place_outbound_call(db, tenant_id=tenant_id, customer=cust,
                                                 purpose=purpose, campaign_id=c.id, actor="dialer")
            t.attempts += 1
            t.last_attempt_at = datetime.now(timezone.utc)
            if decision.allowed:
                t.status = "calling"
                used += 1
            else:
                t.status = "excluded"
                t.exclusion_reason = decision.blocked_by
        credit -= used
        c.stats = {**(c.stats or {}), "pacing_credit": credit}
        total += used
    return total


async def _enqueue_due_callbacks(db: AsyncSession, tenant_id: uuid.UUID) -> int:
    """A 'call me later' that has come due becomes a queued (service) Call once. fulfilled_by
    is stamped when it is enqueued, so the same callback is never enqueued twice."""
    now = datetime.now(timezone.utc)
    due = (await db.execute(select(CallbackRequest).where(
        CallbackRequest.tenant_id == tenant_id, CallbackRequest.status == "pending",
        CallbackRequest.requested_for <= now,
        CallbackRequest.fulfilled_by_call_id.is_(None)))).scalars().all()
    count = 0
    for cb in due:
        cust = (await db.execute(select(Customer).where(
            Customer.id == cb.customer_id))).scalar_one_or_none()
        if cust is None:
            continue
        # A callback the customer asked for is a service contact, not promotional.
        decision = await place_outbound_call(db, tenant_id=tenant_id, customer=cust,
                                             purpose="service", actor="callback")
        if decision.allowed:
            cb.fulfilled_by_call_id = decision.call_id
            count += 1
    return count
