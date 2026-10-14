"""apply_call_event — the one place a Call row's lifecycle is written.

Both the carrier webhook (app/api/telephony_hooks.py) and the stub simulator funnel their
events through here, so retry scheduling, campaign-stat updates and the dashboard events happen
identically no matter how the event arrived. Nothing else mutates Call.status.

Retry policy: a busy / no-answer / voicemail outcome schedules a fresh queued Call (retry_of
the original, attempt + 1, scheduled_for now + backoff) up to dialer_max_attempts. The retry is
still gated at dial time by the dialer, so the calling policy's consecutive-no-answer back-off
and daily caps have the final say on whether it actually goes out — this only decides WHEN and
HOW MANY retries are lined up.

Vocal Backend Tasksheet P5's Campaign.calling_rules can override the tenant-wide
dialer_max_attempts/dialer_retry_backoff_minutes defaults for a call that belongs to that
campaign (e.g. a stricter max_attempts just for this campaign) — see Campaign's own docstring
for why this deliberately does NOT re-implement quiet-hours/DNC, which already live in the
tenant-wide rule engine and are enforced at dial time by check_can_call regardless of campaign.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import AnalyticsEvent, Call, Campaign, CampaignTarget
from app.services.events import publish
from app.services.telephony.base import (ACTIVE_STATUSES, CallEvent, RETRYABLE_STATUSES,
                                         STATUS_ANSWERED, STATUS_COMPLETED, STATUS_QUEUED,
                                         STATUS_RINGING, TERMINAL_STATUSES)

logger = logging.getLogger(__name__)


async def apply_call_event(db: AsyncSession, call: Call, event: CallEvent) -> None:
    """Mutate ``call`` from ``event`` and drive the side effects (retry, campaign, events).

    Idempotent on terminal state: a duplicate webhook for a call that is already finished is
    ignored, because a carrier may deliver the same 'completed' more than once."""
    now = event.occurred_at or datetime.now(timezone.utc)

    if call.status in TERMINAL_STATUSES and event.status in TERMINAL_STATUSES:
        logger.info("ignoring %s for already-terminal call %s (%s)",
                    event.status, call.id, call.status)
        return

    prev = call.status
    call.status = event.status
    if event.hangup_cause:
        call.hangup_cause = event.hangup_cause
    if event.recording_url:
        call.recording_url = event.recording_url
    if event.cost is not None:
        call.cost = event.cost

    if event.status == STATUS_RINGING and call.answered_at is None:
        publish(call.tenant_id, "call.started", call_id=str(call.id),
                customer_id=str(call.customer_id) if call.customer_id else None,
                to_number=call.to_number, purpose=call.purpose, status="ringing")

    if event.status == STATUS_ANSWERED:
        call.answered_at = call.answered_at or now
        publish(call.tenant_id, "call.started", call_id=str(call.id),
                customer_id=str(call.customer_id) if call.customer_id else None,
                to_number=call.to_number, purpose=call.purpose, status="answered")

    if event.status in TERMINAL_STATUSES:
        call.ended_at = now
        if event.duration_seconds is not None:
            call.duration_seconds = event.duration_seconds
        elif call.answered_at is not None:
            call.duration_seconds = max(0, int((now - call.answered_at).total_seconds()))
        await _on_terminal(db, call, event)

    logger.info("call %s %s -> %s (attempt %s)", call.id, prev, call.status, call.attempt)


async def _on_terminal(db: AsyncSession, call: Call, event: CallEvent) -> None:
    connected = event.status == STATUS_COMPLETED and call.answered_at is not None
    retry_scheduled = None
    retry_scheduled_for = None

    if event.status in RETRYABLE_STATUSES:
        retry_scheduled, retry_scheduled_for = await _schedule_retry(db, call, reason=event.status)

    await _update_campaign_target(db, call, connected=connected,
                                  exhausted=(event.status in RETRYABLE_STATUSES
                                             and retry_scheduled is None),
                                  next_retry_at=retry_scheduled_for)

    db.add(AnalyticsEvent(
        tenant_id=call.tenant_id, event_type="call_completed", customer_id=call.customer_id,
        properties={"call_id": str(call.id), "status": event.status,
                    "connected": connected, "purpose": call.purpose,
                    "duration_seconds": call.duration_seconds,
                    "cost": float(call.cost) if call.cost is not None else None,
                    "attempt": call.attempt,
                    "retry_scheduled": str(retry_scheduled) if retry_scheduled else None,
                    "campaign_id": str(call.campaign_id) if call.campaign_id else None}))

    publish(call.tenant_id, "call.ended", call_id=str(call.id),
            customer_id=str(call.customer_id) if call.customer_id else None,
            status=event.status, connected=connected,
            duration_seconds=call.duration_seconds,
            retry_scheduled=str(retry_scheduled) if retry_scheduled else None)


async def _schedule_retry(db: AsyncSession, call: Call,
                          *, reason: str) -> tuple[uuid.UUID | None, datetime | None]:
    """Queue the next attempt after a back-off, unless the attempt ceiling is reached. Returns
    (new call id, its scheduled_for) or (None, None) when no further retry is scheduled."""
    settings = get_settings()
    max_attempts = settings.dialer_max_attempts
    backoff_minutes = settings.dialer_retry_backoff_minutes

    # A campaign-specific calling_rules override (Vocal Backend Tasksheet P5) wins over the
    # tenant-wide default for a call that belongs to it — e.g. a stricter max_attempts just
    # for this campaign. Missing keys fall back to the tenant-wide setting.
    if call.campaign_id is not None:
        campaign = (await db.execute(select(Campaign).where(
            Campaign.id == call.campaign_id))).scalar_one_or_none()
        if campaign is not None and campaign.calling_rules:
            max_attempts = int(campaign.calling_rules.get("max_attempts", max_attempts))
            backoff_minutes = int(campaign.calling_rules.get("retry_backoff_minutes", backoff_minutes))

    if call.attempt >= max_attempts:
        return None, None
    if call.customer_id is None:
        return None, None   # nothing to re-dial without a customer record
    scheduled_for = datetime.now(timezone.utc) + timedelta(minutes=backoff_minutes * call.attempt)
    retry = Call(tenant_id=call.tenant_id, customer_id=call.customer_id,
                 to_number=call.to_number, purpose=call.purpose, campaign_id=call.campaign_id,
                 status=STATUS_QUEUED, provider=None, attempt=call.attempt + 1,
                 scheduled_for=scheduled_for,
                 retry_of_call_id=call.id, retry_reason=reason)
    db.add(retry)
    await db.flush()
    logger.info("scheduled retry %s of %s (attempt %s, in %s min) reason=%s",
                retry.id, call.id, retry.attempt, backoff_minutes * call.attempt, reason)
    return retry.id, scheduled_for


async def _update_campaign_target(db: AsyncSession, call: Call, *, connected: bool,
                                  exhausted: bool, next_retry_at: datetime | None) -> None:
    if call.campaign_id is None or call.customer_id is None:
        return
    target = (await db.execute(select(CampaignTarget).where(
        CampaignTarget.tenant_id == call.tenant_id,
        CampaignTarget.campaign_id == call.campaign_id,
        CampaignTarget.customer_id == call.customer_id))).scalar_one_or_none()
    if target is None:
        return
    if connected:
        target.status = "done"
        target.next_retry_at = None
    elif exhausted:
        target.status = "failed"
        target.next_retry_at = None
    else:
        # A retry is pending: the target stays 'calling' until it connects or is exhausted.
        # next_retry_at mirrors the just-scheduled retry Call's scheduled_for so a caller can
        # see "not now, retry at X" without joining through calls.
        target.next_retry_at = next_retry_at
    publish(call.tenant_id, "campaign.progress", campaign_id=str(call.campaign_id),
            customer_id=str(call.customer_id), target_status=target.status,
            connected=connected)


def call_is_active(status: str) -> bool:
    """A call occupying a concurrency slot (ringing or answered)."""
    return status in ACTIVE_STATUSES
