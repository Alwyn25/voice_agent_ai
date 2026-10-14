#!/usr/bin/env python3
"""Vocal Backend Tasksheet P5 (Campaigns + Contacts) Definition of Done, verified
against the live DB by driving the actual dialer/event-pipeline functions
in-process (app.services.telephony.dialer._enqueue_campaign_targets,
app.services.telephony.events.apply_call_event) — no mocking of that logic:

  - Dialer respects concurrency_limit under load (load test confirms no overshoot)
  - Pause is immediate for new dials (in-flight calls unaffected)
  - Campaign tiles (app.api.campaigns._tile_counts_batch) reconcile against the
    calls/campaign_targets tables with no drift
  - Campaign.calling_rules overrides the tenant-wide max_attempts/retry_backoff_minutes
    for a call's retry chain, and CampaignTarget.next_retry_at tracks the scheduled retry
  - contact_ids on a campaign's audience_filter takes priority over the filter fields

    docker compose exec -T core python scripts/verify_campaigns.py

Creates and cleans up its own tenant-scoped rows (a few customers, campaigns, targets,
calls) — safe to re-run. Sets OUTBOUND_IGNORE_CALLING_HOURS / OUTBOUND_IGNORE_FREQUENCY_CAPS
(test-only flags app/services/outbound.py already supports) so dials aren't blocked by
policy noise unrelated to what this script is checking.
"""
import os

os.environ.setdefault("OUTBOUND_IGNORE_CALLING_HOURS", "true")
os.environ.setdefault("OUTBOUND_IGNORE_FREQUENCY_CAPS", "true")

import asyncio  # noqa: E402
import sys  # noqa: E402
import uuid  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete, select  # noqa: E402

from app.api.campaigns import _audience_query, _tile_counts_batch  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db.base import get_session_factory  # noqa: E402
from app.db.models import Call, Campaign, CampaignTarget, Customer, Tenant  # noqa: E402
from app.services.telephony.base import (STATUS_ANSWERED, STATUS_COMPLETED, STATUS_NO_ANSWER,  # noqa: E402
                                         STATUS_QUEUED, TERMINAL_STATUSES, CallEvent)
from app.services.telephony.dialer import _enqueue_campaign_targets  # noqa: E402
from app.services.telephony.events import apply_call_event  # noqa: E402

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
failures = 0

PHONE_PREFIX = "+19999990"
CAMPAIGN_PREFIX = "Verify Campaign - "


def check(label: str, cond: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if cond else FAIL} {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures += 1


async def _demo_tenant_id(db) -> uuid.UUID:
    tenant = (await db.execute(select(Tenant).where(Tenant.slug == "demo-tenant"))).scalars().first()
    if tenant is None:
        raise SystemExit("No demo tenant found — run scripts/seed.py first.")
    return tenant.id


async def _cleanup(db, tenant_id: uuid.UUID) -> None:
    campaigns = (await db.execute(select(Campaign).where(
        Campaign.tenant_id == tenant_id, Campaign.name.startswith(CAMPAIGN_PREFIX)))).scalars().all()
    for c in campaigns:
        await db.execute(delete(Call).where(Call.tenant_id == tenant_id, Call.campaign_id == c.id))
        await db.execute(delete(CampaignTarget).where(
            CampaignTarget.tenant_id == tenant_id, CampaignTarget.campaign_id == c.id))
        await db.delete(c)
    await db.execute(delete(Customer).where(
        Customer.tenant_id == tenant_id, Customer.phone_normalized.startswith(PHONE_PREFIX)))
    await db.flush()


async def _make_customers(db, tenant_id: uuid.UUID, n: int, *, start: int = 0) -> list[Customer]:
    """`start` offsets the phone-number block so the different tests in one run (each
    keeping its own customers uncleaned-up until the final cleanup) never collide on the
    tenant+phone_normalized unique constraint."""
    out = []
    for i in range(start, start + n):
        cust = Customer(tenant_id=tenant_id, full_name=f"Verify Contact {i}",
                        phone=f"{PHONE_PREFIX}{i:03d}", phone_normalized=f"{PHONE_PREFIX}{i:03d}",
                        current_state="lead")
        db.add(cust)
        out.append(cust)
    await db.flush()
    return out


async def _campaign(db, tenant_id: uuid.UUID, name: str, *, concurrency_limit=None,
                    calling_rules=None) -> Campaign:
    c = Campaign(tenant_id=tenant_id, name=CAMPAIGN_PREFIX + name, status="running",
                schedule={"calls_per_hour": 100_000, "purpose": "marketing"},
                stats={}, concurrency_limit=concurrency_limit, calling_rules=calling_rules or {})
    db.add(c)
    await db.flush()
    return c


async def test_concurrency_limit_no_overshoot(db, tenant_id: uuid.UUID) -> None:
    """DoD: 'Dialer respects concurrency_limit under load — load test confirms no
    overshoot.' Six pending targets, a concurrency_limit of 2, five ticks of the enqueue
    step with nothing originating in between (so enqueued calls stay 'queued', never
    'ringing') — the exact gap the fix to _enqueue_campaign_targets closes: before it,
    the in-flight check only counted ringing/answered, so a queued-but-not-yet-originated
    backlog was invisible to it and every tick could enqueue past the limit."""
    customers = await _make_customers(db, tenant_id, 6, start=0)
    campaign = await _campaign(db, tenant_id, "Concurrency", concurrency_limit=2)
    for cust in customers:
        db.add(CampaignTarget(tenant_id=tenant_id, campaign_id=campaign.id,
                              customer_id=cust.id, status="pending"))
    await db.commit()

    settings = get_settings()
    max_in_flight_seen = 0
    for _ in range(5):
        await _enqueue_campaign_targets(db, tenant_id, settings)
        await db.commit()
        in_flight = (await db.execute(select(Call).where(
            Call.tenant_id == tenant_id, Call.campaign_id == campaign.id,
            Call.status.notin_(TERMINAL_STATUSES)))).scalars().all()
        max_in_flight_seen = max(max_in_flight_seen, len(in_flight))

    check("concurrency_limit=2 never exceeded across 5 enqueue ticks",
         max_in_flight_seen <= 2, f"saw {max_in_flight_seen} in-flight")
    check("concurrency_limit actually got exercised (not just idle at 0)",
         max_in_flight_seen == 2, f"saw {max_in_flight_seen} in-flight")

    total_calls_created = len((await db.execute(select(Call).where(
        Call.tenant_id == tenant_id, Call.campaign_id == campaign.id))).scalars().all())
    check("exactly concurrency_limit calls were ever created (no leak past the cap)",
         total_calls_created == 2, f"created {total_calls_created} Call rows total")


async def test_pause_stops_new_dials(db, tenant_id: uuid.UUID) -> None:
    """DoD: 'Pause is immediate for new dials — in-flight calls unaffected.'"""
    customers = await _make_customers(db, tenant_id, 3, start=100)
    campaign = await _campaign(db, tenant_id, "Pause")
    for cust in customers:
        db.add(CampaignTarget(tenant_id=tenant_id, campaign_id=campaign.id,
                              customer_id=cust.id, status="pending"))
    await db.commit()

    settings = get_settings()
    await _enqueue_campaign_targets(db, tenant_id, settings)
    await db.commit()
    before = {c.id: c.status for c in (await db.execute(select(Call).where(
        Call.tenant_id == tenant_id, Call.campaign_id == campaign.id))).scalars().all()}
    check("at least one call went in-flight before pausing", len(before) > 0, "none enqueued")

    campaign.status = "paused"
    await db.commit()
    await _enqueue_campaign_targets(db, tenant_id, settings)
    await db.commit()

    after = {c.id: c.status for c in (await db.execute(select(Call).where(
        Call.tenant_id == tenant_id, Call.campaign_id == campaign.id))).scalars().all()}
    check("no new calls enqueued for a paused campaign", set(after) == set(before),
         f"before={len(before)} after={len(after)}")
    check("existing in-flight calls left untouched by pause",
         all(after[cid] == status for cid, status in before.items()))


async def test_tiles_reconcile(db, tenant_id: uuid.UUID) -> None:
    """DoD: 'Campaign tiles reconcile against calls table — no drift.' Drives three
    targets through apply_call_event to known terminal states, then cross-checks
    _tile_counts_batch's numbers against independently-computed raw counts."""
    customers = await _make_customers(db, tenant_id, 3, start=200)
    campaign = await _campaign(db, tenant_id, "Tiles", calling_rules={"max_attempts": 1})
    targets = {}
    for cust in customers:
        t = CampaignTarget(tenant_id=tenant_id, campaign_id=campaign.id,
                           customer_id=cust.id, status="calling")
        db.add(t)
        targets[cust.id] = t
    await db.flush()

    # Contact 0: answered and completed -> target 'done', connected.
    call0 = Call(tenant_id=tenant_id, customer_id=customers[0].id, campaign_id=campaign.id,
                to_number=customers[0].phone_normalized, status=STATUS_QUEUED, attempt=1)
    db.add(call0)
    await db.flush()
    await apply_call_event(db, call0, CallEvent(provider="stub", provider_call_id="v0",
                                                status=STATUS_ANSWERED))
    await apply_call_event(db, call0, CallEvent(provider="stub", provider_call_id="v0",
                                                status=STATUS_COMPLETED))

    # Contact 1: no_answer with max_attempts=1 -> exhausted immediately -> target 'failed'.
    call1 = Call(tenant_id=tenant_id, customer_id=customers[1].id, campaign_id=campaign.id,
                to_number=customers[1].phone_normalized, status=STATUS_QUEUED, attempt=1)
    db.add(call1)
    await db.flush()
    await apply_call_event(db, call1, CallEvent(provider="stub", provider_call_id="v1",
                                                status=STATUS_NO_ANSWER))

    # Contact 2: campaign with no override on THIS call (attach directly, bypassing the
    # campaign's own calling_rules by using a fresh campaign) so a retry survives and the
    # target is left 'calling' with next_retry_at set.
    campaign2 = await _campaign(db, tenant_id, "Tiles2")
    t2 = CampaignTarget(tenant_id=tenant_id, campaign_id=campaign2.id,
                        customer_id=customers[2].id, status="calling")
    db.add(t2)
    call2 = Call(tenant_id=tenant_id, customer_id=customers[2].id, campaign_id=campaign2.id,
                to_number=customers[2].phone_normalized, status=STATUS_QUEUED, attempt=1)
    db.add(call2)
    await db.flush()
    await apply_call_event(db, call2, CallEvent(provider="stub", provider_call_id="v2",
                                                status=STATUS_NO_ANSWER))
    await db.commit()

    tiles = await _tile_counts_batch(db, tenant_id, [campaign.id, campaign2.id])

    raw_calls_c1 = len((await db.execute(select(Call).where(
        Call.tenant_id == tenant_id, Call.campaign_id == campaign.id))).scalars().all())
    raw_connected_c1 = len((await db.execute(select(Call).where(
        Call.tenant_id == tenant_id, Call.campaign_id == campaign.id,
        Call.answered_at.is_not(None)))).scalars().all())
    raw_done_c1 = len((await db.execute(select(CampaignTarget).where(
        CampaignTarget.tenant_id == tenant_id, CampaignTarget.campaign_id == campaign.id,
        CampaignTarget.status == "done"))).scalars().all())

    check("tiles.calls_made reconciles against a raw calls-table count",
         tiles[campaign.id]["calls_made"] == raw_calls_c1,
         f"tile={tiles[campaign.id]['calls_made']} raw={raw_calls_c1}")
    check("tiles.connected reconciles against a raw answered_at count",
         tiles[campaign.id]["connected"] == raw_connected_c1,
         f"tile={tiles[campaign.id]['connected']} raw={raw_connected_c1}")
    check("tiles.completed reconciles against a raw done-target count",
         tiles[campaign.id]["completed"] == raw_done_c1,
         f"tile={tiles[campaign.id]['completed']} raw={raw_done_c1}")
    check("contact 1 (no_answer, max_attempts=1) is 'failed' not 'rescheduled'",
         tiles[campaign.id]["rescheduled"] == 0 and tiles[campaign.id]["completed"] == 1,
         str(tiles[campaign.id]))
    check("contact 2 (no_answer, default max_attempts) is 'rescheduled'",
         tiles[campaign2.id]["rescheduled"] == 1, str(tiles[campaign2.id]))

    t1 = (await db.execute(select(CampaignTarget).where(
        CampaignTarget.id == targets[customers[1].id].id))).scalar_one()
    check("target.status == 'failed' after max_attempts exhausted", t1.status == "failed", t1.status)
    check("target.next_retry_at cleared after exhaustion", t1.next_retry_at is None)


async def test_calling_rules_override_retry_backoff(db, tenant_id: uuid.UUID) -> None:
    """Campaign.calling_rules overrides the tenant-wide max_attempts/retry_backoff_minutes
    for calls in that campaign, and CampaignTarget.next_retry_at mirrors the scheduled
    retry's scheduled_for."""
    settings = get_settings()
    check("sanity: tenant default backoff differs from this test's override (or the test "
         "below proves nothing)", settings.dialer_retry_backoff_minutes != 1)

    customers = await _make_customers(db, tenant_id, 1, start=300)
    campaign = await _campaign(db, tenant_id, "Backoff",
                               calling_rules={"max_attempts": 2, "retry_backoff_minutes": 1})
    target = CampaignTarget(tenant_id=tenant_id, campaign_id=campaign.id,
                            customer_id=customers[0].id, status="calling")
    db.add(target)
    call1 = Call(tenant_id=tenant_id, customer_id=customers[0].id, campaign_id=campaign.id,
                to_number=customers[0].phone_normalized, status=STATUS_QUEUED, attempt=1)
    db.add(call1)
    await db.flush()

    before = datetime.now(timezone.utc)
    await apply_call_event(db, call1, CallEvent(provider="stub", provider_call_id="b1",
                                                status=STATUS_NO_ANSWER))
    await db.commit()

    retry = (await db.execute(select(Call).where(
        Call.tenant_id == tenant_id, Call.retry_of_call_id == call1.id))).scalar_one_or_none()
    check("a retry was scheduled after attempt 1 (< max_attempts=2)", retry is not None)
    if retry is not None:
        expected = before + timedelta(minutes=1)  # campaign override, not the 15-min tenant default
        drift = abs((retry.scheduled_for - expected).total_seconds())
        check("retry backoff used the campaign's override (1 min), not the tenant default",
             drift < 10, f"scheduled_for={retry.scheduled_for} expected~={expected}")

        t = (await db.execute(select(CampaignTarget).where(
            CampaignTarget.id == target.id))).scalar_one()
        check("CampaignTarget.next_retry_at mirrors the retry Call's scheduled_for",
             t.next_retry_at is not None and abs(
                 (t.next_retry_at - retry.scheduled_for).total_seconds()) < 1)
        check("CampaignTarget stays 'calling' while a retry is pending", t.status == "calling")

        # Attempt 2 (== max_attempts) — no further retry, target moves to 'failed'.
        await apply_call_event(db, retry, CallEvent(provider="stub", provider_call_id="b2",
                                                    status=STATUS_NO_ANSWER))
        await db.commit()
        no_more_retries = (await db.execute(select(Call).where(
            Call.tenant_id == tenant_id, Call.retry_of_call_id == retry.id))).scalar_one_or_none()
        check("no third attempt once max_attempts=2 is reached", no_more_retries is None)
        t2 = (await db.execute(select(CampaignTarget).where(
            CampaignTarget.id == target.id))).scalar_one()
        check("target 'failed' and next_retry_at cleared once attempts are exhausted",
             t2.status == "failed" and t2.next_retry_at is None)


async def test_contact_ids_override_audience_filter(db, tenant_id: uuid.UUID) -> None:
    """An explicit contact_ids list on a campaign takes priority over the filter fields —
    the mechanism POST /campaigns' contact_ids body field relies on."""
    customers = await _make_customers(db, tenant_id, 4, start=400)
    await db.commit()
    picked = {customers[0].id, customers[2].id}
    af = {"states": ["some_state_that_matches_nobody"], "contact_ids": [str(c) for c in picked]}
    matched = {c.id for c in (await db.execute(_audience_query(tenant_id, af))).scalars().all()}
    check("contact_ids resolves to exactly the given contacts, ignoring other filter fields",
         matched == picked, f"matched={matched} expected={picked}")


async def main() -> int:
    async with get_session_factory()() as db:
        tenant_id = await _demo_tenant_id(db)
        await _cleanup(db, tenant_id)
        await db.commit()

        await test_concurrency_limit_no_overshoot(db, tenant_id)
        await db.commit()
        await test_pause_stops_new_dials(db, tenant_id)
        await db.commit()
        await test_tiles_reconcile(db, tenant_id)
        await db.commit()
        await test_calling_rules_override_retry_backoff(db, tenant_id)
        await db.commit()
        await test_contact_ids_override_audience_filter(db, tenant_id)
        await db.commit()

        await _cleanup(db, tenant_id)
        await db.commit()

    print()
    if failures:
        print(f"{FAIL} {failures} check(s) failed")
        return 1
    print(f"{PASS} all P5 campaign Definition-of-Done checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
