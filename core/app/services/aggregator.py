"""Aggregator Worker (Vocal Backend Tasksheet P7 Overview) — rolls finished
Call-Runtime-driven calls (Call.agent_id set — see app/services/call_runtime.py; the
older dialer/campaign engine's calls have no agent_id and are out of scope for this
Agent-centric dashboard) into `agent_metrics_daily` (per agent) and `daily_org_metrics`
(org-wide), for whichever days actually have calls that haven't been rolled up yet.

Only ever rolls up PAST days (before today) by default — today's own numbers are
computed live by app/api/dashboard.py directly from `calls`/`tool_executions`, since a
call still in progress this afternoon shouldn't wait for tonight's rollup to count.
`aggregate_day()` is still exposed for a specific date (including today, if a caller
explicitly wants a rollup snapshot) since scripts/verify_dashboard.py needs to roll up
its own fixture day deterministically rather than depending on wall-clock "today".
"""
from __future__ import annotations

import logging
import uuid
from datetime import date as date_
from datetime import datetime, time, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_session_factory
from app.db.models import Agent, AgentMetricsDaily, Call, DailyOrgMetrics, ToolExecution
from app.db.models.platform import customer_tenant_ids

logger = logging.getLogger(__name__)

# How many days back a tick looks for unrolled-up completed calls — generous enough to
# catch up after a process outage without scanning the whole calls table every tick.
CATCH_UP_DAYS = 14


def _day_window(target_date: date_) -> tuple[datetime, datetime]:
    start = datetime.combine(target_date, time.min, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def _rollup(rows: list[Call], actions_by_call: dict[uuid.UUID, int]) -> dict:
    completed = [c for c in rows if c.status == "completed"]
    resolved = [c for c in completed if c.hangup_cause != "handoff"]
    durations = [c.duration_seconds for c in completed if c.duration_seconds is not None]
    actions = sum(actions_by_call.get(c.id, 0) for c in rows)
    return {"calls": len(rows), "completed": len(completed), "resolved": len(resolved),
           "actions": actions,
           "avg_duration": int(sum(durations) / len(durations)) if durations else 0,
           "total_duration": sum(durations)}


async def compute_day_metrics(db: AsyncSession, tenant_id: uuid.UUID,
                              target_date: date_) -> dict:
    """Read-only version of the same rollup aggregate_day() persists — what
    app/api/dashboard.py calls directly for TODAY (which is never persisted; see this
    module's own docstring) or as a live fallback for a past day the background loop
    hasn't caught up to yet. Returns {"org": {...}, "by_agent": {agent_id: {...}},
    "active_agents": int} — the same shape aggregate_day writes, just not written."""
    start, end = _day_window(target_date)
    calls = list((await db.execute(select(Call).where(
        Call.tenant_id == tenant_id, Call.agent_id.is_not(None),
        Call.created_at >= start, Call.created_at < end))).scalars().all())

    by_agent_calls: dict[uuid.UUID, list[Call]] = {}
    for c in calls:
        by_agent_calls.setdefault(c.agent_id, []).append(c)

    call_ids = [c.id for c in calls]
    actions_by_call: dict[uuid.UUID, int] = {}
    if call_ids:
        rows = (await db.execute(select(ToolExecution.call_id, func.count()).where(
            ToolExecution.tenant_id == tenant_id, ToolExecution.call_id.in_(call_ids))
            .group_by(ToolExecution.call_id))).all()
        actions_by_call = {cid: int(n) for cid, n in rows}

    active_agents = int((await db.execute(select(func.count()).select_from(Agent).where(
        Agent.tenant_id == tenant_id, Agent.status == "live"))).scalar_one())

    return {"org": _rollup(calls, actions_by_call),
           "by_agent": {aid: _rollup(rows, actions_by_call) for aid, rows in by_agent_calls.items()},
           "active_agents": active_agents}


async def aggregate_day(db: AsyncSession, tenant_id: uuid.UUID, target_date: date_) -> DailyOrgMetrics:
    """Rolls up one tenant's one day. Upserts both agent_metrics_daily (per agent)
    and daily_org_metrics (org-wide); safe to re-run for the same day (idempotent —
    overwrites, never accumulates)."""
    metrics = await compute_day_metrics(db, tenant_id, target_date)

    for agent_id, r in metrics["by_agent"].items():
        row = (await db.execute(select(AgentMetricsDaily).where(
            AgentMetricsDaily.agent_id == agent_id,
            AgentMetricsDaily.date == target_date))).scalar_one_or_none()
        if row is None:
            row = AgentMetricsDaily(tenant_id=tenant_id, agent_id=agent_id, date=target_date)
            db.add(row)
        row.calls = r["calls"]
        row.resolved = r["resolved"]
        row.avg_duration_seconds = r["avg_duration"]
        row.actions_taken = r["actions"]

    org = metrics["org"]
    org_row = (await db.execute(select(DailyOrgMetrics).where(
        DailyOrgMetrics.tenant_id == tenant_id, DailyOrgMetrics.date == target_date
        ))).scalar_one_or_none()
    if org_row is None:
        org_row = DailyOrgMetrics(tenant_id=tenant_id, date=target_date)
        db.add(org_row)
    org_row.active_agents = metrics["active_agents"]
    org_row.calls_placed = org["calls"]
    org_row.calls_completed = org["completed"]
    org_row.resolved = org["resolved"]
    org_row.actions_taken = org["actions"]
    org_row.avg_duration_seconds = org["avg_duration"]
    org_row.total_duration_seconds = org["total_duration"]
    await db.flush()
    return org_row


async def org_metrics_for_day(db: AsyncSession, tenant_id: uuid.UUID, target_date: date_) -> dict:
    """What app/api/dashboard.py actually reads: today is always live-computed
    (never persisted); a past day reads the already-rolled-up daily_org_metrics row
    when the background loop has caught up to it, falling back to a live computation
    (not persisted — a GET must never have a write side effect) when it hasn't yet."""
    today = datetime.now(timezone.utc).date()
    if target_date < today:
        row = (await db.execute(select(DailyOrgMetrics).where(
            DailyOrgMetrics.tenant_id == tenant_id, DailyOrgMetrics.date == target_date
            ))).scalar_one_or_none()
        if row is not None:
            return {"calls": row.calls_placed, "completed": row.calls_completed,
                   "resolved": row.resolved, "actions": row.actions_taken,
                   "avg_duration": row.avg_duration_seconds,
                   "total_duration": row.total_duration_seconds,
                   "active_agents": row.active_agents}
    metrics = await compute_day_metrics(db, tenant_id, target_date)
    return {**metrics["org"], "active_agents": metrics["active_agents"]}


async def agent_metrics_range(db: AsyncSession, tenant_id: uuid.UUID, start: date_,
                              end_inclusive: date_) -> dict[uuid.UUID, dict]:
    """Summed per-agent metrics across [start, end_inclusive] — days before today read
    the persisted agent_metrics_daily rollup (one query, not one per day); today (when
    it falls in range) is live-computed and added in, same "never persist today" rule
    as everywhere else in this module."""
    today = datetime.now(timezone.utc).date()
    totals: dict[uuid.UUID, dict] = {}

    def _add(agent_id: uuid.UUID, calls: int, resolved: int, actions: int,
            avg_duration: int, weight: int) -> None:
        t = totals.setdefault(agent_id, {"calls": 0, "resolved": 0, "actions": 0,
                                         "_duration_weighted": 0, "_duration_weight": 0})
        t["calls"] += calls
        t["resolved"] += resolved
        t["actions"] += actions
        if weight:
            t["_duration_weighted"] += avg_duration * weight
            t["_duration_weight"] += weight

    past_end = min(end_inclusive, today - timedelta(days=1))
    if start <= past_end:
        rows = (await db.execute(select(AgentMetricsDaily).where(
            AgentMetricsDaily.tenant_id == tenant_id,
            AgentMetricsDaily.date >= start, AgentMetricsDaily.date <= past_end))).scalars().all()
        for r in rows:
            _add(r.agent_id, r.calls, r.resolved, r.actions_taken, r.avg_duration_seconds, r.calls)

    if start <= today <= end_inclusive:
        today_metrics = await compute_day_metrics(db, tenant_id, today)
        for agent_id, r in today_metrics["by_agent"].items():
            _add(agent_id, r["calls"], r["resolved"], r["actions"], r["avg_duration"], r["calls"])

    for t in totals.values():
        t["avg_duration"] = (int(t["_duration_weighted"] / t["_duration_weight"])
                             if t["_duration_weight"] else 0)
        del t["_duration_weighted"], t["_duration_weight"]
    return totals


async def org_metrics_series(db: AsyncSession, tenant_id: uuid.UUID, start: date_,
                             end_inclusive: date_) -> dict[date_, dict]:
    """{date: {calls, completed, resolved, actions, avg_duration}} for every day in
    [start, end_inclusive] — the call-activity chart's data source. Batched (one
    query for every already-rolled-up past day, not one per day); today is live-
    computed when it falls in range; a day with no daily_org_metrics row AND not
    today (the background loop hasn't caught up, or genuinely had zero calls) reads
    as all-zero rather than triggering its own live computation — a 14-day chart
    should not run 14 live rollup queries for what is almost always "no data"."""
    today = datetime.now(timezone.utc).date()
    zero = {"calls": 0, "completed": 0, "resolved": 0, "actions": 0, "avg_duration": 0}
    out: dict[date_, dict] = {}

    past_end = min(end_inclusive, today - timedelta(days=1))
    if start <= past_end:
        rows = (await db.execute(select(DailyOrgMetrics).where(
            DailyOrgMetrics.tenant_id == tenant_id,
            DailyOrgMetrics.date >= start, DailyOrgMetrics.date <= past_end))).scalars().all()
        for r in rows:
            out[r.date] = {"calls": r.calls_placed, "completed": r.calls_completed,
                          "resolved": r.resolved, "actions": r.actions_taken,
                          "avg_duration": r.avg_duration_seconds}

    if start <= today <= end_inclusive:
        today_metrics = await compute_day_metrics(db, tenant_id, today)
        out[today] = today_metrics["org"]

    d = start
    while d <= end_inclusive:
        out.setdefault(d, dict(zero))
        d += timedelta(days=1)
    return out


async def _unrolled_days(db: AsyncSession, tenant_id: uuid.UUID) -> list[date_]:
    """Every past day (not today — today is always live-computed, never rolled up
    by the background loop) within CATCH_UP_DAYS that has at least one agent-driven
    call and no daily_org_metrics row yet."""
    today = datetime.now(timezone.utc).date()
    since = today - timedelta(days=CATCH_UP_DAYS)
    since_start, _ = _day_window(since)
    candidate_days = {c.created_at.date() for c in (await db.execute(select(Call).where(
        Call.tenant_id == tenant_id, Call.agent_id.is_not(None),
        Call.created_at >= since_start))).scalars().all()}
    candidate_days.discard(today)
    if not candidate_days:
        return []
    already_rolled = {r.date for r in (await db.execute(select(DailyOrgMetrics).where(
        DailyOrgMetrics.tenant_id == tenant_id,
        DailyOrgMetrics.date.in_(candidate_days)))).scalars().all()}
    return sorted(candidate_days - already_rolled)


async def run_aggregator_tick() -> int:
    """One pass over every tenant, rolling up any past day that has unrolled agent-
    driven calls. Used by the background loop (app/main.py) and POST /dashboard/
    aggregate for an on-demand run. Returns how many (tenant, day) rollups it wrote."""
    written = 0
    async with get_session_factory()() as db:
        # Demo tenants (GTM demo links) stay out of the dashboards' rollups.
        tenant_ids = (await db.execute(customer_tenant_ids())).scalars().all()
    for tid in tenant_ids:
        async with get_session_factory()() as db:
            try:
                for day in await _unrolled_days(db, tid):
                    await aggregate_day(db, tid, day)
                    written += 1
                await db.commit()
            except Exception:
                await db.rollback()
                logger.exception("aggregator tick failed for tenant %s", tid)
    return written
