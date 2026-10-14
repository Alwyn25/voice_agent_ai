"""Vocal Backend Tasksheet P7 (Overview) — the Agent-centric dashboard: summary tiles
with period-comparison deltas, a call-activity chart, who's on a call right now, and a
per-agent performance table. Deliberately a SEPARATE surface from the pre-existing
/analytics/* endpoints (app/api/analytics.py), which are DecisionTrace-driven and
belong to the older state-machine engine — this one is Call/Agent-driven (P4/P6),
reading from tables that engine never touches. All read-only; every number here
traces to a query in this file, never client-side math beyond formatting (DoD).
"""
import asyncio
import json
import uuid
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import Agent, Call, Customer
from app.security.deps import Principal, require
from app.services.aggregator import (agent_metrics_range, org_metrics_for_day,
                                     org_metrics_series, run_aggregator_tick)
from app.services.events import bus
from app.services.telephony.base import ACTIVE_STATUSES

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _pct_delta(current: float | None, prior: float | None) -> float | None:
    if current is None or prior is None or prior == 0:
        return None
    return round(100 * (current - prior) / prior, 1)


@router.get("/summary")
async def summary(p: Principal = Depends(require("analytics:read")),
                  db: AsyncSession = Depends(get_db)):
    """Task 157+163: active agents, calls today, resolved%, avg duration — each with
    a server-computed delta against the prior equivalent period (yesterday for a
    daily gauge/flow number), plus actions taken this week."""
    tid = p.tenant_id
    today = date.today()
    yesterday = today - timedelta(days=1)

    today_m = await org_metrics_for_day(db, tid, today)
    yest_m = await org_metrics_for_day(db, tid, yesterday)

    def resolved_pct(m: dict) -> float | None:
        return round(100 * m["resolved"] / m["completed"], 1) if m["completed"] else None

    today_resolved_pct = resolved_pct(today_m)
    yest_resolved_pct = resolved_pct(yest_m)

    week_start = today - timedelta(days=6)
    week_series = await org_metrics_series(db, tid, week_start, today)
    actions_this_week = sum(d["actions"] for d in week_series.values())

    return {
        "date": today,
        "active_agents": {"value": today_m["active_agents"],
                          "delta": today_m["active_agents"] - yest_m["active_agents"]},
        "calls_today": {"value": today_m["calls"],
                        "delta_pct": _pct_delta(today_m["calls"], yest_m["calls"])},
        "resolved_pct": {"value": today_resolved_pct,
                         "delta_pts": (round(today_resolved_pct - yest_resolved_pct, 1)
                                      if today_resolved_pct is not None
                                      and yest_resolved_pct is not None else None)},
        "avg_duration_seconds": {"value": today_m["avg_duration"],
                                 "delta_seconds": today_m["avg_duration"] - yest_m["avg_duration"]},
        "actions_taken": {"value": actions_this_week, "label": "this week"},
        "total_time_seconds": {"value": today_m["total_duration"],
                               "delta_seconds": today_m["total_duration"] - yest_m["total_duration"]},
    }


@router.get("/call-activity")
async def call_activity(days: int = Query(14, ge=1, le=90),
                        p: Principal = Depends(require("analytics:read")),
                        db: AsyncSession = Depends(get_db)):
    """Task 158: completed-vs-placed series for the Overview chart."""
    end = date.today()
    start = end - timedelta(days=days - 1)
    series = await org_metrics_series(db, p.tenant_id, start, end)
    return {"days": days,
           "series": [{"date": d, "placed": m["calls"], "completed": m["completed"]}
                      for d, m in sorted(series.items())]}


@router.get("/live")
async def live(p: Principal = Depends(require("analytics:read")),
              db: AsyncSession = Depends(get_db)):
    """Task 159: agents with an in-progress call right now, and the "N on shift now"
    banner count. Per-type avg is the in-progress duration so far, grouped by each
    agent's purpose_label (falling back to direction when no label is set)."""
    tid = p.tenant_id
    now = datetime.now(timezone.utc)
    rows = (await db.execute(
        select(Call, Agent, Customer)
        .join(Agent, Agent.id == Call.agent_id)
        .outerjoin(Customer, Customer.id == Call.customer_id)
        .where(Call.tenant_id == tid, Call.agent_id.is_not(None),
              Call.status.in_(ACTIVE_STATUSES)))).all()

    agents_live = []
    by_type: dict[str, list[float]] = {}
    for call, agent, customer in rows:
        started = call.answered_at or call.created_at
        duration = (now - started).total_seconds() if started else 0.0
        type_key = agent.purpose_label or agent.direction
        by_type.setdefault(type_key, []).append(duration)
        agents_live.append({
            "agent_id": str(agent.id), "agent_name": agent.name,
            "call_id": str(call.id), "customer_name": customer.full_name if customer else None,
            "purpose_label": agent.purpose_label, "direction": agent.direction,
            "status": call.status, "duration_so_far_seconds": int(duration)})

    return {"on_shift_now": len(agents_live), "agents": agents_live,
           "avg_duration_by_type": {k: int(sum(v) / len(v)) for k, v in by_type.items()}}


@router.get("/agent-performance")
async def agent_performance(range: str = Query("7d", pattern=r"^\d+d$"),
                            p: Principal = Depends(require("analytics:read")),
                            db: AsyncSession = Depends(get_db)):
    """Task 160: agent / purpose / calls / resolved% / trend / avg / status table.
    `trend` compares this range's call volume against the immediately preceding
    equivalent range (e.g. this-7d vs previous-7d) — up/down/flat."""
    tid = p.tenant_id
    days = int(range[:-1])
    today = date.today()
    start = today - timedelta(days=days - 1)
    prior_end = start - timedelta(days=1)
    prior_start = prior_end - timedelta(days=days - 1)

    current = await agent_metrics_range(db, tid, start, today)
    prior = await agent_metrics_range(db, tid, prior_start, prior_end)

    agents = (await db.execute(select(Agent).where(Agent.tenant_id == tid))).scalars().all()
    rows = []
    for a in agents:
        m = current.get(a.id, {"calls": 0, "resolved": 0, "actions": 0, "avg_duration": 0})
        prior_calls = prior.get(a.id, {}).get("calls", 0)
        if m["calls"] > prior_calls:
            trend = "up"
        elif m["calls"] < prior_calls:
            trend = "down"
        else:
            trend = "flat"
        rows.append({
            "agent_id": str(a.id), "agent_name": a.name, "purpose_label": a.purpose_label,
            "status": a.status, "calls": m["calls"],
            "resolved_pct": round(100 * m["resolved"] / m["calls"], 1) if m["calls"] else None,
            "trend": trend, "avg_duration_seconds": m["avg_duration"]})

    rows.sort(key=lambda r: r["calls"], reverse=True)
    return {"range": range, "from": start, "to": today, "agents": rows}


@router.post("/aggregate")
async def trigger_aggregation(p: Principal = Depends(require("analytics:read"))):
    """On-demand run of the same rollup the background loop does on a timer —
    useful for testing/demos without waiting AGGREGATOR_INTERVAL_SECONDS, and for
    scripts/verify_dashboard.py to make a fixture day's rollup deterministic."""
    written = await run_aggregator_tick()
    return {"rollups_written": written}


@router.get("/activity/stream")
async def activity_stream(request: Request, token: str = Query(...)):
    """Task 164: SSE push of live activity — the same per-tenant EventBus every
    other live feature already publishes through (app/services/events.py; see
    app/api/events_ws.py's WebSocket sibling), just over SSE instead of a socket, since
    the Overview page only ever needs server-to-client updates. Falls back to the
    existing GET /activity poll (frontend concern, not this endpoint's). Auth is a
    query-param token, not a header, for the same reason events_ws.py uses one — an
    EventSource/fetch-stream client cannot set a custom Authorization header on the
    initial request either."""
    import jwt as pyjwt

    from app.db.base import get_session_factory
    from app.db.models import User
    from app.security.auth import decode_token

    async def _tenant_for_token() -> uuid.UUID | None:
        try:
            payload = decode_token(token, expected_type="access")
        except pyjwt.PyJWTError:
            return None
        async with get_session_factory()() as db:
            user = (await db.execute(select(User).where(
                User.id == uuid.UUID(payload["sub"])))).scalar_one_or_none()
            return user.tenant_id if user and user.is_active else None

    tenant_id = await _tenant_for_token()
    if tenant_id is None:
        from fastapi import HTTPException, status
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")

    async def event_source():
        queue = bus.subscribe(tenant_id)
        try:
            recent = [e.as_json() for e in bus.recent(tenant_id)]
            yield f"event: connected\ndata: {json.dumps({'recent': recent}, default=str)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=25)
                    yield f"data: {json.dumps(event.as_json(), default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"  # SSE comment line — keeps proxies from dropping idle
        finally:
            bus.unsubscribe(tenant_id, queue)

    return StreamingResponse(event_source(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
