"""Analytics and audit — business KPIs, agent quality, and the explainability queries."""
import uuid
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.engine.state_machine.machine import load_definition
from app.db.models import (Appointment, AuditEvent, Call, Conversation, Customer,
                           DecisionTrace, Escalation, Ticket, ToolExecution)
from app.observability import pricing
from app.packs import PackRegistry
from app.security.deps import Principal, require

router = APIRouter(tags=["analytics & audit"])

CONNECTED_STATUSES = ("answered", "completed")


def _window(date_from: date | None, date_to: date | None) -> tuple[datetime, datetime]:
    end = datetime.combine(date_to or date.today(), time.max, tzinfo=timezone.utc)
    start = datetime.combine(date_from or (date.today() - timedelta(days=30)),
                             time.min, tzinfo=timezone.utc)
    return start, end


@router.get("/analytics/overview")
async def overview(p: Principal = Depends(require("analytics:read")),
                   db: AsyncSession = Depends(get_db),
                   date_from: date | None = Query(None, alias="from"),
                   date_to: date | None = Query(None, alias="to")):
    start, end = _window(date_from, date_to)
    tid = p.tenant_id

    async def count(model, *conditions):
        q = select(func.count()).select_from(model).where(model.tenant_id == tid, *conditions)
        return int((await db.execute(q)).scalar_one())

    calls_total = await count(Call, Call.created_at.between(start, end))
    calls_connected = await count(Call, Call.created_at.between(start, end),
                                  Call.status.in_(CONNECTED_STATUSES))
    conversations = await count(Conversation, Conversation.created_at.between(start, end))
    appts_booked = await count(Appointment, Appointment.created_at.between(start, end))
    appts_completed = await count(Appointment, Appointment.created_at.between(start, end),
                                  Appointment.status == "completed")
    no_shows = await count(Appointment, Appointment.created_at.between(start, end),
                           Appointment.status == "no_show")
    tickets_open = await count(Ticket, Ticket.status.in_(["open", "in_progress"]))
    tickets_resolved = await count(Ticket, Ticket.created_at.between(start, end),
                                   Ticket.status.in_(["resolved", "closed"]))
    escalations = await count(Escalation, Escalation.created_at.between(start, end))
    # Vocal Backend Tasksheet P7 Generic Sprint 4: "converted" is the real-estate pack's
    # own state name (its state machine literally has one called that); the generic
    # pack's equivalent terminal-success state is "closed". A hardcoded "converted"
    # here meant a generic-pack tenant's conversions were always silently zero.
    pack = await PackRegistry.for_tenant(db, tid)
    converted = await count(Customer, Customer.current_state == pack.conversion_state)

    return {
        "window": {"from": start.date(), "to": end.date()},
        "calls": {"total": calls_total, "connected": calls_connected,
                  "connection_rate": round(calls_connected / calls_total, 3) if calls_total else None},
        "conversations": conversations,
        "appointments": {"booked": appts_booked, "completed": appts_completed,
                         "no_shows": no_shows,
                         "attendance_rate": round(appts_completed / (appts_completed + no_shows), 3)
                         if (appts_completed + no_shows) else None},
        "conversion": {"customers_converted": converted,
                       "conversion_rate": round(appts_completed / conversations, 3)
                       if conversations else None},
        "support": {"tickets_open": tickets_open, "tickets_resolved": tickets_resolved,
                    "escalations": escalations},
    }


@router.get("/analytics/funnel")
async def funnel(p: Principal = Depends(require("analytics:read")),
                 db: AsyncSession = Depends(get_db)):
    """How many customers sit in each journey stage right now."""
    rows = (await db.execute(select(Customer.current_state, func.count()).where(
        Customer.tenant_id == p.tenant_id, Customer.is_deleted == False)  # noqa: E712
        .group_by(Customer.current_state))).all()
    by_state = {state: int(n) for state, n in rows}

    # The stage list comes from THIS TENANT'S state machine, not from a constant.
    #
    # It used to be a hardcoded generic journey — lead, contacted, qualified, interested,
    # appointment_booked and so on. The real-estate pack defines site_visit_scheduled,
    # site_visit_done, offer_shared, negotiation, booking_token, documentation. Almost
    # nothing matched, every real stage fell into other_states, and the funnel chart looked
    # empty on a database full of customers. A funnel that cannot see the journey it is
    # measuring is worse than no funnel, because it reads as "no pipeline" rather than as
    # a bug.
    #
    # Python preserves insertion order, so the definition's own ordering is the funnel's
    # ordering — the pack author already decided what comes after what.
    defn = await load_definition(db, p.tenant_id)
    states = defn.get("states", {}) or {}
    # Terminal states are outcomes, not steps along the way; do_not_contact in the middle
    # of a pipeline chart is noise. They are still reported, separately.
    order = [name for name, cfg in states.items() if not (cfg or {}).get("terminal")]
    terminal = [name for name, cfg in states.items() if (cfg or {}).get("terminal")]

    total = sum(by_state.values()) or 1
    known = set(order) | set(terminal)
    return {
        "total_customers": sum(by_state.values()),
        "journey": [{"stage": s, "count": by_state.get(s, 0),
                     "share": round(by_state.get(s, 0) / total, 3)} for s in order],
        "terminal": [{"stage": s, "count": by_state.get(s, 0),
                      "share": round(by_state.get(s, 0) / total, 3)}
                     for s in terminal if by_state.get(s, 0)],
        # Anything the state machine does not define. Should normally be empty — a value
        # here means a customer is parked in a stage no longer in the definition.
        "other_states": {k: v for k, v in by_state.items() if k not in known},
    }


@router.get("/analytics/agent")
async def agent_quality(p: Principal = Depends(require("analytics:read")),
                        db: AsyncSession = Depends(get_db),
                        date_from: date | None = Query(None, alias="from"),
                        date_to: date | None = Query(None, alias="to")):
    """How well the agent itself is performing — the numbers that decide whether it
    can be trusted with more autonomy."""
    start, end = _window(date_from, date_to)
    tid = p.tenant_id

    traces = (await db.execute(select(DecisionTrace).where(
        DecisionTrace.tenant_id == tid,
        DecisionTrace.created_at.between(start, end)))).scalars().all()
    total = len(traces)

    bands: dict[str, int] = {}
    actions: dict[str, int] = {}
    latencies: list[int] = []
    grounded_answers = ungrounded_answers = 0
    for t in traces:
        bands[t.confidence_band or "unknown"] = bands.get(t.confidence_band or "unknown", 0) + 1
        actions[t.action_taken or "unknown"] = actions.get(t.action_taken or "unknown", 0) + 1
        if t.latency_ms:
            latencies.append(t.latency_ms)
        if t.retrieved_chunks:
            grounded_answers += 1
        elif (t.action_taken or "") in ("spoke", "respond"):
            ungrounded_answers += 1

    latencies.sort()

    def pct(values: list[int], q: float) -> int | None:
        return values[min(int(len(values) * q), len(values) - 1)] if values else None

    tools = (await db.execute(select(ToolExecution.tool_name, ToolExecution.status,
                                     func.count()).where(
        ToolExecution.tenant_id == tid,
        ToolExecution.created_at.between(start, end))
        .group_by(ToolExecution.tool_name, ToolExecution.status))).all()
    tool_stats: dict[str, dict] = {}
    for name, tool_status, n in tools:
        entry = tool_stats.setdefault(name, {"success": 0, "blocked": 0, "failed": 0,
                                             "skipped_duplicate": 0})
        entry[tool_status] = entry.get(tool_status, 0) + int(n)
    for name, entry in tool_stats.items():
        attempted = sum(entry.values())
        entry["attempted"] = attempted
        entry["success_rate"] = round(entry.get("success", 0) / attempted, 3) if attempted else None

    escalations = int((await db.execute(select(func.count()).select_from(Escalation).where(
        Escalation.tenant_id == tid, Escalation.created_at.between(start, end)))).scalar_one())

    return {
        "window": {"from": start.date(), "to": end.date()},
        "turns_analysed": total,
        "confidence_bands": bands,
        "actions_taken": actions,
        "latency_ms": {"p50": pct(latencies, 0.5), "p95": pct(latencies, 0.95),
                       "max": latencies[-1] if latencies else None},
        "escalation_rate": round(escalations / total, 3) if total else None,
        "grounding": {"answers_with_knowledge": grounded_answers,
                      "answers_without_knowledge": ungrounded_answers},
        "tools": tool_stats,
    }


@router.get("/analytics/costs")
async def cost_analytics(p: Principal = Depends(require("analytics:read")),
                         db: AsyncSession = Depends(get_db),
                         date_from: date | None = Query(None, alias="from"),
                         date_to: date | None = Query(None, alias="to")):
    """What each turn actually cost — LLM tokens/$, TTS characters/$, STT seconds/$ — priced
    from DecisionTrace's raw usage via app/observability/pricing.py (never a stored $
    figure, see that module). Broken down BY model_used so a manually-switched model or
    provider (LLM_PROVIDER, or the per-provider *_MODEL_MAIN settings — see app/config.py)
    is directly comparable
    against the one used before the switch. Same windowed-DecisionTrace-aggregate approach
    already proven in /analytics/agent above."""
    start, end = _window(date_from, date_to)
    tid = p.tenant_id

    traces = (await db.execute(select(DecisionTrace).where(
        DecisionTrace.tenant_id == tid,
        DecisionTrace.created_at.between(start, end)))).scalars().all()
    total = len(traces)

    by_model: dict[str, dict] = {}
    daily: dict[str, dict] = {}
    total_llm = total_tts = total_stt = Decimal("0")
    total_tokens = 0

    for t in traces:
        cost = pricing.turn_cost(
            tokens_by_stage=t.tokens, model_used=t.model_used,
            tts_characters=t.tts_characters,
            stt_seconds=float(t.stt_seconds) if t.stt_seconds is not None else None)
        total_llm += cost.llm_usd
        total_tts += cost.tts_usd
        total_stt += cost.stt_usd
        turn_tokens = sum(int(stage.get("total") or 0) for stage in (t.tokens or {}).values())
        total_tokens += turn_tokens

        model_key = t.model_used or "unknown"
        entry = by_model.setdefault(model_key, {
            "calls": 0, "tokens": 0, "latencies": [],
            "llm_usd": Decimal("0"), "tts_usd": Decimal("0"), "stt_usd": Decimal("0")})
        entry["calls"] += 1
        entry["tokens"] += turn_tokens
        entry["llm_usd"] += cost.llm_usd
        entry["tts_usd"] += cost.tts_usd
        entry["stt_usd"] += cost.stt_usd
        if t.latency_ms:
            entry["latencies"].append(t.latency_ms)

        day_key = t.created_at.date().isoformat()
        day_entry = daily.setdefault(day_key, {"calls": 0, "total_usd": Decimal("0")})
        day_entry["calls"] += 1
        day_entry["total_usd"] += cost.total_usd

    by_model_out = {}
    for model_key, entry in by_model.items():
        latencies = sorted(entry.pop("latencies"))
        model_total = entry["llm_usd"] + entry["tts_usd"] + entry["stt_usd"]
        by_model_out[model_key] = {
            "calls": entry["calls"], "tokens": entry["tokens"],
            "avg_latency_ms": int(sum(latencies) / len(latencies)) if latencies else None,
            "llm_usd": str(entry["llm_usd"]), "tts_usd": str(entry["tts_usd"]),
            "stt_usd": str(entry["stt_usd"]), "total_usd": str(model_total)}

    daily_out = [{"date": d, "calls": v["calls"], "total_usd": str(v["total_usd"])}
                for d, v in sorted(daily.items())]

    total_cost = total_llm + total_tts + total_stt
    return {
        "window": {"from": start.date(), "to": end.date()},
        "turns_analysed": total,
        "totals": {
            "tokens": total_tokens,
            "llm_usd": str(total_llm), "tts_usd": str(total_tts), "stt_usd": str(total_stt),
            "total_usd": str(total_cost),
            "avg_cost_per_call_usd": str((total_cost / total).quantize(Decimal("0.000001")))
                                    if total else None},
        "by_model": by_model_out,
        "daily": daily_out,
    }


# ---------------- capabilities (P9 Generic Foundation handoff) ----------------

# Vocal Backend Tasksheet P9, task 181's own Definition of Done: "which P7 rollup
# each future chart/report will read from" — documented here as data (queried by
# GET /analytics/capabilities below), not just as a comment nobody building a future
# report would ever read. Each entry names the endpoint that already answers a
# question, and the table(s) that endpoint reads — so a future saved_reports config
# (or a hand-built chart) has a real source to point at instead of a guess.
_CAPABILITIES = [
    {"question": "Calls placed/completed and connection rate over a date range",
     "endpoint": "GET /analytics/overview", "reads_from": ["calls", "conversations", "appointments"]},
    {"question": "How many customers sit in each journey stage right now",
     "endpoint": "GET /analytics/funnel", "reads_from": ["customers", "state_machine_defs (via PackRegistry)"]},
    {"question": "Agent (planner) quality: confidence bands, latency, escalation rate, tool success",
     "endpoint": "GET /analytics/agent", "reads_from": ["decision_traces", "tool_executions", "escalations"]},
    {"question": "LLM/TTS/STT cost, by model and by day",
     "endpoint": "GET /analytics/costs", "reads_from": ["decision_traces"]},
    {"question": "Active agents, calls today, resolved %, avg duration — each with a "
                "period-comparison delta",
     "endpoint": "GET /dashboard/summary", "reads_from": ["daily_org_metrics", "calls (today, live)"]},
    {"question": "Completed-vs-placed call volume series over N days",
     "endpoint": "GET /dashboard/call-activity", "reads_from": ["daily_org_metrics", "calls (today, live)"]},
    {"question": "Who is on a call right now, and for how long",
     "endpoint": "GET /dashboard/live", "reads_from": ["calls", "agents"]},
    {"question": "Per-agent calls/resolved%/avg duration/trend over a range",
     "endpoint": "GET /dashboard/agent-performance",
     "reads_from": ["agent_metrics_daily", "calls (today, live)"]},
    {"question": "Live activity feed (new calls, escalations, tool connects, …)",
     "endpoint": "GET /dashboard/activity/stream (SSE) or GET /activity (poll fallback)",
     "reads_from": ["audit_events", "in-process EventBus (app/services/events.py)"]},
    {"question": "Org billing: plan, seats used vs purchased, renewal date",
     "endpoint": "GET /billing/subscription", "reads_from": ["tenants", "subscriptions", "users"]},
]


@router.get("/analytics/capabilities")
async def capabilities(p: Principal = Depends(require("analytics:read"))):
    """Task 179: what P7's rollups (and the pre-existing DecisionTrace-driven
    analytics) can already answer — a meta endpoint for a future Reports page to
    discover what it can build a saved_report config against, and the permanent,
    queryable form of task 181's handoff-documentation Definition of Done."""
    return {"capabilities": _CAPABILITIES}


# ---------------- audit ----------------

@router.get("/audit/traces/{trace_id}")
async def get_trace(trace_id: uuid.UUID, p: Principal = Depends(require("audit:read")),
                    db: AsyncSession = Depends(get_db)):
    """The complete record of one decision — why the agent did what it did."""
    t = (await db.execute(select(DecisionTrace).where(
        DecisionTrace.tenant_id == p.tenant_id,
        DecisionTrace.id == trace_id))).scalar_one_or_none()
    if t is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Trace not found")
    executions = (await db.execute(select(ToolExecution).where(
        ToolExecution.tenant_id == p.tenant_id,
        ToolExecution.decision_trace_id == trace_id))).scalars().all()
    return {
        "trace_id": str(t.id), "conversation_id": str(t.conversation_id),
        "customer_id": str(t.customer_id) if t.customer_id else None,
        "turn": t.turn_index, "customer_said": t.input_text,
        "memory_consulted": t.retrieved_memory,
        "knowledge_retrieved": t.retrieved_chunks,
        "rules_evaluated": t.rules_evaluated,
        "rules_matched": [r for r in (t.rules_evaluated or []) if r.get("matched")],
        "agent_reasoning": t.planner_reasoning,
        "proposed_tool": t.proposed_tool, "tool_params": t.tool_params,
        "validation": t.validation_result,
        "confidence": {"score": float(t.confidence) if t.confidence else None,
                       "band": t.confidence_band},
        "action_taken": t.action_taken,
        "tool_executions": [{"tool": e.tool_name, "status": e.status, "error": e.error,
                             "duration_ms": e.duration_ms} for e in executions],
        "performance": {
            "latency_ms": t.latency_ms, "model": t.model_used, "tokens": t.tokens,
            "tts_characters": t.tts_characters,
            "stt_seconds": float(t.stt_seconds) if t.stt_seconds is not None else None,
            "cost_usd": pricing.turn_cost(
                tokens_by_stage=t.tokens, model_used=t.model_used,
                tts_characters=t.tts_characters,
                stt_seconds=float(t.stt_seconds) if t.stt_seconds is not None else None,
            ).as_dict()},
        "at": t.created_at,
    }


@router.get("/audit/blocked")
async def blocked_actions(p: Principal = Depends(require("audit:read")),
                          db: AsyncSession = Depends(get_db),
                          date_from: date | None = Query(None, alias="from"),
                          date_to: date | None = Query(None, alias="to"),
                          limit: int = Query(100, ge=1, le=500)):
    """Everything policy stopped, and which rule stopped it — the compliance report."""
    start, end = _window(date_from, date_to)
    tid = p.tenant_id

    traces = (await db.execute(select(DecisionTrace).where(
        DecisionTrace.tenant_id == tid, DecisionTrace.created_at.between(start, end),
        DecisionTrace.action_taken.in_(["blocked", "escalated"]))
        .order_by(DecisionTrace.created_at.desc()).limit(limit))).scalars().all()

    blocked_tools = (await db.execute(select(ToolExecution).where(
        ToolExecution.tenant_id == tid, ToolExecution.created_at.between(start, end),
        ToolExecution.status == "blocked")
        .order_by(ToolExecution.created_at.desc()).limit(limit))).scalars().all()

    blocked_calls = (await db.execute(select(AuditEvent).where(
        AuditEvent.tenant_id == tid, AuditEvent.created_at.between(start, end),
        AuditEvent.action == "calls.outbound_trigger")
        .order_by(AuditEvent.created_at.desc()).limit(limit))).scalars().all()

    return {
        "window": {"from": start.date(), "to": end.date()},
        "blocked_turns": [{"trace_id": str(t.id), "customer_id": str(t.customer_id) if t.customer_id else None,
                           "input": t.input_text, "action": t.action_taken,
                           "rules": [r.get("rule") for r in (t.rules_evaluated or [])
                                     if r.get("matched")], "at": t.created_at}
                          for t in traces],
        "blocked_tool_calls": [{"tool": e.tool_name, "error": e.error,
                                "customer_id": str(e.customer_id) if e.customer_id else None,
                                "at": e.created_at} for e in blocked_tools],
        "blocked_outbound_calls": [{"customer_id": e.resource_id,
                                    "blocked_by": (e.after or {}).get("blocked_by"),
                                    "allowed": (e.after or {}).get("allowed"),
                                    "at": e.created_at}
                                   for e in blocked_calls if (e.after or {}).get("allowed") is False],
    }


@router.get("/audit/events")
async def audit_events(p: Principal = Depends(require("audit:read")),
                       db: AsyncSession = Depends(get_db),
                       action: str | None = None, actor: str | None = None,
                       date_from: date | None = Query(None, alias="from"),
                       date_to: date | None = Query(None, alias="to"),
                       page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200)):
    """Who changed what, when — configuration and data-access history."""
    start, end = _window(date_from, date_to)
    q = select(AuditEvent).where(AuditEvent.tenant_id == p.tenant_id,
                                 AuditEvent.created_at.between(start, end))
    if action:
        q = q.where(AuditEvent.action.startswith(action))
    if actor:
        q = q.where(AuditEvent.actor.contains(actor))
    total = int((await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one())
    rows = (await db.execute(q.order_by(AuditEvent.created_at.desc())
                             .offset((page - 1) * page_size).limit(page_size))).scalars().all()
    return {"items": [{"id": str(e.id), "actor": e.actor, "action": e.action,
                       "resource_type": e.resource_type, "resource_id": e.resource_id,
                       "before": e.before, "after": e.after, "at": e.created_at}
                      for e in rows],
            "page": page, "page_size": page_size, "total": total}