"""Call Runtime Service (Vocal Backend Tasksheet P6) — resolves an Agent (P4) to its
published template_version (P3), runs the compiled LangGraph call-flow graph for one
call end-to-end, and persists everything the tasksheet asks this wave to produce onto
a real `calls` row: the node-transition trace (call_events), per-step captures
(call_captures), tool invocations (tool_executions — P6's "tool_call_logs", reconciled
onto that pre-existing table, see its own docstring), and a post-call summary.

SCOPE FOR THIS WAVE (confirmed with the user before building this): a SIMULATED turn
loop. The "STT/TTS turn loop" the tasksheet asks for is, for the four node types P3's
engine actually has today (SAY/THINK/TOOL/RETRIEVE — there is no LISTEN/interactive
node type yet, so a call is not genuinely multi-turn-interactive at the graph level),
already fully exercised by seeding one scripted customer utterance and letting the
graph run to completion — exactly scripts/verify_templates.py's own "dry-run against a
scripted fake caller" pattern, just wired onto a real Call row instead of a throwaway
one. No real audio I/O (pipecat/Deepgram/ElevenLabs) is wired here; that pipeline
already exists for the OLDER state-machine engine (app/api/voice_ws.py) and connecting
an Agent's compiled graph to real speech is future work, not this wave's.

Tracing uses LangGraph's own `astream(..., stream_mode="updates")` (verified live
against the actual installed langgraph==1.2.11 before writing this, per this project's
own standing lesson about a fast-moving library's training-data reliability) rather
than modifying anything in app/engine/templates/{compiler,nodes}.py: that graph object
is cross-call CACHED by (tenant_id, graph_hash) in GraphCompiler, so baking a
call-specific tracer closure into the node functions themselves would leak one call's
identity into every later call that reuses the same cached graph — astream() gives a
per-node update sequence from OUTSIDE the cached graph, with no such risk.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (Agent, AnalyticsEvent, AuditEvent, Call, CallCapture, CallEvent,
                           Customer, PhoneNumber, TemplateVersion, ToolExecution)
from app.engine.templates.compiler import GraphCompiler
from app.engine.templates.nodes import _render
from app.engine.templates.state import Turn, new_call_state
from app.providers.llm.base import LLMError
from app.providers.llm.router import get_llm
from app.services.events import publish
from app.services.phone import normalize_phone
from app.services.telephony.base import CallEvent as TelephonyCallEvent
from app.services.telephony.base import STATUS_ANSWERED, STATUS_COMPLETED, STATUS_RINGING
from app.services.telephony.events import apply_call_event

logger = logging.getLogger(__name__)


class CallRuntimeError(Exception):
    """A precondition the caller (an API endpoint, typically) should turn into a 4xx —
    not an infrastructure fault. Distinguishes "this agent isn't callable" from a
    genuine bug, the same split app/services/telephony/base.py's TelephonyError draws."""


async def resolve_agent_by_number(db: AsyncSession, tenant_id: uuid.UUID,
                                  to_number: str) -> Agent | None:
    """The agent whose assigned phone number matches the dialled number — task 140's
    "resolve agent by to_number". None when no agent owns that number (a real carrier
    webhook would then 404, exactly like telephony_hooks.py's unknown-call handling)."""
    normalized = normalize_phone(to_number)
    return (await db.execute(
        select(Agent).join(PhoneNumber, PhoneNumber.id == Agent.phone_number_id)
        .where(Agent.tenant_id == tenant_id, PhoneNumber.e164 == normalized,
               Agent.is_deleted == False))).scalar_one_or_none()  # noqa: E712


async def resolve_tenant_and_agent_by_number(db: AsyncSession,
                                             to_number: str) -> tuple[uuid.UUID, Agent] | None:
    """Same resolution as resolve_agent_by_number, for the one caller that genuinely
    cannot know the tenant ahead of time: an inbound telephony webhook. A carrier
    dials a number, not a tenant — the number IS the tenant lookup, exactly how
    app/api/telephony_hooks.py's existing webhook resolves a Call's tenant from the
    matched row rather than from anything the carrier supplied."""
    normalized = normalize_phone(to_number)
    row = (await db.execute(
        select(PhoneNumber.tenant_id, Agent)
        .join(Agent, Agent.phone_number_id == PhoneNumber.id)
        .where(PhoneNumber.e164 == normalized,
               Agent.is_deleted == False))).first()  # noqa: E712
    return (row[0], row[1]) if row else None


async def run_call(db: AsyncSession, *, tenant_id: uuid.UUID, agent: Agent,
                   from_number: str, to_number: str, opening_utterance: str,
                   actor: str = "call_runtime", interrupt_before: list[str] | None = None) -> Call:
    """Runs one simulated inbound call against `agent`'s published template, and
    returns the Call row (still uncommitted — caller commits, matching every other
    service function in this codebase). `interrupt_before` (a list of step keys) is
    for exercising checkpoint-resume (see resume_call below) — normal calls leave it
    None and run to completion in one pass."""
    if agent.status != "live":
        raise CallRuntimeError(f"agent {agent.id} is not live (status={agent.status})")
    if agent.template_version_id is None:
        raise CallRuntimeError(f"agent {agent.id} has no template_version assigned")

    version = (await db.execute(select(TemplateVersion).where(
        TemplateVersion.tenant_id == tenant_id,
        TemplateVersion.id == agent.template_version_id))).scalar_one_or_none()
    if version is None:
        raise CallRuntimeError(f"template_version {agent.template_version_id} not found")
    steps: list[dict] = version.snapshot
    step_by_key = {s["key"]: s for s in steps}

    from_normalized = normalize_phone(from_number)
    customer = (await db.execute(select(Customer).where(
        Customer.tenant_id == tenant_id, Customer.phone_normalized == from_normalized,
        Customer.is_deleted == False))).scalar_one_or_none()  # noqa: E712

    call = Call(tenant_id=tenant_id, agent_id=agent.id, template_version_id=version.id,
               customer_id=customer.id if customer else None, provider="call_runtime",
               from_number=from_normalized, to_number=normalize_phone(to_number),
               purpose="service", status="queued", attempt=1)
    db.add(call)
    await db.flush()  # need call.id for every row below
    call.provider_call_id = str(call.id)

    await apply_call_event(db, call, TelephonyCallEvent(
        provider="call_runtime", provider_call_id=str(call.id), status=STATUS_RINGING))
    await apply_call_event(db, call, TelephonyCallEvent(
        provider="call_runtime", provider_call_id=str(call.id), status=STATUS_ANSWERED))

    state = new_call_state(str(call.id))
    state["transcript"].append(Turn(role="customer", text=opening_utterance))

    final_state, finished = await _run_traced_safely(
        db, tenant_id=tenant_id, call=call, steps=steps, step_by_key=step_by_key,
        astream_input=state, interrupt_before=interrupt_before, seed_state=state)
    call.transcript = final_state.get("transcript", [])
    if not finished:
        # Interrupted before completion (test-only path) — leave the call mid-flight;
        # resume_call() picks up exactly here. No summary yet: the call hasn't ended.
        return call
    await _finalize_call(db, tenant_id=tenant_id, call=call, final_state=final_state,
                         agent_id=agent.id, actor=actor)
    return call


async def resume_call(db: AsyncSession, *, tenant_id: uuid.UUID, call_id: uuid.UUID,
                      actor: str = "call_runtime") -> Call:
    """Resumes a call previously interrupted by run_call(interrupt_before=...) from
    exactly where its LangGraph checkpoint left off — the Call Runtime Service's own
    version of Wave 4's proven interrupt/ainvoke(None)/resume mechanic (Definition of
    Done: "Kill runtime mid-call, resume, confirm identical graph state")."""
    call = (await db.execute(select(Call).where(
        Call.tenant_id == tenant_id, Call.id == call_id))).scalar_one_or_none()
    if call is None:
        raise CallRuntimeError(f"call {call_id} not found")
    if call.template_version_id is None:
        raise CallRuntimeError(f"call {call_id} has no template_version pinned — nothing to resume")

    version = (await db.execute(select(TemplateVersion).where(
        TemplateVersion.id == call.template_version_id))).scalar_one_or_none()
    if version is None:
        raise CallRuntimeError(f"template_version {call.template_version_id} not found")
    steps: list[dict] = version.snapshot
    step_by_key = {s["key"]: s for s in steps}

    final_state, finished = await _run_traced_safely(
        db, tenant_id=tenant_id, call=call, steps=steps, step_by_key=step_by_key,
        astream_input=None, interrupt_before=None, seed_state=None)
    call.transcript = final_state.get("transcript", [])
    if not finished:
        return call
    await _finalize_call(db, tenant_id=tenant_id, call=call, final_state=final_state,
                         agent_id=call.agent_id, actor=actor)
    return call


async def _finalize_call(db: AsyncSession, *, tenant_id: uuid.UUID, call: Call, final_state: dict,
                         agent_id: uuid.UUID | None, actor: str) -> None:
    """The once-the-graph-is-actually-done tail shared by run_call and resume_call:
    the isolated post-call summarization pass (task 149), the terminal status
    transition, and the audit trail."""
    summary = await _summarize_call(final_state.get("transcript", []))
    handoff = bool(final_state.get("handoff"))
    call.summary = summary["summary"]
    call.intent = summary["intent"] or final_state.get("intent")
    call.sentiment = summary["sentiment"] or final_state.get("sentiment")
    # A handoff (a guardrail denial, most commonly) is itself a deterministic, always-known
    # signal that a human needs to look at this call — that must hold regardless of whether
    # the LLM summarization pass above even ran, let alone whether it agreed.
    call.follow_up_required = handoff or summary["follow_up_required"]
    call.follow_up_reason = summary["follow_up_reason"] or ("handed off mid-call" if handoff else None)

    await apply_call_event(db, call, TelephonyCallEvent(
        provider="call_runtime", provider_call_id=str(call.id), status=STATUS_COMPLETED,
        hangup_cause="handoff" if handoff else "completed"))

    db.add(AuditEvent(tenant_id=tenant_id, actor=actor, action="calls.agent_run",
                      resource_type="call", resource_id=str(call.id),
                      after={"agent_id": str(agent_id) if agent_id else None,
                            "handoff": final_state.get("handoff", False)}))
    publish(tenant_id, "call.agent_run_completed", call_id=str(call.id),
           agent_id=str(agent_id) if agent_id else None, handoff=final_state.get("handoff", False))


async def _run_traced_safely(db: AsyncSession, *, tenant_id: uuid.UUID, call: Call,
                             steps: list[dict], step_by_key: dict[str, dict],
                             astream_input: dict | None, interrupt_before: list[str] | None,
                             seed_state: dict | None) -> tuple[dict, bool]:
    try:
        return await _run_traced(db, tenant_id=tenant_id, call=call, steps=steps,
                                 step_by_key=step_by_key, astream_input=astream_input,
                                 interrupt_before=interrupt_before, seed_state=seed_state)
    except Exception as e:
        logger.exception("call runtime graph run failed (call=%s)", call.id)
        db.add(CallEvent(tenant_id=tenant_id, call_id=call.id, step_key="__runtime__",
                         step_type="ERROR", input={}, output={}, error=str(e)))
        await apply_call_event(db, call, TelephonyCallEvent(
            provider="call_runtime", provider_call_id=str(call.id), status=STATUS_COMPLETED,
            hangup_cause="runtime_error"))
        raise


async def _run_traced(db: AsyncSession, *, tenant_id: uuid.UUID, call: Call, steps: list[dict],
                      step_by_key: dict[str, dict], astream_input: dict | None,
                      interrupt_before: list[str] | None, seed_state: dict | None,
                      ) -> tuple[dict, bool]:
    """Runs the compiled (cache-shared) graph via astream(stream_mode="updates"),
    writing one call_events row per node — and, for TOOL/THINK nodes, a matching
    tool_executions/call_captures row — without ever touching the cached graph
    object itself (see this module's docstring for why). `astream_input=None` means
    RESUME (LangGraph's own convention for "continue from the last checkpoint");
    `seed_state` seeds this function's own input/output logging in that case, since
    resuming has no fresh initial_state to start from — the checkpointer's own
    `aget_state` supplies it when the caller doesn't (see resume_call).

    Returns (final_state, finished) — `finished` is False when `interrupt_before`
    stopped the run early (the checkpoint is left exactly where LangGraph stopped
    it; a caller runs this again with `astream_input=None` to continue)."""
    graph, _ = GraphCompiler.compile(steps, tenant_id)
    from app.engine.templates.checkpointer import open_checkpointer

    async with open_checkpointer() as saver:
        await saver.setup()
        compiled = graph.compile(checkpointer=saver, interrupt_before=interrupt_before or [])
        config = {"configurable": {"thread_id": str(call.id)}}

        if seed_state is not None:
            state: dict[str, Any] = dict(seed_state)
        else:
            snap = await compiled.aget_state(config)
            state = dict(snap.values) if snap.values else {}

        t0 = time.monotonic()
        async for chunk in compiled.astream(astream_input, config=config, stream_mode="updates"):
            latency_ms = int((time.monotonic() - t0) * 1000)
            t0 = time.monotonic()
            for step_key, delta in chunk.items():
                if step_key not in step_by_key:
                    # LangGraph's own internal sentinel keys (e.g. "__interrupt__", yielded
                    # the moment an interrupt_before pause fires) — not a real template step,
                    # so nothing to trace and nothing meaningful to merge into CallState.
                    continue
                step = step_by_key[step_key]
                step_type = step.get("step_type", "?")
                state_before = dict(state)
                db.add(CallEvent(tenant_id=tenant_id, call_id=call.id, step_key=step_key,
                                 step_type=step_type, input=_loggable(state_before),
                                 output=_loggable(delta), latency_ms=latency_ms))

                # Any step type can now carry a generic `config.captures` list (see
                # app/engine/templates/nodes.py's _apply_generic_captures) — not just
                # THINK's older singular `capture_key`. One CallCapture row per key
                # that this node's delta actually touched, whichever config style
                # produced it; `dict.fromkeys` on the concatenation dedupes the case
                # where a legacy `capture_key` also appears in `captures`.
                step_config = step.get("config") or {}
                capture_keys = list(dict.fromkeys(
                    ([step_config["capture_key"]] if step_config.get("capture_key") else [])
                    + (step_config.get("captures") or [])))
                for capture_key in capture_keys:
                    if capture_key in (delta.get("captures") or {}):
                        db.add(CallCapture(tenant_id=tenant_id, call_id=call.id, step_key=step_key,
                                           capture_key=capture_key,
                                           value=_loggable(delta["captures"][capture_key])))

                if step_type == "TOOL":
                    await _log_tool_call(db, tenant_id=tenant_id, call=call, step=step,
                                         step_key=step_key, state_before=state_before,
                                         delta=delta, latency_ms=latency_ms)

                state.update(delta)
        await db.flush()

        snap_after = await compiled.aget_state(config)
        finished = not snap_after.next
    return state, finished


async def _log_tool_call(db: AsyncSession, *, tenant_id: uuid.UUID, call: Call, step: dict,
                         step_key: str, state_before: dict, delta: dict, latency_ms: int) -> None:
    config = step.get("config") or {}
    action = config.get("action", "?")
    result = (delta.get("tool_results") or {}).get(action, {})
    payload_template: dict = config.get("payload_template", {})
    rendered_input = {k: _render(str(v), state_before) for k, v in payload_template.items()}

    if "denied" in result:
        status_, error = "blocked", str(result.get("denied"))
    elif result.get("success"):
        status_, error = "success", None
    else:
        status_, error = "failed", result.get("error")

    db.add(ToolExecution(tenant_id=tenant_id, call_id=call.id, step_key=step_key,
                         customer_id=call.customer_id, tool_name=action,
                         input=_loggable(rendered_input), output=_loggable(result),
                         status=status_, error=error, duration_ms=latency_ms))
    db.add(AnalyticsEvent(tenant_id=tenant_id, event_type="tool_call", customer_id=call.customer_id,
                          properties={"call_id": str(call.id), "tool_name": action, "status": status_}))


_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "intent": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "follow_up_required": {"type": "boolean"},
        "follow_up_reason": {"type": "string"},
    },
    "required": ["summary", "intent", "sentiment", "follow_up_required"],
}

_SUMMARY_SYSTEM_PROMPT = (
    "You summarize a finished customer service call transcript. Write a brief (1-3 "
    "sentence) neutral summary, the caller's primary intent as a short label, the "
    "overall sentiment, and whether a human follow-up is needed (and why, if so). "
    "Never invent details the transcript does not support.")


async def _summarize_call(transcript: list[dict]) -> dict:
    """Task 149's post-call summarization pass — ISOLATED from the deterministic
    graph run above (a separate LLM call after the call is already over, never in
    the control-flow path itself). Degrades honestly on an LLM failure — no summary
    is not a crash, matching app/engine/templates/nodes.py's think_node_factory's
    own precedent for handling LLMError."""
    default = {"summary": None, "intent": None, "sentiment": None,
              "follow_up_required": False, "follow_up_reason": None}
    if not transcript:
        return default
    user_text = "\n".join(f"{t['role']}: {t['text']}" for t in transcript)
    try:
        result = await get_llm().complete_json(system=_SUMMARY_SYSTEM_PROMPT, user=user_text,
                                               schema=_SUMMARY_SCHEMA, schema_name="call_summary")
        data = result.data
        return {"summary": data.get("summary"), "intent": data.get("intent"),
               "sentiment": data.get("sentiment"),
               "follow_up_required": bool(data.get("follow_up_required", False)),
               "follow_up_reason": data.get("follow_up_reason")}
    except LLMError as e:
        logger.warning("post-call summarization failed: %s", e)
        return default


def _loggable(value: Any) -> dict | list:
    """JSONB columns need a dict/list/scalar, never a bare non-JSON-serialisable
    object — every state/delta value here already is one (CallState is a plain
    TypedDict of JSON-safe values), this just guards a non-dict top-level value
    (e.g. a capture's own value) so it still lands in a JSONB column without error."""
    return value if isinstance(value, (dict, list)) else {"value": value}
