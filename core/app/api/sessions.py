"""Session lifecycle wrapper — one shape for every channel.

A "session" is a call from the engine's point of view: it starts, takes turns, and
ends. Chat, browser-mic voice, and Plivo telephony all drive the SAME four endpoints;
only `channel` differs. That is the point of this module — Stage 10 and Stage 11 plug
in here instead of each inventing their own orchestration.

Two things this adds over calling /conversations directly:

1. Outbound sessions pass through services.outbound, so a test call hits the same
   DNC / open-escalation / calling-hours gate a real dial will. A harness that skips
   the gate would lie to you about the exact machinery Stage 9 exists to prove.

2. Date simulation is held on the session, so you can rewind the clock once at start
   and every subsequent turn retrieves knowledge as of that date.
"""
import re
import time
import uuid
from datetime import date, datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import Call, Conversation, Customer, DecisionTrace, Message, Tenant
from app.engine.memory import working
from app.engine.memory.pipeline import run_post_call, run_post_call_task
from app.engine.planner import outbound_flow
from app.engine.planner.turn_runner import process_turn
from app.security.deps import Principal, require
from app.services.outbound import place_outbound_call
from app.services.phone import normalize_phone

router = APIRouter(prefix="/sessions", tags=["sessions"])




# On an inbound call the agent speaks first — there is no caller waiting to be prompted.
# This belongs in the API rather than in whatever client happens to be connected: the
# telephony stage has no page to invent a greeting, and the transcript stored against the
# call has to match what the caller actually heard.
def _default_greeting(tenant_name: str | None, customer) -> str:
    business = (tenant_name or "").strip() or "our service centre"
    first = ((getattr(customer, "full_name", "") or "").strip().split(" ") or [""])[0]
    if first:
        return f"Thanks for calling {business}. Hello {first} — how can I help you today?"
    return f"Thanks for calling {business}. How can I help you today?"


def with_recording_disclosure(opening: str | None, tenant) -> str | None:
    """The voice call's opening line with the tenant's recording disclosure spoken in it.

    A tenant that records calls sets ``settings.recording_disclosure`` (e.g. "This call may be
    recorded for quality and training purposes."). It is placed as its own sentence just
    before the opening's closing question, so the caller hears it before saying anything and
    the line still ends on the question they answer. Unset -> the opening is unchanged.

    WHY HERE AND NOT AS A RULE: the disclosure is owed before the caller's first word, and the
    agent's opening line is the only thing guaranteed to run then. It used to be a turn rule
    (DSR's recording_disclosure_first) that could never fire -- it keyed on turn_index == 0,
    but the caller's first turn is 1 -- and whose flag nothing could ever set; had it fired, it
    would have replaced the caller's first request with a generic "couple of details" ask and
    still never spoken the disclosure.
    """
    disclosure = " ".join(str(((getattr(tenant, "settings", None) or {})
                               .get("recording_disclosure") or "")).split())
    if not opening or not disclosure:
        return opening
    sentences = re.split(r"(?<=[.!?])\s+", opening.strip())
    if len(sentences) > 1 and sentences[-1].endswith("?"):
        return " ".join([*sentences[:-1], disclosure, sentences[-1]])
    return f"{opening.strip()} {disclosure}"


# --------------------------------------------------------------------------- start


class SessionStart(BaseModel):
    customer_id: uuid.UUID | None = None
    phone: str | None = None
    channel: str = "chat"            # chat | voice
    direction: str = "inbound"       # inbound | outbound
    purpose: str = "marketing"       # only used for outbound policy evaluation
    as_of_date: date | None = None   # simulate "today" for knowledge expiry
    opening_line: str | None = None  # agent speaks first (outbound); recorded, not generated
    enforce_outbound_policy: bool = True


@router.post("", status_code=201)
async def start_session(body: SessionStart,
                        p: Principal = Depends(require("conversations:read")),
                        db: AsyncSession = Depends(get_db)):
    """Begin a session. Outbound goes through the dial gate and can be refused.

    A refusal is a 403 carrying the rule that blocked it — the harness shows you the
    same answer the telephony stage will get, without a phone existing yet.
    """
    customer = await _resolve_customer(db, p.tenant_id, body.customer_id, body.phone)

    if body.direction not in ("inbound", "outbound"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "direction must be inbound or outbound")

    call_id = None
    policy: dict = {"evaluated": False}

    if body.direction == "outbound" and body.enforce_outbound_policy:
        if customer is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                "An outbound session needs a known customer to evaluate policy")
        decision = await place_outbound_call(db, tenant_id=p.tenant_id, customer=customer,
                                             purpose=body.purpose, actor=p.actor)
        policy = {"evaluated": True, "allowed": decision.allowed,
                  "blocked_by": decision.blocked_by, "reason": decision.reason}
        if not decision.allowed:
            await db.commit()   # keep the call_blocked analytics event
            raise HTTPException(status.HTTP_403_FORBIDDEN,
                                {"error": "outbound_blocked", **policy})
        call_id = decision.call_id

    conv = Conversation(tenant_id=p.tenant_id,
                        customer_id=customer.id if customer else None,
                        channel=body.channel, direction=body.direction, status="active",
                        started_at=datetime.now(timezone.utc),
                        # Durable copy of an unknown caller's number, normalised to the shape
                        # create_customer stores. Written to Postgres here so the number the
                        # caller dialled from survives a Redis eviction/restart/TTL lapse — the
                        # session cache below is now just a fast path, not the only record.
                        # None for a known caller: their number is already on the customer row.
                        caller_phone=(normalize_phone(body.phone)
                                      if customer is None and body.phone else None))
    db.add(conv)
    await db.flush()

    if call_id is not None:
        call = (await db.execute(select(Call).where(Call.id == call_id))).scalar_one()
        call.conversation_id = conv.id
        call.status = "answered"            # the harness stands in for the carrier
        call.answered_at = datetime.now(timezone.utc)

    # An explicit opening_line wins; otherwise the agent speaks first either way. Inbound gets
    # the standard greeting; outbound opens by confirming it reached the right person (the
    # identity gate in turn_runner then reads the caller's answer). The agent speaking first on
    # outbound is what makes the identity precondition testable in the chat console exactly as
    # it runs on a real call.
    opening = body.opening_line
    tenant = None
    if not opening and body.direction in ("inbound", "outbound"):
        tenant = (await db.execute(
            select(Tenant).where(Tenant.id == p.tenant_id))).scalar_one_or_none()
        if body.direction == "outbound":
            opening = outbound_flow.identity_opener(
                getattr(tenant, "name", None), getattr(customer, "full_name", None))
        else:
            opening = _default_greeting(getattr(tenant, "name", None), customer)
    if body.channel == "voice" and opening:
        # Same opening a real voice call speaks -- recording disclosure included.
        if tenant is None:
            tenant = (await db.execute(
                select(Tenant).where(Tenant.id == p.tenant_id))).scalar_one_or_none()
        opening = with_recording_disclosure(opening, tenant)

    if opening:
        db.add(Message(tenant_id=p.tenant_id, conversation_id=conv.id,
                       role="agent", content=opening))

    await db.commit()

    # Seed working memory so every turn inherits the simulated date without resending it.
    session = await working.load(conv.id)
    dirty = False
    if body.as_of_date:
        session["as_of_date"] = body.as_of_date.isoformat()
        dirty = True
    if customer is None and body.phone:
        # The caller's number, kept for the length of the call.
        #
        # With no customer record this is the ONLY place their number exists — a
        # Conversation has no phone column. Without it, registering them mid-call means
        # asking for a number they have already dialled from, which is the sort of question
        # that makes an agent feel like a form.
        session["caller_phone"] = body.phone
        dirty = True
    if dirty:
        await working.save(conv.id, session)

    return {
        "session_id": str(conv.id),
        "call_id": str(call_id) if call_id else None,
        "channel": conv.channel, "direction": conv.direction, "status": conv.status,
        "as_of_date": body.as_of_date,
        "customer": _customer_brief(customer),
        "policy": policy,
        "opening_line": opening,
    }


# --------------------------------------------------------------------------- turn


class SessionTurn(BaseModel):
    text: str
    stt_confidence: float | None = None
    as_of_date: date | None = None   # overrides the session date for this turn only


@router.post("/{session_id}/turns")
async def take_session_turn(session_id: uuid.UUID, body: SessionTurn,
                            p: Principal = Depends(require("conversations:read")),
                            db: AsyncSession = Depends(get_db)):
    """One turn, with the full debug envelope the test UI renders in its side panel."""
    conv = await _conv(db, p.tenant_id, session_id)
    if conv.status != "active":
        raise HTTPException(status.HTTP_409_CONFLICT, f"Session is {conv.status}")

    session = await working.load(conv.id)
    as_of = body.as_of_date or _parse_date(session.get("as_of_date"))

    started = time.perf_counter()
    result = await process_turn(db, tenant_id=p.tenant_id, conversation=conv,
                                user_input=body.text, stt_confidence=body.stt_confidence,
                                actor=p.actor, as_of_date=as_of)
    wall_ms = int((time.perf_counter() - started) * 1000)

    debug = dict(result.debug or {})
    timing = dict(debug.get("timing") or {})
    timing["wall_ms"] = wall_ms
    debug["timing"] = timing

    return {
        "session_id": str(conv.id),
        "response_text": result.response_text,
        "response_segments": result.response_segments,
        "action": result.action,
        "executed_tools": result.executed_tools,
        "state_changed_to": result.state_changed_to,
        "escalation": result.escalation,
        "end_call": result.end_call,
        "trace_id": str(result.trace_id) if result.trace_id else None,
        "confidence": result.confidence,
        "pending_confirmation": result.pending_confirmation,
        "as_of_date": as_of,
        "debug": debug,
    }


# --------------------------------------------------------------------------- end


class SessionEnd(BaseModel):
    outcome: str | None = None
    hangup_cause: str | None = None
    analyse: str = "sync"   # sync | background | skip


@router.post("/{session_id}/end")
async def end_session(session_id: uuid.UUID, body: SessionEnd, background: BackgroundTasks,
                      p: Principal = Depends(require("conversations:read")),
                      db: AsyncSession = Depends(get_db)):
    """Hang up and run the post-call memory pipeline.

    Default is sync so the UI can show what the agent learned the moment the call ends —
    that is the demo beat. Voice stages should pass analyse="background" instead, since
    a real caller must never wait for extraction.
    """
    conv = await _conv(db, p.tenant_id, session_id)
    ended = datetime.now(timezone.utc)
    conv.status = "completed"
    conv.ended_at = ended
    if body.outcome:
        conv.outcome = body.outcome

    call = (await db.execute(select(Call).where(
        Call.tenant_id == p.tenant_id, Call.conversation_id == conv.id))).scalars().first()
    if call is not None:
        call.status = "completed"
        call.ended_at = ended
        call.hangup_cause = body.hangup_cause
        if call.answered_at:
            call.duration_seconds = int((ended - call.answered_at).total_seconds())

    await db.commit()
    await working.clear(conv.id)

    report = None
    if body.analyse == "background":
        background.add_task(run_post_call_task, p.tenant_id, conv.id)
    elif body.analyse == "sync":
        r = await run_post_call(db, tenant_id=p.tenant_id, conversation_id=conv.id)
        await db.commit()
        report = {
            "skipped": r.skipped, "summary": r.summary, "outcome": r.outcome,
            "sentiment": r.sentiment, "facts_added": r.facts_added,
            "facts_superseded": r.facts_superseded, "facts_unchanged": r.facts_unchanged,
            "candidates_for_review": r.candidates, "profile_updated": r.profile_updated,
            "product_interests": r.product_interests,
            "model": r.model, "tokens": r.tokens,
        }

    return {"session_id": str(conv.id), "status": conv.status, "outcome": conv.outcome,
            "duration_seconds": call.duration_seconds if call else None,
            "analysis": report, "analysis_queued": body.analyse == "background"}


# --------------------------------------------------------------------------- read


@router.get("/{session_id}")
async def get_session(session_id: uuid.UUID,
                      p: Principal = Depends(require("conversations:read")),
                      db: AsyncSession = Depends(get_db),
                      include_traces: bool = True):
    """Full session state — survives a page refresh mid-call."""
    conv = await _conv(db, p.tenant_id, session_id)
    msgs = (await db.execute(select(Message).where(
        Message.conversation_id == conv.id).order_by(Message.created_at))).scalars().all()
    customer = None
    if conv.customer_id:
        customer = (await db.execute(select(Customer).where(
            Customer.id == conv.customer_id))).scalar_one_or_none()

    session = await working.load(conv.id)
    traces = []
    if include_traces:
        rows = (await db.execute(select(DecisionTrace).where(
            DecisionTrace.tenant_id == p.tenant_id,
            DecisionTrace.conversation_id == conv.id)
            .order_by(DecisionTrace.turn_index))).scalars().all()
        traces = [{"turn": t.turn_index, "input": t.input_text, "action": t.action_taken,
                   "tool": t.proposed_tool, "confidence": float(t.confidence) if t.confidence else None,
                   "band": t.confidence_band,
                   "rules_matched": [r.get("rule") for r in (t.rules_evaluated or [])
                                     if r.get("matched")],
                   "knowledge_used": t.retrieved_chunks, "latency_ms": t.latency_ms}
                  for t in rows]

    return {
        "session_id": str(conv.id), "channel": conv.channel, "direction": conv.direction,
        "status": conv.status, "outcome": conv.outcome, "summary": conv.summary,
        "started_at": conv.started_at, "ended_at": conv.ended_at,
        "as_of_date": session.get("as_of_date"),
        "pending_confirmation": session.get("pending_action"),
        "identity_verified": bool(session.get("identity_verified")),
        "customer": _customer_brief(customer),
        "messages": [{"role": m.role, "content": m.content, "at": m.created_at} for m in msgs],
        "traces": traces,
    }


@router.get("")
async def list_sessions(p: Principal = Depends(require("conversations:read")),
                        db: AsyncSession = Depends(get_db),
                        channel: str | None = None,
                        limit: int = Query(25, ge=1, le=100)):
    q = select(Conversation).where(Conversation.tenant_id == p.tenant_id)
    if channel:
        q = q.where(Conversation.channel == channel)
    rows = (await db.execute(q.order_by(Conversation.created_at.desc())
                             .limit(limit))).scalars().all()
    return {"items": [{"session_id": str(c.id), "channel": c.channel,
                       "direction": c.direction, "status": c.status, "outcome": c.outcome,
                       "customer_id": str(c.customer_id) if c.customer_id else None,
                       "started_at": c.started_at, "ended_at": c.ended_at} for c in rows]}


# --------------------------------------------------------------------------- helpers


async def _conv(db: AsyncSession, tenant_id: uuid.UUID, cid: uuid.UUID) -> Conversation:
    conv = (await db.execute(select(Conversation).where(
        Conversation.tenant_id == tenant_id, Conversation.id == cid))).scalar_one_or_none()
    if conv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    return conv


async def _resolve_customer(db: AsyncSession, tenant_id: uuid.UUID,
                            customer_id: uuid.UUID | None, phone: str | None) -> Customer | None:
    if customer_id:
        c = (await db.execute(select(Customer).where(
            Customer.tenant_id == tenant_id, Customer.id == customer_id))).scalar_one_or_none()
        if c is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found")
        return c
    if phone:
        return (await db.execute(select(Customer).where(
            Customer.tenant_id == tenant_id,
            Customer.phone_normalized == normalize_phone(phone)))).scalar_one_or_none()
    return None


def _customer_brief(c: Customer | None) -> dict | None:
    if c is None:
        return None
    return {"id": str(c.id), "name": c.full_name, "phone": c.phone,
            "state": c.current_state}


def _parse_date(value) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None