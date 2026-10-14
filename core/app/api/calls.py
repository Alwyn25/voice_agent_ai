"""Call log, manual outbound trigger, recordings, and the callback queue."""
import csv
import io
import uuid
from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import (AuditEvent, Call, CallbackRequest, CallCapture, CallEvent, Customer,
                           Escalation, ToolExecution)
from app.security.deps import Principal, require
from app.services.events import publish
from app.services.outbound import check_can_call, place_outbound_call

router = APIRouter(tags=["calls"])


def _filtered_calls_query(tenant_id: uuid.UUID, *, customer_id: uuid.UUID | None = None,
                          status_filter: str | None = None, direction: str | None = None,
                          campaign_id: uuid.UUID | None = None,
                          intent: str | None = None, sentiment: str | None = None,
                          search: str | None = None, date_from: date | None = None,
                          date_to: date | None = None):
    """Shared by GET /calls (paginated) and GET /calls/export (unpaginated CSV) so the
    two never drift — exporting should always match exactly what the table shows."""
    q = select(Call).where(Call.tenant_id == tenant_id)
    if customer_id:
        q = q.where(Call.customer_id == customer_id)
    if status_filter:
        q = q.where(Call.status == status_filter)
    if campaign_id:
        q = q.where(Call.campaign_id == campaign_id)
    if intent:
        q = q.where(Call.intent.ilike(f"%{intent}%"))
    if sentiment:
        q = q.where(Call.sentiment == sentiment)
    if search:
        like = f"%{search}%"
        q = q.where(or_(Call.summary.ilike(like), Call.to_number.ilike(like),
                        Call.from_number.ilike(like)))
    if date_from:
        q = q.where(Call.created_at >= datetime.combine(date_from, datetime.min.time(),
                                                        tzinfo=timezone.utc))
    if date_to:
        q = q.where(Call.created_at <= datetime.combine(date_to, datetime.max.time(),
                                                        tzinfo=timezone.utc))
    return q


@router.get("/calls")
async def list_calls(p: Principal = Depends(require("calls:read")),
                     db: AsyncSession = Depends(get_db),
                     customer_id: uuid.UUID | None = None,
                     status_filter: str | None = Query(None, alias="status"),
                     direction: str | None = None,
                     campaign_id: uuid.UUID | None = None,
                     intent: str | None = None,
                     sentiment: str | None = None,
                     search: str | None = None,
                     date_from: date | None = Query(None, alias="from"),
                     date_to: date | None = Query(None, alias="to"),
                     page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=100)):
    q = _filtered_calls_query(p.tenant_id, customer_id=customer_id, status_filter=status_filter,
                              direction=direction, campaign_id=campaign_id, intent=intent,
                              sentiment=sentiment, search=search, date_from=date_from,
                              date_to=date_to)
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
    rows = (await db.execute(q.order_by(Call.created_at.desc())
                             .offset((page - 1) * page_size).limit(page_size))).scalars().all()
    return {"items": [_call_json(c) for c in rows], "page": page,
            "page_size": page_size, "total": total}


EXPORT_LIMIT = 500


@router.get("/calls/export")
async def export_calls(p: Principal = Depends(require("calls:read")),
                       db: AsyncSession = Depends(get_db),
                       status_filter: str | None = Query(None, alias="status"),
                       campaign_id: uuid.UUID | None = None,
                       intent: str | None = None, sentiment: str | None = None,
                       search: str | None = None):
    """CSV export matching the Calls page's Export button — same filters as GET
    /calls, unpaginated (capped at EXPORT_LIMIT rows, matching the campaign export's
    own per-request cap on tenant-scale queries)."""
    q = _filtered_calls_query(p.tenant_id, status_filter=status_filter, campaign_id=campaign_id,
                              intent=intent, sentiment=sentiment, search=search)
    rows = (await db.execute(q.order_by(Call.created_at.desc()).limit(EXPORT_LIMIT))).scalars().all()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["call_id", "customer_id", "phone", "status", "campaign_id", "intent",
                     "sentiment", "summary", "duration_seconds", "follow_up_required",
                     "created_at"])
    for c in rows:
        writer.writerow([str(c.id), str(c.customer_id) if c.customer_id else "",
                         c.to_number or c.from_number or "", c.status,
                         str(c.campaign_id) if c.campaign_id else "", c.intent or "",
                         c.sentiment or "", c.summary or "", c.duration_seconds or "",
                         c.follow_up_required, c.created_at])

    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": 'attachment; filename="calls.csv"'})


@router.get("/calls/summary")
async def calls_summary(p: Principal = Depends(require("calls:read")),
                        db: AsyncSession = Depends(get_db)):
    """Tenant-wide tile counts for the Calls page's metrics strip — the same shape
    of number CallResultsPage already shows per-campaign (_tile_counts_batch), just
    rolled up across every call regardless of campaign."""
    base = select(Call).where(Call.tenant_id == p.tenant_id).subquery()
    total = (await db.execute(select(func.count()).select_from(base))).scalar_one()
    connected = (await db.execute(select(func.count()).select_from(base).where(
        base.c.status.in_(("answered", "completed"))))).scalar_one()
    completed = (await db.execute(select(func.count()).select_from(base).where(
        base.c.status == "completed"))).scalar_one()
    follow_ups = (await db.execute(select(func.count()).select_from(base).where(
        base.c.follow_up_required == True))).scalar_one()  # noqa: E712
    avg_duration = (await db.execute(select(func.avg(base.c.duration_seconds))
                                     .select_from(base).where(base.c.duration_seconds.is_not(None)))).scalar_one()
    return {"total": int(total), "connected": int(connected), "completed": int(completed),
           "follow_ups": int(follow_ups),
           "avg_duration_seconds": round(float(avg_duration), 1) if avg_duration is not None else None}


@router.get("/calls/{call_id}")
async def get_call(call_id: uuid.UUID, p: Principal = Depends(require("calls:read")),
                   db: AsyncSession = Depends(get_db)):
    """Task 143: transcript, events, captures, tool logs — all nested in the one
    response, unlike the older engine's split across /conversations/{id}/traces and
    /audit/traces/{id}. Only a Call the Call Runtime Service ran (agent_id set) has
    anything in the three nested lists; every other call's are simply empty."""
    call = await _call(db, p.tenant_id, call_id)
    chain = (await db.execute(select(Call).where(
        Call.tenant_id == p.tenant_id, Call.retry_of_call_id == call_id))).scalars().all()
    events = (await db.execute(select(CallEvent).where(
        CallEvent.tenant_id == p.tenant_id, CallEvent.call_id == call_id)
        .order_by(CallEvent.created_at))).scalars().all()
    captures = (await db.execute(select(CallCapture).where(
        CallCapture.tenant_id == p.tenant_id, CallCapture.call_id == call_id)
        .order_by(CallCapture.created_at))).scalars().all()
    tool_logs = (await db.execute(select(ToolExecution).where(
        ToolExecution.tenant_id == p.tenant_id, ToolExecution.call_id == call_id)
        .order_by(ToolExecution.created_at))).scalars().all()
    return {**_call_json(call),
            "retries": [_call_json(c) for c in chain],
            "conversation_id": str(call.conversation_id) if call.conversation_id else None,
            "agent_id": str(call.agent_id) if call.agent_id else None,
            "template_version_id": str(call.template_version_id) if call.template_version_id else None,
            "transcript": call.transcript, "summary": call.summary, "intent": call.intent,
            "sentiment": call.sentiment, "follow_up_required": call.follow_up_required,
            "follow_up_reason": call.follow_up_reason, "follow_up_at": call.follow_up_at,
            "events": [{"step_key": e.step_key, "step_type": e.step_type, "input": e.input,
                       "output": e.output, "latency_ms": e.latency_ms, "error": e.error,
                       "created_at": e.created_at} for e in events],
            "captures": [{"step_key": c.step_key, "capture_key": c.capture_key, "value": c.value,
                         "created_at": c.created_at} for c in captures],
            "tool_logs": [{"step_key": t.step_key, "tool_name": t.tool_name, "input": t.input,
                          "output": t.output, "status": t.status, "error": t.error,
                          "duration_ms": t.duration_ms, "created_at": t.created_at}
                         for t in tool_logs]}


class HandoffIn(BaseModel):
    reason: str | None = None
    notes: str | None = None


@router.post("/calls/{call_id}/handoff", status_code=201)
async def handoff_call(call_id: uuid.UUID, body: HandoffIn,
                       p: Principal = Depends(require("calls:write")),
                       db: AsyncSession = Depends(get_db)):
    """Task 144: manual escalate to human — the first direct way to CREATE an
    escalation (every existing /escalations endpoint only lists/accepts/resolves
    one); reuses the Escalation resource itself rather than inventing a second
    handoff mechanism, exactly how the Call Runtime Service's own guardrail-denial
    handoff (app/services/call_runtime.py) is a Call-level fact, not a separate
    table."""
    call = await _call(db, p.tenant_id, call_id)
    if call.customer_id is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Call has no known customer to escalate on behalf of")
    esc = Escalation(tenant_id=p.tenant_id, customer_id=call.customer_id,
                     conversation_id=call.conversation_id, reason=body.reason or "manual_handoff",
                     briefing=body.notes, status="pending")
    db.add(esc)
    await db.flush()
    call.follow_up_required = True
    call.follow_up_reason = body.reason or "manual_handoff"
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="calls.handoff",
                      resource_type="call", resource_id=str(call_id),
                      after={"escalation_id": str(esc.id), "reason": esc.reason}))
    await db.commit()
    publish(p.tenant_id, "escalation.created", escalation_id=str(esc.id),
           customer_id=str(call.customer_id), reason=esc.reason, call_id=str(call_id))
    return {"call_id": str(call_id), "escalation_id": str(esc.id), "status": esc.status}


@router.get("/calls/{call_id}/recording")
async def get_recording(call_id: uuid.UUID,
                        p: Principal = Depends(require("recordings:listen")),
                        db: AsyncSession = Depends(get_db)):
    """Short-lived access to the audio. Every playback is audited — a recording is the
    most sensitive artefact the platform holds."""
    call = await _call(db, p.tenant_id, call_id)
    if not call.recording_url:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No recording for this call")
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="recordings.access",
                      resource_type="call", resource_id=str(call_id),
                      after={"recording_url": call.recording_url}))
    await db.commit()
    return {"call_id": str(call_id), "url": call.recording_url, "expires_in": 300,
            "note": "Signed URLs arrive with object storage in the telephony stage."}


class OutboundRequest(BaseModel):
    customer_id: uuid.UUID
    purpose: str = "marketing"      # marketing | service
    reason: str | None = None


@router.post("/calls/outbound", status_code=201)
async def trigger_outbound(body: OutboundRequest,
                           p: Principal = Depends(require("calls:write")),
                           db: AsyncSession = Depends(get_db)):
    """Manually queue an outbound call. Passes the same policy gate as campaigns —
    a human trigger is not an override."""
    customer = await _customer(db, p.tenant_id, body.customer_id)
    decision = await place_outbound_call(db, tenant_id=p.tenant_id, customer=customer,
                                         purpose=body.purpose, actor=p.actor)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="calls.outbound_trigger",
                      resource_type="customer", resource_id=str(body.customer_id),
                      after={"allowed": decision.allowed, "blocked_by": decision.blocked_by,
                             "reason": body.reason}))
    await db.commit()
    if not decision.allowed:
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            {"code": "BLOCKED_BY_RULE", "rule": decision.blocked_by,
                             "message": decision.reason})
    return {"call_id": str(decision.call_id), "status": "queued",
            "customer_id": str(customer.id)}


@router.get("/calls/can-call/{customer_id}")
async def can_call(customer_id: uuid.UUID, purpose: str = "marketing",
                   p: Principal = Depends(require("calls:read")),
                   db: AsyncSession = Depends(get_db)):
    """Dry-run the calling policy for one customer — what the dashboard shows before
    a supervisor clicks 'call now'."""
    customer = await _customer(db, p.tenant_id, customer_id)
    decision = await check_can_call(db, tenant_id=p.tenant_id, customer=customer, purpose=purpose)
    return {"customer_id": str(customer_id), "allowed": decision.allowed,
            "blocked_by": decision.blocked_by, "reason": decision.reason,
            "rules_evaluated": [r for r in decision.rules_evaluated if r.get("matched")]}


# ---------------- callbacks ----------------

@router.get("/callbacks")
async def list_callbacks(p: Principal = Depends(require("calls:read")),
                         db: AsyncSession = Depends(get_db),
                         status_filter: str | None = Query("pending", alias="status"),
                         due_only: bool = False):
    q = select(CallbackRequest).where(CallbackRequest.tenant_id == p.tenant_id)
    if status_filter:
        q = q.where(CallbackRequest.status == status_filter)
    if due_only:
        q = q.where(CallbackRequest.requested_for <= datetime.now(timezone.utc))
    rows = (await db.execute(q.order_by(CallbackRequest.requested_for))).scalars().all()
    out = []
    for r in rows:
        cust = (await db.execute(select(Customer).where(
            Customer.id == r.customer_id))).scalar_one_or_none()
        out.append({"id": str(r.id), "customer_id": str(r.customer_id),
                    "customer_name": cust.full_name if cust else None,
                    "phone": cust.phone_normalized if cust else None,
                    "requested_for": r.requested_for, "status": r.status,
                    "overdue": r.status == "pending" and r.requested_for < datetime.now(timezone.utc)})
    return out


class CallbackPatch(BaseModel):
    requested_for: datetime | None = None
    status: str | None = None       # pending | completed | cancelled


@router.patch("/callbacks/{callback_id}")
async def update_callback(callback_id: uuid.UUID, body: CallbackPatch,
                          p: Principal = Depends(require("calls:write")),
                          db: AsyncSession = Depends(get_db)):
    row = (await db.execute(select(CallbackRequest).where(
        CallbackRequest.tenant_id == p.tenant_id,
        CallbackRequest.id == callback_id))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Callback not found")
    if body.requested_for:
        row.requested_for = body.requested_for
    if body.status:
        if body.status not in ("pending", "completed", "cancelled"):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid status")
        row.status = body.status
    await db.commit()
    return {"id": str(row.id), "requested_for": row.requested_for, "status": row.status}


# ---------------- helpers ----------------

def _call_json(c: Call) -> dict:
    return {"id": str(c.id), "customer_id": str(c.customer_id) if c.customer_id else None,
            "status": c.status,
            "to_number": c.to_number, "from_number": c.from_number,
            "provider": c.provider, "provider_call_id": c.provider_call_id,
            "purpose": c.purpose, "attempt": c.attempt,
            "campaign_id": str(c.campaign_id) if c.campaign_id else None,
            "agent_id": str(c.agent_id) if c.agent_id else None,
            "scheduled_for": c.scheduled_for, "duration_seconds": c.duration_seconds,
            "answered_at": c.answered_at, "ended_at": c.ended_at,
            "hangup_cause": c.hangup_cause, "has_recording": bool(c.recording_url),
            "disclosure_played": c.disclosure_played,
            "retry_of": str(c.retry_of_call_id) if c.retry_of_call_id else None,
            "retry_reason": c.retry_reason, "cost": float(c.cost) if c.cost else None,
            # Plain columns on Call itself (no join needed) — intent/sentiment/summary
            # only ever got returned by the single-call GET before, even though list_calls
            # fetches the same row and could return them for free.
            "intent": c.intent, "sentiment": c.sentiment, "summary": c.summary,
            "follow_up_required": c.follow_up_required,
            "created_at": c.created_at}


async def _call(db: AsyncSession, tenant_id: uuid.UUID, call_id: uuid.UUID) -> Call:
    row = (await db.execute(select(Call).where(
        Call.tenant_id == tenant_id, Call.id == call_id))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Call not found")
    return row


async def _customer(db: AsyncSession, tenant_id: uuid.UUID, cid: uuid.UUID) -> Customer:
    row = (await db.execute(select(Customer).where(
        Customer.tenant_id == tenant_id, Customer.id == cid,
        Customer.is_deleted == False))).scalar_one_or_none()  # noqa: E712
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found")
    return row
