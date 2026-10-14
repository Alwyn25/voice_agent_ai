"""Tickets and the escalation queue — the human side of the platform."""
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import AuditEvent, Conversation, Customer, Escalation, Ticket
from app.security.deps import Principal, require
from app.services.events import publish

router = APIRouter(tags=["support"])

TICKET_STATUSES = ("open", "in_progress", "resolved", "escalated", "closed")


# ---------------- tickets ----------------

class TicketIn(BaseModel):
    customer_id: uuid.UUID
    subject: str
    description: str | None = None
    category: str | None = None
    priority: str = "medium"          # low | medium | high | urgent
    conversation_id: uuid.UUID | None = None


@router.post("/tickets", status_code=201)
async def create_ticket(body: TicketIn, p: Principal = Depends(require("tickets:write")),
                        db: AsyncSession = Depends(get_db)):
    t = Ticket(tenant_id=p.tenant_id, customer_id=body.customer_id,
               conversation_id=body.conversation_id, subject=body.subject[:300],
               description=body.description, category=body.category, priority=body.priority)
    db.add(t)
    await db.commit()
    await db.refresh(t)
    publish(p.tenant_id, "ticket.created", ticket_id=str(t.id),
            customer_id=str(t.customer_id), subject=t.subject, priority=t.priority)
    return _ticket_json(t)


@router.get("/tickets")
async def list_tickets(p: Principal = Depends(require("tickets:read")),
                       db: AsyncSession = Depends(get_db),
                       status_filter: str | None = Query(None, alias="status"),
                       customer_id: uuid.UUID | None = None,
                       assigned_to: uuid.UUID | None = None,
                       page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=100)):
    q = select(Ticket).where(Ticket.tenant_id == p.tenant_id)
    if status_filter:
        q = q.where(Ticket.status == status_filter)
    if customer_id:
        q = q.where(Ticket.customer_id == customer_id)
    if assigned_to:
        q = q.where(Ticket.assigned_to == assigned_to)
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
    rows = (await db.execute(q.order_by(Ticket.created_at.desc())
                             .offset((page - 1) * page_size).limit(page_size))).scalars().all()
    return {"items": [_ticket_json(t) for t in rows], "page": page,
            "page_size": page_size, "total": total}


@router.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: uuid.UUID, p: Principal = Depends(require("tickets:read")),
                     db: AsyncSession = Depends(get_db)):
    t = await _ticket(db, p.tenant_id, ticket_id)
    cust = (await db.execute(select(Customer).where(
        Customer.id == t.customer_id))).scalar_one_or_none()
    return {**_ticket_json(t),
            "customer": {"id": str(cust.id), "name": cust.full_name,
                         "phone": cust.phone_normalized} if cust else None}


class TicketPatch(BaseModel):
    status: str | None = None
    priority: str | None = None
    assigned_to: uuid.UUID | None = None
    resolution: str | None = None
    category: str | None = None


@router.patch("/tickets/{ticket_id}")
async def update_ticket(ticket_id: uuid.UUID, body: TicketPatch,
                        p: Principal = Depends(require("tickets:write")),
                        db: AsyncSession = Depends(get_db)):
    t = await _ticket(db, p.tenant_id, ticket_id)
    if body.status and body.status not in TICKET_STATUSES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"status must be one of {', '.join(TICKET_STATUSES)}")
    before = {"status": t.status, "priority": t.priority,
              "assigned_to": str(t.assigned_to) if t.assigned_to else None}
    for field_name in ("status", "priority", "assigned_to", "resolution", "category"):
        value = getattr(body, field_name)
        if value is not None:
            setattr(t, field_name, value)
    if body.status in ("resolved", "closed") and t.resolved_at is None:
        t.resolved_at = datetime.now(timezone.utc)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="tickets.update",
                      resource_type="ticket", resource_id=str(ticket_id), before=before,
                      after={"status": t.status, "priority": t.priority}))
    await db.commit()
    await db.refresh(t)
    return _ticket_json(t)


# ---------------- escalations ----------------

@router.get("/escalations")
async def list_escalations(p: Principal = Depends(require("escalations:read")),
                           db: AsyncSession = Depends(get_db),
                           status_filter: str | None = Query("pending", alias="status")):
    """The human work queue. Ordered oldest first — the longest-waiting customer is
    the one being let down."""
    q = select(Escalation).where(Escalation.tenant_id == p.tenant_id)
    if status_filter:
        q = q.where(Escalation.status == status_filter)
    rows = (await db.execute(q.order_by(Escalation.created_at))).scalars().all()
    out = []
    for e in rows:
        cust = (await db.execute(select(Customer).where(
            Customer.id == e.customer_id))).scalar_one_or_none()
        waiting = (datetime.now(timezone.utc) - e.created_at).total_seconds() if e.created_at else 0
        out.append({"id": str(e.id), "reason": e.reason, "trigger_rule": e.trigger_rule,
                    "status": e.status, "created_at": e.created_at,
                    "waiting_seconds": int(waiting),
                    "confidence": float(e.confidence) if e.confidence else None,
                    "customer": {"id": str(e.customer_id),
                                 "name": cust.full_name if cust else None,
                                 "phone": cust.phone_normalized if cust else None,
                                 "state": cust.current_state if cust else None},
                    "conversation_id": str(e.conversation_id) if e.conversation_id else None})
    return out


@router.post("/escalations/{escalation_id}/accept")
async def accept_escalation(escalation_id: uuid.UUID,
                            p: Principal = Depends(require("escalations:write")),
                            db: AsyncSession = Depends(get_db)):
    """Take ownership and receive the agent's briefing — so the customer never has to
    repeat themselves to the human who picks up."""
    e = await _escalation(db, p.tenant_id, escalation_id)
    if e.status == "resolved":
        raise HTTPException(status.HTTP_409_CONFLICT, "Escalation already resolved")
    e.status = "accepted"
    e.assigned_to = p.user_id
    cust = (await db.execute(select(Customer).where(
        Customer.id == e.customer_id))).scalar_one_or_none()
    conv = None
    if e.conversation_id:
        conv = (await db.execute(select(Conversation).where(
            Conversation.id == e.conversation_id))).scalar_one_or_none()
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="escalations.accept",
                      resource_type="escalation", resource_id=str(escalation_id)))
    await db.commit()
    publish(p.tenant_id, "escalation.accepted", escalation_id=str(escalation_id),
            assigned_to=str(p.user_id) if p.user_id else None)
    return {"id": str(e.id), "status": e.status, "reason": e.reason,
            "briefing": e.briefing or "No briefing recorded.",
            "customer": {"id": str(e.customer_id), "name": cust.full_name if cust else None,
                         "phone": cust.phone_normalized if cust else None,
                         "state": cust.current_state if cust else None},
            "conversation": {"id": str(conv.id), "channel": conv.channel,
                             "summary": conv.summary} if conv else None}


class ResolveIn(BaseModel):
    note: str
    create_ticket: bool = False


@router.post("/escalations/{escalation_id}/resolve")
async def resolve_escalation(escalation_id: uuid.UUID, body: ResolveIn,
                             p: Principal = Depends(require("escalations:write")),
                             db: AsyncSession = Depends(get_db)):
    """Closing an escalation releases the customer back to automated contact —
    which is exactly why it needs a note and an audit entry."""
    e = await _escalation(db, p.tenant_id, escalation_id)
    e.status = "resolved"
    ticket_id = None
    if body.create_ticket:
        t = Ticket(tenant_id=p.tenant_id, customer_id=e.customer_id,
                   conversation_id=e.conversation_id,
                   subject=f"Follow-up: {e.reason}"[:300], description=body.note,
                   category="escalation", priority="high", status="open")
        db.add(t)
        await db.flush()
        e.ticket_id = t.id
        ticket_id = str(t.id)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="escalations.resolve",
                      resource_type="escalation", resource_id=str(escalation_id),
                      after={"note": body.note, "ticket_id": ticket_id}))
    await db.commit()
    publish(p.tenant_id, "escalation.resolved", escalation_id=str(escalation_id),
            customer_id=str(e.customer_id))
    return {"id": str(e.id), "status": e.status, "ticket_id": ticket_id,
            "customer_unlocked": True,
            "note": "Automated outbound contact with this customer is permitted again."}


# ---------------- helpers ----------------

def _ticket_json(t: Ticket) -> dict:
    return {"id": str(t.id), "customer_id": str(t.customer_id), "subject": t.subject,
            "description": t.description, "category": t.category, "priority": t.priority,
            "status": t.status, "assigned_to": str(t.assigned_to) if t.assigned_to else None,
            "resolution": t.resolution, "resolved_at": t.resolved_at,
            "created_at": t.created_at}


async def _ticket(db: AsyncSession, tenant_id: uuid.UUID, tid: uuid.UUID) -> Ticket:
    row = (await db.execute(select(Ticket).where(
        Ticket.tenant_id == tenant_id, Ticket.id == tid))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket not found")
    return row


async def _escalation(db: AsyncSession, tenant_id: uuid.UUID, eid: uuid.UUID) -> Escalation:
    row = (await db.execute(select(Escalation).where(
        Escalation.tenant_id == tenant_id, Escalation.id == eid))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Escalation not found")
    return row
