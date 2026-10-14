import uuid
from datetime import datetime, time

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import (Appointment, AuditEvent, Conversation, Customer,
                           CustomerFact, StateTransition, Ticket)
from app.engine.state_machine.machine import StateMachineError, apply_trigger, manual_override
from app.security.deps import Principal, require
from app.services.phone import is_valid_indian_mobile, normalize_phone

router = APIRouter(prefix="/customers", tags=["customers"])


# ---------- schemas ----------

class CustomerCreate(BaseModel):
    full_name: str | None = None
    phone: str
    email: str | None = None
    location: str | None = None
    region_code: str | None = None
    preferred_language: str = "en"
    preferred_contact_start: time | None = None
    preferred_contact_end: time | None = None
    lead_source: str | None = None
    tags: list[str] = []
    external_ref: str | None = None


class CustomerUpdate(BaseModel):
    full_name: str | None = None
    email: str | None = None
    location: str | None = None
    region_code: str | None = None
    preferred_language: str | None = None
    preferred_contact_start: time | None = None
    preferred_contact_end: time | None = None
    lead_source: str | None = None
    tags: list[str] | None = None
    external_ref: str | None = None
    priority_score: float | None = None


class CustomerOut(BaseModel):
    id: uuid.UUID
    full_name: str | None
    phone: str
    phone_normalized: str
    email: str | None
    location: str | None
    region_code: str | None
    preferred_language: str
    preferred_contact_start: time | None = None
    preferred_contact_end: time | None = None
    current_state: str
    state_stack: list
    tags: list
    priority_score: float
    lead_source: str | None
    created_at: datetime

    class Config:
        from_attributes = True


class Page(BaseModel):
    items: list[CustomerOut]
    page: int
    page_size: int
    total: int


# ---------- endpoints ----------

@router.post("", response_model=CustomerOut, status_code=201)
async def create_customer(body: CustomerCreate,
                          principal: Principal = Depends(require("customers:write")),
                          db: AsyncSession = Depends(get_db)):
    if not is_valid_indian_mobile(body.phone):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"'{body.phone}' isn't a valid Indian mobile number "
                                   "(10 digits, starting 6-9).")
    normalized = normalize_phone(body.phone)
    dup = (await db.execute(select(Customer).where(
        Customer.tenant_id == principal.tenant_id,
        Customer.phone_normalized == normalized))).scalar_one_or_none()
    if dup:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            detail=f"Customer with this phone already exists (id: {dup.id})")
    c = Customer(tenant_id=principal.tenant_id, phone=body.phone, phone_normalized=normalized,
                 **body.model_dump(exclude={"phone"}))
    db.add(c)
    db.add(AuditEvent(tenant_id=principal.tenant_id, actor=principal.actor,
                      action="customers.create", resource_type="customer",
                      after={"phone": normalized, "name": body.full_name}))
    await db.commit()
    await db.refresh(c)
    return c


@router.get("/lookup", response_model=CustomerOut)
async def lookup(phone: str, principal: Principal = Depends(require("customers:read")),
                 db: AsyncSession = Depends(get_db)):
    c = (await db.execute(select(Customer).where(
        Customer.tenant_id == principal.tenant_id,
        Customer.phone_normalized == normalize_phone(phone),
        Customer.is_deleted == False))).scalar_one_or_none()  # noqa: E712
    if c is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No customer with this phone")
    return c


@router.get("", response_model=Page)
async def list_customers(
    principal: Principal = Depends(require("customers:read")),
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=100),
    state: str | None = None, tag: str | None = None,
    language: str | None = None, search: str | None = None,
):
    q = select(Customer).where(Customer.tenant_id == principal.tenant_id,
                               Customer.is_deleted == False)  # noqa: E712
    if state:
        q = q.where(Customer.current_state == state)
    if tag:
        q = q.where(Customer.tags.contains([tag]))
    if language:
        q = q.where(Customer.preferred_language == language)
    if search:
        like = f"%{search}%"
        q = q.where(or_(Customer.full_name.ilike(like), Customer.phone_normalized.ilike(like),
                        Customer.email.ilike(like)))
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
    rows = (await db.execute(q.order_by(Customer.created_at.desc())
                             .offset((page - 1) * page_size).limit(page_size))).scalars().all()
    return Page(items=rows, page=page, page_size=page_size, total=total)


@router.get("/{customer_id}", response_model=CustomerOut)
async def get_customer(customer_id: uuid.UUID,
                       principal: Principal = Depends(require("customers:read")),
                       db: AsyncSession = Depends(get_db)):
    c = await _get(db, principal.tenant_id, customer_id)
    return c


@router.patch("/{customer_id}", response_model=CustomerOut)
async def update_customer(customer_id: uuid.UUID, body: CustomerUpdate,
                          principal: Principal = Depends(require("customers:write")),
                          db: AsyncSession = Depends(get_db)):
    c = await _get(db, principal.tenant_id, customer_id)
    changes = body.model_dump(exclude_unset=True)
    before = {k: str(getattr(c, k)) for k in changes}
    for k, v in changes.items():
        setattr(c, k, v)
    db.add(AuditEvent(tenant_id=principal.tenant_id, actor=principal.actor,
                      action="customers.update", resource_type="customer",
                      resource_id=str(customer_id), before=before,
                      after={k: str(v) for k, v in changes.items()}))
    await db.commit()
    await db.refresh(c)
    return c


@router.delete("/{customer_id}", status_code=204)
async def delete_customer(customer_id: uuid.UUID,
                          principal: Principal = Depends(require("customers:write")),
                          db: AsyncSession = Depends(get_db)):
    c = await _get(db, principal.tenant_id, customer_id)
    c.is_deleted = True
    db.add(AuditEvent(tenant_id=principal.tenant_id, actor=principal.actor,
                      action="customers.soft_delete", resource_type="customer",
                      resource_id=str(customer_id)))
    await db.commit()


# ---------- facts ----------

@router.get("/{customer_id}/facts")
async def list_facts(customer_id: uuid.UUID, include_inactive: bool = False,
                     principal: Principal = Depends(require("customers:read")),
                     db: AsyncSession = Depends(get_db)):
    await _get(db, principal.tenant_id, customer_id)
    q = select(CustomerFact).where(CustomerFact.tenant_id == principal.tenant_id,
                                   CustomerFact.customer_id == customer_id)
    if not include_inactive:
        q = q.where(CustomerFact.status.in_(["active", "candidate"]))
    rows = (await db.execute(q.order_by(CustomerFact.created_at.desc()))).scalars().all()
    return [{"id": str(f.id), "category": f.category, "key": f.key, "value": f.value,
             "confidence": float(f.confidence), "status": f.status,
             "source_conversation_id": str(f.source_conversation_id) if f.source_conversation_id else None,
             "created_at": f.created_at} for f in rows]


@router.patch("/{customer_id}/facts/{fact_id}")
async def update_fact(customer_id: uuid.UUID, fact_id: uuid.UUID, body: dict,
                      principal: Principal = Depends(require("facts:write")),
                      db: AsyncSession = Depends(get_db)):
    f = (await db.execute(select(CustomerFact).where(
        CustomerFact.tenant_id == principal.tenant_id, CustomerFact.id == fact_id,
        CustomerFact.customer_id == customer_id))).scalar_one_or_none()
    if f is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Fact not found")
    before = {"value": f.value, "status": f.status}
    if "status" in body and body["status"] in ("active", "retracted"):
        f.status = body["status"]
    if "value" in body:
        f.value = body["value"]
    db.add(AuditEvent(tenant_id=principal.tenant_id, actor=principal.actor,
                      action="facts.update", resource_type="customer_fact",
                      resource_id=str(fact_id), before=before,
                      after={"value": f.value, "status": f.status}))
    await db.commit()
    return {"id": str(f.id), "status": f.status, "value": f.value}


# ---------- journey ----------

@router.get("/{customer_id}/state-history")
async def state_history(customer_id: uuid.UUID,
                        principal: Principal = Depends(require("customers:read")),
                        db: AsyncSession = Depends(get_db)):
    await _get(db, principal.tenant_id, customer_id)
    rows = (await db.execute(select(StateTransition).where(
        StateTransition.tenant_id == principal.tenant_id,
        StateTransition.customer_id == customer_id)
        .order_by(StateTransition.occurred_at.desc()))).scalars().all()
    return [{"from": r.from_state, "to": r.to_state, "trigger": r.trigger,
             "actor": r.actor, "reason": r.reason, "occurred_at": r.occurred_at} for r in rows]


class StateOverride(BaseModel):
    to_state: str
    reason: str


@router.post("/{customer_id}/state")
async def override_state(customer_id: uuid.UUID, body: StateOverride,
                         principal: Principal = Depends(require("customers:state_override")),
                         db: AsyncSession = Depends(get_db)):
    c = await _get(db, principal.tenant_id, customer_id)
    try:
        result = await manual_override(db, c, body.to_state, actor=principal.actor, reason=body.reason)
    except StateMachineError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))
    db.add(AuditEvent(tenant_id=principal.tenant_id, actor=principal.actor,
                      action="customers.state_override", resource_type="customer",
                      resource_id=str(customer_id),
                      before={"state": result.from_state}, after={"state": result.to_state,
                                                                 "reason": body.reason}))
    await db.commit()
    return {"from": result.from_state, "to": result.to_state}


class SignalBody(BaseModel):
    trigger: str  # e.g. "signal:call_me_later", "system:call_connected"
    reason: str | None = None


@router.post("/{customer_id}/signal")
async def fire_signal(customer_id: uuid.UUID, body: SignalBody,
                      principal: Principal = Depends(require("customers:write")),
                      db: AsyncSession = Depends(get_db)):
    """Manual/system trigger firing — used for testing now; tools fire these internally later."""
    c = await _get(db, principal.tenant_id, customer_id)
    result = await apply_trigger(db, c, body.trigger, actor=principal.actor, reason=body.reason)
    await db.commit()
    if result is None:
        return {"applied": False, "state": c.current_state,
                "note": "No transition for this trigger in the current state"}
    return {"applied": True, "from": result.from_state, "to": result.to_state,
            "state_stack": c.state_stack}


@router.get("/{customer_id}/timeline")
async def timeline(customer_id: uuid.UUID,
                   principal: Principal = Depends(require("customers:read")),
                   db: AsyncSession = Depends(get_db)):
    await _get(db, principal.tenant_id, customer_id)
    tid = principal.tenant_id
    events: list[dict] = []

    for r in (await db.execute(select(StateTransition).where(
            StateTransition.tenant_id == tid, StateTransition.customer_id == customer_id))).scalars():
        events.append({"type": "state_change", "at": r.occurred_at,
                       "detail": f"{r.from_state} → {r.to_state}", "trigger": r.trigger})
    for r in (await db.execute(select(Conversation).where(
            Conversation.tenant_id == tid, Conversation.customer_id == customer_id))).scalars():
        events.append({"type": "conversation", "at": r.created_at, "channel": r.channel,
                       "direction": r.direction, "outcome": r.outcome, "summary": r.summary})
    for r in (await db.execute(select(Appointment).where(
            Appointment.tenant_id == tid, Appointment.customer_id == customer_id))).scalars():
        events.append({"type": "appointment", "at": r.created_at, "starts_at": r.starts_at,
                       "status": r.status})
    for r in (await db.execute(select(Ticket).where(
            Ticket.tenant_id == tid, Ticket.customer_id == customer_id))).scalars():
        events.append({"type": "ticket", "at": r.created_at, "subject": r.subject,
                       "status": r.status})

    events.sort(key=lambda e: e["at"], reverse=True)
    return {"customer_id": str(customer_id), "events": events}


# ---------- helper ----------

async def _get(db: AsyncSession, tenant_id: uuid.UUID, customer_id: uuid.UUID) -> Customer:
    c = (await db.execute(select(Customer).where(
        Customer.tenant_id == tenant_id, Customer.id == customer_id,
        Customer.is_deleted == False))).scalar_one_or_none()  # noqa: E712
    if c is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found")
    return c
