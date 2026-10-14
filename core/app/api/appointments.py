"""Slot search + appointment booking/reschedule/cancel/complete."""
import uuid
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import Appointment, AuditEvent, Customer, Location, StaffMember
from app.engine.state_machine.machine import apply_trigger
from app.security.deps import Principal, require
from app.services import scheduling as sched
from app.services.events import publish
from app.services.scheduling import SchedulingError

router = APIRouter(tags=["appointments"])


def _err(e: SchedulingError):
    code_map = {"SLOT_NOT_AVAILABLE": status.HTTP_409_CONFLICT,
                "CENTRE_NOT_FOUND": status.HTTP_404_NOT_FOUND,
                "EXECUTIVE_NOT_FOUND": status.HTTP_404_NOT_FOUND,
                "NOT_ACTIVE": status.HTTP_422_UNPROCESSABLE_ENTITY}
    return HTTPException(code_map.get(e.code, 400), {"code": e.code, "message": str(e)})


@router.get("/scheduling/slots")
async def slots(centre_id: uuid.UUID,
                date_from: date = Query(..., alias="from"),
                date_to: date = Query(..., alias="to"),
                executive_id: uuid.UUID | None = None,
                p: Principal = Depends(require("appointments:read")),
                db: AsyncSession = Depends(get_db)):
    try:
        result = await sched.compute_slots(db, p.tenant_id, centre_id, date_from, date_to,
                                           executive_id=executive_id)
    except SchedulingError as e:
        raise _err(e)
    return {"slots": [{"starts_at": s.starts_at.isoformat(), "ends_at": s.ends_at.isoformat(),
                       "capacity_left": s.capacity_left} for s in result]}


class BookIn(BaseModel):
    customer_id: uuid.UUID
    centre_id: uuid.UUID
    starts_at: datetime
    executive_id: uuid.UUID | None = None
    source_conversation_id: uuid.UUID | None = None


class ApptOut(BaseModel):
    id: uuid.UUID
    customer_id: uuid.UUID
    centre_id: uuid.UUID
    executive_id: uuid.UUID | None
    starts_at: datetime
    ends_at: datetime
    status: str

    class Config:
        from_attributes = True


@router.post("/appointments", response_model=ApptOut, status_code=201)
async def book(body: BookIn, p: Principal = Depends(require("appointments:write")),
               db: AsyncSession = Depends(get_db)):
    customer = (await db.execute(select(Customer).where(
        Customer.tenant_id == p.tenant_id, Customer.id == body.customer_id,
        Customer.is_deleted == False))).scalar_one_or_none()  # noqa: E712
    if customer is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found")
    try:
        appt = await sched.book(db, tenant_id=p.tenant_id, customer=customer,
                                centre_id=body.centre_id, starts_at=body.starts_at,
                                executive_id=body.executive_id, booked_via="api",
                                source_conversation_id=body.source_conversation_id)
    except SchedulingError as e:
        await db.rollback()
        raise _err(e)
    await apply_trigger(db, customer, "tool:book_appointment", actor=p.actor)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="appointments.book",
                      resource_type="appointment", resource_id=str(appt.id),
                      after={"customer_id": str(customer.id), "starts_at": body.starts_at.isoformat()}))
    await db.commit()
    await db.refresh(appt)
    publish(p.tenant_id, "appointment.booked", appointment_id=str(appt.id),
            customer_id=str(appt.customer_id), starts_at=appt.starts_at.isoformat())
    return appt


class RescheduleIn(BaseModel):
    starts_at: datetime
    executive_id: uuid.UUID | None = None


@router.patch("/appointments/{appt_id}/reschedule", response_model=ApptOut)
async def reschedule(appt_id: uuid.UUID, body: RescheduleIn,
                     p: Principal = Depends(require("appointments:write")),
                     db: AsyncSession = Depends(get_db)):
    appt = await _appt(db, p.tenant_id, appt_id)
    old = appt.starts_at
    try:
        appt = await sched.reschedule(db, tenant_id=p.tenant_id, appt=appt,
                                      new_starts_at=body.starts_at, executive_id=body.executive_id)
    except SchedulingError as e:
        await db.rollback()
        raise _err(e)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="appointments.reschedule",
                      resource_type="appointment", resource_id=str(appt_id),
                      before={"starts_at": old.isoformat()},
                      after={"starts_at": body.starts_at.isoformat()}))
    await db.commit()
    await db.refresh(appt)
    return appt


class CancelIn(BaseModel):
    reason: str | None = None


@router.post("/appointments/{appt_id}/cancel", response_model=ApptOut)
async def cancel(appt_id: uuid.UUID, body: CancelIn,
                 p: Principal = Depends(require("appointments:write")),
                 db: AsyncSession = Depends(get_db)):
    appt = await _appt(db, p.tenant_id, appt_id)
    if appt.status not in sched.ACTIVE_APPT_STATUSES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"Cannot cancel a {appt.status} appointment")
    appt.status = "cancelled"
    appt.cancel_reason = body.reason
    customer = (await db.execute(select(Customer).where(Customer.id == appt.customer_id))).scalar_one()
    await apply_trigger(db, customer, "tool:cancel_appointment", actor=p.actor)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="appointments.cancel",
                      resource_type="appointment", resource_id=str(appt_id),
                      after={"reason": body.reason}))
    await db.commit()
    await db.refresh(appt)
    return appt


class CompleteIn(BaseModel):
    outcome: str  # attended | no_show


@router.post("/appointments/{appt_id}/complete", response_model=ApptOut)
async def complete(appt_id: uuid.UUID, body: CompleteIn,
                   p: Principal = Depends(require("appointments:write")),
                   db: AsyncSession = Depends(get_db)):
    if body.outcome not in ("attended", "no_show"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "outcome must be attended|no_show")
    appt = await _appt(db, p.tenant_id, appt_id)
    if appt.status not in sched.ACTIVE_APPT_STATUSES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"Cannot complete a {appt.status} appointment")
    appt.status = "completed" if body.outcome == "attended" else "no_show"
    customer = (await db.execute(select(Customer).where(Customer.id == appt.customer_id))).scalar_one()
    trigger = "signal:appointment_attended" if body.outcome == "attended" else "signal:no_show"
    await apply_trigger(db, customer, trigger, actor=p.actor)
    await db.commit()
    await db.refresh(appt)
    return appt


@router.get("/appointments")
async def list_appointments(p: Principal = Depends(require("appointments:read")),
                            db: AsyncSession = Depends(get_db),
                            customer_id: uuid.UUID | None = None,
                            centre_id: uuid.UUID | None = None,
                            status_f: str | None = Query(None, alias="status"),
                            date_from: date | None = Query(None, alias="from"),
                            date_to: date | None = Query(None, alias="to")):
    q = select(Appointment).where(Appointment.tenant_id == p.tenant_id)
    if customer_id:
        q = q.where(Appointment.customer_id == customer_id)
    if centre_id:
        q = q.where(Appointment.centre_id == centre_id)
    if status_f:
        q = q.where(Appointment.status == status_f)
    if date_from:
        q = q.where(Appointment.starts_at >= datetime.combine(date_from, datetime.min.time()).astimezone())
    if date_to:
        q = q.where(Appointment.starts_at < datetime.combine(date_to, datetime.max.time()).astimezone())
    rows = (await db.execute(q.order_by(Appointment.starts_at))).scalars().all()

    # Names, not just ids. A schedule listing rows of dashes is unusable to the person it
    # is for — an operator reading this needs to know WHO is coming and WHERE, and the
    # console cannot invent that from a UUID. Resolved in three queries rather than one
    # per row, so a long list does not become a hundred round trips.
    names = {}
    for model, ids, attr in (
        (Customer, {r.customer_id for r in rows}, "full_name"),
        (Location, {r.centre_id for r in rows}, "name"),
        (StaffMember, {r.executive_id for r in rows if r.executive_id}, "full_name"),
    ):
        if ids:
            found = (await db.execute(select(model).where(model.id.in_(ids)))).scalars().all()
            names.update({x.id: getattr(x, attr, None) for x in found})

    return [{"id": str(r.id),
             "customer_id": str(r.customer_id),
             "customer_name": names.get(r.customer_id),
             "centre_id": str(r.centre_id),
             "centre_name": names.get(r.centre_id),
             "executive_id": str(r.executive_id) if r.executive_id else None,
             "executive_name": names.get(r.executive_id) if r.executive_id else None,
             "starts_at": r.starts_at.isoformat(), "ends_at": r.ends_at.isoformat(),
             "status": r.status} for r in rows]


async def _appt(db: AsyncSession, tenant_id: uuid.UUID, appt_id: uuid.UUID) -> Appointment:
    appt = (await db.execute(select(Appointment).where(
        Appointment.tenant_id == tenant_id, Appointment.id == appt_id))).scalar_one_or_none()
    if appt is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Appointment not found")
    return appt