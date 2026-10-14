"""Setup CRUD: locations, staff members, working hours, holidays, slot policies.

Vocal Backend Tasksheet P4, Generic Sprint 2 — Centre/Executive renamed to
Location/StaffMember at this admin-CRUD layer (see app/db/models/scheduling.py's
Location docstring for why appointments.centre_id/executive_id and the booking
tool schemas keep their existing names for now)."""
import uuid
from datetime import date, time

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import AuditEvent, Holiday, Location, SlotPolicy, StaffMember, WorkingHours
from app.security.deps import Principal, require

router = APIRouter(prefix="/scheduling", tags=["scheduling setup"])


# ---------- locations ----------

class LocationIn(BaseModel):
    name: str
    address: str | None = None
    city: str | None = None
    region_code: str | None = None
    phone: str | None = None
    timezone: str = "Asia/Kolkata"
    is_active: bool = True
    location_type: dict = {}


@router.post("/locations", status_code=201)
async def create_location(body: LocationIn, p: Principal = Depends(require("scheduling:write")),
                          db: AsyncSession = Depends(get_db)):
    row = Location(tenant_id=p.tenant_id, **body.model_dump())
    db.add(row)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="scheduling.location_create",
                      resource_type="location", after=body.model_dump()))
    await db.commit()
    await db.refresh(row)
    return {"id": str(row.id), "name": row.name, "timezone": row.timezone}


@router.get("/locations")
async def list_locations(p: Principal = Depends(require("appointments:read")),
                         db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(Location).where(Location.tenant_id == p.tenant_id))).scalars().all()
    return [{"id": str(r.id), "name": r.name, "city": r.city, "timezone": r.timezone,
             "location_type": r.location_type, "is_active": r.is_active} for r in rows]


@router.patch("/locations/{location_id}")
async def update_location(location_id: uuid.UUID, body: dict,
                          p: Principal = Depends(require("scheduling:write")),
                          db: AsyncSession = Depends(get_db)):
    row = await _one(db, Location, p.tenant_id, location_id)
    allowed = {"name", "address", "city", "region_code", "phone", "timezone", "is_active", "location_type"}
    for k, v in body.items():
        if k in allowed:
            setattr(row, k, v)
    await db.commit()
    return {"id": str(row.id), "is_active": row.is_active}


# ---------- staff members ----------

class StaffMemberIn(BaseModel):
    location_id: uuid.UUID | None = None
    full_name: str
    email: str | None = None
    phone: str | None = None
    specialties: list[str] = []
    is_active: bool = True


@router.post("/staff", status_code=201)
async def create_staff_member(body: StaffMemberIn, p: Principal = Depends(require("scheduling:write")),
                              db: AsyncSession = Depends(get_db)):
    row = StaffMember(tenant_id=p.tenant_id, **body.model_dump())
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return {"id": str(row.id), "full_name": row.full_name}


@router.get("/staff")
async def list_staff_members(location_id: uuid.UUID | None = None,
                             p: Principal = Depends(require("appointments:read")),
                             db: AsyncSession = Depends(get_db)):
    q = select(StaffMember).where(StaffMember.tenant_id == p.tenant_id)
    if location_id:
        q = q.where(StaffMember.location_id == location_id)
    rows = (await db.execute(q)).scalars().all()
    return [{"id": str(r.id), "full_name": r.full_name,
             "location_id": str(r.location_id) if r.location_id else None,
             "is_active": r.is_active} for r in rows]


# ---------- working hours ----------

class WorkingHoursIn(BaseModel):
    owner_type: str  # location | staff_member
    owner_id: uuid.UUID
    day_of_week: int  # 0=Mon .. 6=Sun
    start_time: time
    end_time: time


@router.post("/working-hours", status_code=201)
async def add_working_hours(body: WorkingHoursIn, p: Principal = Depends(require("scheduling:write")),
                            db: AsyncSession = Depends(get_db)):
    if body.owner_type not in ("location", "staff_member"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "owner_type must be location|staff_member")
    if not (0 <= body.day_of_week <= 6):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "day_of_week must be 0..6")
    if body.end_time <= body.start_time:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "end_time must be after start_time")
    row = WorkingHours(tenant_id=p.tenant_id, **body.model_dump())
    db.add(row)
    await db.commit()
    return {"id": str(row.id)}


@router.get("/working-hours")
async def list_working_hours(owner_type: str, owner_id: uuid.UUID,
                             p: Principal = Depends(require("appointments:read")),
                             db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(WorkingHours).where(
        WorkingHours.tenant_id == p.tenant_id, WorkingHours.owner_type == owner_type,
        WorkingHours.owner_id == owner_id).order_by(WorkingHours.day_of_week))).scalars().all()
    return [{"id": str(r.id), "day_of_week": r.day_of_week,
             "start_time": str(r.start_time), "end_time": str(r.end_time)} for r in rows]


@router.delete("/working-hours/{wh_id}", status_code=204)
async def delete_working_hours(wh_id: uuid.UUID, p: Principal = Depends(require("scheduling:write")),
                               db: AsyncSession = Depends(get_db)):
    row = await _one(db, WorkingHours, p.tenant_id, wh_id)
    await db.delete(row)
    await db.commit()


# ---------- holidays ----------

class HolidayIn(BaseModel):
    date: date
    name: str | None = None
    owner_type: str | None = None  # None = tenant-wide
    owner_id: uuid.UUID | None = None


@router.post("/holidays", status_code=201)
async def add_holiday(body: HolidayIn, p: Principal = Depends(require("scheduling:write")),
                      db: AsyncSession = Depends(get_db)):
    row = Holiday(tenant_id=p.tenant_id, **body.model_dump())
    db.add(row)
    await db.commit()
    return {"id": str(row.id)}


@router.get("/holidays")
async def list_holidays(p: Principal = Depends(require("appointments:read")),
                        db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(Holiday).where(Holiday.tenant_id == p.tenant_id)
                             .order_by(Holiday.date))).scalars().all()
    return [{"id": str(r.id), "date": str(r.date), "name": r.name,
             "owner_type": r.owner_type} for r in rows]


@router.delete("/holidays/{h_id}", status_code=204)
async def delete_holiday(h_id: uuid.UUID, p: Principal = Depends(require("scheduling:write")),
                         db: AsyncSession = Depends(get_db)):
    row = await _one(db, Holiday, p.tenant_id, h_id)
    await db.delete(row)
    await db.commit()


# ---------- slot policies ----------

class SlotPolicyIn(BaseModel):
    location_id: uuid.UUID | None = None  # None = tenant default
    slot_duration_min: int = 30
    buffer_before_min: int = 0
    buffer_after_min: int = 0
    max_parallel: int = 1
    booking_horizon_days: int = 30
    min_notice_hours: int = 2


@router.post("/slot-policies", status_code=201)
async def create_slot_policy(body: SlotPolicyIn, p: Principal = Depends(require("scheduling:write")),
                             db: AsyncSession = Depends(get_db)):
    row = SlotPolicy(tenant_id=p.tenant_id, **body.model_dump())
    db.add(row)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="scheduling.policy_create",
                      resource_type="slot_policy", after=body.model_dump(mode="json")))
    await db.commit()
    return {"id": str(row.id)}


@router.get("/slot-policies")
async def list_slot_policies(p: Principal = Depends(require("appointments:read")),
                             db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(SlotPolicy).where(SlotPolicy.tenant_id == p.tenant_id))).scalars().all()
    return [{"id": str(r.id), "location_id": str(r.location_id) if r.location_id else None,
             "slot_duration_min": r.slot_duration_min, "max_parallel": r.max_parallel,
             "buffer_before_min": r.buffer_before_min, "buffer_after_min": r.buffer_after_min,
             "booking_horizon_days": r.booking_horizon_days, "min_notice_hours": r.min_notice_hours}
            for r in rows]


async def _one(db: AsyncSession, model, tenant_id: uuid.UUID, row_id: uuid.UUID):
    row = (await db.execute(select(model).where(model.tenant_id == tenant_id,
                                                model.id == row_id))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{model.__name__} not found")
    return row
