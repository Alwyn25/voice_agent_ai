"""Scheduling engine: slot computation (never materialized) + transactional booking.

Concurrency model: bookings take a Postgres advisory transaction lock keyed on
(centre_id, slot_start), then re-check capacity inside the transaction, then insert.
Two simultaneous bookings of the last slot serialize on the lock; the second sees
full capacity and is rejected. (Row locks alone can't prevent the phantom-insert
race, hence advisory locks.)
"""
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import and_, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (Appointment, Customer, Holiday, Location, SlotPolicy,
                           StaffMember, WorkingHours)

# Function parameter names below (centre_id, executive_id) and Appointment's own
# columns are DELIBERATELY left as-is even though the entities are now Location/
# StaffMember — this is the live booking tool's wire contract with the LLM
# (engine/tools/builtin/scheduling_tools.py calls these by keyword, matching the
# tool JSON schema), out of scope for this rename pass. See
# app/db/models/scheduling.py's Location docstring.

ACTIVE_APPT_STATUSES = ("booked", "rescheduled")

DEFAULT_POLICY = dict(slot_duration_min=30, buffer_before_min=0, buffer_after_min=0,
                      max_parallel=1, booking_horizon_days=30, min_notice_hours=2)


class SchedulingError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


# ---------- pure slot generation (unit-testable) ----------

def generate_day_slot_starts(windows: list[tuple[time, time]], duration_min: int,
                             buffer_before_min: int, buffer_after_min: int) -> list[time]:
    """Slot start times within working windows. Step = buffer_before + duration + buffer_after.
    The slot itself occupies [start, start+duration); buffers pad between slots."""
    step = timedelta(minutes=buffer_before_min + duration_min + buffer_after_min)
    dur = timedelta(minutes=duration_min)
    out: list[time] = []
    anchor = date(2000, 1, 1)
    for w_start, w_end in windows:
        cur = datetime.combine(anchor, w_start) + timedelta(minutes=buffer_before_min)
        end = datetime.combine(anchor, w_end)
        while cur + dur <= end:
            out.append(cur.time())
            cur += step
    return sorted(set(out))


# ---------- data loading helpers ----------

async def _policy_for(db: AsyncSession, tenant_id: uuid.UUID, centre_id: uuid.UUID) -> dict:
    row = (await db.execute(select(SlotPolicy).where(
        SlotPolicy.tenant_id == tenant_id, SlotPolicy.location_id == centre_id))).scalars().first()
    if row is None:
        row = (await db.execute(select(SlotPolicy).where(
            SlotPolicy.tenant_id == tenant_id, SlotPolicy.location_id.is_(None)))).scalars().first()
    if row is None:
        return dict(DEFAULT_POLICY)
    return dict(slot_duration_min=row.slot_duration_min, buffer_before_min=row.buffer_before_min,
                buffer_after_min=row.buffer_after_min, max_parallel=row.max_parallel,
                booking_horizon_days=row.booking_horizon_days, min_notice_hours=row.min_notice_hours)


async def _working_windows(db: AsyncSession, tenant_id: uuid.UUID, owner_type: str,
                           owner_id: uuid.UUID, dow: int) -> list[tuple[time, time]]:
    rows = (await db.execute(select(WorkingHours).where(
        WorkingHours.tenant_id == tenant_id, WorkingHours.owner_type == owner_type,
        WorkingHours.owner_id == owner_id, WorkingHours.day_of_week == dow))).scalars().all()
    return [(r.start_time, r.end_time) for r in rows]


async def _is_holiday(db: AsyncSession, tenant_id: uuid.UUID, d: date,
                      centre_id: uuid.UUID, executive_id: uuid.UUID | None) -> bool:
    conds = [and_(Holiday.owner_type.is_(None), Holiday.owner_id.is_(None)),
             and_(Holiday.owner_type == "location", Holiday.owner_id == centre_id)]
    if executive_id:
        conds.append(and_(Holiday.owner_type == "staff_member", Holiday.owner_id == executive_id))
    row = (await db.execute(select(Holiday.id).where(
        Holiday.tenant_id == tenant_id, Holiday.date == d, or_(*conds)))).first()
    return row is not None


async def _overlap_count(db: AsyncSession, tenant_id: uuid.UUID, centre_id: uuid.UUID,
                         start: datetime, end: datetime,
                         executive_id: uuid.UUID | None = None,
                         exclude_appointment_id: uuid.UUID | None = None) -> int:
    q = select(func.count()).select_from(Appointment).where(
        Appointment.tenant_id == tenant_id,
        Appointment.centre_id == centre_id,
        Appointment.status.in_(ACTIVE_APPT_STATUSES),
        Appointment.starts_at < end,
        Appointment.ends_at > start,
    )
    if executive_id:
        q = q.where(Appointment.executive_id == executive_id)
    if exclude_appointment_id:
        q = q.where(Appointment.id != exclude_appointment_id)
    return (await db.execute(q)).scalar_one()


# ---------- slot search ----------

@dataclass
class Slot:
    starts_at: datetime
    ends_at: datetime
    capacity_left: int


async def compute_slots(db: AsyncSession, tenant_id: uuid.UUID, centre_id: uuid.UUID,
                        date_from: date, date_to: date,
                        executive_id: uuid.UUID | None = None) -> list[Slot]:
    centre = (await db.execute(select(Location).where(
        Location.tenant_id == tenant_id, Location.id == centre_id,
        Location.is_active == True))).scalar_one_or_none()  # noqa: E712
    if centre is None:
        raise SchedulingError("CENTRE_NOT_FOUND", "Location not found or inactive")

    if executive_id:
        ex = (await db.execute(select(StaffMember).where(
            StaffMember.tenant_id == tenant_id, StaffMember.id == executive_id,
            StaffMember.is_active == True))).scalar_one_or_none()  # noqa: E712
        if ex is None:
            raise SchedulingError("EXECUTIVE_NOT_FOUND", "Staff member not found or inactive")

    # GTM demo tenants (tenants.kind='demo', one per demo company record) offer that record's
    # dummy-data visit slots instead of working-hours slots, so every check and tool that
    # asks this function stays unchanged. Customer tenants return None here and carry on
    # exactly as before. See app/demo/scheduling.py.
    from app.demo.scheduling import demo_slot_windows
    demo = await demo_slot_windows(db, tenant_id, date_from, date_to)
    if demo is not None:
        return [Slot(starts_at=s, ends_at=e, capacity_left=1) for s, e in demo]

    pol = await _policy_for(db, tenant_id, centre_id)
    tz = ZoneInfo(centre.timezone or "Asia/Kolkata")
    now = datetime.now(timezone.utc)
    horizon_end = (now + timedelta(days=pol["booking_horizon_days"])).date()
    min_start = now + timedelta(hours=pol["min_notice_hours"])
    date_to = min(date_to, horizon_end)

    # Everything the loop needs, in three queries instead of one per day and one per slot.
    #
    # This used to run _is_holiday and _working_windows per DAY and _overlap_count per
    # SLOT. A one-week search of a centre open 10:00-18:30 on hourly slots is 7 + 7 + 56 =
    # seventy round trips, and the twenty-one-day widened fallback is over two hundred —
    # inside a turn that is supposed to answer in well under a second. The data is tiny;
    # it was the trips that cost.
    holidays = await _holidays_in_range(db, tenant_id, date_from, date_to,
                                        centre_id, executive_id)
    windows_by_dow = await _windows_by_dow(db, tenant_id, centre_id, executive_id)
    booked = await _appointments_in_range(db, tenant_id, centre_id, date_from, date_to, tz)

    slots: list[Slot] = []
    d = date_from
    while d <= date_to:
        if d in holidays:
            d += timedelta(days=1)
            continue
        windows = windows_by_dow.get(d.weekday()) or []
        if not windows:
            d += timedelta(days=1)
            continue

        for t_start in generate_day_slot_starts(windows, pol["slot_duration_min"],
                                                pol["buffer_before_min"], pol["buffer_after_min"]):
            s = datetime.combine(d, t_start, tzinfo=tz)
            e = s + timedelta(minutes=pol["slot_duration_min"])
            if s < min_start:
                continue
            used = sum(1 for a_s, a_e, _ in booked if a_s < e and a_e > s)
            left = pol["max_parallel"] - used
            if executive_id and left > 0:
                if any(a_s < e and a_e > s and a_ex == executive_id for a_s, a_e, a_ex in booked):
                    left = 0
            if left > 0:
                slots.append(Slot(starts_at=s, ends_at=e, capacity_left=left))
        d += timedelta(days=1)
    return slots


async def _holidays_in_range(db: AsyncSession, tenant_id: uuid.UUID, date_from: date,
                             date_to: date, centre_id: uuid.UUID,
                             executive_id: uuid.UUID | None) -> set[date]:
    """Same ownership rules as the per-day check: tenant-wide, this centre, this executive."""
    conds = [and_(Holiday.owner_type.is_(None), Holiday.owner_id.is_(None)),
             and_(Holiday.owner_type == "location", Holiday.owner_id == centre_id)]
    if executive_id:
        conds.append(and_(Holiday.owner_type == "staff_member", Holiday.owner_id == executive_id))
    rows = (await db.execute(select(Holiday.date).where(
        Holiday.tenant_id == tenant_id,
        Holiday.date >= date_from, Holiday.date <= date_to,
        or_(*conds)))).scalars().all()
    return set(rows)


async def _windows_by_dow(db: AsyncSession, tenant_id: uuid.UUID, centre_id: uuid.UUID,
                          executive_id: uuid.UUID | None) -> dict[int, list[tuple[time, time]]]:
    """Working hours for every weekday at once, keeping the inheritance rule: an executive
    with no hours of their own for a given day falls back to the centre's."""
    centre_rows = (await db.execute(select(WorkingHours).where(
        WorkingHours.tenant_id == tenant_id, WorkingHours.owner_type == "location",
        WorkingHours.owner_id == centre_id))).scalars().all()
    by_dow: dict[int, list[tuple[time, time]]] = {}
    for r in centre_rows:
        by_dow.setdefault(r.day_of_week, []).append((r.start_time, r.end_time))
    if not executive_id:
        return by_dow

    ex_rows = (await db.execute(select(WorkingHours).where(
        WorkingHours.tenant_id == tenant_id, WorkingHours.owner_type == "staff_member",
        WorkingHours.owner_id == executive_id))).scalars().all()
    ex_by_dow: dict[int, list[tuple[time, time]]] = {}
    for r in ex_rows:
        ex_by_dow.setdefault(r.day_of_week, []).append((r.start_time, r.end_time))
    return {dow: ex_by_dow.get(dow) or by_dow.get(dow) or []
            for dow in set(by_dow) | set(ex_by_dow)}


async def _appointments_in_range(db: AsyncSession, tenant_id: uuid.UUID, centre_id: uuid.UUID,
                                 date_from: date, date_to: date,
                                 tz) -> list[tuple[datetime, datetime, uuid.UUID | None]]:
    """Active appointments overlapping the window, once. A day either side of the range so
    a booking that straddles midnight is still counted."""
    lo = datetime.combine(date_from - timedelta(days=1), time(0, 0), tzinfo=tz)
    hi = datetime.combine(date_to + timedelta(days=1), time(23, 59), tzinfo=tz)
    rows = (await db.execute(select(
        Appointment.starts_at, Appointment.ends_at, Appointment.executive_id).where(
        Appointment.tenant_id == tenant_id,
        Appointment.centre_id == centre_id,
        Appointment.status.in_(ACTIVE_APPT_STATUSES),
        Appointment.starts_at < hi,
        Appointment.ends_at > lo))).all()
    return [(r[0], r[1], r[2]) for r in rows]


async def _check_bookable(db: AsyncSession, tenant_id: uuid.UUID, centre_id: uuid.UUID,
                          starts_at: datetime, executive_id: uuid.UUID | None,
                          exclude_appointment_id: uuid.UUID | None = None) -> datetime:
    """Validate a specific start time is a real, available slot. Returns ends_at."""
    day_slots = await compute_slots(db, tenant_id, centre_id, starts_at.date(), starts_at.date(),
                                    executive_id=executive_id)
    for s in day_slots:
        if s.starts_at == starts_at:
            return s.ends_at
    # distinguish "never a valid slot" from "was valid but now full"
    raise SchedulingError("SLOT_NOT_AVAILABLE",
                          "Requested slot is not available (taken, outside hours, or too soon)")


# ---------- booking (transactional) ----------

async def _advisory_lock(db: AsyncSession, centre_id: uuid.UUID, starts_at: datetime) -> None:
    key = f"book:{centre_id}:{starts_at.astimezone(timezone.utc).isoformat()}"
    await db.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"), {"k": key})


async def book(db: AsyncSession, *, tenant_id: uuid.UUID, customer: Customer,
               centre_id: uuid.UUID, starts_at: datetime,
               executive_id: uuid.UUID | None = None, booked_via: str = "api",
               source_conversation_id: uuid.UUID | None = None,
               unit_type: str | None = None,
               unit_context: dict | None = None,
               project_id: uuid.UUID | None = None) -> Appointment:
    await _advisory_lock(db, centre_id, starts_at)
    ends_at = await _check_bookable(db, tenant_id, centre_id, starts_at, executive_id)
    appt = Appointment(tenant_id=tenant_id, customer_id=customer.id, centre_id=centre_id,
                       executive_id=executive_id, starts_at=starts_at, ends_at=ends_at,
                       status="booked", booked_via=booked_via,
                       source_conversation_id=source_conversation_id,
                       # Which project this visit is for (multi-project-per-tenant), so a
                       # returning caller's project can be read off their visit next time.
                       project_id=project_id,
                       # What the visit is for, bound now so "the flat I selected" resolves
                       # from the record later instead of from a drifting retrieval hint.
                       unit_type=unit_type, unit_context=unit_context or {})
    db.add(appt)
    await db.flush()
    return appt


async def reschedule(db: AsyncSession, *, tenant_id: uuid.UUID, appt: Appointment,
                     new_starts_at: datetime, executive_id: uuid.UUID | None = None) -> Appointment:
    if appt.status not in ACTIVE_APPT_STATUSES:
        raise SchedulingError("NOT_ACTIVE", f"Cannot reschedule a {appt.status} appointment")
    await _advisory_lock(db, appt.centre_id, new_starts_at)
    ex = executive_id if executive_id is not None else appt.executive_id
    ends_at = await _check_bookable(db, tenant_id, appt.centre_id, new_starts_at, ex,
                                    exclude_appointment_id=appt.id)
    appt.starts_at = new_starts_at
    appt.ends_at = ends_at
    appt.executive_id = ex
    appt.status = "rescheduled"
    await db.flush()
    return appt