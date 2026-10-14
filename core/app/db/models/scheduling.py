import uuid
from datetime import date, datetime, time

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Index, Integer, Numeric, String, Text, Time
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin, TimestampMixin


class Location(IdMixin, TimestampMixin, Base):
    """Vocal Backend Tasksheet P4, Generic Sprint 2 — renamed from Centre.
    Deliberately scoped to the model/admin-CRUD layer only (see
    app/api/scheduling_admin.py): Appointment.centre_id and every engine/tools/
    call-runtime reference to "centre_id" as a tool-parameter/session-state
    name are UNCHANGED here on purpose — that wire-level contract with the live
    LLM call flow (turn_runner.py, scheduling_tools.py's tool schemas) is
    exactly what P6's Generic Sprint 3 (Engine Vocabulary) exists to
    genericize, and touching it here without the ability to verify against a
    live call would risk regressing the most tuned code in this codebase for
    no behavioural gain in this wave."""
    __tablename__ = "locations"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    address: Mapped[str | None] = mapped_column(Text)
    city: Mapped[str | None] = mapped_column(String(100))
    region_code: Mapped[str | None] = mapped_column(String(10))
    lat: Mapped[float | None] = mapped_column(Numeric(9, 6))
    lng: Mapped[float | None] = mapped_column(Numeric(9, 6))
    phone: Mapped[str | None] = mapped_column(String(20))
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Kolkata")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # RE 'site office' attributes (Vocal Backend Tasksheet P4) — a generic bag for
    # whatever a pack's own location concept needs beyond the fields above.
    location_type: Mapped[dict] = mapped_column(JSONB, default=dict)


class StaffMember(IdMixin, TimestampMixin, Base):
    """Renamed from Executive — same scope note as Location above:
    Appointment.executive_id and the booking tool schemas keep their existing
    wire-level name for now."""
    __tablename__ = "staff_members"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    location_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("locations.id"), index=True)
    full_name: Mapped[str] = mapped_column(String(200))
    email: Mapped[str | None] = mapped_column(String(255))
    phone: Mapped[str | None] = mapped_column(String(20))
    specialties: Mapped[list] = mapped_column(ARRAY(String), default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class WorkingHours(IdMixin, TimestampMixin, Base):
    __tablename__ = "working_hours"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    owner_type: Mapped[str] = mapped_column(String(20))  # location|staff_member
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    day_of_week: Mapped[int] = mapped_column(Integer)  # 0=Mon .. 6=Sun
    start_time: Mapped[time] = mapped_column(Time)
    end_time: Mapped[time] = mapped_column(Time)


class Holiday(IdMixin, TimestampMixin, Base):
    __tablename__ = "holidays"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    owner_type: Mapped[str | None] = mapped_column(String(20))  # location|staff_member|None=tenant-wide
    owner_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    date: Mapped[date] = mapped_column(Date)
    name: Mapped[str | None] = mapped_column(String(100))


class SlotPolicy(IdMixin, TimestampMixin, Base):
    __tablename__ = "slot_policies"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    location_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("locations.id"))
    slot_duration_min: Mapped[int] = mapped_column(Integer, default=30)
    buffer_before_min: Mapped[int] = mapped_column(Integer, default=0)
    buffer_after_min: Mapped[int] = mapped_column(Integer, default=0)
    max_parallel: Mapped[int] = mapped_column(Integer, default=1)
    booking_horizon_days: Mapped[int] = mapped_column(Integer, default=30)
    min_notice_hours: Mapped[int] = mapped_column(Integer, default=2)


class Appointment(IdMixin, TimestampMixin, Base):
    __tablename__ = "appointments"
    __table_args__ = (
        Index("ix_appt_centre_time", "tenant_id", "centre_id", "starts_at"),
        Index("ix_appt_customer", "tenant_id", "customer_id"),
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    customer_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("customers.id"), index=True)
    # WHICH project this visit is for. NULL = tenant-wide (single-project tenants). Stamped at
    # book time from the resolved project so a returning caller's project can be read back off
    # their existing visit — a strong resolution signal on the next call.
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"), index=True)
    # Column names deliberately kept as centre_id/executive_id (see Location's
    # docstring above) even though the FK targets are renamed — this table is
    # the live booking tool's wire contract, out of scope for this pass.
    centre_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("locations.id"), index=True)
    executive_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("staff_members.id"))
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ends_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20), default="booked")  # booked|rescheduled|cancelled|completed|no_show
    booked_via: Mapped[str] = mapped_column(String(20), default="agent")  # agent|dashboard|api
    source_conversation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    reschedule_of: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    cancel_reason: Mapped[str | None] = mapped_column(Text)
    reminder_state: Mapped[dict] = mapped_column(JSONB, default=dict)
    # WHAT the visit is FOR — the unit configuration the caller chose ("3 BHK"), bound to the
    # booking at book time. Before this, an appointment recorded only when/where; "show me the
    # details of the flat I selected" had no authoritative referent and was answered from the
    # transient retrieval-focus hint, which drifts — a caller who chose the 3 BHK was quoted
    # the 2.5 BHK. The canonical string is indexed for reporting/segmentation; unit_context
    # carries the richer snapshot (size, facing, price at time of booking, a future product
    # ref) without committing to a rigid schema while the catalogue lives only in documents.
    unit_type: Mapped[str | None] = mapped_column(String(60), index=True)
    unit_context: Mapped[dict] = mapped_column(JSONB, default=dict)
