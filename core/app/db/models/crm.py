import uuid
from datetime import datetime, time

from sqlalchemy import DateTime, ForeignKey, Index, Numeric, String, Text, Time, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin, TimestampMixin


class Customer(IdMixin, TimestampMixin, Base):
    __tablename__ = "customers"
    __table_args__ = (
        UniqueConstraint("tenant_id", "phone_normalized"),
        Index("ix_customers_tenant_state", "tenant_id", "current_state"),
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    full_name: Mapped[str | None] = mapped_column(String(200))
    phone: Mapped[str] = mapped_column(String(20))
    phone_normalized: Mapped[str] = mapped_column(String(20))  # E.164
    email: Mapped[str | None] = mapped_column(String(255))
    location: Mapped[str | None] = mapped_column(String(255))
    region_code: Mapped[str | None] = mapped_column(String(10))
    preferred_language: Mapped[str] = mapped_column(String(10), default="en")
    preferred_contact_start: Mapped[time | None] = mapped_column(Time)
    preferred_contact_end: Mapped[time | None] = mapped_column(Time)
    current_state: Mapped[str] = mapped_column(String(50), default="lead", index=True)
    state_stack: Mapped[list] = mapped_column(JSONB, default=list)  # interrupting-state return stack
    lead_source: Mapped[str | None] = mapped_column(String(100))
    priority_score: Mapped[float] = mapped_column(Numeric(6, 2), default=0)
    assigned_staff_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    tags: Mapped[list] = mapped_column(ARRAY(String), default=list)
    external_ref: Mapped[str | None] = mapped_column(String(255))
    is_deleted: Mapped[bool] = mapped_column(default=False)


class CustomerFact(IdMixin, TimestampMixin, Base):
    __tablename__ = "customer_facts"
    __table_args__ = (Index("ix_facts_lookup", "tenant_id", "customer_id", "category", "status"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    customer_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("customers.id"), index=True)
    category: Mapped[str] = mapped_column(String(50))  # pain_point|objection|preference|buying_intent|decision_maker|product_interest
    key: Mapped[str] = mapped_column(String(100))
    value: Mapped[dict] = mapped_column(JSONB)
    confidence: Mapped[float] = mapped_column(Numeric(4, 3), default=1.0)
    source_conversation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    status: Mapped[str] = mapped_column(String(20), default="active")  # active|superseded|retracted|candidate
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))


class CustomerProductInterest(IdMixin, TimestampMixin, Base):
    __tablename__ = "customer_product_interest"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    customer_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("customers.id"), index=True)
    # Which project this interest is in (multi-project-per-tenant). NULL = tenant-wide. The
    # same product_ref ("3 BHK") can exist in several projects, so interest is deduped on
    # (customer, project, product_ref) — a caller interested in a 3 BHK in two projects has two
    # rows, not one that overwrites the other.
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"), index=True)
    product_ref: Mapped[str] = mapped_column(String(200))
    interest_level: Mapped[str] = mapped_column(String(20))  # interested|purchased|rejected
    noted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_conversation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))


class StateTransition(IdMixin, Base):
    __tablename__ = "state_transitions"
    __table_args__ = (Index("ix_transitions_customer", "tenant_id", "customer_id", "occurred_at"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    customer_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("customers.id"), index=True)
    from_state: Mapped[str | None] = mapped_column(String(50))
    to_state: Mapped[str] = mapped_column(String(50))
    trigger: Mapped[str] = mapped_column(String(100))
    tool_execution_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    actor: Mapped[str] = mapped_column(String(100), default="agent")  # agent|human:<id>|system
    confidence: Mapped[float | None] = mapped_column(Numeric(4, 3))
    reason: Mapped[str | None] = mapped_column(Text)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
