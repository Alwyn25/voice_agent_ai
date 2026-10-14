"""Vocal Backend Tasksheet P8 (Settings tier — Billing). `plan`/`seats_purchased`
already live on `Tenant` (P0) — the one source of truth the seat-count nav chip and
every seat check elsewhere in the codebase already reads. `Subscription` adds only
the genuinely NEW billing fields on top (billing cycle, Stripe ids, renewal date);
`Invoice` is a plain new table — nothing existed for it to reconcile onto.
"""
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin, TimestampMixin


class Subscription(IdMixin, TimestampMixin, Base):
    __tablename__ = "subscriptions"
    __table_args__ = (UniqueConstraint("tenant_id", name="uq_subscription_tenant"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    billing_cycle: Mapped[str] = mapped_column(String(20), default="monthly")  # monthly|annual
    status: Mapped[str] = mapped_column(String(20), default="active")  # active|past_due|canceled|trialing
    stripe_customer_id: Mapped[str | None] = mapped_column(String(100))
    stripe_subscription_id: Mapped[str | None] = mapped_column(String(100))
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Invoice(IdMixin, TimestampMixin, Base):
    __tablename__ = "invoices"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    amount_cents: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(10), default="usd")
    status: Mapped[str] = mapped_column(String(20), default="open")  # paid|open|void|uncollectible
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    pdf_url: Mapped[str | None] = mapped_column(Text)
    stripe_invoice_id: Mapped[str | None] = mapped_column(String(100))
