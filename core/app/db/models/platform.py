import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, func, select
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin, TimestampMixin


class Tenant(IdMixin, TimestampMixin, Base):
    """The tasksheet's "organizations" — kept as the existing `tenants` table (already
    referenced by every other model's tenant_id FK) rather than a parallel table, with
    the plan/seats_purchased fields P0 asks for added onto it."""
    __tablename__ = "tenants"
    name: Mapped[str] = mapped_column(String(200))
    slug: Mapped[str] = mapped_column(String(100), unique=True)
    industry_pack: Mapped[str | None] = mapped_column(String(100))
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Kolkata")
    settings: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="active")  # active|suspended
    plan: Mapped[str] = mapped_column(String(50), default="starter")
    seats_purchased: Mapped[int] = mapped_column(Integer, default=1)
    # customer | demo. A demo tenant holds exactly one GTM demo prospect (docs/gtm-demo-
    # links.md §2a): one tenant per prospect so the engine's tenant scoping isolates demos
    # from each other. Background work that walks every tenant (dialer, aggregator) and
    # anything billed or reported must use customer_tenant_ids() so demos never show up.
    kind: Mapped[str] = mapped_column(String(20), default="customer", server_default="customer")


TENANT_KIND_CUSTOMER = "customer"
TENANT_KIND_DEMO = "demo"


def customer_tenant_ids():
    """SELECT of every real customer's tenant id — the set the dialer, the aggregator and
    any billing or reporting sweep should walk. Demo tenants are deliberately excluded."""
    return select(Tenant.id).where(Tenant.kind == TENANT_KIND_CUSTOMER)


class Role(IdMixin, TimestampMixin, Base):
    __tablename__ = "roles"
    __table_args__ = (UniqueConstraint("tenant_id", "name"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(50))  # admin|supervisor|agent|analyst
    permissions: Mapped[list] = mapped_column(JSONB, default=list)


class User(IdMixin, TimestampMixin, Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("tenant_id", "email"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    email: Mapped[str] = mapped_column(String(255))
    password_hash: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(50), default="agent")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ApiKey(IdMixin, TimestampMixin, Base):
    __tablename__ = "api_keys"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    key_hash: Mapped[str] = mapped_column(String(255), unique=True)
    scopes: Mapped[list] = mapped_column(JSONB, default=list)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class TeamInvite(IdMixin, TimestampMixin, Base):
    """The UniqueConstraint below never actually fires for two pending invites to the
    same email — Postgres treats every NULL accepted_at as distinct, so it only
    guards against two invites that were both accepted at the exact same instant,
    which can't happen. The real "no two pending invites to one email" guarantee is
    migration 0017's partial unique index (`uq_team_invite_pending_email`,
    `WHERE accepted_at IS NULL`), backed up by an application-level check in
    app/api/foundation.py's create_invite()."""
    __tablename__ = "team_invites"
    __table_args__ = (UniqueConstraint("tenant_id", "email", "accepted_at", name="uq_team_invite_pending"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    email: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(50))
    invited_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    token_hash: Mapped[str] = mapped_column(String(255), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PhoneNumber(IdMixin, TimestampMixin, Base):
    __tablename__ = "phone_numbers"
    __table_args__ = (UniqueConstraint("e164", name="uq_phone_number_e164"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    e164: Mapped[str] = mapped_column(String(20))
    provider: Mapped[str] = mapped_column(String(50), default="twilio")
    capability: Mapped[str] = mapped_column(String(20), default="both")  # inbound|outbound|both
    status: Mapped[str] = mapped_column(String(20), default="active")  # active|released
    assigned_to: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))  # agent_id once P4 exists


class Notification(IdMixin, Base):
    """User-scoped read-model over audit_events (Vocal Backend Tasksheet P0) — lets
    the bell-icon badge show an unread count without scanning the append-only
    audit_events log on every request. Populated by append_activity_event()'s
    optional notify_user_ids param (see app/services/activity.py); audit_events
    itself is never retrofitted to write here on its own."""
    __tablename__ = "notifications"
    __table_args__ = (Index("ix_notification_user_unread", "tenant_id", "user_id", "read_at"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("audit_events.id"))
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
