"""Vocal Backend Tasksheet P1 (Tool Library) — a connectable third-party tool
marketplace: org admins connect Shopify/Stripe/Twilio/etc. and set guardrails.

Deliberately separate from ToolDefinition (app/db/models/engine_config.py), which is
this backend's PRE-EXISTING, load-bearing table of built-in engine tools the live
voice call executor reads during a real call (book_appointment, search_slots, ...).
That table and its GET/PATCH /tools API stay untouched. This module's tables are
additive and are served under /integrations/tools instead, to avoid colliding with
the already-working /tools route.
"""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin, TimestampMixin


class ToolCatalogEntry(IdMixin, TimestampMixin, Base):
    """The system catalog of connectable tools (tasksheet's "tool_definitions").
    Tenant-scoped and seeded, like every other "built-in" catalog in this codebase
    (ToolDefinition, RuleSet, StateMachineDef) — not a single global table."""
    __tablename__ = "tool_catalog"
    __table_args__ = (UniqueConstraint("tenant_id", "key", name="uq_tool_catalog_tenant_key"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    key: Mapped[str] = mapped_column(String(100))
    category: Mapped[str] = mapped_column(String(50))
    provider_name: Mapped[str] = mapped_column(String(100))
    actions: Mapped[list] = mapped_column(JSONB, default=list)
    action_schema: Mapped[dict] = mapped_column(JSONB, default=dict)
    is_custom: Mapped[bool] = mapped_column(Boolean, default=False)


class Integration(IdMixin, TimestampMixin, Base):
    """Org-level OAuth/API credential store, one level above tool_connections — e.g.
    one Shopify OAuth grant that several tool_connections can share."""
    __tablename__ = "integrations"
    __table_args__ = (UniqueConstraint("tenant_id", "provider_name", name="uq_integration_tenant_provider"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    provider_name: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), default="disconnected")  # disconnected|connected|error
    credentials_encrypted: Mapped[str | None] = mapped_column(Text)
    connected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ToolConnection(IdMixin, TimestampMixin, Base):
    """An org's instance of a catalog tool. One connection per (tenant, catalog tool) —
    reconnecting updates the existing row (see connect_tool in integrations_tools.py)
    rather than creating a second one."""
    __tablename__ = "tool_connections"
    __table_args__ = (UniqueConstraint("tenant_id", "tool_catalog_id", name="uq_tool_connection_tenant_tool"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    tool_catalog_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tool_catalog.id"), index=True)
    integration_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("integrations.id"))
    status: Mapped[str] = mapped_column(String(20), default="disconnected")  # disconnected|connected|error
    credentials_encrypted: Mapped[str | None] = mapped_column(Text)
    config: Mapped[dict] = mapped_column(JSONB, default=dict)
    guardrails: Mapped[dict] = mapped_column(JSONB, default=dict)


class GuardrailPolicy(IdMixin, TimestampMixin, Base):
    """A named, structured guardrail rule on one connection (e.g.
    refund_limit_usd=200, destructive_actions_enabled=false) — the queryable form of
    ToolConnection.guardrails, which stays as a denormalised convenience blob."""
    __tablename__ = "guardrail_policies"
    __table_args__ = (UniqueConstraint("tool_connection_id", "key", name="uq_guardrail_connection_key"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    tool_connection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tool_connections.id"), index=True)
    key: Mapped[str] = mapped_column(String(100))
    value: Mapped[dict] = mapped_column(JSONB, default=dict)
