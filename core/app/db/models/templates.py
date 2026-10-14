"""Vocal Backend Tasksheet P3 (Templates): call-flow templates compiled into a
LangGraph runtime graph (see app/engine/templates/). A template's call_flow_steps
are the editable working copy; publishing snapshots them into an immutable
template_version, which is what a future agent (P4) binds to — never the live,
still-editable template — so an in-progress edit can never affect a call already
running against a published version.
"""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin, TimestampMixin


class CallFlowTemplate(IdMixin, TimestampMixin, Base):
    __tablename__ = "call_flow_templates"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    is_system_template: Mapped[bool] = mapped_column(Boolean, default=False)


class CallFlowStep(IdMixin, TimestampMixin, Base):
    """The editable working copy of a template's steps. step_type is one of
    SAY|TOOL|THINK|RETRIEVE (see app/engine/templates/nodes.py's node factories,
    one per type). `key` is the step's stable identifier — routing config
    (config["routes"]) and edges reference other steps by key, not by row id, so
    reordering/renaming a template's steps in the UI doesn't break a route that
    names a sibling step."""
    __tablename__ = "call_flow_steps"
    __table_args__ = (UniqueConstraint("template_id", "key", name="uq_call_flow_step_template_key"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    template_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("call_flow_templates.id"), index=True)
    step_order: Mapped[int] = mapped_column(Integer)
    key: Mapped[str] = mapped_column(String(100))
    step_type: Mapped[str] = mapped_column(String(20))  # SAY|TOOL|THINK|RETRIEVE
    config: Mapped[dict] = mapped_column(JSONB, default=dict)


class TemplateVersion(IdMixin, Base):
    """Immutable snapshot of a template's steps at the moment of publish.
    `snapshot` is the ordered step list (same shape GraphCompiler.compile()
    takes), so replaying/compiling a past version never depends on
    call_flow_steps still existing or being unchanged."""
    __tablename__ = "template_versions"
    __table_args__ = (UniqueConstraint("template_id", "version", name="uq_template_version"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    template_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("call_flow_templates.id"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    snapshot: Mapped[list] = mapped_column(JSONB, default=list)
    compiled_graph_hash: Mapped[str] = mapped_column(String(64))
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    published_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
