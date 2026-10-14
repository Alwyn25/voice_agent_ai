import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin, TimestampMixin


class ToolDefinition(IdMixin, TimestampMixin, Base):
    __tablename__ = "tool_definitions"
    __table_args__ = (UniqueConstraint("tenant_id", "name"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(30))  # sales|scheduling|support|utility
    input_schema: Mapped[dict] = mapped_column(JSONB, default=dict)
    allowed_states: Mapped[list] = mapped_column(JSONB, default=list)
    preconditions: Mapped[list] = mapped_column(JSONB, default=list)
    postconditions: Mapped[list] = mapped_column(JSONB, default=list)
    effects: Mapped[dict] = mapped_column(JSONB, default=dict)
    failure_policy: Mapped[dict] = mapped_column(JSONB, default=dict)
    requires_confirmation: Mapped[bool] = mapped_column(Boolean, default=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(200))
    permission_level: Mapped[str] = mapped_column(String(30), default="agent")  # agent|supervisor_only|human_only
    implementation: Mapped[str] = mapped_column(String(200))  # builtin:module.func | http
    http_config: Mapped[dict | None] = mapped_column(JSONB)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)


class RuleSet(IdMixin, TimestampMixin, Base):
    __tablename__ = "rule_sets"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    category: Mapped[str] = mapped_column(String(50))  # calling_policy|retry|dnd|escalation|appointment|eligibility
    yaml_source: Mapped[str] = mapped_column(Text)
    compiled: Mapped[dict] = mapped_column(JSONB, default=dict)
    version: Mapped[int] = mapped_column(Integer, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    activated_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class StateMachineDef(IdMixin, TimestampMixin, Base):
    __tablename__ = "state_machine_defs"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    definition: Mapped[dict] = mapped_column(JSONB, default=dict)


class PromptTemplate(IdMixin, TimestampMixin, Base):
    __tablename__ = "prompt_templates"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    purpose: Mapped[str] = mapped_column(String(50))  # persona|planner|summarizer|objection|pitch
    template: Mapped[str] = mapped_column(Text)
    model_hint: Mapped[str | None] = mapped_column(String(100))
    version: Mapped[int] = mapped_column(Integer, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
