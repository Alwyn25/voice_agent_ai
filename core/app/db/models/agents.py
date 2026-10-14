"""Vocal Backend Tasksheet P4 (Agents): a deployable agent binds a published
template_version (never a live, still-editable call_flow_template — see P3's
TemplateVersion docstring), an optional phone number, and its own knowledge-base/
tool attachments and daily metrics rollup.
"""
import uuid
from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin, TimestampMixin


class Agent(IdMixin, TimestampMixin, Base):
    __tablename__ = "agents"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    purpose_label: Mapped[str | None] = mapped_column(String(100))
    direction: Mapped[str] = mapped_column(String(10), default="inbound")  # inbound|outbound
    status: Mapped[str] = mapped_column(String(20), default="draft")  # draft|live|paused
    voice: Mapped[str | None] = mapped_column(String(100))
    template_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("template_versions.id"))
    phone_number_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("phone_numbers.id"))
    is_deleted: Mapped[bool] = mapped_column(default=False)

    # ── Behaviour tab (added outside the original tasksheet's P4 scope, by
    # explicit user request, to match the Agent Builder reference design) ──
    description: Mapped[str | None] = mapped_column(String(2000))
    language: Mapped[str] = mapped_column(String(20), default="en-US")
    system_instructions: Mapped[str | None] = mapped_column(String(4000))
    # {"warmth": 0-100, "formality": 0-100, "verbosity": 0-100, "persistence": 0-100}
    personality: Mapped[dict] = mapped_column(JSONB, default=dict)
    greeting_message: Mapped[str | None] = mapped_column(String(500))
    closing_message: Mapped[str | None] = mapped_column(String(500))

    # ── Voice tab ──
    stt_model: Mapped[str | None] = mapped_column(String(100))
    tts_model: Mapped[str | None] = mapped_column(String(100))
    voice_accent: Mapped[str | None] = mapped_column(String(100))
    speaking_speed: Mapped[float] = mapped_column(default=1.0)
    response_latency_ms: Mapped[int] = mapped_column(Integer, default=300)
    interruption_sensitivity: Mapped[int] = mapped_column(Integer, default=50)  # 0-100 -> Low/Medium/High
    silence_timeout_seconds: Mapped[int] = mapped_column(Integer, default=6)
    background_noise: Mapped[str | None] = mapped_column(String(50))
    call_quality: Mapped[str | None] = mapped_column(String(20))


class AgentGuardrail(IdMixin, TimestampMixin, Base):
    """Conversational intercept rules for the Behaviour/Guardrails tab — distinct
    from `guardrail_policies` (P1 Tool Library), which gates a TOOL connection's
    own destructive actions. This table is a per-agent "if this happens, do this"
    list checked on every conversation turn (e.g. "caller asks for a human ->
    warm transfer"), not a tool-permission check."""
    __tablename__ = "agent_guardrails"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    agent_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("agents.id"), index=True)
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    condition_text: Mapped[str] = mapped_column(String(300))
    action_text: Mapped[str] = mapped_column(String(300))
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class AgentKnowledgeBase(IdMixin, Base):
    __tablename__ = "agent_knowledge_bases"
    __table_args__ = (UniqueConstraint("agent_id", "knowledge_base_id", name="uq_agent_kb"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    agent_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("agents.id"), index=True)
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_bases.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AgentTool(IdMixin, Base):
    __tablename__ = "agent_tools"
    __table_args__ = (UniqueConstraint("agent_id", "tool_connection_id", name="uq_agent_tool"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    agent_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("agents.id"), index=True)
    tool_connection_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tool_connections.id"))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    config_override: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AgentMetricsDaily(IdMixin, Base):
    """Backfilled zero until P6 (Calls Runtime) and P7 (Overview) exist to write
    real numbers here — see this table's DoD note in the tasksheet."""
    __tablename__ = "agent_metrics_daily"
    __table_args__ = (UniqueConstraint("agent_id", "date", name="uq_agent_metrics_day"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    agent_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("agents.id"), index=True)
    date: Mapped[date] = mapped_column(Date)
    calls: Mapped[int] = mapped_column(Integer, default=0)
    resolved: Mapped[int] = mapped_column(Integer, default=0)
    avg_duration_seconds: Mapped[int] = mapped_column(Integer, default=0)
    actions_taken: Mapped[int] = mapped_column(Integer, default=0)
