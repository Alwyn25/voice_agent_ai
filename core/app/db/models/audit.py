import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin


class DecisionTrace(IdMixin, Base):
    __tablename__ = "decision_traces"
    __table_args__ = (Index("ix_traces_conv", "tenant_id", "conversation_id", "turn_index"),
                      Index("ix_traces_time", "tenant_id", "created_at"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    customer_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    turn_index: Mapped[int] = mapped_column(Integer, default=0)
    input_text: Mapped[str | None] = mapped_column(Text)
    retrieved_memory: Mapped[dict] = mapped_column(JSONB, default=dict)
    retrieved_chunks: Mapped[list] = mapped_column(JSONB, default=list)
    rules_evaluated: Mapped[list] = mapped_column(JSONB, default=list)
    planner_reasoning: Mapped[str | None] = mapped_column(Text)
    proposed_tool: Mapped[str | None] = mapped_column(String(100))
    tool_params: Mapped[dict | None] = mapped_column(JSONB)
    validation_result: Mapped[dict | None] = mapped_column(JSONB)
    confidence: Mapped[float | None] = mapped_column(Numeric(4, 3))
    confidence_band: Mapped[str | None] = mapped_column(String(10))  # high|medium|low
    action_taken: Mapped[str | None] = mapped_column(String(30))  # executed|clarified|escalated|blocked|spoke
    state_transition_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    model_used: Mapped[str | None] = mapped_column(String(100))
    tokens: Mapped[dict] = mapped_column(JSONB, default=dict)
    # Raw usage only -- never a stored $ figure, so re-pricing after a rate change is an edit
    # to app/observability/pricing.py, not a backfill. Both null on a chat turn (no TTS/STT
    # ever runs there); tts_characters is set only once the response text reaches its final
    # form (see turn_runner.py, after the price-rounding/trim passes).
    tts_characters: Mapped[int | None] = mapped_column(Integer)
    stt_seconds: Mapped[float | None] = mapped_column(Numeric(6, 2))
    prompt_versions: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ToolExecution(IdMixin, Base):
    """Vocal Backend Tasksheet P6's `tool_call_logs` reconciles onto this pre-existing
    table rather than duplicating it: "every tool invocation during a call — request,
    response, status, latency_ms" is exactly this row's existing input/output/status/
    duration_ms shape. `call_id`/`step_key` (added for P6) are set only for a TOOL-node
    invocation the new Call Runtime Service made (app/engine/templates/nodes.py's
    tool_node_factory, via the P1 tool marketplace's BaseToolAdapter) — `decision_trace_id`
    stays the field a pre-existing OLDER-engine (turn_runner.py) tool call sets instead.
    The two are mutually exclusive per row, never both, since they come from two different
    engines' tool systems (see app/db/models/calls_runtime.py's own module docstring for
    why call_events/call_captures could not be reconciled the same way)."""
    __tablename__ = "tool_executions"
    __table_args__ = (Index("ix_tool_exec_idem", "tenant_id", "idempotency_key"),
                      Index("ix_tool_exec_conv", "tenant_id", "conversation_id"),
                      Index("ix_tool_exec_call", "tenant_id", "call_id"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    decision_trace_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), index=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    customer_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    idempotency_key: Mapped[str | None] = mapped_column(String(300))
    tool_name: Mapped[str] = mapped_column(String(100))
    input: Mapped[dict] = mapped_column(JSONB, default=dict)
    output: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(20))  # success|failed|blocked
    error: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    call_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("calls.id"))
    step_key: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AuditEvent(IdMixin, Base):
    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_time", "tenant_id", "created_at"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    actor: Mapped[str] = mapped_column(String(150))
    action: Mapped[str] = mapped_column(String(100))
    resource_type: Mapped[str | None] = mapped_column(String(50))
    resource_id: Mapped[str | None] = mapped_column(String(100))
    before: Mapped[dict | None] = mapped_column(JSONB)
    after: Mapped[dict | None] = mapped_column(JSONB)
    ip: Mapped[str | None] = mapped_column(String(45))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AnalyticsEvent(IdMixin, Base):
    __tablename__ = "analytics_events"
    __table_args__ = (Index("ix_analytics_time", "tenant_id", "event_type", "occurred_at"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(50))
    customer_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    properties: Mapped[dict] = mapped_column(JSONB, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
