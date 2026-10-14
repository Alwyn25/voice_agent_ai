"""Vocal Backend Tasksheet P6 (Calls Runtime): the per-call trace tables the new
app/services/call_runtime.py Call Runtime Service writes to as it runs a compiled
LangGraph call-flow graph (P3). Genuinely new — nothing in the schema before this
wave recorded per-LangGraph-node execution or per-step captures (the pre-existing
`decision_traces`/`tool_executions` in app/db/models/audit.py are shaped around the
OLDER state-machine/turn_runner.py engine's per-TURN planner reasoning, not a
graph's per-NODE steps, so reconciling onto them would conflate two different
engines' concepts). `tool_call_logs` is the one exception: see ToolExecution in
audit.py, which gained two additive nullable columns (call_id, step_key) instead of
a third new table, since its existing input/output/status/duration_ms shape already
matches "request, response, status, latency_ms" exactly.
"""
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin


class CallEvent(IdMixin, Base):
    """One row per LangGraph node execution — the node-transition logger's output
    (audit + replay, per the tasksheet's own note). `input` is the accumulated
    CallState immediately before this step ran; `output` is exactly the partial
    state dict the node returned (LangGraph's own `astream(..., stream_mode=
    "updates")` shape) — never the two states diffed, so a step's own contribution
    is unambiguous even when a later step overwrites the same key."""
    __tablename__ = "call_events"
    # No separate index=True on call_id: the composite index below already covers
    # every real query (always tenant_id + call_id together — see app/api/calls.py,
    # app/services/call_runtime.py), matching the same call_id/tenant_id pairing
    # ToolExecution (app/db/models/audit.py) already uses for its own call_id.
    __table_args__ = (Index("ix_call_events_call", "tenant_id", "call_id", "created_at"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    call_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("calls.id"))
    step_key: Mapped[str] = mapped_column(String(100))
    step_type: Mapped[str] = mapped_column(String(20))  # SAY|TOOL|THINK|RETRIEVE
    input: Mapped[dict] = mapped_column(JSONB, default=dict)
    output: Mapped[dict] = mapped_column(JSONB, default=dict)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CallCapture(IdMixin, Base):
    """One row per structured value captured during a call, per the step (usually a
    THINK node) whose config named it — the durable, queryable audit trail
    CallState.captures itself never gets (that dict lives only in the checkpointer's
    process-lifetime state). Distinct from CustomerFact (app/db/models/crm.py):
    that is cross-call, durable CUSTOMER memory the planner reads back on a later
    call; this is a per-CALL record of what this one call's graph run captured,
    kept even for a value nothing durable ever reads again."""
    __tablename__ = "call_captures"
    # Same reasoning as CallEvent above: no separate index=True on call_id.
    __table_args__ = (Index("ix_call_captures_call", "tenant_id", "call_id"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    call_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("calls.id"))
    step_key: Mapped[str] = mapped_column(String(100))
    capture_key: Mapped[str] = mapped_column(String(100))
    value: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
