import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin


class ToolMetric(IdMixin, Base):
    """One row per tool-execution attempt, written by app.observability.tool_metrics.

    Deliberately separate from ToolExecution (the audit/idempotency record): that table
    is only written on success or an execution/postcondition failure, so blocked and
    skipped_duplicate outcomes — permission, schema, rule, precondition failures — never
    reach it. This table records every outcome the executor produces, which is what a
    performance/accuracy view needs.
    """
    __tablename__ = "tool_metrics"
    __table_args__ = (Index("ix_tool_metrics_tool_time", "tenant_id", "tool_name", "created_at"),
                      Index("ix_tool_metrics_status", "tenant_id", "status"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    tool_name: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20))  # success|failed|blocked|skipped_duplicate
    failed_check: Mapped[str | None] = mapped_column(String(100))
    error: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    decision_trace_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
