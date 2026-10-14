import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin, TimestampMixin


class Conversation(IdMixin, TimestampMixin, Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conv_customer", "tenant_id", "customer_id", "started_at"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    customer_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("customers.id"), index=True)
    channel: Mapped[str] = mapped_column(String(20))  # voice|chat
    direction: Mapped[str] = mapped_column(String(10))  # inbound|outbound
    # The number an UNKNOWN caller is reaching us on, persisted the moment the session opens.
    # For an unregistered caller on chat/console there is no Call row (that is telephony-only)
    # and no customer record, so before this the number lived ONLY in the Redis session — a
    # single point of failure that lost the number on eviction, restart, or TTL lapse, after
    # which registration asked the caller to recite a number they had already dialled from.
    # A known caller carries their number on the customer record, so this stays null for them.
    caller_phone: Mapped[str | None] = mapped_column(String(20))
    # The project this call resolved to (dialled number, the caller's existing visit, or an
    # explicit mention). Pinned once and carried for the rest of the call so retrieval, the
    # prompt facts and any booking all scope to the same project. NULL until resolved, and for
    # single-project tenants where there is nothing to resolve.
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="active")  # active|completed|abandoned
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    summary: Mapped[str | None] = mapped_column(Text)
    outcome: Mapped[str | None] = mapped_column(String(50))
    sentiment: Mapped[str | None] = mapped_column(String(20))
    intent_labels: Mapped[list] = mapped_column(ARRAY(String), default=list)
    campaign_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))


class Message(IdMixin, Base):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conv", "conversation_id", "created_at"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String(20))  # customer|agent|system|tool
    content: Mapped[str] = mapped_column(Text)
    audio_offset_ms: Mapped[int | None] = mapped_column(Integer)
    tool_execution_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    meta: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Call(IdMixin, TimestampMixin, Base):
    __tablename__ = "calls"
    __table_args__ = (Index("ix_calls_customer", "tenant_id", "customer_id", "created_at"),
                      Index("ix_calls_provider", "provider", "provider_call_id"))
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id"))
    customer_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("customers.id"), index=True)
    provider: Mapped[str | None] = mapped_column(String(30))
    provider_call_id: Mapped[str | None] = mapped_column(String(100))
    from_number: Mapped[str | None] = mapped_column(String(20))
    to_number: Mapped[str | None] = mapped_column(String(20))
    # Why we are dialling (marketing|service|…). The dialer re-runs the purpose-sensitive
    # calling policy before every originate and retry, so it lives on the Call (migration 0011).
    purpose: Mapped[str] = mapped_column(String(30), default="marketing")
    # The campaign that queued this call, if any — lets the dialer advance target/stats.
    campaign_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued|ringing|answered|busy|no_answer|voicemail|failed|completed
    # Earliest time the dialer may originate this call. NULL = dial as soon as capacity
    # allows; a retry sets it to now + backoff so the dialer waits (migration 0011).
    scheduled_for: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # 1 for a first dial, incremented down the retry chain; the max-attempts ceiling reads this.
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    answered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[int | None] = mapped_column(Integer)
    hangup_cause: Mapped[str | None] = mapped_column(String(50))
    recording_url: Mapped[str | None] = mapped_column(Text)
    transcript_url: Mapped[str | None] = mapped_column(Text)
    disclosure_played: Mapped[bool] = mapped_column(Boolean, default=False)
    retry_of_call_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    retry_reason: Mapped[str | None] = mapped_column(String(100))
    cost: Mapped[float | None] = mapped_column(Numeric(10, 4))
    # Vocal Backend Tasksheet P6 (Calls Runtime) additions. `agent_id`/`template_version_id`
    # are only ever set for a call the NEW app/services/call_runtime.py Call Runtime Service
    # ran (an Agent, P4, bound to a published template_version, P3) — NULL for every call the
    # pre-existing state-machine/turn_runner.py engine drives (the dialer/campaign calls this
    # model has served since before P6), which is not being migrated onto Agents/templates.
    # `template_version_id` is pinned at call start (not the agent's current one) so a template
    # republished mid-call, or even mid-analysis later, never changes what this call actually
    # ran. `contacts`/`contact_id` (tasksheet wording) is `customer_id` above — P5 already
    # reconciled contacts onto customers, so no separate column is needed here.
    agent_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("agents.id"))
    template_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("template_versions.id"))
    # The full structured turn-by-turn transcript for a Call-Runtime-driven call (mirrors
    # CallState.transcript exactly) — distinct from `transcript_url` above, which is a pointer
    # to externally-stored audio/text this project has no object-storage integration for yet.
    transcript: Mapped[list] = mapped_column(JSONB, default=list)
    summary: Mapped[str | None] = mapped_column(Text)
    intent: Mapped[str | None] = mapped_column(String(100))
    sentiment: Mapped[str | None] = mapped_column(String(20))
    follow_up_required: Mapped[bool] = mapped_column(Boolean, default=False)
    follow_up_reason: Mapped[str | None] = mapped_column(Text)
    follow_up_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CallbackRequest(IdMixin, TimestampMixin, Base):
    __tablename__ = "callback_requests"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    customer_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("customers.id"), index=True)
    requested_for: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source_conversation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|completed|cancelled
    fulfilled_by_call_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
