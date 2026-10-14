import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin, TimestampMixin


class Ticket(IdMixin, TimestampMixin, Base):
    __tablename__ = "tickets"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    customer_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("customers.id"), index=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    subject: Mapped[str] = mapped_column(String(300))
    description: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(50))
    priority: Mapped[str] = mapped_column(String(20), default="medium")
    status: Mapped[str] = mapped_column(String(20), default="open")  # open|in_progress|resolved|escalated|closed
    assigned_to: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    resolution: Mapped[str | None] = mapped_column(Text)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    sla_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Escalation(IdMixin, TimestampMixin, Base):
    __tablename__ = "escalations"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    customer_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("customers.id"), index=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    ticket_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    reason: Mapped[str] = mapped_column(String(100))
    trigger_rule: Mapped[str | None] = mapped_column(String(100))
    confidence: Mapped[float | None] = mapped_column()
    briefing: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|accepted|resolved
    assigned_to: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    transferred_call_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    # The resolved transfer-target key a routing instruction routed this handoff to
    # (TransferTarget.key), or NULL for a generic escalation. Lets the right team/queue
    # pick it up; degrades to NULL when the named target is unknown.
    target: Mapped[str | None] = mapped_column(String(60))


class Campaign(IdMixin, TimestampMixin, Base):
    """Vocal Backend Tasksheet P5 additions — agent_id, concurrency_limit,
    calling_rules — layered onto the pre-existing campaign engine (audience_filter/
    schedule/stats + the rule-engine-backed compliance gate in
    app/services/outbound.py) rather than a rebuild. `calling_rules` is
    deliberately NOT a second quiet-hours/DNC mechanism: that already lives in
    the tenant-wide rule engine (engine/rules/), which every dial passes through
    regardless of campaign. This field is for CAMPAIGN-specific overrides on top
    of that (e.g. a stricter max_attempts just for this campaign) — empty by
    default, meaning "defer entirely to the tenant's rule engine"."""
    __tablename__ = "campaigns"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20), default="draft")  # draft|running|paused|completed
    audience_filter: Mapped[dict] = mapped_column(JSONB, default=dict)
    pack_prompt_override: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    schedule: Mapped[dict] = mapped_column(JSONB, default=dict)
    stats: Mapped[dict] = mapped_column(JSONB, default=dict)
    agent_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("agents.id"))
    concurrency_limit: Mapped[int | None] = mapped_column(Integer)  # None = no per-campaign cap
    calling_rules: Mapped[dict] = mapped_column(JSONB, default=dict)


class CampaignTarget(IdMixin, TimestampMixin, Base):
    __tablename__ = "campaign_targets"
    __table_args__ = (UniqueConstraint("campaign_id", "customer_id"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    campaign_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("campaigns.id"), index=True)
    customer_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("customers.id"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|calling|done|failed|excluded
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    exclusion_reason: Mapped[str | None] = mapped_column(String(100))
    # Vocal Backend Tasksheet P5 — mirrors the retry-backoff timing that already
    # lives on Call.scheduled_for (per-attempt), promoted to the target level so
    # a caller doesn't have to join through calls to see "not now, retry at X".
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ConsentRecord(IdMixin, Base):
    __tablename__ = "consent_records"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    customer_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("customers.id"), index=True)
    consent_type: Mapped[str] = mapped_column(String(30))  # calls|recording|marketing|data_processing
    granted: Mapped[bool] = mapped_column()
    channel: Mapped[str | None] = mapped_column(String(30))
    evidence_uri: Mapped[str | None] = mapped_column(Text)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DncEntry(IdMixin, TimestampMixin, Base):
    __tablename__ = "dnc_list"
    __table_args__ = (UniqueConstraint("tenant_id", "phone_normalized", "scope"),
                      Index("ix_dnc_phone", "tenant_id", "phone_normalized"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    phone_normalized: Mapped[str] = mapped_column(String(20))
    customer_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    scope: Mapped[str] = mapped_column(String(20), default="all")  # all|marketing|voice
    source: Mapped[str] = mapped_column(String(30), default="customer_request")  # customer_request|regulator|import
    added_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
