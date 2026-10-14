"""Custom Instructions & Dynamic Knowledge — tables.

Two separate problems that look alike but need different mechanisms (see
``docs/custom-instructions-implementation-plan.md``):

  * ``CustomInstruction`` — a tenant admin's plain-English behavioural change, classified
    at author time into ``advisory`` (prompt injection, this file's Phase-1 focus) or
    ``routing`` (owns a generated ``RuleSet``, Phase 2). The row is the admin-facing
    wrapper: the English, the compiled artifact, the review/version state, the audit.

  * ``ProjectFactDynamic`` — a time-bounded FACT (a seasonal offer, a floor-rise price)
    the agent must cite accurately. Merged into the same PROJECT FACTS block the grounded
    responder already trusts as verified. No vector embedding, no base-knowledge reload —
    just a date-filtered query, so it self-expires when ``valid_until`` passes.

Both mirror ``RuleSet`` / ``PromptTemplate``: per-row CRUD, versioning, soft enable/disable
and audit rather than a ``tenant.settings`` JSON blob.
"""
import uuid
from datetime import date

from sqlalchemy import Boolean, Date, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin, TimestampMixin


class CustomInstruction(IdMixin, TimestampMixin, Base):
    __tablename__ = "custom_instructions"
    # Fetch server-generated defaults (created_at/updated_at) via RETURNING at flush time,
    # so a freshly inserted/updated row can be serialised after commit without triggering a
    # lazy load — which on the async engine raises MissingGreenlet outside an await.
    __mapper_args__ = {"eager_defaults": True}

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    # null = tenant-wide (applies to every project); else scoped to one project.
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True)

    # The admin's English, preserved verbatim — the audit source of record.
    raw_text: Mapped[str] = mapped_column(Text)

    # 'advisory' | 'routing'. Decided at author time by CONSEQUENCE, not phrasing.
    kind: Mapped[str] = mapped_column(String(20))

    # advisory: the clean imperative snippet rendered into the responder prompt.
    compiled_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    # routing (Phase 2): points at the generated RuleSet that IS the executable artifact.
    rule_set_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rule_sets.id"), nullable=True)

    # {direction, state, intent, valid_from, valid_until} — every key optional/null.
    # Only rows whose scope matches the current turn are rendered.
    scope: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Lower = evaluated/rendered first.
    priority: Mapped[int] = mapped_column(Integer, default=100)

    # The last compile-preview payload: {kind, reasoning, consequence_if_missed,
    # grounding_gaps, test_scenarios, ...}. Kept for the review gate and for debugging why
    # the compiler decided what it did. Never read on the hot turn path.
    preview: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Activated only after explicit admin approval of the compiled form (the review gate).
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    # Update = new version (old retained); delete = deactivate (soft). Audit-preserving.
    version: Mapped[int] = mapped_column(Integer, default=1)
    # Version lineage: every version of one logical instruction shares the first version's
    # id as root_id (the first row points at itself). Lets version history be grouped
    # without a separate name, the way RuleSet groups by name.
    root_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)

    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)


class ProjectFactDynamic(IdMixin, TimestampMixin, Base):
    __tablename__ = "project_facts_dynamic"
    __mapper_args__ = {"eager_defaults": True}

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    # null = tenant-wide offer.
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True)

    # Optional short label ("Festive offer"); ``content`` is the fact cited verbatim.
    label: Mapped[str | None] = mapped_column(String(120), nullable=True)
    # The fact in plain English, e.g. "Festive offer: 10% discount on bookings confirmed
    # before 31 Oct 2026. Applicable on 2 BHK and 3 BHK units only."
    content: Mapped[str] = mapped_column(Text)

    # null valid_from = active immediately; null valid_until = no expiry.
    valid_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    valid_until: Mapped[date | None] = mapped_column(Date, nullable=True)

    # Higher-priority facts appear first in the PROJECT FACTS block.
    priority: Mapped[int] = mapped_column(Integer, default=100)
    # Manual kill-switch, independent of the date range.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)


class IntentVocabularyEntry(IdMixin, TimestampMixin, Base):
    """A tenant's EXTENSION to the canonical intent set (Phase 2, spec §4.3).

    The compiler grounds a routing rule's ``turn.intent`` condition against the vocabulary —
    it may only key on an intent the planner can actually emit. The code defaults
    (``app.custom_instructions.intents.BASE_INTENTS``, derived from
    ``confidence.INTENT_ALIASES``) are the base; these rows add tenant-specific intents
    (e.g. ``loan_enquiry``) with the aliases the planner's free-form label is normalised
    through, so the rule fires every time regardless of phrasing.
    """
    __tablename__ = "intent_vocabulary"
    __table_args__ = (UniqueConstraint("tenant_id", "name", name="uq_intent_vocab_tenant_name"),)
    __mapper_args__ = {"eager_defaults": True}

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    # Canonical intent label the rule keys on, e.g. "loan_enquiry". Lowercase, snake_case.
    name: Mapped[str] = mapped_column(String(60))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Surface forms the planner's free-form label may take — normalize_intent maps any of
    # these onto ``name``, e.g. ["loan", "finance", "emi", "home loan"].
    aliases: Mapped[list] = mapped_column(JSONB, default=list)
    # A couple of example caller lines, for the admin and the compiler prompt.
    examples: Mapped[list] = mapped_column(JSONB, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)


class TransferTarget(IdMixin, TimestampMixin, Base):
    """A named human/queue a routing instruction may escalate to (spec §4.4 / §5).

    "Connect loan questions to the finance desk" resolves ``escalate.target = finance_desk``
    to one of these rows. There is no live SIP-transfer engine today, so an escalation
    creates a pending ``Escalation`` stamped with the resolved target (and the event payload
    carries it) for the right team to pick up; an unknown target degrades to a generic
    escalation with the reason recorded.
    """
    __tablename__ = "transfer_targets"
    __table_args__ = (UniqueConstraint("tenant_id", "key", name="uq_transfer_target_tenant_key"),)
    __mapper_args__ = {"eager_defaults": True}

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    # Stable machine key a rule references, e.g. "finance_desk". Lowercase, snake_case.
    key: Mapped[str] = mapped_column(String(60))
    display_name: Mapped[str] = mapped_column(String(120))
    # Where a handoff would go, when a live-transfer engine exists:
    # {"kind": "phone"|"sip"|"queue"|"user", "value": "..."}. Free-form for now.
    destination: Mapped[dict] = mapped_column(JSONB, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
