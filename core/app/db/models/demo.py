"""GTM demo links (docs/gtm-demo-links.md §4).

These tables are the admin-side catalogue across ALL demos, so unlike the rest of the
schema they are not tenant-scoped. Each prospect owns one demo tenant (tenants.kind=
'demo', §2a) and everything the engine touches for that demo (agent, knowledge base,
calls, conversations) lives inside that tenant; the rows here only point at it.

Industries are data, not an enum: a prospect names a demo_industries key, so adding an
industry is an upload, not a migration.

KEEP IN STEP WITH migrations/versions/0038_demo_links.py. Migration 0001 builds a fresh
database with Base.metadata.create_all(), so on a new environment these model definitions
ARE the schema and 0038's guarded create_table is skipped; on an existing database 0038's
DDL is the schema. Every constraint, index and server default is therefore declared in
both places, with the same names. tests/test_demo_schema_parity.py builds both paths and
fails if they differ.
"""
import uuid
from datetime import datetime

from sqlalchemy import (Boolean, CheckConstraint, DateTime, ForeignKey, ForeignKeyConstraint,
                        Index, Integer, String, Text, UniqueConstraint, func, text, true)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin, TimestampMixin


class DemoIndustry(TimestampMixin, Base):
    """One industry a demo can be built for (real estate, D2C, …). The definition itself is
    versioned in demo_industry_versions; current_version is the one new prospects use."""
    __tablename__ = "demo_industries"
    __table_args__ = (
        CheckConstraint("status IN ('draft', 'active', 'archived')",
                        name="ck_demo_industries_status"),
        CheckConstraint("key ~ '^[a-z][a-z0-9_]{1,39}$'", name="ck_demo_industries_key"),
    )
    key: Mapped[str] = mapped_column(String(40), primary_key=True)
    label: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(20), default="draft", server_default="draft")
    current_version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    updated_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))


class DemoIndustryVersion(Base):
    """An immutable version of an industry definition (§4.8). Prospects pin the version they
    were built from, so editing a definition never silently changes a live demo."""
    __tablename__ = "demo_industry_versions"
    __table_args__ = (
        CheckConstraint("version >= 1", name="ck_demo_industry_versions_version"),
    )
    industry_key: Mapped[str] = mapped_column(
        String(40), ForeignKey("demo_industries.key", ondelete="CASCADE"), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    definition: Mapped[dict] = mapped_column(JSONB)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)


class DemoBatch(IdMixin, TimestampMixin, Base):
    """Prospects emailed together; funnel metrics are reported per batch."""
    __tablename__ = "demo_batches"
    name: Mapped[str] = mapped_column(String(120))
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))


class DemoImport(IdMixin, TimestampMixin, Base):
    """One workbook upload. The dry-run report is stored here, and "apply" saves exactly
    this stored report: what the admin previewed is what gets written, with no re-upload in
    between that could differ. An import is applied at most once."""
    __tablename__ = "demo_imports"
    __table_args__ = (
        CheckConstraint("status IN ('previewed', 'applied', 'discarded')",
                        name="ck_demo_imports_status"),
        Index("ix_demo_imports_created_at", "created_at"),
    )
    industry_key: Mapped[str] = mapped_column(String(40))
    industry_version: Mapped[int] = mapped_column(Integer)
    source_name: Mapped[str] = mapped_column(String(255))
    file_sha256: Mapped[str] = mapped_column(String(64))
    file_size: Mapped[int] = mapped_column(Integer)
    report: Mapped[dict] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(20), default="previewed", server_default="previewed")
    batch_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_batches.id", ondelete="SET NULL"))
    apply_result: Mapped[dict | None] = mapped_column(JSONB)
    created_by: Mapped[str | None] = mapped_column(String(255))
    applied_by: Mapped[str | None] = mapped_column(String(255))
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DemoProspect(IdMixin, TimestampMixin, Base):
    """A company data record: one prospect company (and the one project/product its demo is
    about), imported from a row of the industry sheet. Its facts become the demo agent's
    knowledge base. The record has ONE private link (DemoLink); the PEOPLE at the company
    are DemoContact rows, each sent that link with their own recipient code."""
    __tablename__ = "demo_prospects"
    __table_args__ = (
        ForeignKeyConstraint(["industry_key", "industry_version"],
                             ["demo_industry_versions.industry_key",
                              "demo_industry_versions.version"],
                             name="fk_demo_prospects_industry_version"),
        # The record's own demo tenant (§2a): one tenant never serves two records.
        UniqueConstraint("tenant_id", name="uq_demo_prospects_tenant"),
        # Re-importing the same sheet updates a record instead of duplicating it.
        UniqueConstraint("industry_key", "record_key", name="uq_demo_prospects_record_key"),
        CheckConstraint("status IN ('draft', 'ready_for_qa', 'qa_passed', 'live', 'expired', "
                        "'revoked', 'dropped')", name="ck_demo_prospects_status"),
        Index("ix_demo_prospects_industry_key", "industry_key"),
        Index("ix_demo_prospects_status", "status"),
    )
    tenant_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"))
    industry_key: Mapped[str] = mapped_column(String(40))
    industry_version: Mapped[int] = mapped_column(Integer)
    # slug(company) + "--" + slug(project): the natural key a re-import matches on.
    record_key: Mapped[str] = mapped_column(String(160))
    company_name: Mapped[str] = mapped_column(String(200))
    company_name_pronunciation: Mapped[str | None] = mapped_column(String(200))
    city: Mapped[str | None] = mapped_column(String(80))
    website_url: Mapped[str | None] = mapped_column(Text)
    trigger_note: Mapped[str | None] = mapped_column(Text)
    research_verified: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    source_file: Mapped[str | None] = mapped_column(String(255))
    source_row: Mapped[int | None] = mapped_column(Integer)
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Kolkata",
                                          server_default="Asia/Kolkata")
    agent_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("agents.id"))
    knowledge_base_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_bases.id"))
    # draft|ready_for_qa|qa_passed|live|expired|revoked|dropped
    status: Mapped[str] = mapped_column(String(20), default="draft", server_default="draft")
    dropped_reason: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))


class DemoContact(IdMixin, TimestampMixin, Base):
    """A person at a prospect company (a buyer from the sheet). They are sent their company's
    link with their own recipient code, so opens, calls and questions are attributed to one
    person. needs_review marks
    an imported contact whose details the importer could not pair with confidence (the sheet
    lists names and emails as separate comma lists that don't always line up)."""
    __tablename__ = "demo_contacts"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'excluded')", name="ck_demo_contacts_status"),
        Index("ix_demo_contacts_prospect_id", "prospect_id"),
        Index("ix_demo_contacts_batch_id", "batch_id"),
        Index("uq_demo_contacts_name", "prospect_id", text("lower(full_name)"), unique=True),
        Index("uq_demo_contacts_email", "prospect_id", text("lower(email)"), unique=True,
              postgresql_where=text("email IS NOT NULL")),
        # The same person is never emailed twice in one batch.
        Index("uq_demo_contacts_batch_email", "batch_id", text("lower(email)"), unique=True,
              postgresql_where=text("batch_id IS NOT NULL AND email IS NOT NULL")),
    )
    prospect_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_prospects.id", ondelete="CASCADE"))
    batch_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_batches.id", ondelete="SET NULL"))
    full_name: Mapped[str] = mapped_column(String(160))
    first_name: Mapped[str | None] = mapped_column(String(80))
    role: Mapped[str | None] = mapped_column(String(120))
    email: Mapped[str | None] = mapped_column(String(254))
    linkedin_url: Mapped[str | None] = mapped_column(Text)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    review_notes: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="active", server_default="active")


class DemoProspectField(IdMixin, TimestampMixin, Base):
    """One public fact about the prospect (project name, price, return policy…). A null value
    means "not published": the agent says the sales team will confirm instead of guessing."""
    __tablename__ = "demo_prospect_fields"
    __table_args__ = (
        UniqueConstraint("prospect_id", "field_key", name="uq_demo_prospect_field"),
        CheckConstraint("confidence IN ('confirmed', 'needs_check')",
                        name="ck_demo_prospect_fields_confidence"),
    )
    prospect_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_prospects.id", ondelete="CASCADE"))
    field_key: Mapped[str] = mapped_column(String(60))
    value: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    confidence: Mapped[str] = mapped_column(String(12), default="confirmed",
                                            server_default="confirmed")


class DemoSampleDataset(IdMixin, Base):
    """Dummy data for a demo. prospect_id NULL = the industry's template, copied to each new
    prospect. Versions are kept for audit and rollback; exactly one is active per owner."""
    __tablename__ = "demo_sample_datasets"
    __table_args__ = (
        ForeignKeyConstraint(["industry_key", "industry_version"],
                             ["demo_industry_versions.industry_key",
                              "demo_industry_versions.version"],
                             name="fk_demo_sample_datasets_industry_version"),
        Index("ix_demo_sample_datasets_prospect_id", "prospect_id"),
        # Versions are numbered per owner: per prospect, or per industry for the template.
        Index("uq_demo_datasets_prospect_version", "prospect_id", "version", unique=True,
              postgresql_where=text("prospect_id IS NOT NULL")),
        Index("uq_demo_datasets_template_version", "industry_key", "version", unique=True,
              postgresql_where=text("prospect_id IS NULL")),
        # Exactly one active dataset per prospect, and one active template per industry.
        Index("uq_demo_datasets_prospect_active", "prospect_id", unique=True,
              postgresql_where=text("prospect_id IS NOT NULL AND is_active")),
        Index("uq_demo_datasets_template_active", "industry_key", unique=True,
              postgresql_where=text("prospect_id IS NULL AND is_active")),
    )
    prospect_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_prospects.id", ondelete="CASCADE"))
    industry_key: Mapped[str] = mapped_column(String(40))
    industry_version: Mapped[int] = mapped_column(Integer)
    data: Mapped[dict] = mapped_column(JSONB)
    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=true())
    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)


class DemoLink(IdMixin, TimestampMixin, Base):
    """The private link of one company record: everyone emailed about that company gets the
    same link, each with their own recipient code (DemoLinkRecipient) so opens, calls and
    questions are attributed to a person. Lookup is by the SHA-256 of the token; the token
    itself is also kept encrypted (token_ciphertext, DEMO_LINK_TOKEN_KEYS) so the link can
    be exported again for people added later. NULL ciphertext = a link made before 0039,
    which can't be exported (make a new one). The limit columns are optional per-link
    overrides of the DEMO_* settings (NULL = use .env)."""
    __tablename__ = "demo_links"
    __table_args__ = (
        UniqueConstraint("token_hash", name="uq_demo_links_token_hash"),
        CheckConstraint("status IN ('active', 'expired', 'revoked')", name="ck_demo_links_status"),
        CheckConstraint("token_hash ~ '^[0-9a-f]{64}$'", name="ck_demo_links_token_hash"),
        CheckConstraint("expires_at > activated_at", name="ck_demo_links_expiry"),
        CheckConstraint(
            "(max_calls_per_day IS NULL OR max_calls_per_day > 0) AND "
            "(max_call_seconds IS NULL OR max_call_seconds > 0) AND "
            "(max_total_minutes IS NULL OR max_total_minutes > 0) AND "
            "(max_concurrent_calls IS NULL OR max_concurrent_calls > 0)",
            name="ck_demo_links_limits"),
        Index("ix_demo_links_prospect_id", "prospect_id"),
        # One active link per company record; making a new one revokes the old.
        Index("uq_demo_links_one_active_per_record", "prospect_id", unique=True,
              postgresql_where=text("status = 'active'")),
    )
    prospect_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_prospects.id", ondelete="CASCADE"))
    token_hash: Mapped[str] = mapped_column(String(64))
    token_prefix: Mapped[str] = mapped_column(String(8))
    token_ciphertext: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="active", server_default="active")
    activated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    max_calls_per_day: Mapped[int | None] = mapped_column(Integer)
    max_call_seconds: Mapped[int | None] = mapped_column(Integer)
    max_total_minutes: Mapped[int | None] = mapped_column(Integer)
    max_concurrent_calls: Mapped[int | None] = mapped_column(Integer)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_reason: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))


class DemoLinkRecipient(IdMixin, TimestampMixin, Base):
    """One person a company link is sent to. Their copy of the link carries ref_code
    (?r=…), so a visit can be attributed to them without putting their email in the URL.
    The code is attribution, not a credential: it means nothing without the link's token.
    Usually the contact belongs to the link's own company, but an admin may send a
    company's demo to anyone (e.g. a group's head office)."""
    __tablename__ = "demo_link_recipients"
    __table_args__ = (
        UniqueConstraint("ref_code", name="uq_demo_link_recipients_ref"),
        UniqueConstraint("demo_link_id", "contact_id", name="uq_demo_link_recipients_contact"),
        CheckConstraint("ref_code ~ '^[A-Za-z0-9_-]{6,16}$'", name="ck_demo_link_recipients_ref"),
        Index("ix_demo_link_recipients_contact_id", "contact_id"),
    )
    demo_link_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_links.id", ondelete="CASCADE"))
    contact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_contacts.id", ondelete="CASCADE"))
    ref_code: Mapped[str] = mapped_column(String(16))
    # Last time this person's URL went into an export for the email tool ("emailed").
    last_exported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))


class DemoVisitor(IdMixin, TimestampMixin, Base):
    """One browser that opened a company link (visitor_key = the page's first-party id).
    attribution says who we think it is:

      recipient        arrived on a recipient's code, the first browser to do so
      shared           arrived on a recipient's code another browser already used: that
                       person on a second device, or someone they forwarded the email to
      self_identified  typed their name and email on the page
      unknown          no code (the link was copied from the address bar, or the code
                       was stripped) and no name given

    First touch wins for the landing details (code, email step, UTM tags)."""
    __tablename__ = "demo_visitors"
    __table_args__ = (
        UniqueConstraint("demo_link_id", "visitor_key", name="uq_demo_visitors_link_key"),
        CheckConstraint("attribution IN ('recipient', 'shared', 'self_identified', 'unknown')",
                        name="ck_demo_visitors_attribution"),
        Index("ix_demo_visitors_recipient_id", "recipient_id"),
    )
    demo_link_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_links.id", ondelete="CASCADE"))
    visitor_key: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    recipient_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_link_recipients.id", ondelete="SET NULL"))
    attribution: Mapped[str] = mapped_column(String(20), default="unknown", server_default="unknown")
    identified_name: Mapped[str | None] = mapped_column(String(120))
    identified_email: Mapped[str | None] = mapped_column(String(254))
    identified_role: Mapped[str | None] = mapped_column(String(120))
    identified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)
    # Set by the page's own beacon after a person saw it (never by a plain GET, §9.2).
    first_viewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    landing_ref: Mapped[str | None] = mapped_column(String(16))      # the ?r= code as given
    source_step: Mapped[str | None] = mapped_column(String(10))      # e1|e2|e3|e4|li
    utm: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    ip_hash: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(String(400))


class DemoSession(IdMixin, TimestampMixin, Base):
    """One browser call on a demo link. state holds what the demo tools changed during the
    call (bookings, orders, returns); every call starts from a fresh copy of the dummy data."""
    __tablename__ = "demo_sessions"
    __table_args__ = (
        CheckConstraint("outcome IS NULL OR outcome IN ('goal_completed', 'partial', 'none')",
                        name="ck_demo_sessions_outcome"),
        CheckConstraint("end_reason IS NULL OR end_reason IN ('user_hangup', 'max_duration', "
                        "'idle', 'error', 'budget')", name="ck_demo_sessions_end_reason"),
        Index("ix_demo_sessions_link_started", "demo_link_id", "started_at"),
        Index("ix_demo_sessions_created_at", "created_at"),
    )
    demo_link_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_links.id", ondelete="CASCADE"))
    visitor_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    # Who made the call, as known when it started (see DemoVisitor.attribution).
    demo_visitor_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_visitors.id", ondelete="SET NULL"))
    recipient_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_link_recipients.id", ondelete="SET NULL"))
    # The conversation (inside the record's demo tenant) this call runs on: transcript,
    # traces and the sample lead's booking hang off it.
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL"))
    call_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("calls.id", ondelete="SET NULL"))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[int | None] = mapped_column(Integer)
    resolved_context: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    state: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    outcome: Mapped[str | None] = mapped_column(String(20))      # goal_completed|partial|none
    end_reason: Mapped[str | None] = mapped_column(String(20))   # user_hangup|max_duration|idle|error|budget
    ip_hash: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(String(400))


class DemoEvent(IdMixin, Base):
    """Funnel events. page_viewed comes only from the page's own beacon, never a plain GET,
    so mail scanners that pre-fetch links don't count as opens (§9.2)."""
    __tablename__ = "demo_events"
    __table_args__ = (
        CheckConstraint(
            "type IN ('page_viewed', 'kb_tab_viewed', 'sample_tab_viewed', 'call_started', "
            "'call_ended', 'transcript_viewed', 'cta_real_calls_clicked', "
            "'contact_submitted', 'expired_page_viewed', 'visitor_identified')",
            name="ck_demo_events_type"),
        Index("ix_demo_events_link_type_time", "demo_link_id", "type", "occurred_at"),
    )
    demo_link_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_links.id", ondelete="CASCADE"))
    visitor_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    demo_visitor_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_visitors.id", ondelete="SET NULL"))
    recipient_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_link_recipients.id", ondelete="SET NULL"))
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_sessions.id", ondelete="SET NULL"))
    type: Mapped[str] = mapped_column(String(40))
    source_step: Mapped[str | None] = mapped_column(String(10))   # e1|e2|e3|e4
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")


class DemoLead(IdMixin, TimestampMixin, Base):
    """A Contact us / Schedule a call submission. Written here first, in the request, so a
    lead survives a mail or Google outage; the notifier then emails it and appends it to the
    leads Google Sheet, where the sales team works it. call_status / called_by / notes are
    edited in the sheet and read back here (§10.3). Statuses come from DEMO_LEAD_STATUSES, so
    call_status has no CHECK constraint."""
    __tablename__ = "demo_leads"
    __table_args__ = (
        UniqueConstraint("sheet_row_key", name="uq_demo_leads_sheet_row_key"),
        CheckConstraint(
            "kind IN ('contact', 'schedule_call', 'real_calls_pilot', 'rebuild_request')",
            name="ck_demo_leads_kind"),
        CheckConstraint("char_length(message) <= 2000", name="ck_demo_leads_message_len"),
        Index("ix_demo_leads_prospect_id", "prospect_id"),
        Index("ix_demo_leads_created_at", "created_at"),
        # A double-click or refresh must not create a second lead the same (UTC) day.
        Index("uq_demo_leads_daily", text("lower(email)"), "kind",
              text("((created_at AT TIME ZONE 'UTC')::date)"), unique=True),
        # The notifier's outbox sweep: leads still owed an email or a sheet row.
        Index("ix_demo_leads_pending_notify", "created_at",
              postgresql_where=text("email_notified_at IS NULL OR sheet_synced_at IS NULL")),
    )
    demo_link_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_links.id", ondelete="SET NULL"))
    prospect_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_prospects.id", ondelete="SET NULL"))
    contact_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_contacts.id", ondelete="SET NULL"))
    kind: Mapped[str] = mapped_column(String(20))  # contact|schedule_call|real_calls_pilot|rebuild_request
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(254))
    phone: Mapped[str | None] = mapped_column(String(20))
    company: Mapped[str | None] = mapped_column(String(200))
    role: Mapped[str | None] = mapped_column(String(120))
    preferred_time: Mapped[str | None] = mapped_column(String(120))
    message: Mapped[str | None] = mapped_column(Text)
    last_session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("demo_sessions.id", ondelete="SET NULL"))
    call_status: Mapped[str] = mapped_column(String(30), default="New", server_default="New")
    called_by: Mapped[str | None] = mapped_column(String(120))
    notes: Mapped[str | None] = mapped_column(Text)
    sheet_row_key: Mapped[str] = mapped_column(String(40))
    email_notified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    sheet_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notify_attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    last_notify_error: Mapped[str | None] = mapped_column(Text)
