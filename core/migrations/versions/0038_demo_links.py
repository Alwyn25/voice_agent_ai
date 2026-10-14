"""GTM demo links, phase 0 (docs/gtm-demo-links.md §4, §2a).

Adds:
  * ``tenants.kind`` text NOT NULL DEFAULT 'customer' (customer | demo). Each demo prospect
    gets its own tenant so the engine's tenant scoping isolates demos from each other; the
    dialer and aggregator now walk only kind='customer'.
  * demo_industries + demo_industry_versions  industries as versioned data, not an enum
  * demo_imports (stored dry-run reports), demo_batches, demo_prospects (company data records), demo_contacts (the people, one
    private link each), demo_prospect_fields, demo_sample_datasets
  * demo_links (contact + the selected record; token hash only), demo_sessions, demo_events
  * demo_leads (the safety copy of every Contact us / Schedule a call submission)

Additive only: no existing row changes meaning (every tenant becomes 'customer', which is
what it already was), and nothing reads the new tables until DEMO_ENABLED=true. ADD COLUMN
with a constant default is metadata-only on PostgreSQL >= 11, so tenants is not rewritten.
Every step is guarded, so re-running after a partial failure is safe.

Revision ID: 0038
Revises: 0036

NOTE: R5.04 (feat/r5.04-embedding-upgrade) adds 0037 on another branch. Whichever of the
two merges into dev second must re-point its down_revision to the other (0038 → "0037"),
or alembic will see two heads.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0038"
down_revision = "0036"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def _has_column(table: str, column: str) -> bool:
    return column in {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def _has_constraint(table: str, name: str) -> bool:
    return bool(op.get_bind().execute(sa.text(
        "SELECT 1 FROM pg_constraint WHERE conname = :n AND conrelid = CAST(:t AS regclass)"),
        {"n": name, "t": table}).scalar())


def _ts():
    return [sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                      nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                      nullable=False)]


def _users_fk(name: str) -> sa.Column:
    return sa.Column(name, UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True)


def upgrade() -> None:
    # ── tenants.kind ──
    if not _has_column("tenants", "kind"):
        op.add_column("tenants", sa.Column("kind", sa.String(20), nullable=False,
                                           server_default="customer"))
    if not _has_constraint("tenants", "ck_tenants_kind"):
        op.create_check_constraint("ck_tenants_kind", "tenants", "kind IN ('customer', 'demo')")

    # ── industries ──
    if not _has_table("demo_industries"):
        op.create_table(
            "demo_industries",
            sa.Column("key", sa.String(40), primary_key=True),
            sa.Column("label", sa.String(80), nullable=False),
            sa.Column("status", sa.String(20), nullable=False, server_default="draft"),
            sa.Column("current_version", sa.Integer(), nullable=False, server_default="1"),
            _users_fk("updated_by"),
            *_ts(),
            sa.CheckConstraint("status IN ('draft', 'active', 'archived')",
                               name="ck_demo_industries_status"),
            sa.CheckConstraint("key ~ '^[a-z][a-z0-9_]{1,39}$'", name="ck_demo_industries_key"),
        )
    if not _has_table("demo_industry_versions"):
        op.create_table(
            "demo_industry_versions",
            sa.Column("industry_key", sa.String(40),
                      sa.ForeignKey("demo_industries.key", ondelete="CASCADE"), primary_key=True),
            sa.Column("version", sa.Integer(), primary_key=True),
            sa.Column("definition", JSONB(), nullable=False),
            _users_fk("created_by"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                      nullable=False),
            sa.CheckConstraint("version >= 1", name="ck_demo_industry_versions_version"),
        )

    # ── batches + prospects ──
    if not _has_table("demo_batches"):
        op.create_table(
            "demo_batches",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("name", sa.String(120), nullable=False),
            _users_fk("created_by"),
            *_ts(),
        )
    # Workbook uploads: the stored dry-run report is exactly what "apply" saves.
    if not _has_table("demo_imports"):
        op.create_table(
            "demo_imports",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("industry_key", sa.String(40), nullable=False),
            sa.Column("industry_version", sa.Integer(), nullable=False),
            sa.Column("source_name", sa.String(255), nullable=False),
            sa.Column("file_sha256", sa.String(64), nullable=False),
            sa.Column("file_size", sa.Integer(), nullable=False),
            sa.Column("report", JSONB(), nullable=False),
            sa.Column("status", sa.String(20), nullable=False, server_default="previewed"),
            sa.Column("batch_id", UUID(as_uuid=True),
                      sa.ForeignKey("demo_batches.id", ondelete="SET NULL"), nullable=True),
            sa.Column("apply_result", JSONB(), nullable=True),
            sa.Column("created_by", sa.String(255), nullable=True),
            sa.Column("applied_by", sa.String(255), nullable=True),
            sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
            *_ts(),
            sa.CheckConstraint("status IN ('previewed', 'applied', 'discarded')",
                               name="ck_demo_imports_status"),
        )
        op.create_index("ix_demo_imports_created_at", "demo_imports", ["created_at"])

    if not _has_table("demo_prospects"):
        op.create_table(
            "demo_prospects",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=True),
            sa.Column("industry_key", sa.String(40), nullable=False),
            sa.Column("industry_version", sa.Integer(), nullable=False),
            sa.Column("record_key", sa.String(160), nullable=False),
            sa.Column("company_name", sa.String(200), nullable=False),
            sa.Column("company_name_pronunciation", sa.String(200), nullable=True),
            sa.Column("city", sa.String(80), nullable=True),
            sa.Column("website_url", sa.Text(), nullable=True),
            sa.Column("trigger_note", sa.Text(), nullable=True),
            sa.Column("research_verified", sa.Boolean(), nullable=False,
                      server_default=sa.text("false")),
            sa.Column("source_file", sa.String(255), nullable=True),
            sa.Column("source_row", sa.Integer(), nullable=True),
            sa.Column("timezone", sa.String(64), nullable=False, server_default="Asia/Kolkata"),
            sa.Column("agent_id", UUID(as_uuid=True), sa.ForeignKey("agents.id"), nullable=True),
            sa.Column("knowledge_base_id", UUID(as_uuid=True),
                      sa.ForeignKey("knowledge_bases.id"), nullable=True),
            sa.Column("status", sa.String(20), nullable=False, server_default="draft"),
            sa.Column("dropped_reason", sa.Text(), nullable=True),
            _users_fk("created_by"),
            *_ts(),
            sa.ForeignKeyConstraint(["industry_key", "industry_version"],
                                    ["demo_industry_versions.industry_key",
                                     "demo_industry_versions.version"],
                                    name="fk_demo_prospects_industry_version"),
            sa.UniqueConstraint("tenant_id", name="uq_demo_prospects_tenant"),
            sa.UniqueConstraint("industry_key", "record_key", name="uq_demo_prospects_record_key"),
            sa.CheckConstraint(
                "status IN ('draft', 'ready_for_qa', 'qa_passed', 'live', 'expired', "
                "'revoked', 'dropped')", name="ck_demo_prospects_status"),
        )
        op.create_index("ix_demo_prospects_industry_key", "demo_prospects", ["industry_key"])
        op.create_index("ix_demo_prospects_status", "demo_prospects", ["status"])

    # People at a prospect company; each gets their own private link.
    if not _has_table("demo_contacts"):
        op.create_table(
            "demo_contacts",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("prospect_id", UUID(as_uuid=True),
                      sa.ForeignKey("demo_prospects.id", ondelete="CASCADE"), nullable=False),
            sa.Column("batch_id", UUID(as_uuid=True),
                      sa.ForeignKey("demo_batches.id", ondelete="SET NULL"), nullable=True),
            sa.Column("full_name", sa.String(160), nullable=False),
            sa.Column("first_name", sa.String(80), nullable=True),
            sa.Column("role", sa.String(120), nullable=True),
            sa.Column("email", sa.String(254), nullable=True),
            sa.Column("linkedin_url", sa.Text(), nullable=True),
            sa.Column("needs_review", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            sa.Column("review_notes", sa.Text(), nullable=True),
            sa.Column("status", sa.String(20), nullable=False, server_default="active"),
            *_ts(),
            sa.CheckConstraint("status IN ('active', 'excluded')", name="ck_demo_contacts_status"),
        )
        op.create_index("ix_demo_contacts_prospect_id", "demo_contacts", ["prospect_id"])
        op.create_index("ix_demo_contacts_batch_id", "demo_contacts", ["batch_id"])
        op.execute("CREATE UNIQUE INDEX uq_demo_contacts_name ON demo_contacts "
                   "(prospect_id, lower(full_name))")
        op.execute("CREATE UNIQUE INDEX uq_demo_contacts_email ON demo_contacts "
                   "(prospect_id, lower(email)) WHERE email IS NOT NULL")
        # The same person is never emailed twice in one batch.
        op.execute("CREATE UNIQUE INDEX uq_demo_contacts_batch_email ON demo_contacts "
                   "(batch_id, lower(email)) WHERE batch_id IS NOT NULL AND email IS NOT NULL")

    if not _has_table("demo_prospect_fields"):
        op.create_table(
            "demo_prospect_fields",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("prospect_id", UUID(as_uuid=True),
                      sa.ForeignKey("demo_prospects.id", ondelete="CASCADE"), nullable=False),
            sa.Column("field_key", sa.String(60), nullable=False),
            sa.Column("value", sa.Text(), nullable=True),
            sa.Column("source_url", sa.Text(), nullable=True),
            sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("confidence", sa.String(12), nullable=False, server_default="confirmed"),
            *_ts(),
            sa.UniqueConstraint("prospect_id", "field_key", name="uq_demo_prospect_field"),
            sa.CheckConstraint("confidence IN ('confirmed', 'needs_check')",
                               name="ck_demo_prospect_fields_confidence"),
        )

    if not _has_table("demo_sample_datasets"):
        op.create_table(
            "demo_sample_datasets",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("prospect_id", UUID(as_uuid=True),
                      sa.ForeignKey("demo_prospects.id", ondelete="CASCADE"), nullable=True),
            sa.Column("industry_key", sa.String(40), nullable=False),
            sa.Column("industry_version", sa.Integer(), nullable=False),
            sa.Column("data", JSONB(), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
            _users_fk("uploaded_by"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                      nullable=False),
            sa.ForeignKeyConstraint(["industry_key", "industry_version"],
                                    ["demo_industry_versions.industry_key",
                                     "demo_industry_versions.version"],
                                    name="fk_demo_sample_datasets_industry_version"),
        )
        op.create_index("ix_demo_sample_datasets_prospect_id", "demo_sample_datasets",
                        ["prospect_id"])
        # Versions are numbered per owner: per prospect, or per industry for the template.
        op.execute("CREATE UNIQUE INDEX uq_demo_datasets_prospect_version "
                   "ON demo_sample_datasets (prospect_id, version) WHERE prospect_id IS NOT NULL")
        op.execute("CREATE UNIQUE INDEX uq_demo_datasets_template_version "
                   "ON demo_sample_datasets (industry_key, version) WHERE prospect_id IS NULL")
        # Exactly one active dataset per prospect, and one active template per industry.
        op.execute("CREATE UNIQUE INDEX uq_demo_datasets_prospect_active "
                   "ON demo_sample_datasets (prospect_id) "
                   "WHERE prospect_id IS NOT NULL AND is_active")
        op.execute("CREATE UNIQUE INDEX uq_demo_datasets_template_active "
                   "ON demo_sample_datasets (industry_key) WHERE prospect_id IS NULL AND is_active")

    # ── links, sessions, events ──
    if not _has_table("demo_links"):
        op.create_table(
            "demo_links",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("contact_id", UUID(as_uuid=True),
                      sa.ForeignKey("demo_contacts.id", ondelete="CASCADE"), nullable=False),
            # The company record the admin selected for this link (default: the contact's own).
            sa.Column("prospect_id", UUID(as_uuid=True),
                      sa.ForeignKey("demo_prospects.id", ondelete="CASCADE"), nullable=False),
            sa.Column("token_hash", sa.String(64), nullable=False),
            sa.Column("token_prefix", sa.String(8), nullable=False),
            sa.Column("status", sa.String(20), nullable=False, server_default="active"),
            sa.Column("activated_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                      nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("max_calls_per_day", sa.Integer(), nullable=True),
            sa.Column("max_call_seconds", sa.Integer(), nullable=True),
            sa.Column("max_total_minutes", sa.Integer(), nullable=True),
            sa.Column("max_concurrent_calls", sa.Integer(), nullable=True),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("revoked_reason", sa.Text(), nullable=True),
            _users_fk("created_by"),
            *_ts(),
            sa.UniqueConstraint("token_hash", name="uq_demo_links_token_hash"),
            sa.CheckConstraint("status IN ('active', 'expired', 'revoked')",
                               name="ck_demo_links_status"),
            sa.CheckConstraint("token_hash ~ '^[0-9a-f]{64}$'", name="ck_demo_links_token_hash"),
            sa.CheckConstraint("expires_at > activated_at", name="ck_demo_links_expiry"),
            sa.CheckConstraint(
                "(max_calls_per_day IS NULL OR max_calls_per_day > 0) AND "
                "(max_call_seconds IS NULL OR max_call_seconds > 0) AND "
                "(max_total_minutes IS NULL OR max_total_minutes > 0) AND "
                "(max_concurrent_calls IS NULL OR max_concurrent_calls > 0)",
                name="ck_demo_links_limits"),
        )
        op.create_index("ix_demo_links_prospect_id", "demo_links", ["prospect_id"])
        op.create_index("ix_demo_links_contact_id", "demo_links", ["contact_id"])
        # One active link per person; making a new one revokes the old.
        op.execute("CREATE UNIQUE INDEX uq_demo_links_one_active ON demo_links (contact_id) "
                   "WHERE status = 'active'")

    if not _has_table("demo_sessions"):
        op.create_table(
            "demo_sessions",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("demo_link_id", UUID(as_uuid=True),
                      sa.ForeignKey("demo_links.id", ondelete="CASCADE"), nullable=False),
            sa.Column("visitor_id", UUID(as_uuid=True), nullable=True),
            sa.Column("conversation_id", UUID(as_uuid=True),
                      sa.ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True),
            sa.Column("call_id", UUID(as_uuid=True),
                      sa.ForeignKey("calls.id", ondelete="SET NULL"), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("duration_seconds", sa.Integer(), nullable=True),
            sa.Column("resolved_context", JSONB(), nullable=False, server_default="{}"),
            sa.Column("state", JSONB(), nullable=False, server_default="{}"),
            sa.Column("outcome", sa.String(20), nullable=True),
            sa.Column("end_reason", sa.String(20), nullable=True),
            sa.Column("ip_hash", sa.String(64), nullable=True),
            sa.Column("user_agent", sa.String(400), nullable=True),
            *_ts(),
            sa.CheckConstraint("outcome IS NULL OR outcome IN ('goal_completed', 'partial', 'none')",
                               name="ck_demo_sessions_outcome"),
            sa.CheckConstraint(
                "end_reason IS NULL OR end_reason IN ('user_hangup', 'max_duration', 'idle', "
                "'error', 'budget')", name="ck_demo_sessions_end_reason"),
        )
        op.create_index("ix_demo_sessions_link_started", "demo_sessions",
                        ["demo_link_id", "started_at"])
        op.create_index("ix_demo_sessions_created_at", "demo_sessions", ["created_at"])

    if not _has_table("demo_events"):
        op.create_table(
            "demo_events",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("demo_link_id", UUID(as_uuid=True),
                      sa.ForeignKey("demo_links.id", ondelete="CASCADE"), nullable=False),
            sa.Column("visitor_id", UUID(as_uuid=True), nullable=True),
            sa.Column("session_id", UUID(as_uuid=True),
                      sa.ForeignKey("demo_sessions.id", ondelete="SET NULL"), nullable=True),
            sa.Column("type", sa.String(40), nullable=False),
            sa.Column("source_step", sa.String(10), nullable=True),
            sa.Column("occurred_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                      nullable=False),
            sa.Column("meta", JSONB(), nullable=False, server_default="{}"),
            sa.CheckConstraint(
                "type IN ('page_viewed', 'kb_tab_viewed', 'sample_tab_viewed', 'call_started', "
                "'call_ended', 'transcript_viewed', 'cta_real_calls_clicked', "
                "'contact_submitted', 'expired_page_viewed')", name="ck_demo_events_type"),
        )
        op.create_index("ix_demo_events_link_type_time", "demo_events",
                        ["demo_link_id", "type", "occurred_at"])

    # ── leads ──
    if not _has_table("demo_leads"):
        op.create_table(
            "demo_leads",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("demo_link_id", UUID(as_uuid=True),
                      sa.ForeignKey("demo_links.id", ondelete="SET NULL"), nullable=True),
            sa.Column("prospect_id", UUID(as_uuid=True),
                      sa.ForeignKey("demo_prospects.id", ondelete="SET NULL"), nullable=True),
            sa.Column("contact_id", UUID(as_uuid=True),
                      sa.ForeignKey("demo_contacts.id", ondelete="SET NULL"), nullable=True),
            sa.Column("kind", sa.String(20), nullable=False),
            sa.Column("name", sa.String(120), nullable=False),
            sa.Column("email", sa.String(254), nullable=False),
            sa.Column("phone", sa.String(20), nullable=True),
            sa.Column("company", sa.String(200), nullable=True),
            sa.Column("role", sa.String(120), nullable=True),
            sa.Column("preferred_time", sa.String(120), nullable=True),
            sa.Column("message", sa.Text(), nullable=True),
            sa.Column("last_session_id", UUID(as_uuid=True),
                      sa.ForeignKey("demo_sessions.id", ondelete="SET NULL"), nullable=True),
            sa.Column("call_status", sa.String(30), nullable=False, server_default="New"),
            sa.Column("called_by", sa.String(120), nullable=True),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column("sheet_row_key", sa.String(40), nullable=False),
            sa.Column("email_notified_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("sheet_synced_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("notify_attempts", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("last_notify_error", sa.Text(), nullable=True),
            *_ts(),
            sa.UniqueConstraint("sheet_row_key", name="uq_demo_leads_sheet_row_key"),
            sa.CheckConstraint(
                "kind IN ('contact', 'schedule_call', 'real_calls_pilot', 'rebuild_request')",
                name="ck_demo_leads_kind"),
            sa.CheckConstraint("char_length(message) <= 2000", name="ck_demo_leads_message_len"),
        )
        op.create_index("ix_demo_leads_prospect_id", "demo_leads", ["prospect_id"])
        op.create_index("ix_demo_leads_created_at", "demo_leads", ["created_at"])
        # A double-click or refresh must not create a second lead the same (UTC) day.
        op.execute("CREATE UNIQUE INDEX uq_demo_leads_daily ON demo_leads "
                   "(lower(email), kind, ((created_at AT TIME ZONE 'UTC')::date))")
        # The notifier's outbox sweep: leads still owed an email or a sheet row.
        op.execute("CREATE INDEX ix_demo_leads_pending_notify ON demo_leads (created_at) "
                   "WHERE email_notified_at IS NULL OR sheet_synced_at IS NULL")


def downgrade() -> None:
    # Dropping tenants.kind would silently turn every demo tenant into a "customer" that the
    # dialer, aggregator and billing then pick up. Refuse instead; erase the demos first.
    if _has_column("tenants", "kind"):
        demo_tenants = op.get_bind().execute(
            sa.text("SELECT count(*) FROM tenants WHERE kind = 'demo'")).scalar()
        if demo_tenants:
            raise RuntimeError(
                f"Refusing to downgrade 0038: {demo_tenants} demo tenant(s) still exist. "
                "Delete the demo prospects and their tenants first.")
    for table in ("demo_leads", "demo_events", "demo_sessions", "demo_links",
                  "demo_sample_datasets", "demo_prospect_fields", "demo_contacts", "demo_prospects",
                  "demo_imports", "demo_batches", "demo_industry_versions", "demo_industries"):
        if _has_table(table):
            op.drop_table(table)   # drops its indexes and constraints with it
    if _has_constraint("tenants", "ck_tenants_kind"):
        op.drop_constraint("ck_tenants_kind", "tenants", type_="check")
    if _has_column("tenants", "kind"):
        op.drop_column("tenants", "kind")
