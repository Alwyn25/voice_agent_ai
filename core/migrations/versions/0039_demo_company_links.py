"""GTM demo links: one link per company record, with per-person attribution.

Before: one private link per PERSON (demo_links.contact_id). After: one link per company
record, sent to everyone at that company, each with their own recipient code (?r=…) so a
visit is still attributed to a person without putting their email in the URL.

  * demo_link_recipients  (link, contact, ref_code): who a company link was sent to
  * demo_visitors         one row per browser that opened a link, and who we think it is
                          (recipient · shared · self_identified · unknown)
  * demo_links            - contact_id; + token_ciphertext (the token, encrypted with
                          DEMO_LINK_TOKEN_KEYS, so a link can be exported again for people
                          added later); one active link per company record
  * demo_sessions, demo_events  + demo_visitor_id, recipient_id
  * demo_events.type      + 'visitor_identified'

Existing per-person links are carried over as recipients (so old calls keep their person)
and then revoked: their tokens were never stored, so they could not be exported again.
Nothing has been sent to prospects yet; generate company links after upgrading.

Every step is guarded. On a fresh database migration 0001's create_all() already built the
new shape from the models, and this migration changes nothing (tests/test_demo_schema_parity.py).

Revision ID: 0039
Revises: 0038
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0039"
down_revision = "0038"
branch_labels = None
depends_on = None

EVENT_TYPES_OLD = ("'page_viewed', 'kb_tab_viewed', 'sample_tab_viewed', 'call_started', "
                   "'call_ended', 'transcript_viewed', 'cta_real_calls_clicked', "
                   "'contact_submitted', 'expired_page_viewed'")
EVENT_TYPES_NEW = EVENT_TYPES_OLD + ", 'visitor_identified'"


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def _has_column(table: str, column: str) -> bool:
    return column in {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def _constraint_def(table: str, name: str) -> str | None:
    return op.get_bind().execute(sa.text(
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
        "WHERE conname = :n AND conrelid = CAST(:t AS regclass)"), {"n": name, "t": table}).scalar()


def _ts():
    return [sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                      nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                      nullable=False)]


def _set_event_types(types: str) -> None:
    op.drop_constraint("ck_demo_events_type", "demo_events", type_="check")
    op.create_check_constraint("ck_demo_events_type", "demo_events", f"type IN ({types})")


def _repair_early_0038() -> None:
    """Databases that ran an early draft of 0038 (during development, before demo calls
    were stored) lack demo_sessions.conversation_id and ix_demo_sessions_created_at."""
    if not _has_column("demo_sessions", "conversation_id"):
        op.add_column("demo_sessions", sa.Column("conversation_id", UUID(as_uuid=True),
                                                 sa.ForeignKey("conversations.id", ondelete="SET NULL"),
                                                 nullable=True))
    op.execute("CREATE INDEX IF NOT EXISTS ix_demo_sessions_created_at ON demo_sessions (created_at)")


def upgrade() -> None:
    _repair_early_0038()
    if not _has_table("demo_link_recipients"):
        op.create_table(
            "demo_link_recipients",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("demo_link_id", UUID(as_uuid=True),
                      sa.ForeignKey("demo_links.id", ondelete="CASCADE"), nullable=False),
            sa.Column("contact_id", UUID(as_uuid=True),
                      sa.ForeignKey("demo_contacts.id", ondelete="CASCADE"), nullable=False),
            sa.Column("ref_code", sa.String(16), nullable=False),
            sa.Column("last_exported_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
            *_ts(),
            sa.UniqueConstraint("ref_code", name="uq_demo_link_recipients_ref"),
            sa.UniqueConstraint("demo_link_id", "contact_id", name="uq_demo_link_recipients_contact"),
            sa.CheckConstraint("ref_code ~ '^[A-Za-z0-9_-]{6,16}$'", name="ck_demo_link_recipients_ref"),
        )
        op.create_index("ix_demo_link_recipients_contact_id", "demo_link_recipients", ["contact_id"])

    if not _has_table("demo_visitors"):
        op.create_table(
            "demo_visitors",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("demo_link_id", UUID(as_uuid=True),
                      sa.ForeignKey("demo_links.id", ondelete="CASCADE"), nullable=False),
            sa.Column("visitor_key", UUID(as_uuid=True), nullable=False),
            sa.Column("recipient_id", UUID(as_uuid=True),
                      sa.ForeignKey("demo_link_recipients.id", ondelete="SET NULL"), nullable=True),
            sa.Column("attribution", sa.String(20), nullable=False, server_default="unknown"),
            sa.Column("identified_name", sa.String(120), nullable=True),
            sa.Column("identified_email", sa.String(254), nullable=True),
            sa.Column("identified_role", sa.String(120), nullable=True),
            sa.Column("identified_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("first_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                      nullable=False),
            sa.Column("last_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                      nullable=False),
            sa.Column("first_viewed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("landing_ref", sa.String(16), nullable=True),
            sa.Column("source_step", sa.String(10), nullable=True),
            sa.Column("utm", JSONB(), nullable=False, server_default="{}"),
            sa.Column("ip_hash", sa.String(64), nullable=True),
            sa.Column("user_agent", sa.String(400), nullable=True),
            *_ts(),
            sa.UniqueConstraint("demo_link_id", "visitor_key", name="uq_demo_visitors_link_key"),
            sa.CheckConstraint("attribution IN ('recipient', 'shared', 'self_identified', 'unknown')",
                               name="ck_demo_visitors_attribution"),
        )
        op.create_index("ix_demo_visitors_recipient_id", "demo_visitors", ["recipient_id"])

    for table in ("demo_sessions", "demo_events"):
        if not _has_column(table, "demo_visitor_id"):
            op.add_column(table, sa.Column("demo_visitor_id", UUID(as_uuid=True),
                                           sa.ForeignKey("demo_visitors.id", ondelete="SET NULL"),
                                           nullable=True))
        if not _has_column(table, "recipient_id"):
            op.add_column(table, sa.Column("recipient_id", UUID(as_uuid=True),
                                           sa.ForeignKey("demo_link_recipients.id", ondelete="SET NULL"),
                                           nullable=True))

    if "visitor_identified" not in (_constraint_def("demo_events", "ck_demo_events_type") or ""):
        _set_event_types(EVENT_TYPES_NEW)

    if not _has_column("demo_links", "token_ciphertext"):
        op.add_column("demo_links", sa.Column("token_ciphertext", sa.Text(), nullable=True))

    if _has_column("demo_links", "contact_id"):
        # Each per-person link's person becomes its recipient, so old calls and events keep
        # their attribution. md5 hex is within the ref_code alphabet.
        op.execute("INSERT INTO demo_link_recipients (id, demo_link_id, contact_id, ref_code) "
                   "SELECT gen_random_uuid(), id, contact_id, "
                   "substr(md5(random()::text || id::text), 1, 12) FROM demo_links "
                   "ON CONFLICT DO NOTHING")
        for table in ("demo_sessions", "demo_events"):
            op.execute(f"UPDATE {table} t SET recipient_id = r.id FROM demo_link_recipients r "
                       "WHERE r.demo_link_id = t.demo_link_id AND t.recipient_id IS NULL")
        # Their tokens were never stored, so they can't be exported again: revoke them.
        op.execute("UPDATE demo_links SET status = 'revoked', revoked_at = now(), "
                   "revoked_reason = 'replaced by per-company links (migration 0039)' "
                   "WHERE status = 'active'")
        op.execute("DROP INDEX IF EXISTS uq_demo_links_one_active")
        op.execute("DROP INDEX IF EXISTS ix_demo_links_contact_id")
        op.drop_column("demo_links", "contact_id")

    # One active link per company record; making a new one revokes the old.
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_demo_links_one_active_per_record "
               "ON demo_links (prospect_id) WHERE status = 'active'")


def downgrade() -> None:
    # Back to one link per person: each company link goes to its first recipient (links
    # nobody was sent are dropped), and only the newest active link per person stays active.
    if not _has_column("demo_links", "contact_id"):
        op.add_column("demo_links", sa.Column("contact_id", UUID(as_uuid=True),
                                              sa.ForeignKey("demo_contacts.id", ondelete="CASCADE"),
                                              nullable=True))
        op.execute("UPDATE demo_links l SET contact_id = (SELECT r.contact_id FROM demo_link_recipients r "
                   "WHERE r.demo_link_id = l.id ORDER BY r.created_at, r.id LIMIT 1)")
        op.execute("DELETE FROM demo_links WHERE contact_id IS NULL")
        op.alter_column("demo_links", "contact_id", nullable=False)
    op.execute("DROP INDEX IF EXISTS uq_demo_links_one_active_per_record")
    op.execute("UPDATE demo_links SET status = 'revoked', revoked_at = now(), "
               "revoked_reason = 'downgrade of migration 0039' WHERE status = 'active' AND id NOT IN ("
               "SELECT DISTINCT ON (contact_id) id FROM demo_links WHERE status = 'active' "
               "ORDER BY contact_id, created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_demo_links_contact_id ON demo_links (contact_id)")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_demo_links_one_active ON demo_links (contact_id) "
               "WHERE status = 'active'")
    if _has_column("demo_links", "token_ciphertext"):
        op.drop_column("demo_links", "token_ciphertext")

    if "visitor_identified" in (_constraint_def("demo_events", "ck_demo_events_type") or ""):
        op.execute("DELETE FROM demo_events WHERE type = 'visitor_identified'")
        _set_event_types(EVENT_TYPES_OLD)
    for table in ("demo_sessions", "demo_events"):
        for column in ("recipient_id", "demo_visitor_id"):
            if _has_column(table, column):
                op.drop_column(table, column)
    for table in ("demo_visitors", "demo_link_recipients"):
        if _has_table(table):
            op.drop_table(table)
