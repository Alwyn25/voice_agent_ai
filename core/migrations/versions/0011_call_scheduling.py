"""Call scheduling + attempt count — what the dialer worker needs to pace and retry.

Two columns on ``calls``:

  * ``scheduled_for`` (nullable) — the earliest time the dialer may originate this call. NULL
    means "dial as soon as capacity allows" (the normal case for a first attempt). A retry
    scheduled after a busy/no-answer sets this to now + backoff, and the dialer only picks up
    queued calls whose scheduled_for is NULL or already due. Without it a retry Call would be
    dialled again on the very next tick, defeating the back-off the calling policy assumes.
  * ``attempt`` — 1 for a first dial, incremented down the retry chain, so the max-attempts
    ceiling is a single column read rather than a recursive walk of retry_of_call_id.

Both are additive and nullable/defaulted, so existing rows and a tenant that never dials are
unaffected. See app/db/models/conversations.py (Call) and app/services/telephony/dialer.py.

Revision ID: 0011
Revises: 0010
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    cols = _columns("calls")
    if "scheduled_for" not in cols:
        op.add_column("calls", sa.Column("scheduled_for", sa.DateTime(timezone=True),
                                         nullable=True))
    if "attempt" not in cols:
        op.add_column("calls", sa.Column("attempt", sa.Integer(), nullable=False,
                                         server_default="1"))
    # Why we are calling — the dialer re-runs the purpose-sensitive gate before every
    # originate and every retry, so the purpose must live on the Call, not only in the
    # analytics event place_outbound_call writes.
    if "purpose" not in cols:
        op.add_column("calls", sa.Column("purpose", sa.String(length=30), nullable=False,
                                         server_default="marketing"))
    # Which campaign queued this call, so the dialer can advance the campaign's targets and
    # stats as calls connect and fail without re-deriving the link from analytics rows.
    if "campaign_id" not in cols:
        op.add_column("calls", sa.Column("campaign_id", UUID(as_uuid=True), nullable=True))
    # The dialer's hot query is "queued calls due now, oldest first" scoped to a tenant.
    existing = {i["name"] for i in sa.inspect(op.get_bind()).get_indexes("calls")}
    if "ix_calls_dispatch" not in existing:
        op.create_index("ix_calls_dispatch", "calls",
                        ["tenant_id", "status", "scheduled_for"])


def downgrade() -> None:
    existing = {i["name"] for i in sa.inspect(op.get_bind()).get_indexes("calls")}
    if "ix_calls_dispatch" in existing:
        op.drop_index("ix_calls_dispatch", table_name="calls")
    cols = _columns("calls")
    for col in ("campaign_id", "purpose", "attempt", "scheduled_for"):
        if col in cols:
            op.drop_column("calls", col)
