"""Vocal Backend Tasksheet P5 (Campaigns + Contacts): additive columns on the
pre-existing campaigns/campaign_targets tables — agent_id, concurrency_limit,
calling_rules on campaigns; next_retry_at on campaign_targets. See
app/db/models/support.py's Campaign docstring for why calling_rules doesn't
duplicate the tenant-wide rule engine's compliance gate.

Additive and idempotent (guarded), so a re-run is safe.

Revision ID: 0024
Revises: 0023
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    cols = _columns("campaigns")
    if "agent_id" not in cols:
        op.add_column("campaigns", sa.Column("agent_id", UUID(as_uuid=True),
                                             sa.ForeignKey("agents.id"), nullable=True))
    if "concurrency_limit" not in cols:
        op.add_column("campaigns", sa.Column("concurrency_limit", sa.Integer(), nullable=True))
    if "calling_rules" not in cols:
        op.add_column("campaigns", sa.Column("calling_rules", JSONB(), nullable=False, server_default="{}"))

    target_cols = _columns("campaign_targets")
    if "next_retry_at" not in target_cols:
        op.add_column("campaign_targets", sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    target_cols = _columns("campaign_targets")
    if "next_retry_at" in target_cols:
        op.drop_column("campaign_targets", "next_retry_at")

    cols = _columns("campaigns")
    if "calling_rules" in cols:
        op.drop_column("campaigns", "calling_rules")
    if "concurrency_limit" in cols:
        op.drop_column("campaigns", "concurrency_limit")
    if "agent_id" in cols:
        op.drop_column("campaigns", "agent_id")
