"""Vocal Backend Tasksheet P7 (Overview): new `daily_org_metrics` table — the
org-wide counterpart to the pre-existing `agent_metrics_daily`, both written by the
new Aggregator Worker (app/services/aggregator.py).

Additive and idempotent (guarded), so a re-run is safe.

Revision ID: 0026
Revises: 0025
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0026"
down_revision = "0025"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def upgrade() -> None:
    if not _has_table("daily_org_metrics"):
        op.create_table(
            "daily_org_metrics",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
            sa.Column("date", sa.Date(), nullable=False),
            sa.Column("active_agents", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("calls_placed", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("calls_completed", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("resolved", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("actions_taken", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("avg_duration_seconds", sa.Integer(), nullable=False, server_default="0"),
            sa.UniqueConstraint("tenant_id", "date", name="uq_daily_org_metrics_day"),
        )
        op.create_index("ix_daily_org_metrics_tenant_id", "daily_org_metrics", ["tenant_id"])


def downgrade() -> None:
    if _has_table("daily_org_metrics"):
        op.drop_table("daily_org_metrics")
