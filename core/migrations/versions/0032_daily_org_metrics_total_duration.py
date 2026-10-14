"""Adds `daily_org_metrics.total_duration_seconds` — the sum (not average) of a
day's completed-call durations, backing the Overview "Total time" tile. Sits
alongside the existing avg_duration_seconds column, which is a mean and can't
answer "how much talk time did the org rack up today".

Additive and idempotent (guarded), so a re-run is safe.

Revision ID: 0032
Revises: 0031
"""
import sqlalchemy as sa
from alembic import op

revision = "0032"
down_revision = "0031"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    return {c["name"] for c in insp.get_columns(table)}


def upgrade() -> None:
    if "total_duration_seconds" not in _columns("daily_org_metrics"):
        op.add_column("daily_org_metrics", sa.Column(
            "total_duration_seconds", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    if "total_duration_seconds" in _columns("daily_org_metrics"):
        op.drop_column("daily_org_metrics", "total_duration_seconds")
