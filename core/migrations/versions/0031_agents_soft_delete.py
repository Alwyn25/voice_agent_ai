"""Adds `agents.is_deleted` so DELETE /agents/{id} can soft-delete (matching
customers.py's is_deleted convention) instead of a hard row delete — agents are
referenced by calls, tickets, agent_metrics_daily etc. via a plain FK with no
ondelete rule, so removing the row outright would either violate those
constraints or silently orphan call history. A soft-deleted agent is excluded
from the roster/summary but its history stays intact.

Additive and idempotent (guarded), so a re-run is safe.

Revision ID: 0031
Revises: 0030
"""
import sqlalchemy as sa
from alembic import op

revision = "0031"
down_revision = "0030"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    return {c["name"] for c in insp.get_columns(table)}


def upgrade() -> None:
    if "is_deleted" not in _columns("agents"):
        op.add_column("agents", sa.Column("is_deleted", sa.Boolean(),
                                          nullable=False, server_default=sa.false()))


def downgrade() -> None:
    if "is_deleted" in _columns("agents"):
        op.drop_column("agents", "is_deleted")
