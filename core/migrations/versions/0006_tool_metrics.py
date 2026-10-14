"""Add tool_metrics table for per-tool-call performance tracking.

Revision ID: 0006
Revises: 0005

Defensive like 0003: on a fresh database this table is already created by 0001's
create_all, which builds from the live app.db.models package — so ToolMetric exists
before Alembic ever runs. On a database that migrated before ToolMetric was added, the
table is missing and this creates it.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

TABLE = "tool_metrics"


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if TABLE in inspector.get_table_names():
        return

    op.create_table(
        TABLE,
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("tool_name", sa.String(100), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("failed_check", sa.String(100), nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("duration_ms", sa.Integer, nullable=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("decision_trace_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_tool_metrics_tenant_id", TABLE, ["tenant_id"])
    op.create_index("ix_tool_metrics_tool_time", TABLE, ["tenant_id", "tool_name", "created_at"])
    op.create_index("ix_tool_metrics_status", TABLE, ["tenant_id", "status"])


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return
    op.drop_table(TABLE)
