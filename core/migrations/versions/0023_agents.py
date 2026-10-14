"""Vocal Backend Tasksheet P4 (Agents): agents, agent_knowledge_bases,
agent_tools, agent_metrics_daily.

Additive and idempotent (guarded), so a re-run is safe.

Revision ID: 0023
Revises: 0022
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def upgrade() -> None:
    if not _has_table("agents"):
        op.create_table(
            "agents",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("name", sa.String(length=200), nullable=False),
            sa.Column("purpose_label", sa.String(length=100), nullable=True),
            sa.Column("direction", sa.String(length=10), nullable=False, server_default="inbound"),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="draft"),
            sa.Column("voice", sa.String(length=100), nullable=True),
            sa.Column("template_version_id", UUID(as_uuid=True),
                      sa.ForeignKey("template_versions.id"), nullable=True),
            sa.Column("phone_number_id", UUID(as_uuid=True),
                      sa.ForeignKey("phone_numbers.id"), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )

    if not _has_table("agent_knowledge_bases"):
        op.create_table(
            "agent_knowledge_bases",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("agent_id", UUID(as_uuid=True),
                      sa.ForeignKey("agents.id"), nullable=False, index=True),
            sa.Column("knowledge_base_id", UUID(as_uuid=True),
                      sa.ForeignKey("knowledge_bases.id"), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("agent_id", "knowledge_base_id", name="uq_agent_kb"),
        )

    if not _has_table("agent_tools"):
        op.create_table(
            "agent_tools",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("agent_id", UUID(as_uuid=True),
                      sa.ForeignKey("agents.id"), nullable=False, index=True),
            sa.Column("tool_connection_id", UUID(as_uuid=True),
                      sa.ForeignKey("tool_connections.id"), nullable=False),
            sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("config_override", JSONB(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("agent_id", "tool_connection_id", name="uq_agent_tool"),
        )

    if not _has_table("agent_metrics_daily"):
        op.create_table(
            "agent_metrics_daily",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("agent_id", UUID(as_uuid=True),
                      sa.ForeignKey("agents.id"), nullable=False, index=True),
            sa.Column("date", sa.Date(), nullable=False),
            sa.Column("calls", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("resolved", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("avg_duration_seconds", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("actions_taken", sa.Integer(), nullable=False, server_default="0"),
            sa.UniqueConstraint("agent_id", "date", name="uq_agent_metrics_day"),
        )


def downgrade() -> None:
    if _has_table("agent_metrics_daily"):
        op.drop_table("agent_metrics_daily")
    if _has_table("agent_tools"):
        op.drop_table("agent_tools")
    if _has_table("agent_knowledge_bases"):
        op.drop_table("agent_knowledge_bases")
    if _has_table("agents"):
        op.drop_table("agents")
