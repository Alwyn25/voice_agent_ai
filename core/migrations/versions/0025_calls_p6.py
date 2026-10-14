"""Vocal Backend Tasksheet P6 (Calls Runtime): additive columns on the pre-existing
`calls` table (agent_id, template_version_id, transcript, summary, intent, sentiment,
follow_up_required/follow_up_reason/follow_up_at — campaign_id/status/contact
(customer_id) already existed); two genuinely new tables (call_events, call_captures
— see app/db/models/calls_runtime.py's module docstring for why these could not be
reconciled onto the older engine's decision_traces/tool_executions); and two
additive nullable columns on `tool_executions` (call_id, step_key) so P6's
`tool_call_logs` reconciles onto that pre-existing table instead of duplicating it.

Additive and idempotent (guarded), so a re-run is safe.

Revision ID: 0025
Revises: 0024
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0025"
down_revision = "0024"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def _columns(table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    cols = _columns("calls")
    if "agent_id" not in cols:
        op.add_column("calls", sa.Column("agent_id", UUID(as_uuid=True),
                                         sa.ForeignKey("agents.id"), nullable=True))
    if "template_version_id" not in cols:
        op.add_column("calls", sa.Column("template_version_id", UUID(as_uuid=True),
                                         sa.ForeignKey("template_versions.id"), nullable=True))
    if "transcript" not in cols:
        op.add_column("calls", sa.Column("transcript", JSONB(), nullable=False, server_default="[]"))
    if "summary" not in cols:
        op.add_column("calls", sa.Column("summary", sa.Text(), nullable=True))
    if "intent" not in cols:
        op.add_column("calls", sa.Column("intent", sa.String(100), nullable=True))
    if "sentiment" not in cols:
        op.add_column("calls", sa.Column("sentiment", sa.String(20), nullable=True))
    if "follow_up_required" not in cols:
        op.add_column("calls", sa.Column("follow_up_required", sa.Boolean(),
                                         nullable=False, server_default=sa.false()))
    if "follow_up_reason" not in cols:
        op.add_column("calls", sa.Column("follow_up_reason", sa.Text(), nullable=True))
    if "follow_up_at" not in cols:
        op.add_column("calls", sa.Column("follow_up_at", sa.DateTime(timezone=True), nullable=True))

    tool_exec_cols = _columns("tool_executions")
    if "call_id" not in tool_exec_cols:
        op.add_column("tool_executions", sa.Column("call_id", UUID(as_uuid=True),
                                                    sa.ForeignKey("calls.id"), nullable=True))
        op.create_index("ix_tool_exec_call", "tool_executions", ["tenant_id", "call_id"])
    if "step_key" not in tool_exec_cols:
        op.add_column("tool_executions", sa.Column("step_key", sa.String(100), nullable=True))

    if not _has_table("call_events"):
        op.create_table(
            "call_events",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
            sa.Column("call_id", UUID(as_uuid=True), sa.ForeignKey("calls.id"), nullable=False),
            sa.Column("step_key", sa.String(100), nullable=False),
            sa.Column("step_type", sa.String(20), nullable=False),
            sa.Column("input", JSONB(), nullable=False, server_default="{}"),
            sa.Column("output", JSONB(), nullable=False, server_default="{}"),
            sa.Column("latency_ms", sa.Integer(), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
        op.create_index("ix_call_events_tenant_id", "call_events", ["tenant_id"])
        op.create_index("ix_call_events_call", "call_events", ["tenant_id", "call_id", "created_at"])

    if not _has_table("call_captures"):
        op.create_table(
            "call_captures",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
            sa.Column("call_id", UUID(as_uuid=True), sa.ForeignKey("calls.id"), nullable=False),
            sa.Column("step_key", sa.String(100), nullable=False),
            sa.Column("capture_key", sa.String(100), nullable=False),
            sa.Column("value", JSONB(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
        op.create_index("ix_call_captures_tenant_id", "call_captures", ["tenant_id"])
        op.create_index("ix_call_captures_call", "call_captures", ["tenant_id", "call_id"])


def downgrade() -> None:
    if _has_table("call_captures"):
        op.drop_table("call_captures")
    if _has_table("call_events"):
        op.drop_table("call_events")

    tool_exec_cols = _columns("tool_executions")
    if "step_key" in tool_exec_cols:
        op.drop_column("tool_executions", "step_key")
    if "call_id" in tool_exec_cols:
        op.drop_index("ix_tool_exec_call", table_name="tool_executions")
        op.drop_column("tool_executions", "call_id")

    cols = _columns("calls")
    for col in ("follow_up_at", "follow_up_reason", "follow_up_required", "sentiment",
               "intent", "summary", "transcript", "template_version_id", "agent_id"):
        if col in cols:
            op.drop_column("calls", col)
