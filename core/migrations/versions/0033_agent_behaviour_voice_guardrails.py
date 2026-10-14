"""Adds the Agent Builder "Behaviour"/"Voice" fields to `agents` and a new
`agent_guardrails` table for the "Guardrails" tab — explicitly out of the
original Vocal Backend Tasksheet P4 scope (which only ever specified name/
purpose_label/direction/status/template_version_id/phone_number_id), added by
direct user request to match the Agent Builder reference design. See
app/db/models/agents.py's Agent/AgentGuardrail docstrings for the reasoning.

Additive and idempotent (guarded), so a re-run is safe.

Revision ID: 0033
Revises: 0032
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0033"
down_revision = "0032"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    return {c["name"] for c in insp.get_columns(table)}


def _tables() -> set[str]:
    bind = op.get_bind()
    return set(sa.inspect(bind).get_table_names())


AGENT_COLUMNS = [
    ("description", sa.String(2000), None),
    ("language", sa.String(20), "'en-US'"),
    ("system_instructions", sa.String(4000), None),
    ("personality", postgresql.JSONB, "'{}'::jsonb"),
    ("stt_model", sa.String(100), None),
    ("tts_model", sa.String(100), None),
    ("voice_accent", sa.String(100), None),
    ("speaking_speed", sa.Float, "1.0"),
    ("response_latency_ms", sa.Integer, "300"),
    ("interruption_sensitivity", sa.Integer, "50"),
    ("silence_timeout_seconds", sa.Integer, "6"),
    ("background_noise", sa.String(50), None),
    ("call_quality", sa.String(20), None),
]


def upgrade() -> None:
    existing = _columns("agents")
    for name, coltype, default in AGENT_COLUMNS:
        if name not in existing:
            nullable = default is None
            op.add_column("agents", sa.Column(
                name, coltype, nullable=nullable,
                server_default=sa.text(default) if default else None))

    if "agent_guardrails" not in _tables():
        op.create_table(
            "agent_guardrails",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                     onupdate=sa.func.now(), nullable=False),
            sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
            sa.Column("agent_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("agents.id"), nullable=False),
            sa.Column("order_index", sa.Integer, nullable=False, server_default="0"),
            sa.Column("condition_text", sa.String(300), nullable=False),
            sa.Column("action_text", sa.String(300), nullable=False),
            sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
        )
        op.create_index("ix_agent_guardrails_tenant_id", "agent_guardrails", ["tenant_id"])
        op.create_index("ix_agent_guardrails_agent_id", "agent_guardrails", ["agent_id"])


def downgrade() -> None:
    if "agent_guardrails" in _tables():
        op.drop_table("agent_guardrails")
    existing = _columns("agents")
    for name, _coltype, _default in AGENT_COLUMNS:
        if name in existing:
            op.drop_column("agents", name)
