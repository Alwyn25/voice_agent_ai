"""Vocal Backend Tasksheet P3 (Templates): call_flow_templates, call_flow_steps
(the editable working copy), template_versions (immutable, published snapshots —
a future agent binds here, never to the live template).

Additive and idempotent (guarded), so a re-run is safe. Does NOT create the
LangGraph checkpointer's own tables (checkpoints/checkpoint_writes/
checkpoint_blobs/checkpoint_migrations) — those are owned and versioned by
langgraph-checkpoint-postgres itself via AsyncPostgresSaver.setup(), called
lazily on first use (see app/engine/templates/checkpointer.py); forcing that
library's own schema through Alembic would fight its own migration mechanism.

Revision ID: 0021
Revises: 0020
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def upgrade() -> None:
    if not _has_table("call_flow_templates"):
        op.create_table(
            "call_flow_templates",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("name", sa.String(length=200), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("is_system_template", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )

    if not _has_table("call_flow_steps"):
        op.create_table(
            "call_flow_steps",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("template_id", UUID(as_uuid=True),
                      sa.ForeignKey("call_flow_templates.id"), nullable=False, index=True),
            sa.Column("step_order", sa.Integer(), nullable=False),
            sa.Column("key", sa.String(length=100), nullable=False),
            sa.Column("step_type", sa.String(length=20), nullable=False),
            sa.Column("config", JSONB(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("template_id", "key", name="uq_call_flow_step_template_key"),
        )

    if not _has_table("template_versions"):
        op.create_table(
            "template_versions",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("template_id", UUID(as_uuid=True),
                      sa.ForeignKey("call_flow_templates.id"), nullable=False, index=True),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("snapshot", JSONB(), nullable=False, server_default="[]"),
            sa.Column("compiled_graph_hash", sa.String(length=64), nullable=False),
            sa.Column("published_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("published_by", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
            sa.UniqueConstraint("template_id", "version", name="uq_template_version"),
        )


def downgrade() -> None:
    if _has_table("template_versions"):
        op.drop_table("template_versions")
    if _has_table("call_flow_steps"):
        op.drop_table("call_flow_steps")
    if _has_table("call_flow_templates"):
        op.drop_table("call_flow_templates")
