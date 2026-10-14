"""Vocal Backend Tasksheet P1 (Tool Library) — a connectable third-party tool
marketplace, entirely separate from the pre-existing tool_definitions table (the
voice engine's built-in, load-bearing tool catalog, left untouched).

Additive and idempotent (guarded), so a re-run is safe.

Revision ID: 0016
Revises: 0015
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0016b"
down_revision = "0015b"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def upgrade() -> None:
    if not _has_table("tool_catalog"):
        op.create_table(
            "tool_catalog",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("key", sa.String(length=100), nullable=False),
            sa.Column("category", sa.String(length=50), nullable=False),
            sa.Column("provider_name", sa.String(length=100), nullable=False),
            sa.Column("actions", JSONB(), nullable=False, server_default="[]"),
            sa.Column("action_schema", JSONB(), nullable=False, server_default="{}"),
            sa.Column("is_custom", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("tenant_id", "key", name="uq_tool_catalog_tenant_key"),
        )

    if not _has_table("integrations"):
        op.create_table(
            "integrations",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("provider_name", sa.String(length=100), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="disconnected"),
            sa.Column("credentials_encrypted", sa.Text(), nullable=True),
            sa.Column("connected_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("tenant_id", "provider_name", name="uq_integration_tenant_provider"),
        )

    if not _has_table("tool_connections"):
        op.create_table(
            "tool_connections",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("tool_catalog_id", UUID(as_uuid=True),
                      sa.ForeignKey("tool_catalog.id"), nullable=False, index=True),
            sa.Column("integration_id", UUID(as_uuid=True), sa.ForeignKey("integrations.id"), nullable=True),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="disconnected"),
            sa.Column("credentials_encrypted", sa.Text(), nullable=True),
            sa.Column("config", JSONB(), nullable=False, server_default="{}"),
            sa.Column("guardrails", JSONB(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )

    if not _has_table("guardrail_policies"):
        op.create_table(
            "guardrail_policies",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("tool_connection_id", UUID(as_uuid=True),
                      sa.ForeignKey("tool_connections.id"), nullable=False, index=True),
            sa.Column("key", sa.String(length=100), nullable=False),
            sa.Column("value", JSONB(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("tool_connection_id", "key", name="uq_guardrail_connection_key"),
        )


def downgrade() -> None:
    if _has_table("guardrail_policies"):
        op.drop_table("guardrail_policies")
    if _has_table("tool_connections"):
        op.drop_table("tool_connections")
    if _has_table("integrations"):
        op.drop_table("integrations")
    if _has_table("tool_catalog"):
        op.drop_table("tool_catalog")
