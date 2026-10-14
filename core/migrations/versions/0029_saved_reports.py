"""Vocal Backend Tasksheet P9 (Analytics + Reports, "deferred — not part of design
pass" per the sheet, the lowest-priority track in the build): a bare `saved_reports`
stub table.

Additive and idempotent (guarded), so a re-run is safe.

Revision ID: 0029
Revises: 0028
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0029"
down_revision = "0028"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def upgrade() -> None:
    if not _has_table("saved_reports"):
        op.create_table(
            "saved_reports",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
            sa.Column("name", sa.String(200), nullable=False),
            sa.Column("config", JSONB(), nullable=False, server_default="{}"),
            sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
        op.create_index("ix_saved_reports_tenant_id", "saved_reports", ["tenant_id"])


def downgrade() -> None:
    if _has_table("saved_reports"):
        op.drop_table("saved_reports")
