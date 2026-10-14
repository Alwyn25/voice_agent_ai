"""Structured unit inventory — a units table so budget/size/configuration questions are
answered against real rows, not by a model reading prose.

Adds:
  * units — one row per buyable configuration/size in a project's inventory, with the all-in
    published total, the rate, availability, and an explicit price_published flag for
    configurations that exist but have no cost sheet.

tenant_id is required; project_id is NULLABLE, mirroring 0008 — a tenant that seeds no units
is unaffected, and units fall back to tenant-wide when a tenant runs a single project.

Revision ID: 0009
Revises: 0008
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if _has_table("units"):
        return
    op.create_table(
        "units",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=True),
                  sa.ForeignKey("tenants.id"), nullable=False, index=True),
        sa.Column("project_id", UUID(as_uuid=True),
                  sa.ForeignKey("projects.id"), nullable=True),
        sa.Column("bhk_type", sa.String(length=30), nullable=False),
        sa.Column("super_builtup_sqft", sa.Integer(), nullable=True),
        sa.Column("carpet_sqft", sa.Integer(), nullable=True),
        sa.Column("facing", sa.String(length=40), nullable=True),
        sa.Column("tower", sa.String(length=40), nullable=True),
        sa.Column("rate_psf", sa.Integer(), nullable=True),
        sa.Column("flat_cost", sa.Integer(), nullable=True),
        sa.Column("all_in_total", sa.Integer(), nullable=True),
        sa.Column("price_published", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("availability", sa.String(length=20), nullable=False,
                  server_default="available"),
        sa.Column("source_note", sa.String(length=200), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_units_tenant_project", "units", ["tenant_id", "project_id"])


def downgrade() -> None:
    if _has_table("units"):
        op.drop_index("ix_units_tenant_project", table_name="units")
        op.drop_table("units")
