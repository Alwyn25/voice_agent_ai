"""Vocal Backend Tasksheet P8 (Settings tier — Billing): new `subscriptions` (one row
per tenant; `plan`/`seats_purchased` stay on the pre-existing `tenants` table, not
duplicated here) and `invoices` tables. `integrations`/`team` (the other two P8
areas) needed no schema — `Integration` (migration 0016) and `TeamInvite`/`User`
(migration 0015) already existed with no API surface on top of them; this wave only
adds that.

Additive and idempotent (guarded), so a re-run is safe.

Revision ID: 0027
Revises: 0026
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0027"
down_revision = "0026"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def upgrade() -> None:
    if not _has_table("subscriptions"):
        op.create_table(
            "subscriptions",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
            sa.Column("billing_cycle", sa.String(20), nullable=False, server_default="monthly"),
            sa.Column("status", sa.String(20), nullable=False, server_default="active"),
            sa.Column("stripe_customer_id", sa.String(100), nullable=True),
            sa.Column("stripe_subscription_id", sa.String(100), nullable=True),
            sa.Column("current_period_end", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("tenant_id", name="uq_subscription_tenant"),
        )
        op.create_index("ix_subscriptions_tenant_id", "subscriptions", ["tenant_id"])

    if not _has_table("invoices"):
        op.create_table(
            "invoices",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
            sa.Column("amount_cents", sa.Integer(), nullable=False),
            sa.Column("currency", sa.String(10), nullable=False, server_default="usd"),
            sa.Column("status", sa.String(20), nullable=False, server_default="open"),
            sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
            sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
            sa.Column("pdf_url", sa.Text(), nullable=True),
            sa.Column("stripe_invoice_id", sa.String(100), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
        op.create_index("ix_invoices_tenant_id", "invoices", ["tenant_id"])


def downgrade() -> None:
    if _has_table("invoices"):
        op.drop_table("invoices")
    if _has_table("subscriptions"):
        op.drop_table("subscriptions")
