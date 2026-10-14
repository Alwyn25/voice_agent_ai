"""Vocal Backend Tasksheet P0 (Foundation): plan/seats_purchased on tenants,
team_invites, phone_numbers.

organizations/users/api_keys/activity_events are intentionally NOT new tables here —
they already exist as tenants/users/api_keys/audit_events respectively (see
app/db/models/platform.py and app/db/models/audit.py docstrings/comments for the
mapping). Only the genuinely-missing pieces are added: invite flow and a phone-number
inventory.

Additive and idempotent (guarded), so a re-run is safe.

Revision ID: 0015b
Revises: 0015

Renumbered from a plain "0015"/down_revision="0014" onto "0015b" — this migration and
0015_hnsw_index_reconcile.py were authored independently on separate branches and both
originally claimed revision "0015" off the same parent (0014), which Alembic cannot
support (two files, one revision id). Resequenced to run AFTER the HNSW reconcile
instead of parallel to it; nothing in either migration's own body changed.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0015b"
down_revision = "0015"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def _columns(table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    if "plan" not in _columns("tenants"):
        op.add_column("tenants", sa.Column("plan", sa.String(length=50), nullable=False,
                                           server_default="starter"))
    if "seats_purchased" not in _columns("tenants"):
        op.add_column("tenants", sa.Column("seats_purchased", sa.Integer(), nullable=False,
                                           server_default="1"))

    if not _has_table("team_invites"):
        op.create_table(
            "team_invites",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("email", sa.String(length=255), nullable=False),
            sa.Column("role", sa.String(length=50), nullable=False),
            sa.Column("invited_by", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
            sa.Column("token_hash", sa.String(length=255), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True),
                      server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True),
                      server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("tenant_id", "email", "accepted_at", name="uq_team_invite_pending"),
            sa.UniqueConstraint("token_hash", name="uq_team_invite_token"),
        )

    if not _has_table("phone_numbers"):
        op.create_table(
            "phone_numbers",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("e164", sa.String(length=20), nullable=False),
            sa.Column("provider", sa.String(length=50), nullable=False, server_default="twilio"),
            sa.Column("capability", sa.String(length=20), nullable=False, server_default="both"),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
            sa.Column("assigned_to", UUID(as_uuid=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True),
                      server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True),
                      server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("e164", name="uq_phone_number_e164"),
        )


def downgrade() -> None:
    if _has_table("phone_numbers"):
        op.drop_table("phone_numbers")
    if _has_table("team_invites"):
        op.drop_table("team_invites")
    if "seats_purchased" in _columns("tenants"):
        op.drop_column("tenants", "seats_purchased")
    if "plan" in _columns("tenants"):
        op.drop_column("tenants", "plan")
