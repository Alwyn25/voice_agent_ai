"""Vocal Backend Tasksheet P0 (Foundation) — notifications, a user-scoped
read-model over audit_events for the bell-icon badge (see
app/services/activity.py's append_activity_event(notify_user_ids=...)).

Additive and idempotent (guarded), so a re-run is safe.

Revision ID: 0018
Revises: 0017
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def upgrade() -> None:
    if not _has_table("notifications"):
        op.create_table(
            "notifications",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("user_id", UUID(as_uuid=True),
                      sa.ForeignKey("users.id"), nullable=False, index=True),
            sa.Column("event_id", UUID(as_uuid=True),
                      sa.ForeignKey("audit_events.id"), nullable=False),
            sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
        op.create_index("ix_notification_user_unread", "notifications",
                        ["tenant_id", "user_id", "read_at"])


def downgrade() -> None:
    if _has_table("notifications"):
        op.drop_index("ix_notification_user_unread", table_name="notifications")
        op.drop_table("notifications")
