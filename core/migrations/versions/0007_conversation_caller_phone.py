"""Persist an unknown caller's phone on the conversation.

Adds conversations.caller_phone. Before it, an unregistered caller's number (someone with
no customer record and, on chat/console, no Call row) lived ONLY in the Redis working-memory
session. That made Redis the single source of truth for the one fact registration needs most:
lose it to eviction, a restart, or the 1h TTL and create_customer had nothing to register, so
the agent asked the caller to read out a number they had already dialled from. Writing it to
Postgres at session start makes Redis a cache and the database the durable record.

Revision ID: 0007
Revises: 0006
"""
import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


TABLE = "conversations"
COLUMN = "caller_phone"


def _has_column() -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(col["name"] == COLUMN for col in inspector.get_columns(TABLE))


def upgrade() -> None:
    # Defensive like 0006: caller_phone was first introduced under an earlier
    # migration number, so a database that ran that build already has the column
    # even though Alembic is only at 0006 here. Adding it again raises
    # DuplicateColumn, so skip when it already exists.
    if _has_column():
        return
    op.add_column(TABLE, sa.Column(COLUMN, sa.String(length=20), nullable=True))


def downgrade() -> None:
    if not _has_column():
        return
    op.drop_column(TABLE, COLUMN)
