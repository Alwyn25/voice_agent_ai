"""Fix frozen timestamp defaults: quoted 'now()' string literals were evaluated
once at table creation. Replace with proper now() function defaults.

Revision ID: 0002
Revises: 0001
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

COLUMNS = [
    ("state_transitions", "occurred_at"),
    ("messages", "created_at"),
    ("consent_records", "recorded_at"),
    ("decision_traces", "created_at"),
    ("tool_executions", "created_at"),
    ("audit_events", "created_at"),
    ("analytics_events", "occurred_at"),
]


def upgrade() -> None:
    for table, col in COLUMNS:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN {col} SET DEFAULT now()")


def downgrade() -> None:
    pass  # no need to restore a broken default
