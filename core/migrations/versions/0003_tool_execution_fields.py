"""Add conversation scoping + idempotency key to tool_executions.

Revision ID: 0003
Revises: 0002

Written defensively. The application creates tables from the SQLAlchemy models at
startup, so on a FRESH database tool_executions already has these columns before Alembic
runs — and a plain add_column then fails with DuplicateColumn, leaving the container in a
crash loop. On an existing database the columns are missing and must be added. Both paths
have to work, so each object is checked before it is created.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

TABLE = "tool_executions"

NEW_COLUMNS = [
    ("conversation_id", postgresql.UUID(as_uuid=True), {}),
    ("customer_id", postgresql.UUID(as_uuid=True), {}),
    ("idempotency_key", sa.String(300), {}),
]

NEW_INDEXES = [
    ("ix_tool_exec_idem", ["tenant_id", "idempotency_key"]),
    ("ix_tool_exec_conv", ["tenant_id", "conversation_id"]),
]


def _existing_columns(inspector) -> set[str]:
    return {c["name"] for c in inspector.get_columns(TABLE)}


def _existing_indexes(inspector) -> set[str]:
    return {i["name"] for i in inspector.get_indexes(TABLE)}


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return  # nothing to extend yet

    present = _existing_columns(inspector)
    for name, type_, kwargs in NEW_COLUMNS:
        if name not in present:
            op.add_column(TABLE, sa.Column(name, type_, nullable=True, **kwargs))

    inspector = sa.inspect(op.get_bind())
    indexes = _existing_indexes(inspector)
    columns = _existing_columns(inspector)
    for name, cols in NEW_INDEXES:
        if name in indexes:
            continue
        if not set(cols).issubset(columns):
            continue
        op.create_index(name, TABLE, cols)


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return
    indexes = _existing_indexes(inspector)
    for name, _cols in NEW_INDEXES:
        if name in indexes:
            op.drop_index(name, table_name=TABLE)
    present = _existing_columns(inspector)
    for name, _type, _kwargs in reversed(NEW_COLUMNS):
        if name in present:
            op.drop_column(TABLE, name)
