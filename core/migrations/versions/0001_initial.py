"""Initial schema — all 34 tables + pgvector extension.

Revision ID: 0001
Revises:
Create Date: stage 0
"""
from alembic import op

from app.db.base import Base
import app.db.models  # noqa: F401  (registers all tables on Base.metadata)

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
