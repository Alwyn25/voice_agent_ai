"""Adds `knowledge_bases.reindex_policy` — the Knowledge Base editor modal's
"Re-index" dropdown. UI-only setting for now (see the model's own docstring);
nothing schedules an automatic reindex off of it yet.

Additive and idempotent (guarded), so a re-run is safe.

Revision ID: 0035
Revises: 0034
"""
import sqlalchemy as sa
from alembic import op

revision = "0035"
down_revision = "0034"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    return {c["name"] for c in insp.get_columns(table)}


def upgrade() -> None:
    if "reindex_policy" not in _columns("knowledge_bases"):
        op.add_column("knowledge_bases", sa.Column(
            "reindex_policy", sa.String(30), nullable=False, server_default="on_source_change"))


def downgrade() -> None:
    if "reindex_policy" in _columns("knowledge_bases"):
        op.drop_column("knowledge_bases", "reindex_policy")
