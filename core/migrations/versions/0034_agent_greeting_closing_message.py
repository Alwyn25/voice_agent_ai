"""Adds `agents.greeting_message`/`closing_message` — the opening and closing
lines an agent speaks on a call, edited from the Behaviour tab (replacing the
inline template/phone "Deployment" box there, by user request — that
assignment still exists as a capability, it's just not surfaced there anymore).

Additive and idempotent (guarded), so a re-run is safe.

Revision ID: 0034
Revises: 0033
"""
import sqlalchemy as sa
from alembic import op

revision = "0034"
down_revision = "0033"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    return {c["name"] for c in insp.get_columns(table)}


def upgrade() -> None:
    existing = _columns("agents")
    if "greeting_message" not in existing:
        op.add_column("agents", sa.Column("greeting_message", sa.String(500), nullable=True))
    if "closing_message" not in existing:
        op.add_column("agents", sa.Column("closing_message", sa.String(500), nullable=True))


def downgrade() -> None:
    existing = _columns("agents")
    if "closing_message" in existing:
        op.drop_column("agents", "closing_message")
    if "greeting_message" in existing:
        op.drop_column("agents", "greeting_message")
