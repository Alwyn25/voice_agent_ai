"""Hardening pass on P0/P1: a missing uniqueness guarantee on tool_connections (a
race between two concurrent "Connect" clicks could create two rows for the same
tool), and a proper partial-unique guard on team_invites so two pending invites to
the same email can't both exist (the plain UniqueConstraint added in 0015 doesn't
actually work for this — Postgres treats every NULL accepted_at as distinct, so it
never fires).

Additive and idempotent (guarded), so a re-run is safe. Dedupes any existing
duplicate tool_connections rows (keeping the newest) before adding the constraint,
since ALTER TABLE ADD CONSTRAINT fails if violating rows already exist.

Revision ID: 0017
Revises: 0016
"""
import sqlalchemy as sa
from alembic import op

revision = "0017"
down_revision = "0016b"
branch_labels = None
depends_on = None


def _has_constraint(table: str, name: str) -> bool:
    return any(c["name"] == name for c in sa.inspect(op.get_bind()).get_unique_constraints(table))


def _has_index(name: str) -> bool:
    bind = op.get_bind()
    row = bind.execute(sa.text("SELECT 1 FROM pg_indexes WHERE indexname = :name"), {"name": name}).first()
    return row is not None


def upgrade() -> None:
    # Dedupe tool_connections: keep the newest row per (tenant_id, tool_catalog_id),
    # deleting its guardrail_policies dependents first (FK has no ON DELETE CASCADE).
    op.execute("""
        DELETE FROM guardrail_policies WHERE tool_connection_id IN (
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY tenant_id, tool_catalog_id ORDER BY created_at DESC
                ) AS rn
                FROM tool_connections
            ) ranked WHERE rn > 1
        )
    """)
    op.execute("""
        DELETE FROM tool_connections WHERE id IN (
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY tenant_id, tool_catalog_id ORDER BY created_at DESC
                ) AS rn
                FROM tool_connections
            ) ranked WHERE rn > 1
        )
    """)
    if not _has_constraint("tool_connections", "uq_tool_connection_tenant_tool"):
        op.create_unique_constraint("uq_tool_connection_tenant_tool", "tool_connections",
                                    ["tenant_id", "tool_catalog_id"])

    # A plain UniqueConstraint on (tenant_id, email, accepted_at) never fires for two
    # pending rows, since Postgres never considers two NULLs equal. A partial index
    # is the correct way to say "unique while still pending".
    if not _has_index("uq_team_invite_pending_email"):
        op.execute("""
            CREATE UNIQUE INDEX uq_team_invite_pending_email
            ON team_invites (tenant_id, email)
            WHERE accepted_at IS NULL
        """)


def downgrade() -> None:
    if _has_index("uq_team_invite_pending_email"):
        op.execute("DROP INDEX uq_team_invite_pending_email")
    if _has_constraint("tool_connections", "uq_tool_connection_tenant_tool"):
        op.drop_constraint("uq_tool_connection_tenant_tool", "tool_connections", type_="unique")
