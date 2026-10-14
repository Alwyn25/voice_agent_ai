"""R5.03 — parent-child ("small-to-big") chunking columns on knowledge_chunks.

Adds:
  * ``chunk_role``  text NOT NULL DEFAULT 'standalone'  — standalone | parent | child
  * ``parent_id``   uuid NULL → knowledge_chunks.id ON DELETE CASCADE
  * ``ck_chunks_role``         chunk_role is one of the three values
  * ``ck_chunks_role_parent``  a row has a parent_id exactly when it is a child
  * ``ix_chunks_parent``       partial btree on parent_id (children only) — serves the parent
                               lookup at retrieval and the cascade on delete

Every existing row becomes 'standalone', which retrieval treats identically in both the "flat"
and "parent_child" modes — so this migration changes no behaviour on its own. Children appear
only when a document is (re-)ingested with CHUNK_PARENT_CHILD_ENABLED=true, or when
scripts/backfill_parent_child.py is run.

Zero downtime on a large table:
  * ADD COLUMN with a constant default is metadata-only on PostgreSQL >= 11 (no rewrite).
  * Constraints are added NOT VALID, then VALIDATEd — validation takes only a SHARE UPDATE
    EXCLUSIVE lock, so reads and writes keep flowing.
  * The index is built CONCURRENTLY (hence the autocommit block).
Every step is guarded, so re-running after a partial failure is safe.

Revision ID: 0036
Revises: 0035
"""
import logging

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0036"
down_revision = "0035"
branch_labels = None
depends_on = None

log = logging.getLogger("alembic.runtime.migration")

TABLE = "knowledge_chunks"


def _has_column(column: str) -> bool:
    return column in {c["name"] for c in sa.inspect(op.get_bind()).get_columns(TABLE)}


def _has_constraint(name: str) -> bool:
    return bool(op.get_bind().execute(sa.text(
        "SELECT 1 FROM pg_constraint WHERE conname = :n"), {"n": name}).scalar())


def upgrade() -> None:
    bind = op.get_bind()
    if not bind.execute(sa.text("SELECT to_regclass(:t)"), {"t": TABLE}).scalar():
        log.info("R5.03: %s does not exist yet; nothing to do", TABLE)
        return

    if not _has_column("chunk_role"):
        op.add_column(TABLE, sa.Column("chunk_role", sa.String(12), nullable=False,
                                       server_default="standalone"))
    if not _has_column("parent_id"):
        op.add_column(TABLE, sa.Column("parent_id", UUID(as_uuid=True), nullable=True))

    if not _has_constraint("fk_chunks_parent"):
        op.execute(f"ALTER TABLE {TABLE} ADD CONSTRAINT fk_chunks_parent FOREIGN KEY (parent_id) "
                   f"REFERENCES {TABLE}(id) ON DELETE CASCADE NOT VALID")
    if not _has_constraint("ck_chunks_role"):
        op.execute(f"ALTER TABLE {TABLE} ADD CONSTRAINT ck_chunks_role "
                   "CHECK (chunk_role IN ('standalone', 'parent', 'child')) NOT VALID")
    if not _has_constraint("ck_chunks_role_parent"):
        op.execute(f"ALTER TABLE {TABLE} ADD CONSTRAINT ck_chunks_role_parent "
                   "CHECK ((chunk_role = 'child') = (parent_id IS NOT NULL)) NOT VALID")
    for name in ("fk_chunks_parent", "ck_chunks_role", "ck_chunks_role_parent"):
        op.execute(f"ALTER TABLE {TABLE} VALIDATE CONSTRAINT {name}")

    with op.get_context().autocommit_block():
        op.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_chunks_parent ON {TABLE} "
                   "(parent_id) WHERE parent_id IS NOT NULL")
    log.info("R5.03: knowledge_chunks has chunk_role/parent_id; existing rows are 'standalone'")


def downgrade() -> None:
    # Children are meaningless without the columns that link them, and would otherwise be
    # served as ordinary chunks (duplicating their parents' text) after the downgrade.
    op.execute(f"DELETE FROM {TABLE} WHERE parent_id IS NOT NULL")
    op.execute("DROP INDEX IF EXISTS ix_chunks_parent")
    for name in ("ck_chunks_role_parent", "ck_chunks_role", "fk_chunks_parent"):
        op.execute(f"ALTER TABLE {TABLE} DROP CONSTRAINT IF EXISTS {name}")
    if _has_column("parent_id"):
        op.drop_column(TABLE, "parent_id")
    if _has_column("chunk_role"):
        op.drop_column(TABLE, "chunk_role")
