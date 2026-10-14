"""R5.02 — make sure knowledge_chunks.embedding has exactly one valid HNSW index.

Migration 0004 already creates ``ix_chunks_embedding_hnsw`` (HNSW, vector_cosine_ops, default
m=16 / ef_construction=64), so on an environment that ran 0004 cleanly this is a NO-OP. It
exists to REPAIR environments that drifted from that:

  * an IVFFlat index created by hand (the state the R5.02 task sheet assumed),
  * an INVALID index left by a failed/cancelled ``CREATE INDEX CONCURRENTLY`` — ignored by the
    planner but still maintained on every INSERT,
  * an HNSW index built with other parameters or another opclass (which <=> would not use),
  * duplicate vector indexes (each one doubles ingestion write cost for no read benefit).

Zero downtime: the replacement is built with ``CREATE INDEX CONCURRENTLY`` (reads and writes
keep flowing) under a temporary name BEFORE anything is dropped, so vector search always has an
index; old indexes are then dropped CONCURRENTLY and the new one renamed into place. CONCURRENTLY
cannot run inside a transaction, hence Alembic's ``autocommit_block``.

The decision is made by the pure, unit-tested ``plan_index_migration``
(app/engine/rag/vector_index.py). What it decided is logged either way.

Operator controls (env):
  VECTOR_INDEX_MIGRATION=plan_only   log the plan + SQL but do not execute it (e.g. to run the
                                     build by hand in a maintenance window on a very large
                                     table; ``python scripts/vector_index_report.py --plan``
                                     prints the same SQL later).
  VECTOR_INDEX_BUILD_MEM=2GB         maintenance_work_mem for the build session. An HNSW build
                                     whose graph does not fit is several times slower.

Re-running is safe: every step is guarded (IF EXISTS / decided from live catalog state).

Revision ID: 0015
Revises: 0014
"""
import logging
import os
import re

from alembic import op
from sqlalchemy import text

from app.engine.rag.vector_index import (
    COLUMN, INSPECT_INDEXES_SQL, MIN_VERSION_HNSW, PGVECTOR_VERSION_SQL, TABLE,
    IndexInfo, fmt_version, parse_version, plan_index_migration,
)

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None

log = logging.getLogger("alembic.runtime.migration")


def upgrade() -> None:
    bind = op.get_bind()
    if not bind.execute(text("SELECT to_regclass(:t)"), {"t": TABLE}).scalar():
        log.info("R5.02: %s does not exist yet; nothing to reconcile", TABLE)
        return

    rows = bind.execute(INSPECT_INDEXES_SQL, {"table": TABLE, "column": COLUMN}).mappings().all()
    existing = [IndexInfo.from_row(r) for r in rows]
    plan = plan_index_migration(existing)

    for ix in existing:
        log.info("R5.02: found %s method=%s valid=%s opclass=%s options=%s size=%dkB",
                 ix.name, ix.method, ix.valid, ix.opclass, list(ix.options) or "defaults",
                 ix.size_bytes // 1024)
    log.info("R5.02: plan=%s — %s", plan.action, plan.reason)
    if plan.action == "noop":
        return

    for stmt in plan.statements:
        log.info("R5.02:   %s", stmt)

    if os.environ.get("VECTOR_INDEX_MIGRATION", "auto").strip().lower() == "plan_only":
        log.warning("R5.02: VECTOR_INDEX_MIGRATION=plan_only — NOT executing the plan above. "
                    "Run it by hand (scripts/vector_index_report.py --plan prints it).")
        return

    if plan.action == "build":
        version = parse_version(bind.execute(PGVECTOR_VERSION_SQL).scalar())
        if version is None or version < MIN_VERSION_HNSW:
            raise RuntimeError(
                f"R5.02: pgvector {fmt_version(version)} cannot build HNSW indexes "
                f"(needs >= {fmt_version(MIN_VERSION_HNSW)}). Upgrade the extension "
                f"(ALTER EXTENSION vector UPDATE) and re-run the migration.")

    build_mem = os.environ.get("VECTOR_INDEX_BUILD_MEM", "").strip()
    # CONCURRENTLY is illegal inside a transaction block: commit what Alembic has open and run
    # each statement in autocommit. A failure mid-way leaves a state the NEXT run's plan repairs
    # (e.g. an INVALID build index is dropped and rebuilt), so the migration is re-runnable.
    with op.get_context().autocommit_block():
        if build_mem:
            if not re.fullmatch(r"\d+\s*(kB|MB|GB)", build_mem):
                raise ValueError(f"VECTOR_INDEX_BUILD_MEM must look like '2GB', got {build_mem!r}")
            op.execute(f"SET maintenance_work_mem = '{build_mem}'")
        for stmt in plan.statements:
            log.info("R5.02: executing: %s", stmt)
            op.execute(stmt)
        if build_mem:
            op.execute("RESET maintenance_work_mem")
    log.info("R5.02: done — %s now serves vector search", plan.keep)


def downgrade() -> None:
    # Intentionally a no-op. 0015 only REPAIRS the index 0004 owns; downgrading must not take
    # vector search's index away (0004's own downgrade drops it), and re-creating an IVFFlat
    # index nobody should use would be a regression, not a rollback.
    pass
