"""R5.02 — the pgvector HNSW index on knowledge_chunks.embedding, and how queries use it.

Two separate concerns live here, both about the ONE vector index retrieval depends on:

1. QUERY-TIME SCAN TUNING (hot path, every retrieval).

   An HNSW index scan does not find "the N nearest rows that match the WHERE clause". It walks
   the graph, collects the ``hnsw.ef_search`` (default 40) nearest vectors across the WHOLE
   table, and only THEN do the tenant / project / status / effective-date filters run. Every
   candidate belonging to another tenant is thrown away after the fact. On a multi-tenant table
   a small tenant can therefore get 3 chunks back, or none, for a question its documents answer
   — and nothing errors: retrieval just looks like "the knowledge base doesn't know".
   (Measured on the dev DB with the index forced on: LIMIT 32 returned 19 of 35 rows even with
   NO filter at all.) Today the table is small enough that Postgres seq-scans and the problem
   is hidden; it appears exactly when the corpus grows and the index starts being used.

   pgvector 0.8 fixes this with ITERATIVE index scans: when the filters discard candidates the
   scan resumes the graph walk and keeps producing rows until the LIMIT is satisfied (bounded
   by ``hnsw.max_scan_tuples`` so a pathological filter cannot scan forever). We enable that per
   query with ``set_config(..., is_local => true)`` — transaction-scoped, so a value can never
   leak onto a pooled connection another request reuses.

2. INDEX SHAPE (migration 0015 and the report script).

   The pure ``plan_index_migration`` decides, from what is ACTUALLY in the database, whether
   the index is already correct (no-op), or must be (re)built — an IVFFlat index, a missing
   index, an INVALID leftover of a failed ``CREATE INDEX CONCURRENTLY``, or an HNSW built with
   other parameters. It is pure so every one of those states is unit-tested without a DB.

Debugging: every retrieval records a ``VectorScanConfig.describe()`` string (e.g.
``hnsw ef_search=100 iterative=relaxed_order max_tuples=20000 pgvector=0.8.6``) in its result
and in the ``rag.vector`` log line, so "which scan settings served this turn" is never a guess.
Kill switch: ``RETRIEVAL_HNSW_TUNING_ENABLED=false`` restores the exact pre-R5.02 query path.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from sqlalchemy import text as sql_text

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------------
# Index spec
# ---------------------------------------------------------------------------------------------

TABLE = "knowledge_chunks"
COLUMN = "embedding"
INDEX_NAME = "ix_chunks_embedding_hnsw"          # the name migration 0004 created
BUILD_NAME = "ix_chunks_embedding_hnsw_build"    # temporary name while a rebuild is in flight
OPCLASS = "vector_cosine_ops"                     # must match the <=> operator in VECTOR_SQL

# m / ef_construction: pgvector's defaults, and the values R5.02 specifies. Written EXPLICITLY
# into the index so a future pgvector default change cannot silently alter a rebuild.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64

MIN_VERSION_HNSW = (0, 5, 0)       # HNSW index access method
MIN_VERSION_ITERATIVE = (0, 8, 0)  # hnsw.iterative_scan / hnsw.max_scan_tuples

# pgvector's own bounds, validated here so set_config can never be handed a value the server
# rejects — an error there would abort the caller's transaction on the hot path.
EF_SEARCH_MIN, EF_SEARCH_MAX = 1, 1000
ITERATIVE_MODES = ("off", "relaxed_order", "strict_order")


def parse_version(raw: str | None) -> tuple[int, int, int] | None:
    """'0.8.6' -> (0, 8, 6). Tolerates suffixes ('0.8.0-dev') and missing parts ('0.8')."""
    if not raw:
        return None
    nums = [int(n) for n in re.findall(r"\d+", raw)[:3]]
    if not nums:
        return None
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def fmt_version(v: tuple[int, int, int] | None) -> str:
    return ".".join(map(str, v)) if v else "unknown"


# ---------------------------------------------------------------------------------------------
# 1. Query-time scan tuning
# ---------------------------------------------------------------------------------------------

PGVECTOR_VERSION_SQL = sql_text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")

# One round trip; is_local => true makes every value transaction-scoped.
#
# plan_cache_mode: asyncpg PREPARES every statement, and after 5 executions Postgres may switch
# a prepared statement to a GENERIC plan — one planned without knowing :tenant_id. The right
# plan depends heavily on that value: a tenant owning 1% of the table is served fastest (and
# exactly) by its btree tenant index, a tenant owning 50% by the HNSW index. A generic plan picks
# one for everybody; measured, it sent a 20-row tenant through a full iterative HNSW walk
# (~7 ms instead of ~1 ms). force_custom_plan re-plans with the real values (~0.3 ms).
# Passed as :pcm so it can be turned off by config ('auto' = Postgres' default behaviour).
_SET_EF_SQL = sql_text("SELECT set_config('hnsw.ef_search', :ef, true), "
                       "set_config('plan_cache_mode', :pcm, true)")
_SET_ITERATIVE_SQL = sql_text("SELECT set_config('hnsw.ef_search', :ef, true), "
                              "set_config('hnsw.iterative_scan', :mode, true), "
                              "set_config('hnsw.max_scan_tuples', :max_tuples, true), "
                              "set_config('plan_cache_mode', :pcm, true)")

# Process-level cache of the server's pgvector version. The extension version only changes on
# an ALTER EXTENSION + restart, so reading it once per process is enough. A failed read is NOT
# cached (retried next retrieval), so one transient error cannot pin a process to "unknown".
_pgvector_version: tuple[int, int, int] | None = None


async def get_pgvector_version(db) -> tuple[int, int, int] | None:
    global _pgvector_version
    if _pgvector_version is None:
        raw = (await db.execute(PGVECTOR_VERSION_SQL)).scalar()
        _pgvector_version = parse_version(raw)
        log.info("rag.vector pgvector version detected: %s (iterative scan %s)",
                 fmt_version(_pgvector_version),
                 "available" if supports_iterative(_pgvector_version) else "UNAVAILABLE")
    return _pgvector_version


def supports_iterative(version: tuple[int, int, int] | None) -> bool:
    return version is not None and version >= MIN_VERSION_ITERATIVE


@dataclass(frozen=True)
class VectorScanConfig:
    """The HNSW scan settings one retrieval actually ran with (what the logs report)."""
    enabled: bool                       # False -> tuning off, server defaults apply
    ef_search: int | None = None
    iterative_scan: str | None = None   # None when the server's pgvector is too old
    max_scan_tuples: int | None = None
    pgvector: tuple[int, int, int] | None = None
    reason: str = ""                    # why a setting was skipped, if one was
    plan_cache_mode: str = "auto"       # 'force_custom_plan' = re-plan with real tenant_id

    @property
    def relaxed(self) -> bool:
        """relaxed_order may return rows slightly out of distance order; the vector SQL
        re-sorts them (see retrieval.VECTOR_SQL_TUNED)."""
        return self.iterative_scan == "relaxed_order"

    def describe(self) -> str:
        if not self.enabled:
            return f"hnsw defaults (tuning off{': ' + self.reason if self.reason else ''})"
        parts = [f"hnsw ef_search={self.ef_search}",
                 f"iterative={self.iterative_scan or 'n/a'}"]
        if self.max_scan_tuples:
            parts.append(f"max_tuples={self.max_scan_tuples}")
        if self.plan_cache_mode != "auto":
            parts.append(f"plan={self.plan_cache_mode}")
        parts.append(f"pgvector={fmt_version(self.pgvector)}")
        if self.reason:
            parts.append(f"note={self.reason}")
        return " ".join(parts)


def plan_scan_config(settings, *, pool: int,
                     version: tuple[int, int, int] | None) -> VectorScanConfig:
    """Pure: the scan settings for a query that wants ``pool`` rows.

    ef_search is raised to at least ``pool``: HNSW can never return more rows than ef_search
    in a non-iterative scan, so a pool of 32 with ef_search 20 would be silently truncated.
    """
    if not getattr(settings, "retrieval_hnsw_tuning_enabled", True):
        return VectorScanConfig(enabled=False, pgvector=version, reason="disabled by config")

    ef = max(int(settings.retrieval_hnsw_ef_search), int(pool))
    ef = min(max(ef, EF_SEARCH_MIN), EF_SEARCH_MAX)

    mode = str(settings.retrieval_hnsw_iterative_scan)
    if mode not in ITERATIVE_MODES:          # config validation should prevent this; belt+braces
        mode = "relaxed_order"
    pcm = ("force_custom_plan" if getattr(settings, "retrieval_vector_force_custom_plan", True)
           else "auto")
    if not supports_iterative(version):
        return VectorScanConfig(
            enabled=True, ef_search=ef, iterative_scan=None, pgvector=version,
            plan_cache_mode=pcm,
            reason=f"pgvector {fmt_version(version)} < {fmt_version(MIN_VERSION_ITERATIVE)}: "
                   f"no iterative scan, filtered queries may underfill")
    return VectorScanConfig(enabled=True, ef_search=ef, iterative_scan=mode,
                            max_scan_tuples=int(settings.retrieval_hnsw_max_scan_tuples),
                            pgvector=version, plan_cache_mode=pcm)


def scan_gucs(cfg: VectorScanConfig) -> dict[str, str]:
    """The GUC name -> value map a config applies (used by the report script to reproduce
    production settings exactly)."""
    if not cfg.enabled:
        return {}
    gucs = {"hnsw.ef_search": str(cfg.ef_search), "plan_cache_mode": cfg.plan_cache_mode}
    if cfg.iterative_scan is not None:
        gucs.update({"hnsw.iterative_scan": cfg.iterative_scan,
                     "hnsw.max_scan_tuples": str(cfg.max_scan_tuples)})
    return gucs


async def apply_scan_config(db, settings, *, pool: int) -> VectorScanConfig:
    """Set the HNSW scan GUCs for the CURRENT transaction and return what was applied.

    Must run in the same transaction as the vector query (SQLAlchemy's session autobegins,
    so it always is). Values are range-checked in plan_scan_config, so the server cannot
    reject them. Note: inside a SAVEPOINT the values survive RELEASE and last until the outer
    transaction ends — harmless, since only retrieval issues HNSW scans and it re-applies its
    own settings every time.
    """
    s = settings
    if not getattr(s, "retrieval_hnsw_tuning_enabled", True):
        return plan_scan_config(s, pool=pool, version=None)
    version = await get_pgvector_version(db)
    cfg = plan_scan_config(s, pool=pool, version=version)
    if cfg.iterative_scan is not None:
        await db.execute(_SET_ITERATIVE_SQL, {"ef": str(cfg.ef_search),
                                              "mode": cfg.iterative_scan,
                                              "max_tuples": str(cfg.max_scan_tuples),
                                              "pcm": cfg.plan_cache_mode})
    else:
        await db.execute(_SET_EF_SQL, {"ef": str(cfg.ef_search), "pcm": cfg.plan_cache_mode})
    return cfg


# ---------------------------------------------------------------------------------------------
# 2. Index shape — inspection + the (pure) migration decision
# ---------------------------------------------------------------------------------------------

# Every index on knowledge_chunks whose key includes the embedding column, with what we need to
# judge it: access method, validity (a failed CONCURRENTLY build leaves an INVALID index that
# the planner ignores but every INSERT still pays to maintain), opclass and build options.
INSPECT_INDEXES_SQL = sql_text("""
    SELECT i.relname                         AS name,
           am.amname                         AS method,
           ix.indisvalid                     AS valid,
           ix.indisready                     AS ready,
           i.reloptions                      AS options,
           opc.opcname                       AS opclass,
           pg_relation_size(i.oid)           AS size_bytes,
           pg_get_indexdef(i.oid)            AS definition
    FROM pg_index ix
    JOIN pg_class i   ON i.oid = ix.indexrelid
    JOIN pg_class t   ON t.oid = ix.indrelid
    JOIN pg_namespace n ON n.oid = t.relnamespace
    JOIN pg_am am     ON am.oid = i.relam
    JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
    LEFT JOIN pg_opclass opc ON opc.oid = ix.indclass[0]
    WHERE t.relname = :table
      AND n.nspname = current_schema()
      AND a.attname = :column
    ORDER BY i.relname
""")


@dataclass(frozen=True)
class IndexInfo:
    name: str
    method: str                      # 'hnsw' | 'ivfflat' | 'btree' | ...
    valid: bool
    opclass: str | None = None
    options: tuple[str, ...] = ()    # e.g. ('m=16', 'ef_construction=64'); () = defaults
    size_bytes: int = 0
    definition: str = ""

    @classmethod
    def from_row(cls, row) -> "IndexInfo":
        return cls(name=row["name"], method=row["method"],
                   valid=bool(row["valid"]) and bool(row.get("ready", True)),
                   opclass=row.get("opclass"), options=tuple(row.get("options") or ()),
                   size_bytes=int(row.get("size_bytes") or 0),
                   definition=row.get("definition") or "")

    def option(self, key: str, default: int) -> int:
        for opt in self.options:
            k, _, v = opt.partition("=")
            if k.strip() == key:
                try:
                    return int(v)
                except ValueError:
                    return -1
        return default               # unset reloption = pgvector's default

    def is_target(self) -> bool:
        """A valid HNSW cosine index built with the R5.02 parameters."""
        return (self.method == "hnsw" and self.valid and self.opclass == OPCLASS
                and self.option("m", 16) == HNSW_M
                and self.option("ef_construction", 64) == HNSW_EF_CONSTRUCTION)


@dataclass
class IndexPlan:
    """What migration 0015 will do, in order. ``statements`` are executed OUTSIDE a
    transaction (CONCURRENTLY cannot run inside one)."""
    action: str                                   # 'noop' | 'cleanup' | 'build'
    reason: str
    statements: list[str] = field(default_factory=list)
    keep: str | None = None                       # the index that serves queries afterwards


def create_index_sql(name: str) -> str:
    return (f"CREATE INDEX CONCURRENTLY {name} ON {TABLE} "
            f"USING hnsw ({COLUMN} {OPCLASS}) "
            f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})")


def drop_index_sql(name: str) -> str:
    return f"DROP INDEX CONCURRENTLY IF EXISTS {name}"


def plan_index_migration(existing: list[IndexInfo]) -> IndexPlan:
    """Pure decision: from the embedding indexes that exist, what brings us to exactly ONE
    valid HNSW cosine index named INDEX_NAME with the R5.02 parameters.

    Ordering is chosen for zero downtime: the new index is built (CONCURRENTLY, so reads and
    writes continue) BEFORE any old one is dropped, so vector search always has an index to
    use; the rename at the end is a catalog-only change.
    """
    vector_ix = [ix for ix in existing if ix.method in ("hnsw", "ivfflat")]
    target = next((ix for ix in vector_ix if ix.name == INDEX_NAME and ix.is_target()), None)
    if target is None:
        # A correct index under another name (e.g. a manual rebuild that never got renamed).
        target = next((ix for ix in vector_ix if ix.is_target()), None)

    others = [ix for ix in vector_ix if target is None or ix.name != target.name]

    if target is not None:
        stmts = [drop_index_sql(ix.name) for ix in others]
        if target.name != INDEX_NAME:
            stmts.append(f"ALTER INDEX {target.name} RENAME TO {INDEX_NAME}")
        if not stmts:
            return IndexPlan("noop", f"{target.name} is already a valid HNSW "
                                     f"(m={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}, "
                                     f"{OPCLASS}) index", keep=INDEX_NAME)
        dropped = ", ".join(f"{ix.name} ({_why_not(ix)})" for ix in others) or "none"
        return IndexPlan("cleanup", f"target index present as {target.name}; "
                                    f"removing redundant: {dropped}", stmts, keep=INDEX_NAME)

    # No correct index anywhere: build one under a temporary name, then swap.
    stmts = [drop_index_sql(BUILD_NAME),            # leftover of an earlier failed attempt
             create_index_sql(BUILD_NAME)]
    stmts += [drop_index_sql(ix.name) for ix in others if ix.name != BUILD_NAME]
    stmts.append(f"ALTER INDEX {BUILD_NAME} RENAME TO {INDEX_NAME}")
    found = ", ".join(f"{ix.name} ({_why_not(ix)})" for ix in others) or "no vector index"
    return IndexPlan("build", f"found {found}; building HNSW", stmts, keep=INDEX_NAME)


def _why_not(ix: IndexInfo) -> str:
    if ix.method != "hnsw":
        return ix.method
    if not ix.valid:
        return "INVALID"
    if ix.opclass != OPCLASS:
        return f"opclass {ix.opclass}"
    if ix.is_target():
        return "duplicate"
    return f"params m={ix.option('m', 16)} ef_construction={ix.option('ef_construction', 64)}"
