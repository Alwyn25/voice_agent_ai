"""Unit tests for R5.02 — HNSW index reconciliation and query-time scan tuning.

What must hold, and is tested below (no live DB — fakes only):
  * migration planner: a correct index is a NO-OP; IVFFlat / INVALID / wrong-parameter / missing
    indexes are rebuilt; a new index is always built BEFORE an old one is dropped (zero downtime);
    leftovers of a failed CONCURRENTLY build are cleaned up.
  * scan config: ef_search is never below the query's pool, is clamped to pgvector's bounds;
    iterative scan is skipped (not errored) on pgvector < 0.8; the kill switch restores defaults.
  * retrieval vector arm: tuned SQL when enabled, the exact pre-R5.02 SQL when disabled; a short
    result is classified as underfilled ONLY when more eligible rows exist; a failing diagnostic
    never fails the search.

Run: cd core && python -m pytest tests/test_vector_index.py -q
"""
import asyncio
import logging
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from app.engine.rag import retrieval, vector_index as vi  # noqa: E402
from app.engine.rag.vector_index import (  # noqa: E402
    BUILD_NAME, INDEX_NAME, IndexInfo, parse_version, plan_index_migration, plan_scan_config,
)


def _settings(**over):
    base = dict(retrieval_hnsw_tuning_enabled=True, retrieval_hnsw_ef_search=100,
                retrieval_hnsw_iterative_scan="relaxed_order",
                retrieval_hnsw_max_scan_tuples=20000, retrieval_vector_underfill_check=True,
                retrieval_vector_explain=False)
    base.update(over)
    return SimpleNamespace(**base)


def _hnsw(name=INDEX_NAME, valid=True, options=(), opclass="vector_cosine_ops"):
    return IndexInfo(name=name, method="hnsw", valid=valid, opclass=opclass, options=options)


def _ivf(name="ix_chunks_embedding_ivfflat"):
    return IndexInfo(name=name, method="ivfflat", valid=True, opclass="vector_cosine_ops",
                     options=("lists=100",))


# --------------------------------------------------------------------------- version parsing

@pytest.mark.parametrize("raw,expected", [
    ("0.8.6", (0, 8, 6)), ("0.8", (0, 8, 0)), ("0.8.0-dev", (0, 8, 0)), ("", None), (None, None),
])
def test_parse_version(raw, expected):
    assert parse_version(raw) == expected


# --------------------------------------------------------------------------- migration planner

def test_default_hnsw_is_noop():
    # 0004 built the index with no reloptions => pgvector defaults m=16, ef_construction=64.
    plan = plan_index_migration([_hnsw()])
    assert plan.action == "noop" and plan.statements == []


def test_explicit_params_hnsw_is_noop():
    plan = plan_index_migration([_hnsw(options=("m=16", "ef_construction=64"))])
    assert plan.action == "noop"


def test_no_index_builds():
    plan = plan_index_migration([])
    assert plan.action == "build"
    assert any("CREATE INDEX CONCURRENTLY" in s and "hnsw" in s and "m = 16" in s
               and "ef_construction = 64" in s for s in plan.statements)
    assert plan.statements[-1] == f"ALTER INDEX {BUILD_NAME} RENAME TO {INDEX_NAME}"


def test_ivfflat_is_replaced_build_before_drop():
    plan = plan_index_migration([_ivf()])
    assert plan.action == "build"
    create = next(i for i, s in enumerate(plan.statements) if s.startswith("CREATE INDEX"))
    drop_old = next(i for i, s in enumerate(plan.statements)
                    if "DROP INDEX" in s and "ivfflat" in s)
    assert create < drop_old, "the old index must keep serving until the new one exists"
    assert all("CONCURRENTLY" in s for s in plan.statements if s.startswith(("CREATE", "DROP")))


def test_ivfflat_alongside_valid_hnsw_is_cleaned_up():
    plan = plan_index_migration([_hnsw(), _ivf()])
    assert plan.action == "cleanup"
    assert plan.statements == ["DROP INDEX CONCURRENTLY IF EXISTS ix_chunks_embedding_ivfflat"]


def test_invalid_hnsw_is_rebuilt():
    plan = plan_index_migration([_hnsw(valid=False)])
    assert plan.action == "build"
    assert f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}" in plan.statements


def test_wrong_params_or_opclass_rebuilt():
    assert plan_index_migration([_hnsw(options=("m=32",))]).action == "build"
    assert plan_index_migration([_hnsw(opclass="vector_l2_ops")]).action == "build"


def test_finished_build_index_is_renamed_into_place():
    # A previous run built the new index but died before the swap.
    plan = plan_index_migration([_hnsw(valid=False), _hnsw(name=BUILD_NAME)])
    assert plan.action == "cleanup"
    assert plan.statements == [f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}",
                               f"ALTER INDEX {BUILD_NAME} RENAME TO {INDEX_NAME}"]


def test_invalid_build_leftover_dropped_when_target_fine():
    plan = plan_index_migration([_hnsw(), _hnsw(name=BUILD_NAME, valid=False)])
    assert plan.action == "cleanup"
    assert plan.statements == [f"DROP INDEX CONCURRENTLY IF EXISTS {BUILD_NAME}"]


def test_btree_indexes_are_ignored():
    btree = IndexInfo(name="ix_other", method="btree", valid=True)
    assert plan_index_migration([_hnsw(), btree]).action == "noop"


# --------------------------------------------------------------------------- scan config

def test_scan_config_raises_ef_to_pool_and_clamps():
    cfg = plan_scan_config(_settings(retrieval_hnsw_ef_search=20), pool=32, version=(0, 8, 6))
    assert cfg.ef_search == 32
    cfg = plan_scan_config(_settings(retrieval_hnsw_ef_search=5000), pool=32, version=(0, 8, 6))
    assert cfg.ef_search == 1000


def test_scan_config_iterative_on_new_pgvector():
    cfg = plan_scan_config(_settings(), pool=32, version=(0, 8, 6))
    assert cfg.iterative_scan == "relaxed_order" and cfg.relaxed
    assert "iterative=relaxed_order" in cfg.describe() and "pgvector=0.8.6" in cfg.describe()


def test_scan_config_old_pgvector_degrades_not_errors():
    cfg = plan_scan_config(_settings(), pool=32, version=(0, 7, 4))
    assert cfg.enabled and cfg.iterative_scan is None and cfg.ef_search == 100
    assert "0.7.4 < 0.8.0" in cfg.describe()


def test_scan_config_kill_switch():
    cfg = plan_scan_config(_settings(retrieval_hnsw_tuning_enabled=False), pool=32,
                           version=(0, 8, 6))
    assert not cfg.enabled and "tuning off" in cfg.describe()


# --------------------------------------------------------------------------- fake DB

class _Result:
    def __init__(self, value=None, rows=()):
        self._value, self._rows = value, list(rows)

    def scalar(self):
        return self._value

    def mappings(self):
        return self

    def scalars(self):
        return self

    def all(self):
        return self._rows


class FakeDB:
    """Records every statement; answers by matching on the SQL text."""

    def __init__(self, *, version="0.8.6", vector_rows=0, eligible=0, eligible_fails=False):
        self.calls: list[tuple[str, dict | None]] = []
        self.version, self.vector_rows = version, vector_rows
        self.eligible, self.eligible_fails = eligible, eligible_fails

    async def execute(self, sql, params=None):
        q = getattr(sql, "text", str(sql))
        self.calls.append((q, params))
        if "pg_extension" in q:
            return _Result(self.version)
        if "set_config" in q:
            return _Result()
        if "eligible" in q:
            if self.eligible_fails:
                raise RuntimeError("boom")
            return _Result(self.eligible)
        if "knowledge_chunks" in q:
            return _Result(rows=[{"id": uuid.uuid4()} for _ in range(self.vector_rows)])
        raise AssertionError(f"unexpected SQL: {q[:80]}")


@pytest.fixture(autouse=True)
def _reset_version_cache():
    vi._pgvector_version = None
    yield
    vi._pgvector_version = None


def _run(coro):
    return asyncio.run(coro)


def test_apply_scan_config_sets_transaction_local_values_once_per_process():
    db = FakeDB()
    cfg = _run(vi.apply_scan_config(db, _settings(), pool=32))
    sets = [(q, p) for q, p in db.calls if "set_config" in q]
    assert len(sets) == 1
    q, p = sets[0]
    assert q.count(", true)") == 4, "every value must be is_local (transaction-scoped)"
    assert p == {"ef": "100", "mode": "relaxed_order", "max_tuples": "20000",
                 "pcm": "force_custom_plan"}
    assert cfg.ef_search == 100
    _run(vi.apply_scan_config(db, _settings(), pool=32))
    assert sum("pg_extension" in q for q, _ in db.calls) == 1, "version is cached per process"


def test_apply_scan_config_old_pgvector_sets_only_ef():
    db = FakeDB(version="0.7.4")
    _run(vi.apply_scan_config(db, _settings(), pool=32))
    sets = [p for q, p in db.calls if "set_config" in q]
    assert sets == [{"ef": "100", "pcm": "force_custom_plan"}]


def test_custom_plan_can_be_turned_off():
    cfg = plan_scan_config(_settings(retrieval_vector_force_custom_plan=False), pool=32,
                           version=(0, 8, 6))
    assert cfg.plan_cache_mode == "auto" and "plan=" not in cfg.describe()
    assert vi.scan_gucs(cfg)["plan_cache_mode"] == "auto"
    on = plan_scan_config(_settings(), pool=32, version=(0, 8, 6))
    assert "plan=force_custom_plan" in on.describe()


def test_apply_scan_config_disabled_touches_nothing():
    db = FakeDB()
    cfg = _run(vi.apply_scan_config(db, _settings(retrieval_hnsw_tuning_enabled=False), pool=32))
    assert db.calls == [] and not cfg.enabled


# --------------------------------------------------------------------------- retrieval vector arm

def _vsearch(db, s, pool=32):
    result = retrieval.RetrievalResult(query="q")
    rows = _run(retrieval._vector_search(db, s, result, {"limit": pool}, pool=pool,
                                         tenant_id=uuid.uuid4()))
    return result, rows


def test_vector_search_uses_tuned_sql_and_records_diagnostics():
    db = FakeDB(vector_rows=32)
    result, rows = _vsearch(db, _settings())
    assert len(rows) == 32
    assert any(q == retrieval.VECTOR_SQL_TUNED.text for q, _ in db.calls)
    assert not any("eligible" in q for q, _ in db.calls), "full result => no diagnostic query"
    d = result.vector_diagnostics()
    assert d["pool"] == 32 and "iterative=relaxed_order" in d["scan"] and not d["underfilled"]


def test_vector_search_kill_switch_uses_original_sql():
    db = FakeDB(vector_rows=32)
    _vsearch(db, _settings(retrieval_hnsw_tuning_enabled=False))
    assert any(q == retrieval.VECTOR_SQL.text for q, _ in db.calls)
    assert not any("set_config" in q for q, _ in db.calls)


def test_short_result_small_tenant_is_not_underfilled():
    db = FakeDB(vector_rows=10, eligible=10)      # tenant genuinely has only 10 chunks
    result, _ = _vsearch(db, _settings())
    assert result.vector_eligible == 10 and not result.vector_underfilled


def test_short_result_with_more_eligible_is_underfilled_and_warns(caplog):
    db = FakeDB(vector_rows=10, eligible=32)      # the index dropped rows that match
    with caplog.at_level(logging.WARNING, logger="app.engine.rag.retrieval"):
        result, _ = _vsearch(db, _settings())
    assert result.vector_underfilled
    assert any("rag.vector.underfilled" in r.message for r in caplog.records)


def test_underfill_diagnostic_failure_never_fails_search():
    db = FakeDB(vector_rows=10, eligible_fails=True)
    result, rows = _vsearch(db, _settings())
    assert len(rows) == 10 and result.vector_eligible is None and not result.vector_underfilled


def test_underfill_check_can_be_disabled():
    db = FakeDB(vector_rows=10, eligible=32)
    _vsearch(db, _settings(retrieval_vector_underfill_check=False))
    assert not any("eligible" in q for q, _ in db.calls)


def test_tuned_sql_keeps_filters_and_reorders_exactly():
    tuned, orig = retrieval.VECTOR_SQL_TUNED.text, retrieval.VECTOR_SQL.text
    for clause in ("c.tenant_id = CAST(:tenant_id AS uuid)", "d.status = 'ready'",
                   "c.embedding IS NOT NULL", "effective_from", "effective_to",
                   "CAST(:doc_type AS text)", "CAST(:project_id AS uuid)"):
        assert clause in tuned and clause in orig, clause
    assert "AS MATERIALIZED" in tuned and "ORDER BY distance, id" in tuned  # R5.03 tie-break
