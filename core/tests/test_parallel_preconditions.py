"""Unit tests for P2.05 — parallel DB queries in the tool preconditions step.

The whole point of the change is to run a tool's DB-backed preconditions concurrently instead
of one-after-another, WITHOUT altering which check blocks a call. The orchestration in
executor._run_preconditions is where that guarantee lives, so it is pinned here directly with
fakes — no live DB, no pipecat — the same way test_customer_prefetch pins P2.04.

What must hold, and is tested below:
  * flag OFF  -> the original sequential, short-circuiting loop (byte-for-byte behaviour).
  * flag ON   -> DB checks actually overlap (real concurrency), bounded by the pool cap.
  * either way -> the blocking check is the FIRST not-ok one in DECLARED order.
  * an uncommitted intra-turn tool write forces the safe sequential path.
  * any infra failure in the parallel path degrades to sequential, never raises.

Run: cd core && python -m pytest tests/test_parallel_preconditions.py -q
"""
import asyncio
import sys
import uuid
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

# core/ on the path so `import app...` resolves the same way the app does at runtime.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.tools import contracts, executor
from app.engine.tools.types import CheckResult, ToolContext

# Real check names, so the REAL contracts.is_in_memory_check routes them correctly:
#   slot_available / no_conflicting_appointment / centre_exists / executive_exists -> DB
#   customer_identified / slot_was_offered                                         -> in-memory
_DB_CHECKS = ["slot_available", "no_conflicting_appointment", "centre_exists", "executive_exists"]


class _Settings:
    def __init__(self, enabled, *, maxc=8, trace=False):
        self.parallel_preconditions_enabled = enabled
        self.parallel_preconditions_max_concurrency = maxc
        self.parallel_preconditions_trace = trace


class _Tool:
    def __init__(self, checks):
        self.name = "book_appointment"
        self.preconditions = [{"check": c, "args": {}} for c in checks]


class _Db:
    """Stand-in for the shared AsyncSession — the runner only touches `.info` on it."""
    def __init__(self, info=None):
        self.info = dict(info or {})


class _SideSession:
    """A parallel side session. Its own async context manager; holds nothing."""
    def __init__(self, raises=False):
        self._raises = raises

    async def __aenter__(self):
        if self._raises:
            raise RuntimeError("simulated pool checkout failure")
        return self

    async def __aexit__(self, *a):
        return False


class _Recorder:
    """Fake run_check that records call order and, crucially, how many checks are in flight at
    once — so a test can prove DB checks really overlapped (max_active > 1) or really did
    not (== 1)."""
    def __init__(self, failing=(), delay=0.02):
        self.failing = set(failing)
        self.delay = delay
        self.order = []
        self.active = 0
        self.max_active = 0

    async def run_check(self, name, ctx, args):
        self.order.append(name)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)      # a real DB round-trip; overlap shows here
        finally:
            self.active -= 1
        ok = name not in self.failing
        return CheckResult(ok, name, "" if ok else f"{name} failed", {"n": name})


def _run(checks, *, enabled, failing=(), maxc=8, pending=False, side_raises=False):
    """Drive executor._run_preconditions with everything faked. Returns (result_list, recorder,
    factory_call_count)."""
    rec = _Recorder(failing=failing)
    factory_calls = {"n": 0}

    def fake_factory():                    # get_session_factory() -> factory
        def factory():                     # factory() -> async-CM session
            factory_calls["n"] += 1
            return _SideSession(raises=side_raises)
        return factory

    ctx = ToolContext(db=_Db({"_pending_tool_write": True} if pending else {}),
                      tenant_id=uuid.uuid4(), extra={})
    tool = _Tool(checks)

    with ExitStack() as st:
        st.enter_context(patch.object(executor, "get_settings",
                                      lambda: _Settings(enabled, maxc=maxc)))
        st.enter_context(patch.object(executor, "get_session_factory", fake_factory))
        st.enter_context(patch.object(contracts, "run_check", rec.run_check))
        result = asyncio.run(executor._run_preconditions(ctx, tool, {}))
    return result, rec, factory_calls["n"]


# --- flag OFF: unchanged sequential behaviour -------------------------------------------------

def test_flag_off_runs_sequentially_in_order():
    result, rec, factory_calls = _run(_DB_CHECKS, enabled=False)
    assert [r["check"] for r in result] == _DB_CHECKS   # declared order preserved
    assert rec.max_active == 1                          # never overlapped
    assert factory_calls == 0                           # no side sessions opened
    assert all(r["ok"] for r in result)


def test_flag_off_short_circuits_on_first_failure():
    # Original loop stops at the first failing check and never runs the rest.
    result, rec, _ = _run(_DB_CHECKS, enabled=False, failing={"no_conflicting_appointment"})
    assert [r["check"] for r in result] == ["slot_available", "no_conflicting_appointment"]
    assert rec.order == ["slot_available", "no_conflicting_appointment"]
    assert result[-1]["ok"] is False


# --- flag ON: real concurrency ----------------------------------------------------------------

def test_flag_on_runs_db_checks_concurrently():
    result, rec, factory_calls = _run(_DB_CHECKS, enabled=True)
    assert rec.max_active > 1                           # checks genuinely overlapped
    assert factory_calls == len(_DB_CHECKS)             # one side session per DB check
    assert [r["check"] for r in result] == _DB_CHECKS   # still returned in declared order
    assert all(r["ok"] for r in result)
    assert all("ms" in r for r in result)               # per-check timing recorded


def test_concurrency_is_bounded_by_the_cap():
    result, rec, _ = _run(_DB_CHECKS, enabled=True, maxc=2)
    assert rec.max_active <= 2                           # semaphore held the line
    assert len(result) == len(_DB_CHECKS)


def test_in_memory_checks_run_inline_not_on_a_side_session():
    checks = ["customer_identified", "slot_available", "slot_was_offered",
              "no_conflicting_appointment"]
    result, _, factory_calls = _run(checks, enabled=True)
    # Only the two DB checks get a side session; the two in-memory ones run inline on ctx.
    assert factory_calls == 2
    assert [r["check"] for r in result] == checks       # order still declared order


# --- the blocking check is deterministic ------------------------------------------------------

def test_first_failure_in_declared_order_wins_even_when_a_later_check_also_fails():
    # c1 (index 1) and c3 (index 3) both fail; the caller blocks on the FIRST in declared
    # order, so the runner must return no_conflicting_appointment as the first not-ok row.
    result, _, _ = _run(_DB_CHECKS, enabled=True,
                        failing={"no_conflicting_appointment", "executive_exists"})
    first_bad = next(r for r in result if not r["ok"])
    assert first_bad["check"] == "no_conflicting_appointment"
    assert first_bad["message"] == "no_conflicting_appointment failed"


# --- safety gates -----------------------------------------------------------------------------

def test_pending_tool_write_forces_sequential():
    # A prior tool in the same turn flushed an uncommitted write; side connections would not
    # see it, so the runner must stay on the shared session (sequential, no overlap).
    _, rec, factory_calls = _run(_DB_CHECKS, enabled=True, pending=True)
    assert rec.max_active == 1
    assert factory_calls == 0


def test_parallel_infra_failure_degrades_to_sequential():
    # Opening a side session blows up. The runner must fall back to a clean sequential run and
    # still return correct results rather than propagating the error into the turn.
    result, rec, _ = _run(_DB_CHECKS, enabled=True, side_raises=True)
    assert [r["check"] for r in result] == _DB_CHECKS
    assert all(r["ok"] for r in result)
    # The fallback ran every check sequentially on the shared session.
    assert rec.max_active == 1


if __name__ == "__main__":       # allow plain `python tests/test_parallel_preconditions.py`
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
