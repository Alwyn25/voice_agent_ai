"""Unit tests for P2.05 support — concurrent DB pool warming (db.base.warm_pool).

The whole reason warm_pool exists is that parallel preconditions fan a booking turn out to
several DB connections AT ONCE; if the pool holds only one warm connection the rest are built
cold mid-call (a ~145ms TLS spike, seen live). So the one property that matters is: warm_pool(N)
must hold N connections open AT THE SAME TIME (forcing the pool to create N distinct physical
connections), not open one N times. That, the pool_size clamp, and the single-connection path
are pinned here with a fake session factory — no live DB.

Run: cd core && python -m pytest tests/test_warm_pool.py -q
"""
import asyncio
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import base


class _Settings:
    def __init__(self, pool_size):
        self.db_pool_size = pool_size


class _Tracker:
    """Counts how many fake sessions are open simultaneously."""
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.total = 0


class _FakeSession:
    def __init__(self, tracker):
        self._t = tracker

    async def __aenter__(self):
        self._t.active += 1
        self._t.total += 1
        self._t.max_active = max(self._t.max_active, self._t.active)
        return self

    async def __aexit__(self, *a):
        self._t.active -= 1
        return False

    async def execute(self, *a, **k):
        await asyncio.sleep(0.01)     # stand in for a real round-trip
        return None


def _run(connections, *, pool_size=10):
    tracker = _Tracker()

    def fake_get_factory():
        def factory():
            return _FakeSession(tracker)
        return factory

    with ExitStack() as st:
        st.enter_context(patch.object(base, "get_session_factory", fake_get_factory))
        st.enter_context(patch.object(base, "get_settings", lambda: _Settings(pool_size)))
        primed = asyncio.run(base.warm_pool(connections))
    return primed, tracker


def test_warms_connections_concurrently():
    primed, tracker = _run(4)
    assert primed == 4
    assert tracker.max_active == 4          # all four held open AT ONCE — the whole point
    assert tracker.total == 4


def test_single_connection_path():
    primed, tracker = _run(1)
    assert primed == 1
    assert tracker.max_active == 1
    assert tracker.total == 1


def test_clamped_to_pool_size():
    # Asking for more than pool_size is pointless (overflow connections are discarded on
    # return), so warm_pool must clamp — never open more than the pool can keep warm.
    primed, tracker = _run(8, pool_size=3)
    assert primed == 3
    assert tracker.max_active == 3


def test_zero_or_negative_floors_to_one():
    primed, tracker = _run(0)
    assert primed == 1                       # always primes at least one connection
    assert tracker.total == 1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
