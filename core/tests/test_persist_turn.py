"""turn_runner._persist_turn: working memory (Redis) + commit (Postgres) must not be split
by a barge-in cancellation, and the caller must be told whether the turn was saved.

    cd core && python -m pytest tests/test_persist_turn.py -q
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.planner import turn_runner as tr  # noqa: E402
from app.observability import StageTimer  # noqa: E402


class _FakeDB:
    def __init__(self, commit_delay=0.0, fail=False):
        self.commit_delay, self.fail, self.committed = commit_delay, fail, False

    async def commit(self):
        await asyncio.sleep(self.commit_delay)
        if self.fail:
            raise RuntimeError("commit failed")
        self.committed = True


@pytest.fixture
def saved(monkeypatch):
    calls = []

    async def _save(conv_id, session):
        calls.append(dict(session))
    monkeypatch.setattr(tr.working, "save", _save)
    return calls


def _run(coro):
    return asyncio.run(coro)


def test_normal_turn_saves_then_commits(saved):
    db = _FakeDB()
    _run(tr._persist_turn(db, "conv", {"turn_index": 3}, StageTimer()))
    assert saved == [{"turn_index": 3}] and db.committed


def test_cancel_mid_commit_finishes_and_reports_saved(saved):
    db = _FakeDB(commit_delay=0.05)

    async def scenario():
        task = asyncio.ensure_future(tr._persist_turn(db, "conv", {"turn_index": 4},
                                                      StageTimer()))
        await asyncio.sleep(0.01)           # inside the commit
        task.cancel()                       # the caller barged in
        with pytest.raises(tr.TurnCancelledAfterCommit):
            await task
        # Finished BEFORE the cancellation surfaced: the caller may touch the session now.
        assert db.committed and saved == [{"turn_index": 4}]
    _run(scenario())


def test_turn_cancelled_after_commit_is_still_a_cancellation():
    # pipecat's task machinery must see an ordinary cancellation.
    assert issubclass(tr.TurnCancelledAfterCommit, asyncio.CancelledError)


def test_write_failure_during_cancel_reports_plain_cancellation(saved):
    db = _FakeDB(commit_delay=0.05, fail=True)

    async def scenario():
        task = asyncio.ensure_future(tr._persist_turn(db, "conv", {}, StageTimer()))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as e:
            await task
        # Not saved -> NOT the "after commit" subclass, so the bridge rolls back and carries.
        assert not isinstance(e.value, tr.TurnCancelledAfterCommit)
        assert not db.committed
    _run(scenario())
