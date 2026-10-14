"""Unit tests for P4-1 — running the fast UNDERSTANDING classifier concurrently with propose().

Live replay (scripts/replay_outliers.py) showed the understanding call is a ~1.3s serial cost:
it "overlaps" retrieval, but retrieval is only ~15ms, so the overlap hides nothing. P4-1 moves
the classifier off the retrieval leg. The seam that makes that possible is
``planner._understand_and_retrieve(skip_understanding=True)`` — retrieve ONLY, keep the result,
and DO NOT call the classifier (plan() runs it later, alongside propose). This pins that seam
directly with fakes, and pins that the ordinary (serial) path is byte-for-byte unchanged, so the
default-off flag can never alter today's routing.

Run: cd core && python -m pytest tests/test_concurrent_understanding.py -q
"""
import asyncio
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.planner import planner
from app.engine.planner.understanding import Understanding


class _FakeNested:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeDB:
    """Only the surface _understand_and_retrieve touches: a savepoint context manager."""
    def begin_nested(self):
        return _FakeNested()


class _FakeRag:
    """Duck-types the retrieval result _understand_and_retrieve reads."""
    def __init__(self, chunks):
        self._chunks = chunks
        self.vector_hits = len(chunks)
        self.text_hits = 0
        self.latency_ms = 15          # the measured reality: retrieval is tiny
        self.excluded_expired = 0
        self.error = None

    def as_context(self):
        return list(self._chunks)


def _tc():
    """A minimal TurnContext stand-in with only the attributes the retrieval leg reads."""
    return SimpleNamespace(
        history=[], active_preference=None, retrieval_synonyms=[],
        retrieval_project_id=None, project_ambiguous=False, project_candidates=[],
        project_candidate_ids=[], project_id=None, project_name=None, project_signal=None,
        knowledge=[])


_CHUNKS = [{"source": "brochure", "content": "The clubhouse has a creche."}]


def _run(skip_understanding, understanding_result):
    """Drive _understand_and_retrieve with retrieve() and understand() faked."""
    async def _fake_retrieve(*a, **k):
        return _FakeRag(_CHUNKS)

    calls = {"understand": 0}

    async def _fake_understand(*a, **k):
        calls["understand"] += 1
        return understanding_result

    with patch.object(planner, "retrieve", _fake_retrieve), \
         patch.object(planner, "understand", _fake_understand):
        result = asyncio.run(planner._understand_and_retrieve(
            _FakeDB(), tenant_id=uuid.uuid4(), tc=_tc(), user_input="is there a daycare?",
            active_entities={}, as_of_date=None, skip_understanding=skip_understanding))
    return result, calls


# --- skip_understanding=True: retrieve only, classifier NOT called, knowledge KEPT -----------
def test_skip_understanding_does_not_call_classifier():
    (understanding, knowledge, meta), calls = _run(
        skip_understanding=True,
        understanding_result=Understanding(needs_knowledge=True))
    assert calls["understand"] == 0, "understand() must not run in skip mode"
    assert understanding is None
    # Knowledge is kept regardless of routing — plan() applies do_retrieve after propose.
    assert knowledge == _CHUNKS
    assert meta["used"] is True
    assert meta["routing"] == {}          # no classifier ran, so no routing trace yet


def test_skip_understanding_keeps_knowledge_even_when_a_scheduling_turn():
    # Even a turn the classifier WOULD route away from documents keeps its chunks in skip mode —
    # that is the whole point: propose sees them; the responder gating is applied later.
    (understanding, knowledge, _meta), calls = _run(
        skip_understanding=True,
        understanding_result=Understanding(needs_knowledge=False, is_scheduling=True))
    assert calls["understand"] == 0
    assert knowledge == _CHUNKS


# --- serial path (default): unchanged — classifier runs and its routing gates knowledge ------
def test_serial_runs_classifier_and_keeps_knowledge_when_do_retrieve():
    (understanding, knowledge, meta), calls = _run(
        skip_understanding=False,
        understanding_result=Understanding(needs_knowledge=True))
    assert calls["understand"] == 1
    assert understanding is not None and understanding.do_retrieve is True
    assert knowledge == _CHUNKS
    assert meta["used"] is True


def test_serial_drops_knowledge_when_routed_away():
    # do_retrieve False (scheduling/status) -> serial path DISCARDS the speculative chunks, so
    # the responder is never handed off-topic material. This is the behaviour skip mode defers.
    (understanding, knowledge, meta), calls = _run(
        skip_understanding=False,
        understanding_result=Understanding(needs_knowledge=False, is_status_question=True))
    assert calls["understand"] == 1
    assert understanding.do_retrieve is False
    assert knowledge == []
    assert meta["used"] is False
    assert "routed away" in meta.get("skipped_reason", "")


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-q"]))
