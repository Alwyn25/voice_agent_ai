"""Unit tests for R5.01 — cross-encoder re-ranking.

What must hold, and is tested below (fakes only unless noted):
  * fusion: "fuse" lets a strong cross-encoder judgement overturn a near-tie but keeps the base
    ranking's vote; "replace" is pure cross-encoder order; results are deterministic.
  * reranker.score never raises and never blocks: not-ready, load failure (with back-off),
    too-few candidates, busy (concurrency cap), timeout and inference errors all come back as
    a non-ok status; a timed-out job still frees its slot and leaves its scores cached.
  * retrieve(): on "ok" the fused order decides the window (the 2.5 BHK vs 3 BHK regression);
    on ANY failure the result is identical to reranking being off; rerank=False bypasses it;
    diagnostics say what happened.
  * (integration, skipped unless onnxruntime + the model files are available locally) the real
    MiniLM cross-encoder ranks the 2.5 BHK cost chunk above the 3 BHK sheet.

Run: cd core && python -m pytest tests/test_reranker.py -q
"""
import asyncio
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.engine.rag import reranker, retrieval  # noqa: E402
from app.engine.rag.reranker import RerankConfig, RerankOutcome, fuse_rankings  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean_reranker():
    reranker._reset_for_tests()
    yield
    reranker._reset_for_tests()


def _cfg(**over) -> RerankConfig:
    base = dict(model="fake/model", revision="abc", onnx_file="model.onnx", cache_dir=None,
                allow_download=False, max_tokens=64, threads=1, max_concurrency=2, timeout_ms=500)
    base.update(over)
    return RerankConfig(**base)


# --------------------------------------------------------------------------- fusion (pure)

def test_fuse_lets_cross_encoder_overturn_a_near_tie():
    # RRF had the 3 BHK sheet first and the 2.5 BHK rows a hair behind; the CE is sure of 2.5.
    base = ["3bhk", "2.5bhk", "amenities"]
    ce = {"3bhk": 1.0, "2.5bhk": 8.0, "amenities": -5.0}
    fused = fuse_rankings(base, ce, mode="fuse", k=60, base_weight=1.0, ce_weight=1.0)
    assert max(fused, key=fused.get) == "2.5bhk"
    assert min(fused, key=fused.get) == "amenities"


def test_fuse_keeps_the_base_vote():
    # CE alone would put d first; in fuse mode a chunk the base ranking put LAST cannot jump
    # the chunk both rankings agree is near the top.
    base = ["a", "b", "c", "d"]
    ce = {"a": 5.0, "b": 1.0, "c": 0.0, "d": 6.0}
    fused = fuse_rankings(base, ce, mode="fuse", k=60, base_weight=1.0, ce_weight=1.0)
    assert sorted(fused, key=fused.get, reverse=True)[0] == "a"
    replaced = fuse_rankings(base, ce, mode="replace", k=60, base_weight=1.0, ce_weight=1.0)
    assert sorted(replaced, key=replaced.get, reverse=True) == ["d", "a", "b", "c"]


def test_fuse_with_zero_base_weight_equals_replace_order():
    base = ["a", "b", "c"]
    ce = {"a": 0.1, "b": 0.9, "c": 0.5}
    f = fuse_rankings(base, ce, mode="fuse", k=60, base_weight=0.0, ce_weight=1.0)
    r = fuse_rankings(base, ce, mode="replace", k=60, base_weight=0.0, ce_weight=1.0)
    assert sorted(f, key=f.get, reverse=True) == sorted(r, key=r.get, reverse=True)


def test_fusion_ties_are_deterministic():
    base = ["a", "b", "c"]
    ce = {"a": 1.0, "b": 1.0, "c": 1.0}
    f = fuse_rankings(base, ce, mode="replace", k=60, base_weight=1.0, ce_weight=1.0)
    assert sorted(f, key=f.get, reverse=True) == ["a", "b", "c"]   # equal CE -> base order


# --------------------------------------------------------------------------- score(): fakes

class _FakeHandle:
    input_names = ("input_ids",)


def _ready(monkeypatch, infer):
    """Pretend the model is loaded and route inference to ``infer(query, passages)``."""
    handle = _FakeHandle()
    monkeypatch.setattr(reranker, "_ensure_loading", lambda cfg: (handle, None))
    monkeypatch.setattr(reranker, "_infer", lambda h, q, ps: infer(q, ps))


def _length_scores(q, passages):
    return [float(len(p)) for p in passages]


def test_score_ok_then_served_from_cache(monkeypatch):
    calls = []

    def infer(q, ps):
        calls.append(len(ps))
        return _length_scores(q, ps)

    _ready(monkeypatch, infer)
    cands = [("a", "x"), ("b", "xxx"), ("c", "xx")]
    out = _run(reranker.score(_cfg(), "q", cands))
    assert out.ok and out.scores == {"a": 1.0, "b": 3.0, "c": 2.0}
    assert out.scored == 3 and out.cache_hits == 0
    again = _run(reranker.score(_cfg(), "q", cands))
    assert again.ok and again.scores == out.scores
    assert again.scored == 0 and again.cache_hits == 3 and calls == [3], "no second inference"


def test_score_skips_fewer_than_two_candidates(monkeypatch):
    _ready(monkeypatch, _length_scores)
    assert _run(reranker.score(_cfg(), "q", [("a", "x")])).status == reranker.SKIPPED_FEW


def test_score_not_ready_while_loading(monkeypatch):
    monkeypatch.setattr(reranker, "_ensure_loading", lambda cfg: (None, reranker.NOT_READY))
    out = _run(reranker.score(_cfg(), "q", [("a", "x"), ("b", "y")]))
    assert out.status == reranker.NOT_READY and not out.scores


def test_load_failure_backs_off_and_never_raises(monkeypatch):
    def boom(cfg):
        raise FileNotFoundError("model missing")

    monkeypatch.setattr(reranker, "_load_sync", boom)
    cfg = _cfg()
    first = _run(reranker.score(cfg, "q", [("a", "x"), ("b", "y")]))
    assert first.status == reranker.NOT_READY          # load kicked off in the background
    for _ in range(100):                                # wait for the loader thread
        if reranker._load_errors:
            break
        time.sleep(0.01)
    second = _run(reranker.score(cfg, "q", [("a", "x"), ("b", "y")]))
    assert second.status == reranker.UNAVAILABLE       # in back-off, no reload storm
    status = reranker.reranker_status(cfg)
    assert status["load_errors"] and "model missing" in status["load_errors"][0]["error"]
    assert status["configured"]["ready"] is False


def test_score_timeout_frees_slot_and_keeps_scores(monkeypatch):
    release = threading.Event()

    def slow(q, ps):
        release.wait(2)
        return _length_scores(q, ps)

    _ready(monkeypatch, slow)
    cfg = _cfg(timeout_ms=50, max_concurrency=1)
    cands = [("a", "x"), ("b", "xx")]
    out = _run(reranker.score(cfg, "q", cands))
    assert out.status == reranker.TIMEOUT and not out.scores
    release.set()
    for _ in range(200):                                # the worker finishes in the background
        if reranker._inflight == 0:
            break
        time.sleep(0.01)
    assert reranker._inflight == 0, "slot must be released when the worker finishes"
    later = _run(reranker.score(cfg, "q", cands))
    assert later.ok and later.cache_hits == 2, "the late result was cached for the next call"


def test_score_busy_when_concurrency_cap_reached(monkeypatch):
    _ready(monkeypatch, _length_scores)
    reranker._inflight = 1
    out = _run(reranker.score(_cfg(max_concurrency=1), "q", [("a", "x"), ("b", "y")]))
    assert out.status == reranker.SKIPPED_BUSY


def test_score_inference_error_is_contained(monkeypatch):
    def broken(q, ps):
        raise RuntimeError("onnx exploded")

    _ready(monkeypatch, broken)
    out = _run(reranker.score(_cfg(), "q", [("a", "x"), ("b", "y")]))
    assert out.status == reranker.ERROR and "onnx exploded" in out.error
    for _ in range(100):
        if reranker._inflight == 0:
            break
        time.sleep(0.01)
    assert reranker._inflight == 0


# --------------------------------------------------------------------------- retrieve()

DOC_3, DOC_25, DOC_AM = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
ROWS = {
    "c3": {"id": "c3", "document_id": DOC_3, "title": "3 BHK Cost Sheet", "doc_type": "pricing",
           "content": "3 BHK 1,390 sq ft Total 1,72,06,110", "metadata": {"heading": "3 BHK"}},
    "c25": {"id": "c25", "document_id": DOC_25, "title": "2 BHK and 2.5 BHK Cost Sheet",
            "doc_type": "pricing", "content": "2.5 BHK 1,325 sq ft Total 1,55,31,250",
            "metadata": {"heading": "2.5 BHK"}},
    "cam": {"id": "cam", "document_id": DOC_AM, "title": "Amenities", "doc_type": "brochure",
            "content": "Swimming pool, squash court", "metadata": {"heading": "Clubhouse"}},
}


class _Result:
    def __init__(self, value=None, rows=()):
        self._value, self._rows = value, list(rows)

    def scalar(self):
        return self._value

    def mappings(self):
        return self

    def all(self):
        return self._rows

    def scalar_one_or_none(self):
        return self._value


class _FakeDB:
    """Keyword arm returns the same order as the vector arm; expired count is 0. The tenant
    row carries ``tenant_settings`` (for the rerank glossary); tenant_fails simulates a DB error."""

    tenant_settings: dict = {}
    tenant_fails = False
    tenant_reads = 0

    async def execute(self, sql, params=None):
        q = getattr(sql, "text", None) or str(sql)
        if "ts_rank" in q:
            return _Result(rows=[ROWS["c3"], ROWS["c25"], ROWS["cam"]])
        if "count(*)" in q:
            return _Result(0)
        if "tenants" in q:
            type(self).tenant_reads += 1
            if self.tenant_fails:
                raise RuntimeError("db down")
            return _Result(SimpleNamespace(settings=self.tenant_settings))
        raise AssertionError(f"unexpected SQL: {q[:80]}")


@pytest.fixture
def rag_env(monkeypatch):
    """retrieve() over three fixed chunks where plain RRF ranks the 3 BHK sheet first."""
    async def vector(db, s, result, params, *, pool, tenant_id):
        return [ROWS["c3"], ROWS["c25"], ROWS["cam"]]

    async def embed(query, session_cache=None):
        return [0.0, 1.0]

    monkeypatch.setattr(retrieval, "_vector_search", vector)
    monkeypatch.setattr(retrieval, "_embed_query", embed)
    retrieval._GLOSSARY_CACHE.clear()
    monkeypatch.setattr(_FakeDB, "tenant_settings", {})
    monkeypatch.setattr(_FakeDB, "tenant_fails", False)
    monkeypatch.setattr(_FakeDB, "tenant_reads", 0)
    s = get_settings()
    for key, value in dict(retrieval_rerank_enabled=True, retrieval_rerank_mode="fuse",
                           retrieval_rerank_languages="english",
                           retrieval_rerank_translate=True, retrieval_rerank_glossary={},
                           retrieval_max_chunks_per_doc=2,
                           retrieval_rerank_candidates=12, retrieval_rerank_weight=1.0,
                           retrieval_rerank_base_weight=1.0, retrieval_rerank_rrf_k=60,
                           retrieval_rerank_debug_log=True, retrieval_top_k=8,
                           retrieval_min_score=0.15).items():
        monkeypatch.setattr(s, key, value)
    return s


def _fake_score(status="ok", scores=None, seen=None):
    async def score(cfg, query, candidates):
        if seen is not None:
            seen.append((query, candidates))
        return RerankOutcome(status=status, model="fake", scores=dict(scores or {}),
                             candidates=len(candidates), scored=len(candidates))
    return score


def _search(**kw):
    return _run(retrieval.retrieve(_FakeDB(), tenant_id=uuid.uuid4(),
                                   query="price of a 2.5 BHK?", synonyms=[], **kw))


def test_rerank_promotes_the_right_cost_sheet(monkeypatch, rag_env):
    seen = []
    monkeypatch.setattr(reranker, "score", _fake_score(
        scores={"c3": 1.0, "c25": 9.0, "cam": -8.0}, seen=seen))
    res = _search()
    assert [c.chunk_id for c in res.chunks][0] == "c25"
    top = res.chunks[0]
    assert top.base_rank == 2 and top.rerank_rank == 1 and top.rerank_score == 9.0
    assert res.rerank["status"] == "ok" and res.rerank["top1_changed"] is True
    # The CE reads the document title too — the chunk body alone may not name the config.
    query, cands = seen[0]
    assert query == "price of a 2.5 bhk?"
    assert any(text.startswith("2 BHK and 2.5 BHK Cost Sheet") for _, text in cands)


@pytest.mark.parametrize("status", [reranker.TIMEOUT, reranker.ERROR, reranker.NOT_READY,
                                    reranker.SKIPPED_BUSY, reranker.UNAVAILABLE])
def test_any_rerank_failure_equals_rerank_off(monkeypatch, rag_env, status):
    off = _search(rerank=False)
    monkeypatch.setattr(reranker, "score", _fake_score(status=status))
    failed = _search()
    assert [c.chunk_id for c in failed.chunks] == [c.chunk_id for c in off.chunks]
    assert [c.relevance for c in failed.chunks] == [c.relevance for c in off.chunks]
    assert failed.rerank["status"] == status
    assert all(c.rerank_score is None for c in failed.chunks)


def test_rerank_false_bypasses_reranker(monkeypatch, rag_env):
    called = []
    monkeypatch.setattr(reranker, "score", _fake_score(seen=called))
    res = _search(rerank=False)
    assert not called and res.rerank is None
    assert res.chunks[0].chunk_id == "c3", "plain RRF order (the pre-R5.01 behaviour)"


def test_disabled_setting_is_the_old_path(monkeypatch, rag_env):
    monkeypatch.setattr(rag_env, "retrieval_rerank_enabled", False)
    called = []
    monkeypatch.setattr(reranker, "score", _fake_score(seen=called))
    res = _search()
    assert not called and res.rerank is None


def test_language_gate_skips_hinglish_and_keeps_rrf_order(monkeypatch, rag_env):
    called = []
    monkeypatch.setattr(reranker, "score", _fake_score(scores={"c3": 1.0, "c25": 9.0, "cam": 0.0},
                                                       seen=called))
    q = "2.5 BHK ka price kya hai"
    off = _run(retrieval.retrieve(_FakeDB(), tenant_id=uuid.uuid4(), query=q, synonyms=[],
                                  rerank=False))
    gated = _run(retrieval.retrieve(_FakeDB(), tenant_id=uuid.uuid4(), query=q, synonyms=[]))
    assert not called, "the model is never invoked for a gated language"
    assert gated.rerank["status"] == reranker.SKIPPED_LANGUAGE
    assert gated.rerank["language"] == "hinglish"
    assert [c.chunk_id for c in gated.chunks] == [c.chunk_id for c in off.chunks]
    # "all" opens the gate.
    monkeypatch.setattr(rag_env, "retrieval_rerank_languages", "all")
    opened = _run(retrieval.retrieve(_FakeDB(), tenant_id=uuid.uuid4(), query=q, synonyms=[]))
    assert called and opened.rerank["status"] == "ok"


def test_chunks_below_candidate_cut_stay_in_the_pool(monkeypatch, rag_env):
    # Only the top 2 are re-scored; the 3rd chunk must still reach the window (below them),
    # otherwise the per-document cap loses the breadth it reaches down the pool for.
    monkeypatch.setattr(rag_env, "retrieval_rerank_candidates", 2)
    seen = []
    monkeypatch.setattr(reranker, "score", _fake_score(scores={"c3": 1.0, "c25": 9.0}, seen=seen))
    res = _search()
    assert [key for key, _ in seen[0][1]] == ["c3", "c25"]
    assert [c.chunk_id for c in res.chunks] == ["c25", "c3", "cam"]
    assert res.chunks[2].rerank_score is None and res.chunks[2].base_rank == 3


def test_focus_anchor_still_orders_after_rerank(monkeypatch, rag_env):
    # The CE prefers 2.5 BHK, but a caller whose standing focus is "3 BHK" keeps the existing
    # tie-break behaviour: focus only reorders the window, it never evicts.
    monkeypatch.setattr(reranker, "score", _fake_score(scores={"c3": 1.0, "c25": 9.0, "cam": 0.0}))
    res = _search(focus_terms=["3 BHK"])
    ids = [c.chunk_id for c in res.chunks]
    assert ids[0] == "c3" and set(ids) == {"c3", "c25", "cam"}


def test_cross_encoder_reads_the_english_gloss_of_hinglish(monkeypatch, rag_env):
    monkeypatch.setattr(rag_env, "retrieval_rerank_languages", "all")
    seen = []
    monkeypatch.setattr(reranker, "score", _fake_score(scores={"c3": 1.0, "c25": 9.0, "cam": 0.0},
                                                       seen=seen))
    res = _run(retrieval.retrieve(_FakeDB(), tenant_id=uuid.uuid4(), synonyms=[],
                                  query="2.5 BHK ka price kya hai"))
    assert seen[0][0] == "2.5 bhk of price what is"
    assert res.rerank["ce_query"] == "2.5 bhk of price what is"
    assert res.query == "2.5 BHK ka price kya hai", "the caller's words are kept for the trace"


def test_tenant_glossary_is_applied_and_cached(monkeypatch, rag_env):
    monkeypatch.setattr(rag_env, "retrieval_rerank_languages", "all")
    monkeypatch.setattr(_FakeDB, "tenant_settings", {"rerank_glossary": {"khel kood": "sports"}})
    seen = []
    monkeypatch.setattr(reranker, "score", _fake_score(scores={"c3": 1.0, "c25": 2.0, "cam": 3.0},
                                                       seen=seen))
    tenant = uuid.uuid4()
    for _ in range(2):
        _run(retrieval.retrieve(_FakeDB(), tenant_id=tenant, synonyms=[],
                                granularity="flat",
                                query="khel kood ki kya suvidha hai"))
    assert seen[0][0] == "sports of what suvidha is"
    assert _FakeDB.tenant_reads == 1, "glossary is cached, not re-read every turn"


def test_glossary_db_failure_never_fails_retrieval(monkeypatch, rag_env):
    monkeypatch.setattr(rag_env, "retrieval_rerank_languages", "all")
    monkeypatch.setattr(_FakeDB, "tenant_fails", True)
    monkeypatch.setattr(reranker, "score", _fake_score(scores={"c3": 1.0, "c25": 2.0, "cam": 3.0}))
    res = _run(retrieval.retrieve(_FakeDB(), tenant_id=uuid.uuid4(), synonyms=[],
                                  query="swimming pool hai kya"))
    assert res.rerank["status"] == "ok" and res.rerank["ce_query"] == "swimming pool is what"


def test_per_doc_cap_setting(monkeypatch, rag_env):
    # Three chunks of ONE document: the cap decides how many make the window before the
    # overflow back-fill; with rerank off and a top_k of 2 the difference is visible directly.
    doc = uuid.uuid4()
    rows = [dict(ROWS["c3"], id=f"x{i}", document_id=doc) for i in range(3)] + [ROWS["cam"]]

    async def vector(db, s, result, params, *, pool, tenant_id):
        return rows
    monkeypatch.setattr(retrieval, "_vector_search", vector)
    monkeypatch.setattr(_FakeDB, "execute", _only_expired)
    ids = lambda: [c.chunk_id for c in _run(retrieval.retrieve(   # noqa: E731
        _FakeDB(), tenant_id=uuid.uuid4(), query="price of 3 bhk", synonyms=[], top_k=3,
        rerank=False)).chunks]
    assert set(ids()) == {"x0", "x1", "cam"}             # cap 2: breadth wins the 3rd slot
    monkeypatch.setattr(rag_env, "retrieval_max_chunks_per_doc", 3)
    assert set(ids()) == {"x0", "x1", "x2"}              # cap 3: the sibling fact gets in


async def _only_expired(self, sql, params=None):
    q = getattr(sql, "text", None) or str(sql)
    if "ts_rank" in q:
        return _Result(rows=[])
    return _Result(0)


# --------------------------------------------------------------------------- gloss (pure)

@pytest.mark.parametrize("query", [
    "tell me the price of 3 bhk", "is there a swimming pool", "what is the corpus fund",
    "Can I park my car near the door?",
])
def test_gloss_leaves_english_untouched(query):
    assert reranker.gloss_query(query) == query.lower()


@pytest.mark.parametrize("query,expected", [
    ("corner aur premium mein charge ka farak kya hai",
     "corner and premium in charge of difference what is"),
    ("Wipro se kitni door hai", "wipro from how much door is"),          # "door" is not glossed
    ("kitchen modular milegaa kia", "kitchen modular milegaa what"),     # unknown word kept
    ("1239 wale flat ka total kitna padega", "1239 flat of total how much will cost"),
])
def test_gloss_hinglish(query, expected):
    assert reranker.gloss_query(query) == expected


def test_gloss_ambiguous_words_need_a_hindi_marker():
    # "me" / "se" / "par" are English words too: without a strong Hindi word, nothing changes.
    assert reranker.gloss_query("send me the brochure") == "send me the brochure"
    assert reranker.gloss_query("mujhe brochure bhejo, me busy hoon") == "me brochure bhejo, in busy hoon"


def test_gloss_extra_phrases_longest_first_whole_word():
    extra = {"khel": "games", "khel kood": "sports", "gym": "fitness centre"}
    assert reranker.gloss_query("khel kood ki suvidha", extra) == "sports ki suvidha"
    assert reranker.gloss_query("gymkhana hai kya", extra) == "gymkhana is what"


# --------------------------------------------------------------------------- real model

def _real_cfg():
    s = get_settings()
    return reranker.config_from_settings(SimpleNamespace(**{
        **{k: getattr(s, k) for k in type(s).model_fields if k.startswith("retrieval_rerank_")},
        "retrieval_rerank_allow_download": False, "retrieval_rerank_timeout_ms": 10000}))


def _real_model_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
        cfg = _real_cfg()
        reranker._resolve_file(cfg, cfg.onnx_file)
        reranker._resolve_file(cfg, "tokenizer.json")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _real_model_available(),
                    reason="onnxruntime or the pinned model files not available locally")
def test_real_cross_encoder_prefers_the_asked_configuration():
    cfg = _real_cfg()
    assert _run(reranker.warm_up(cfg, timeout_s=120))
    out = _run(reranker.score(cfg, "what is the price of a 2.5 bhk?", [
        ("3bhk", "3 BHK Cost Sheet — 11,499 per square foot\n3 BHK cost sheet. Rate 11,499 "
                 "rupees per square foot. 1,390 sq ft. Total 1,72,06,110."),
        ("2.5bhk", "2 BHK and 2.5 BHK Cost Sheet — 11,000 per square foot\n2.5 BHK cost — what "
                   "a 2.5 BHK flat costs. 2.5 BHK, 1,325 sq ft. Total 1,55,31,250."),
        ("pool", "Amenities and Clubhouses\nSwimming pool, pool deck, squash court."),
    ]))
    assert out.ok, out
    assert out.scores["2.5bhk"] > out.scores["3bhk"] > out.scores["pool"]
