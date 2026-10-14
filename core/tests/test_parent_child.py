"""Unit tests for R5.03 — parent-child ("small-to-big") chunking and retrieval.

What must hold (no live DB, no network — fakes only):
  * child splitting: table children keep the header; lists split into small groups; prose never
    carries overlap across paragraphs (one cost-sheet unit per paragraph); COVERAGE — no text of
    a parent is missing from every child; child-sized parents are not split; tiny fragments are
    merged, never dropped; children inherit the breadcrumb.
  * ingestion plan: flag off = every chunk standalone (pre-R5.03); flag on = parents + children.
  * collapse: children of one parent become ONE unit with the parent's text; best child
    dominates the score, the sibling bonus is capped; ranks are the best member's; standalone
    hits pass through; a child whose parent is missing is served as-is and counted.
  * retrieve(): flat mode excludes children and never collapses; parent_child mode excludes
    parents, dedups, returns parent text with matched_children; a failing parent fetch degrades
    to child-level results instead of failing the turn; the cache key separates the two modes.
  * ingest_document(): roles/parent links written correctly; a parent whose children all failed
    to embed is stored standalone; flag off writes only standalone rows.

Run: cd core && python -m pytest tests/test_parent_child.py -q
"""
import asyncio
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from app.engine.rag import parent_child as pc, retrieval  # noqa: E402
from app.engine.rag.chunkers import Chunk, split_children, uncovered_text  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


SHEET = ("Rate 11,499 rupees per square foot on super built-up area. Clubhouse charges "
         "3,25,000 on all three. Car parking is 5,50,000 on the two smaller units.\n\n"
         "1,239 sq ft, West facing. Carpet 808. Balcony and utility 99. Flat cost 1,42,47,261. "
         "Total 1,54,32,011.\n\n"
         "1,390 sq ft, East and West facing. Carpet 923. Balcony and utility 100. Flat cost "
         "1,59,83,610. Total 1,72,06,110.\n\n"
         "1,560 sq ft, East and West facing. Carpet 1,032. Flat cost 1,79,38,440. "
         "Total 1,92,53,440.")

# One long paragraph of many short records — the real 2 BHK sheet's shape.
LONG_PARA = " ".join(f"{875 + i * 5} sq ft, North facing. Carpet {590 + i}. "
                     f"Total 1,0{i},00,000." for i in range(8))


def _parent(text, kind="prose", heading="3 BHK cost sheet"):
    return Chunk(0, text, heading, {"kind": kind, "heading_path": heading})


# ─────────────────────────────────────────────────────────────── child splitting

def test_prose_children_cover_parent_and_keep_breadcrumb():
    parent = _parent(SHEET)
    kids = split_children(parent, target_chars=200, overlap_chars=60, min_chars=40)
    assert len(kids) >= 3
    assert uncovered_text(parent, kids) == []
    assert all(k.heading == "3 BHK cost sheet" for k in kids)
    assert all(k.embed_text.startswith("3 BHK cost sheet\n") for k in kids)
    assert [k.metadata["child_index"] for k in kids] == list(range(len(kids)))


def test_no_overlap_across_paragraphs():
    kids = split_children(_parent(SHEET), target_chars=200, overlap_chars=60, min_chars=40)
    totals = ("1,54,32,011", "1,72,06,110", "1,92,53,440")
    for k in kids:
        assert sum(t in k.content for t in totals) <= 1, k.content


def test_long_paragraph_is_sentence_split_with_overlap_and_full_coverage():
    parent = _parent(LONG_PARA)
    kids = split_children(parent, target_chars=150, overlap_chars=40, min_chars=30)
    assert len(kids) >= 3
    assert uncovered_text(parent, kids) == []
    # overlap inside one paragraph: consecutive children share a seam sentence
    assert any(set(a.content.split(". ")) & set(b.content.split(". "))
               for a, b in zip(kids, kids[1:]))
    # no child ends or starts mid-number
    assert all(not k.content.rstrip().endswith(",") for k in kids)


def test_table_children_all_keep_header():
    rows = "\n".join(f"| Unit {i} | {1200 + i * 10} | 1,{50 + i},00,000 |" for i in range(12))
    parent = _parent(f"| Unit | Area | Total |\n|---|---|---|\n{rows}", kind="table")
    kids = split_children(parent, target_chars=150, min_chars=20)
    assert len(kids) >= 2
    assert all(k.content.startswith("| Unit | Area | Total |") for k in kids)
    assert uncovered_text(parent, kids) == []


def test_inline_list_children_are_small_groups():
    items = ["amphitheatre", "party lawn with stage", "barbeque zone", "informal seating",
             "youth hangout zone", "picnic lawn", "play lawn", "dog park", "jogging track",
             "cycle lane", "bird watching deck", "elders seating plaza"]
    parent = _parent(", ".join(items) + ".", kind="list", heading="Amenities > Outdoor")
    # ~170 chars: below 1.3 x 200 it is already child-sized and is (correctly) not split ...
    assert split_children(parent, target_chars=200, list_group_size=3, min_chars=10) == []
    # ... with a smaller target it splits into groups of at most 3 items.
    kids = split_children(parent, target_chars=100, list_group_size=3, min_chars=10)
    assert len(kids) == 4
    assert all(k.content.count(",") + 1 <= 3 for k in kids)
    assert uncovered_text(parent, kids) == []
    assert sum("dog park" in k.content for k in kids) == 1


def test_bullet_list_children_keep_bullets():
    parent = _parent("\n".join(f"- feature number {i} of the tower" for i in range(9)),
                     kind="list")
    kids = split_children(parent, target_chars=200, list_group_size=3, min_chars=10)
    assert len(kids) == 3
    assert all(line.startswith("- ") for k in kids for line in k.content.splitlines())


def test_child_sized_parent_is_not_split():
    assert split_children(_parent("Swimming pool, gym and a clubhouse."), target_chars=200) == []


def test_tiny_trailing_fragment_is_merged_not_dropped():
    parent = _parent("A" * 150 + ".\n\n" + "B" * 150 + ".\n\nEnd.")
    kids = split_children(parent, target_chars=160, overlap_chars=0, min_chars=40)
    assert uncovered_text(parent, kids) == []
    assert all(len(k.content) >= 40 for k in kids)


# ─────────────────────────────────────────────────────────────── ingestion plan

def test_plan_families_flag_off_is_pre_r503():
    chunks = [_parent(SHEET), _parent("short one here")]
    fams = pc.plan_families(chunks, enabled=False, target_chars=200, overlap_chars=60,
                            list_group_size=3, min_chars=40)
    assert [f.role for f in fams] == ["standalone", "standalone"]
    assert all(not f.children for f in fams)


def test_plan_families_flag_on():
    fams = pc.plan_families([_parent(SHEET), _parent("short one here")], enabled=True,
                            target_chars=200, overlap_chars=60, list_group_size=3,
                            min_chars=40)
    assert [f.role for f in fams] == ["parent", "standalone"]


def test_excluded_role():
    assert pc.excluded_role("flat") == "child"
    assert pc.excluded_role("parent_child") == "parent"
    assert pc.excluded_role("garbage") == "child", "unknown mode must fall back to flat"


# ─────────────────────────────────────────────────────────────── collapse

def test_parent_score_best_dominates_and_bonus_is_capped():
    assert pc.parent_score([0.02], sibling_weight=0.1, sibling_cap=0.25) == 0.02
    assert pc.parent_score([0.01, 0.02], sibling_weight=0.1,
                           sibling_cap=0.25) == pytest.approx(0.021)
    many = [0.02] + [0.019] * 30              # a long parent with many weak siblings
    assert pc.parent_score(many, sibling_weight=0.1,
                           sibling_cap=0.25) == pytest.approx(0.025)
    assert pc.parent_score([], sibling_weight=0.1, sibling_cap=0.25) == 0.0


def _child(cid, pid, **kw):
    return {"id": cid, "parent_id": pid, "chunk_role": "child", "document_id": "D1",
            "content": f"child text {cid}", **kw}


def _standalone(cid, **kw):
    return {"id": cid, "parent_id": None, "chunk_role": "standalone", "document_id": "D2",
            "content": f"standalone {cid}", **kw}


def _parent_row(pid):
    return {"id": pid, "parent_id": None, "chunk_role": "parent", "document_id": "D1",
            "content": f"PARENT TEXT {pid}", "title": "3 BHK cost sheet", "doc_type": "pricing",
            "metadata": {"heading": "3 BHK cost sheet"}}


def test_collapse_dedups_children_into_one_parent():
    rows = {"c1": _child("c1", "P1", vector_rank=3), "c2": _child("c2", "P1", text_rank=1),
            "c3": _child("c3", "P1", vector_rank=9), "s1": _standalone("s1", vector_rank=1)}
    scores = {"c1": 0.016, "c2": 0.013, "c3": 0.010, "s1": 0.017}
    out_scores, out_rows, st = pc.collapse_to_parents(
        scores, rows, {"P1": _parent_row("P1")}, sibling_weight=0.1, sibling_cap=0.25)
    assert set(out_scores) == {"P1", "s1"}
    p = out_rows["P1"]
    assert p["content"] == "PARENT TEXT P1" and p["doc_type"] == "pricing"
    assert p["vector_rank"] == 3 and p["text_rank"] == 1
    assert [m["chunk_id"] for m in p["matched_children"]] == ["c1", "c2", "c3"]
    assert out_scores["P1"] == pytest.approx(0.016 + 0.1 * (0.013 + 0.010))
    assert out_rows["s1"] is rows["s1"] and out_scores["s1"] == 0.017
    assert (st.hits, st.child_hits, st.parents, st.merged_siblings, st.parents_missing,
            st.units) == (4, 3, 1, 2, 0, 2)


def test_collapse_orphan_child_is_kept_and_counted():
    rows = {"c1": _child("c1", "GONE")}
    out_scores, out_rows, st = pc.collapse_to_parents({"c1": 0.01}, rows, {},
                                                      sibling_weight=0.1, sibling_cap=0.25)
    assert list(out_scores) == ["c1"] and out_rows["c1"]["parent_missing"] is True
    assert st.parents_missing == 1


def test_collapse_merges_a_parent_hit_with_its_children():
    rows = {"P1": {**_parent_row("P1"), "vector_rank": 5}, "c1": _child("c1", "P1",
                                                                          vector_rank=1)}
    out_scores, out_rows, _ = pc.collapse_to_parents(
        {"P1": 0.012, "c1": 0.016}, rows, {"P1": _parent_row("P1")}, sibling_weight=0.1,
        sibling_cap=0.25)
    assert list(out_scores) == ["P1"]
    assert [m["chunk_id"] for m in out_rows["P1"]["matched_children"]] == ["c1"]
    assert out_rows["P1"]["vector_rank"] == 1


def test_parent_ids_of_is_distinct_and_children_only():
    rows = {"c1": _child("c1", "P1"), "c2": _child("c2", "P1"), "c3": _child("c3", "P2"),
            "s1": _standalone("s1")}
    assert pc.parent_ids_of(rows) == ["P1", "P2"]


# ─────────────────────────────────────────────────────────────── retrieve() with fakes

class _Res:
    def __init__(self, rows=None, scalar=None):
        self._rows, self._scalar = rows or [], scalar

    def mappings(self):
        return self

    def all(self):
        return self._rows

    def scalar(self):
        return self._scalar


class RetrievalDB:
    """Answers TEXT_SQL / EXPIRED_SQL / PARENTS_SQL; the vector arm is patched separately."""

    def __init__(self, text_rows, parents, parents_fail=False):
        self.text_rows, self.parents, self.parents_fail = text_rows, parents, parents_fail
        self.params = []

    async def execute(self, sql, params=None):
        q = getattr(sql, "text", str(sql))
        self.params.append((q, params))
        if "ANY(CAST(:ids" in q:
            if self.parents_fail:
                raise RuntimeError("db hiccup")
            wanted = {str(i) for i in params["ids"]}
            return _Res([r for r in self.parents if str(r["id"]) in wanted])
        if "ts_rank" in q:
            return _Res(self.text_rows)
        if "count(*)" in q:
            return _Res(scalar=0)
        # R5.08 pre-filter check: direct VECTOR_SQL execute before _vector_search; return
        # empty — the patched _vector_search mock drives the actual vector results.
        if "embedding" in q or "similarity" in q or "qvec" in (params or {}):
            return _Res([])
        raise AssertionError(f"unexpected SQL: {q[:80]}")


def _settings(granularity, **over):
    base = dict(
        retrieval_top_k=8, retrieval_min_score=0.0, retrieval_vector_weight=1.0,
        retrieval_text_weight=0.8, retrieval_smalltalk_skip_enabled=False,
        retrieval_smalltalk_max_chars=120, retrieval_granularity=granularity,
        retrieval_parent_sibling_weight=0.1, retrieval_parent_sibling_cap=0.25,
        retrieval_parent_child_pool_factor=2, retrieval_pc_vector_scope="children",
        retrieval_pc_text_scope="children", retrieval_pc_fusion="parent_rank",
        retrieval_pc_parent_vector_weight=1.0,
        # R5.08 metadata enrichment settings (off for parent-child tests)
        retrieval_metadata_mode="off",
        retrieval_meta_price_boost=1.3, retrieval_meta_bhk_boost=1.2,
        # R5.01 cross-encoder rerank settings (disabled for parent-child tests)
        retrieval_rerank_enabled=False, retrieval_rerank_mode="fuse",
        retrieval_rerank_candidates=12, retrieval_rerank_debug_log=False,
        retrieval_rerank_glossary={}, retrieval_rerank_languages="all",
        retrieval_rerank_rrf_k=60, retrieval_rerank_translate=True,
        retrieval_rerank_weight=1.0, retrieval_rerank_base_weight=1.0,
        retrieval_max_chunks_per_doc=2,
        # R5.02 vector index diagnostics
        retrieval_vector_explain=False, retrieval_vector_underfill_check=True)
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture
def patched(monkeypatch):
    """Stub everything retrieve() touches outside the DB session."""
    state = {"vector_rows": [], "vector_params": None}

    async def _embed(q, session_cache=None):
        return [0.0, 1.0]

    async def _vsearch(db, s, result, params, *, pool, tenant_id):
        state["vector_params"] = params
        return state["vector_rows"]

    async def _none(*a, **k):
        return None

    async def _key(**k):
        return "k"

    state["tenant_settings"] = {}
    state["tenant_lookups"] = 0

    async def _tsettings(db, tenant_id):
        state["tenant_lookups"] += 1
        return state["tenant_settings"]

    monkeypatch.setattr(retrieval, "_embed_query", _embed)
    monkeypatch.setattr(retrieval, "_vector_search", _vsearch)
    monkeypatch.setattr(retrieval, "_cache_get", _none)
    monkeypatch.setattr(retrieval, "_cache_set", _none)
    monkeypatch.setattr(retrieval, "_cache_key", _key)
    monkeypatch.setattr(retrieval, "_tenant_settings", _tsettings)
    return state


def _vrow(cid, role, pid=None, doc="D1", content=None, title="3 BHK cost sheet"):
    return {"id": cid, "document_id": doc, "content": content or f"text {cid}",
            "metadata": {"heading": title}, "chunk_role": role, "parent_id": pid,
            "title": title, "doc_type": "pricing", "similarity": 0.5}


def test_retrieve_parent_child_mode_returns_deduped_parent_text(monkeypatch, patched):
    monkeypatch.setattr(retrieval, "get_settings", lambda: _settings("parent_child"))
    pid = "00000000-0000-0000-0000-000000000001"      # parent ids are uuids in the DB
    patched["vector_rows"] = [_vrow("c1", "child", pid), _vrow("s1", "standalone", doc="D2"),
                              _vrow("c2", "child", pid)]
    db = RetrievalDB(text_rows=[_vrow("c2", "child", pid)], parents=[_parent_row(pid)])
    res = _run(retrieval.retrieve(db, tenant_id=uuid.uuid4(), query="3 bhk total price"))
    assert patched["vector_params"]["exclude_role"] == "parent"
    assert patched["vector_params"]["limit"] == 64, "pool widened by the pool factor"
    assert [c.chunk_id for c in res.chunks] == [pid, "s1"]
    top = res.chunks[0]
    assert top.content == f"PARENT TEXT {pid}"
    assert {m["chunk_id"] for m in top.matched_children} == {"c1", "c2"}
    d = res.parent_child_diagnostics()
    assert d["granularity"] == "parent_child" and d["parents"] == 1 and d["child_hits"] == 2
    assert d["merged_siblings"] == 1 and d["parents_missing"] == 0


def test_retrieve_flat_mode_excludes_children_and_never_collapses(monkeypatch, patched):
    monkeypatch.setattr(retrieval, "get_settings", lambda: _settings("flat"))
    patched["vector_rows"] = [_vrow("P1", "parent"), _vrow("s1", "standalone", doc="D2")]
    db = RetrievalDB(text_rows=[], parents=[])
    res = _run(retrieval.retrieve(db, tenant_id=uuid.uuid4(), query="3 bhk total price"))
    assert patched["vector_params"]["exclude_role"] == "child"
    assert patched["vector_params"]["limit"] == 32
    assert not any("ANY(CAST(:ids" in q for q, _ in db.params), "flat must not fetch parents"
    assert [c.chunk_id for c in res.chunks] == ["P1", "s1"]
    assert all(c.matched_children == [] for c in res.chunks)
    assert res.parent_child_diagnostics() == {"granularity": "flat"}


def test_retrieve_parent_fetch_failure_degrades_to_children(monkeypatch, patched, caplog):
    monkeypatch.setattr(retrieval, "get_settings", lambda: _settings("parent_child"))
    patched["vector_rows"] = [_vrow("c1", "child", "00000000-0000-0000-0000-000000000009")]
    db = RetrievalDB(text_rows=[], parents=[], parents_fail=True)
    res = _run(retrieval.retrieve(db, tenant_id=uuid.uuid4(), query="3 bhk total price"))
    assert [c.chunk_id for c in res.chunks] == ["c1"], "the child is still served"
    d = res.parent_child_diagnostics()
    assert d["fetch_error"] and "db hiccup" in d["fetch_error"]
    assert any("rag.parent_expand fetch failed" in r.message for r in caplog.records)


def test_cache_signature_separates_granularity():
    common = dict(query="3 bhk price", doc_type=None, project_id=None, top_k=8)
    assert (retrieval._cache_signature(**common, granularity="flat")
            != retrieval._cache_signature(**common, granularity="parent_child"))
    assert retrieval._cache_signature(**common) == retrieval._cache_signature(
        **common, granularity="flat"), "default must stay the flat signature"


# ─────────────────────────────────────────────────────────────── ingest_document() with fakes

class _Scalars:
    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items


class _IngestResult:
    def __init__(self, one=None, items=None):
        self._one, self._items = one, items or []

    def scalar_one_or_none(self):
        return self._one

    def scalars(self):
        return _Scalars(self._items)


class IngestDB:
    def __init__(self, doc):
        self.doc, self.added, self.calls = doc, [], 0

    async def execute(self, stmt, params=None):
        self.calls += 1
        if self.calls == 1:
            return _IngestResult(one=self.doc)           # SELECT document
        return _IngestResult(items=[])                   # SELECT old chunk ids / DELETE

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass


class FakeEmbedder:
    def __init__(self, fail_texts=()):
        self.fail_texts = fail_texts

    async def embed(self, texts):
        from app.providers.embeddings.base import EmbeddingError
        if any(any(f in t for f in self.fail_texts) for t in texts):
            raise EmbeddingError("provider timeout")
        return SimpleNamespace(vectors=[[0.1, 0.2] for _ in texts], model="fake", dimensions=2)


def _ingest(monkeypatch, *, enabled, fail_texts=()):
    from app.engine.rag import ingestion
    s = SimpleNamespace(chunk_target_chars=900, chunk_overlap_chars=150,
                        chunk_list_group_size=8, chunk_min_chars=20,
                        chunk_parent_child_enabled=enabled, chunk_child_target_chars=200,
                        chunk_child_overlap_chars=60, chunk_child_list_group_size=3,
                        chunk_child_min_chars=40, chunk_child_split_min_chars=260,
                        chunk_child_title_prefix=False, chunk_child_record_split=False,
                        ingest_normalize_mode="auto",
                        ingestion_enrich_metadata=False)
    monkeypatch.setattr(ingestion, "get_settings", lambda: s)
    monkeypatch.setattr(ingestion, "get_embeddings", lambda: FakeEmbedder(fail_texts))

    async def _inv(tid):
        return None
    monkeypatch.setattr(retrieval, "invalidate_tenant_knowledge_cache", _inv)
    doc = SimpleNamespace(project_id=None, knowledge_base_id=None,
                          doc_type="pricing", title="3 BHK", status="x",
                          version=1)
    db = IngestDB(doc)
    text = "# 3 BHK cost sheet\n\n" + SHEET + "\n\n# Contact\n\nCall the sales office."
    report = _run(ingestion.ingest_document(db, tenant_id=uuid.uuid4(),
                                            document_id=uuid.uuid4(), text=text))
    return report, db.added, doc


def test_ingest_flag_off_writes_only_standalone(monkeypatch):
    report, added, doc = _ingest(monkeypatch, enabled=False)
    assert {r.chunk_role for r in added} == {"standalone"}
    assert all(r.parent_id is None for r in added)
    assert report.parents_created == report.children_created == 0
    assert report.chunks_created == len(added) and doc.status == "ready"


def test_ingest_flag_on_links_children_to_parents(monkeypatch):
    report, added, doc = _ingest(monkeypatch, enabled=True)
    parents = [r for r in added if r.chunk_role == "parent"]
    children = [r for r in added if r.chunk_role == "child"]
    assert len(parents) == 1 and len(children) >= 3
    assert all(c.parent_id == parents[0].id for c in children)
    assert all(c.content.startswith("3 BHK cost sheet\n") for c in children)
    assert parents[0].embedding is not None, "parents stay embedded so flat mode works"
    assert parents[0].meta["children"] == len(children)
    assert report.chunks_created == len(added) and doc.status == "ready"
    assert added.index(parents[0]) < min(added.index(c) for c in children), \
        "parent must be inserted before its children (FK)"


def test_ingest_parent_whose_children_all_fail_is_stored_standalone(monkeypatch):
    # The batch call fails, then the per-chunk retry succeeds only for the (long) parent text:
    # every (short) child fails. The parent must still be stored — as standalone.
    from app.engine.rag import ingestion

    class ChildFails(FakeEmbedder):
        async def embed(self, texts):
            from app.providers.embeddings.base import EmbeddingError
            if len(texts) > 1 or len(texts[0]) < 400:
                raise EmbeddingError("timeout")
            return SimpleNamespace(vectors=[[0.1] for _ in texts], model="fake", dimensions=1)

    monkeypatch.setattr(ingestion, "get_embeddings", lambda: ChildFails())
    s = SimpleNamespace(chunk_target_chars=900, chunk_overlap_chars=150,
                        chunk_list_group_size=8, chunk_min_chars=20,
                        chunk_parent_child_enabled=True, chunk_child_target_chars=200,
                        chunk_child_overlap_chars=60, chunk_child_list_group_size=3,
                        chunk_child_min_chars=40, chunk_child_split_min_chars=260,
                        chunk_child_title_prefix=False, chunk_child_record_split=False,
                        ingest_normalize_mode="auto",
                        ingestion_enrich_metadata=False)
    monkeypatch.setattr(ingestion, "get_settings", lambda: s)

    async def _inv(tid):
        return None
    monkeypatch.setattr(retrieval, "invalidate_tenant_knowledge_cache", _inv)
    doc = SimpleNamespace(project_id=None, knowledge_base_id=None,
                          doc_type="pricing", title="3 BHK", status="x",
                          version=1)
    db = IngestDB(doc)
    report = _run(ingestion.ingest_document(db, tenant_id=uuid.uuid4(), document_id=uuid.uuid4(),
                                            text="# 3 BHK cost sheet\n\n" + SHEET))
    roles = [r.chunk_role for r in db.added]
    assert "child" not in roles and "parent" not in roles
    assert roles == ["standalone"], "the sheet is still searchable, in both modes"
    assert doc.status == "partial" and report.embeddings_failed > 0


# ─────────────────────────────────────────────────────────────── parent_rank fusion + scopes

def test_scope_exclusion():
    assert pc.scope_exclusion("children") == "parent"
    assert pc.scope_exclusion("parents") == "child"
    assert pc.scope_exclusion("both") not in ("standalone", "parent", "child")
    assert pc.scope_exclusion("garbage") == "parent"


def _unit_order(scores):
    return [u for u, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def test_parent_rank_stops_one_documents_children_crowding_the_arm():
    """The measured EN027 failure: 'how many flats in TOTAL' — every cost-sheet child says
    'Total ...', so raw child hits fill the keyword arm's top ranks and the one chunk that
    answers ('900 flats', a standalone) is pushed down by as many places."""
    parents = {"SHEET": _parent_row("SHEET")}
    text = [_child(f"c{i}", "SHEET") for i in range(6)] + [_standalone("project")]
    vector = [_standalone("project"), _child("c0", "SHEET")]
    # child_rank (old): fuse raw hits, collapse after — 'project' is text rank 7.
    raw_scores, raw_rows = {}, {}
    for rank, r in enumerate(vector, 1):
        raw_rows.setdefault(r["id"], dict(r))["vector_rank"] = rank
        raw_scores[r["id"]] = raw_scores.get(r["id"], 0) + 1.0 / (60 + rank)
    for rank, r in enumerate(text, 1):
        raw_rows.setdefault(r["id"], dict(r))["text_rank"] = rank
        raw_scores[r["id"]] = raw_scores.get(r["id"], 0) + 0.8 / (60 + rank)
    old_scores, _, _ = pc.collapse_to_parents(raw_scores, raw_rows, parents,
                                              sibling_weight=0.0, sibling_cap=0.25)
    # parent_rank (new): the sheet holds ONE rank per arm, so 'project' is text rank 2.
    new_scores, new_rows, st = pc.fuse_parent_rank(
        [("vector", vector, 1.0), ("text", text, 0.8)], parents, rrf_k=60)
    assert new_rows["project"]["text_rank"] == 2 and new_rows["SHEET"]["text_rank"] == 1
    assert _unit_order(new_scores)[0] == "project", "vector #1 + text #2 must now win"
    assert new_scores["project"] - new_scores["SHEET"] > old_scores["project"] - old_scores["SHEET"]
    assert st.merged_siblings == 5 and st.parents == 1 and st.child_hits == 6
    assert len(new_rows["SHEET"]["matched_children"]) == 6


def test_parent_rank_dense_ranks_and_trace():
    parents = {"P1": _parent_row("P1"), "P2": _parent_row("P2")}
    vector = [_child("a", "P1"), _child("b", "P1"), _child("c", "P2"), _standalone("s")]
    scores, rows, st = pc.fuse_parent_rank([("vector", vector, 1.0)], parents, rrf_k=60)
    assert [rows[u]["vector_rank"] for u in ("P1", "P2", "s")] == [1, 2, 3]
    assert scores["P1"] == pytest.approx(1 / 61) and scores["s"] == pytest.approx(1 / 63)
    kids = rows["P1"]["matched_children"]
    assert [k["chunk_id"] for k in kids] == ["a", "b"]
    assert [k["vector_rank"] for k in kids] == [1, 2], "children keep their RAW arm rank"
    assert rows["P1"]["content"] == "PARENT TEXT P1"


def test_parent_rank_parent_hit_and_child_hit_share_one_unit():
    parents = {"P1": _parent_row("P1")}
    vector = [{**_parent_row("P1")}, _child("a", "P1")]          # scope "both"
    scores, rows, st = pc.fuse_parent_rank([("vector", vector, 1.0)], parents, rrf_k=60)
    assert list(scores) == ["P1"] and rows["P1"]["vector_rank"] == 1
    assert st.merged_siblings == 1


def test_parent_rank_orphan_child_is_its_own_unit():
    scores, rows, st = pc.fuse_parent_rank([("text", [_child("a", "GONE")], 0.8)], {},
                                           rrf_k=60)
    assert list(scores) == ["a"] and rows["a"]["parent_missing"] is True
    assert st.parents_missing == 1 and st.parents == 0


def test_retrieve_uses_per_arm_scopes(monkeypatch, patched):
    monkeypatch.setattr(retrieval, "get_settings", lambda: _settings(
        "parent_child", retrieval_pc_vector_scope="both", retrieval_pc_text_scope="parents"))
    patched["vector_rows"] = [_vrow("s1", "standalone")]
    db = RetrievalDB(text_rows=[], parents=[])
    res = _run(retrieval.retrieve(db, tenant_id=uuid.uuid4(), query="how many flats in total"))
    assert patched["vector_params"]["exclude_role"] == pc.scope_exclusion("both")
    text_params = next(p for q, p in db.params if "ts_rank" in q)
    assert text_params["exclude_role"] == "child", "keyword arm searches parents only"
    d = res.parent_child_diagnostics()
    assert (d["fusion"], d["vector_scope"], d["text_scope"]) == ("parent_rank", "both", "parents")


def test_retrieve_child_rank_fusion_still_available(monkeypatch, patched):
    monkeypatch.setattr(retrieval, "get_settings",
                        lambda: _settings("parent_child", retrieval_pc_fusion="child_rank"))
    pid = "00000000-0000-0000-0000-000000000001"
    patched["vector_rows"] = [_vrow("c1", "child", pid), _vrow("c2", "child", pid)]
    db = RetrievalDB(text_rows=[], parents=[_parent_row(pid)])
    res = _run(retrieval.retrieve(db, tenant_id=uuid.uuid4(), query="3 bhk total price"))
    assert [c.chunk_id for c in res.chunks] == [pid]
    assert res.parent_child_diagnostics()["fusion"] == "child_rank"


def test_cache_key_mode_signature_includes_scopes_and_fusion():
    common = dict(query="q", doc_type=None, project_id=None, top_k=8)
    a = retrieval._cache_signature(**common, granularity="parent_child:v=children:t=children:f=parent_rank")
    b = retrieval._cache_signature(**common, granularity="parent_child:v=both:t=parents:f=parent_rank")
    assert a != b


# ─────────────────────────────────────────────────────────────── split options (fix 3)

def test_split_min_chars_threshold():
    body = "The project is 10.10 acres off Sarjapur Road in Bengaluru. "            "900 flats across 4 towers and 8 wings. One basement plus ground plus 23 floors. "            "Possession is expected in December 2028 for all towers."   # ~220 chars
    parent = _parent(body, heading="What we offer > The project")
    assert split_children(parent, target_chars=200) == [], "default 1.3 x target keeps it whole"
    kids = split_children(parent, target_chars=120, split_min_chars=150, min_chars=30)
    assert len(kids) >= 2 and uncovered_text(parent, kids) == []
    assert any(k.content.startswith("900 flats") or "900 flats" in k.content for k in kids)


def test_title_prefix_added_once():
    kids = split_children(_parent(SHEET, heading="Rates"), target_chars=200, min_chars=40,
                          title="3 BHK Cost Sheet")
    assert all(k.heading == "3 BHK Cost Sheet > Rates" for k in kids)
    assert all(k.metadata["heading_path"] == "Rates" for k in kids), "parent path unchanged"
    dup = split_children(_parent(SHEET, heading="3 BHK cost sheet"), target_chars=200,
                         min_chars=40, title="3 BHK Cost Sheet")
    assert all(k.heading == "3 BHK cost sheet" for k in dup), "no duplicate when contained"


def test_record_split_keeps_every_record_whole_without_overlap():
    import re as _re
    parent = _parent(LONG_PARA)
    recs = _re.findall(r"\d[\d,]* sq ft,.*?Total [\d,]+\.", LONG_PARA)
    assert len(recs) == 8
    kids = split_children(parent, target_chars=150, overlap_chars=40, min_chars=30,
                          record_split=True)
    assert uncovered_text(parent, kids) == []
    for r in recs:
        assert sum(r in k.content for k in kids) == 1, "each record whole in exactly ONE child"
    assert all(_re.match(r"\d", k.content) for k in kids), "children start at a record"


def test_record_split_ignores_ordinary_prose():
    prose = ("Rate 11,499 rupees per square foot on super built-up area. 3 BHK is the largest. "
             "Clubhouse charges are 3,25,000 on every unit, and car parking is 5,50,000 on the "
             "two smaller layouts. Possession is December 2028.")
    a = split_children(_parent(prose), target_chars=100, min_chars=20)
    b = split_children(_parent(prose), target_chars=100, min_chars=20, record_split=True)
    assert [k.content for k in a] == [k.content for k in b]


def test_child_split_kwargs_single_source():
    s = SimpleNamespace(chunk_child_target_chars=200, chunk_child_overlap_chars=100,
                        chunk_child_list_group_size=3, chunk_child_min_chars=40,
                        chunk_child_split_min_chars=180, chunk_child_title_prefix=True,
                        chunk_child_record_split=True)
    kw = pc.child_split_kwargs(s, title="Doc")
    assert kw["split_min_chars"] == 180 and kw["title"] == "Doc" and kw["record_split"]
    s.chunk_child_title_prefix = False
    assert pc.child_split_kwargs(s, title="Doc")["title"] is None


def test_retrieve_separate_scope_runs_parent_vector_list(monkeypatch, patched):
    """vector scope 'separate': children and parents are searched as TWO lists, and a section
    gets an RRF vote from each (its best child + its whole-section vector)."""
    calls = []
    pid = "00000000-0000-0000-0000-000000000001"

    async def _vsearch(db, s, result, params, *, pool, tenant_id):
        calls.append(params["exclude_role"])
        if params["exclude_role"] == "parent":                 # the children list
            return [_vrow("c1", "child", pid), _vrow("s1", "standalone", doc="D2")]
        return [{**_parent_row(pid), "similarity": 0.4}, _vrow("s1", "standalone", doc="D2")]

    monkeypatch.setattr(retrieval, "_vector_search", _vsearch)
    monkeypatch.setattr(retrieval, "get_settings", lambda: _settings(
        "parent_child", retrieval_pc_vector_scope="separate", retrieval_pc_text_scope="both",
        retrieval_pc_parent_vector_weight=1.0))
    db = RetrievalDB(text_rows=[], parents=[_parent_row(pid)])
    res = _run(retrieval.retrieve(db, tenant_id=uuid.uuid4(), query="how many flats in total"))
    assert calls == ["parent", "child"], "children list first, then the parents list"
    assert [c.chunk_id for c in res.chunks] == [pid, "s1"]
    # parent: 1/61 (child list #1) + 1/61 (parent list #1); s1: 1/62 + 1/62
    assert res.chunks[0].score == pytest.approx(2 / 61)
    assert [m["chunk_id"] for m in res.chunks[0].matched_children] == ["c1"]



# ─────────────────────────────────────────────────────────────── per-tenant rollout

def test_tenant_granularity_override_enables_parent_child(monkeypatch, patched):
    monkeypatch.setattr(retrieval, "get_settings", lambda: _settings("flat"))
    patched["tenant_settings"] = {"retrieval_granularity": "parent_child"}
    patched["vector_rows"] = [_vrow("s1", "standalone")]
    res = _run(retrieval.retrieve(RetrievalDB(text_rows=[], parents=[]), tenant_id=uuid.uuid4(),
                                  query="3 bhk total price"))
    assert res.parent_child_diagnostics()["granularity"] == "parent_child"
    assert patched["tenant_lookups"] == 1, "settings read ONCE for granularity + synonyms"


def test_invalid_tenant_override_falls_back_to_global(monkeypatch, patched, caplog):
    monkeypatch.setattr(retrieval, "get_settings", lambda: _settings("flat"))
    patched["tenant_settings"] = {"retrieval_granularity": "parent-child"}      # typo
    patched["vector_rows"] = [_vrow("s1", "standalone")]
    res = _run(retrieval.retrieve(RetrievalDB(text_rows=[], parents=[]), tenant_id=uuid.uuid4(),
                                  query="3 bhk total price"))
    assert res.parent_child_diagnostics()["granularity"] == "flat"
    assert any("invalid tenant override" in r.message for r in caplog.records)


def test_explicit_granularity_and_synonyms_need_no_tenant_query(monkeypatch, patched):
    """The planner path: turn context already resolved both — the hot path adds no query."""
    monkeypatch.setattr(retrieval, "get_settings", lambda: _settings("flat"))
    patched["vector_rows"] = [_vrow("s1", "standalone")]
    res = _run(retrieval.retrieve(RetrievalDB(text_rows=[], parents=[]), tenant_id=uuid.uuid4(),
                                  query="3 bhk total price", synonyms=[],
                                  granularity="parent_child"))
    assert patched["tenant_lookups"] == 0
    assert res.parent_child_diagnostics()["granularity"] == "parent_child"


def test_resolve_granularity():
    assert pc.resolve_granularity(None, "flat") == "flat"
    assert pc.resolve_granularity("", "parent_child") == "parent_child"
    assert pc.resolve_granularity("parent_child", "flat") == "parent_child"
    assert pc.resolve_granularity(42, "flat") == "flat"


def test_parent_child_enabled_for_tenant():
    from app.engine.rag.ingestion import parent_child_enabled_for
    s = SimpleNamespace(chunk_parent_child_enabled=False)
    assert parent_child_enabled_for(s, {}) is False
    assert parent_child_enabled_for(s, {"chunk_parent_child_enabled": True}) is True
    assert parent_child_enabled_for(SimpleNamespace(chunk_parent_child_enabled=True),
                                    {"chunk_parent_child_enabled": False}) is False
    assert parent_child_enabled_for(s, {"chunk_parent_child_enabled": "yes"}) is False, \
        "only a real bool overrides"


# ─────────────────────────────────────────────────────────────── resilient embedding

class CountingEmbedder:
    """Fails any request containing a text with "BAD"; counts requests."""
    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        from app.providers.embeddings.base import EmbeddingError
        self.calls.append(len(texts))
        if any("BAD" in t for t in texts):
            raise EmbeddingError("timeout")
        return SimpleNamespace(vectors=[[0.1] for _ in texts], model="m", dimensions=1)


def test_embed_resilient_retries_only_the_failed_batch():
    from app.engine.rag.ingestion import embed_resilient
    texts = [f"t{i}" for i in range(10)] + ["BAD"] + [f"u{i}" for i in range(9)]   # 20 texts
    emb = CountingEmbedder()
    res = _run(embed_resilient(emb, texts, batch_size=5))
    # batches: [0-4] ok, [5-9] ok, [10-14] FAILS -> 5 single retries, [15-19] ok
    assert emb.calls == [5, 5, 5, 1, 1, 1, 1, 1, 5]
    assert res.failed == 1 and res.retried_batches == 1 and len(res.vectors) == 20
    assert res.vectors[10] is None and all(v is not None for i, v in enumerate(res.vectors)
                                           if i != 10)
    assert res.model == "m"


def test_embed_resilient_all_fail():
    from app.engine.rag.ingestion import embed_resilient
    res = _run(embed_resilient(CountingEmbedder(), ["BAD1", "BAD2"], batch_size=5))
    assert res.failed == 2 and res.vectors == [None, None] and res.last_error


def test_ingest_normalizes_extracted_text_and_reports_it(monkeypatch):
    from app.engine.rag import ingestion
    s = SimpleNamespace(chunk_target_chars=900, chunk_overlap_chars=150,
                        chunk_list_group_size=8, chunk_min_chars=20,
                        chunk_parent_child_enabled=False, chunk_child_target_chars=200,
                        chunk_child_overlap_chars=60, chunk_child_list_group_size=3,
                        chunk_child_min_chars=40, chunk_child_split_min_chars=260,
                        chunk_child_title_prefix=False, chunk_child_record_split=False,
                        ingest_normalize_mode="auto",
                        ingestion_enrich_metadata=False)
    monkeypatch.setattr(ingestion, "get_settings", lambda: s)
    monkeypatch.setattr(ingestion, "get_embeddings", lambda: FakeEmbedder())

    async def _inv(tid):
        return None
    monkeypatch.setattr(retrieval, "invalidate_tenant_knowledge_cache", _inv)
    doc = SimpleNamespace(project_id=None, knowledge_base_id=None,
                          doc_type="pricing", title="Price list", status="x",
                          version=1)
    db = IngestDB(doc)
    header = "Sunrise Meadows Brochure        Page {}"
    pages = [
        header.format(1) + "\n\nPRICE LIST\nType     Carpet     Price\n"
        "2 BHK    690 sq ft  89,50,000\n3 BHK    980 sq ft  1,26,00,000\n",
        header.format(2) + "\n\nThe project has 1,100 homes across six towers.\n",
        header.format(3) + "\n\nPossession is planned for March 2029.\n"]
    text = "\f".join(pages)                        # form feed = extractor page break
    report = _run(ingestion.ingest_document(db, tenant_id=uuid.uuid4(),
                                            document_id=uuid.uuid4(), text=text))
    n = report.normalization
    assert n["detected_extracted"] and n["boilerplate_lines_removed"] == 3
    assert n["tables_converted"] == 1
    contents = " ".join(r.content for r in db.added)
    assert "Page 2" not in contents and "89,50,000" in contents and "| Type |" in contents
