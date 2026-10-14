"""Unit tests for scripts/build_retrieval_qrels.py — the draft eval-set generator (no LLM, no DB).

What must hold:
  * an anchor is kept only if it is a literal (normalised) substring of its source chunk, short,
    and not a duplicate — the LLM can never invent evidence;
  * sampling spreads across documents (one long document cannot dominate) and is deterministic;
  * the qrels file loader rejects malformed files (no items, duplicate ids, bad evidence shape)
    and defaults version/reviewed.

Run: cd core && python -m pytest tests/test_qrels_builder.py -q
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest  # noqa: E402

from build_retrieval_qrels import load_qrels_file, sample_chunks, valid_anchors  # noqa: E402

CHUNK = ("3 BHK cost sheet\nRate 11,499 rupees per square foot on super built-up area. "
         "1,390 sq ft, East and West facing. Total 1,72,06,110.")


def test_valid_anchors_are_verbatim_only():
    got = valid_anchors(["Total 1,72,06,110", "total 1,72,06,110", "Rs 1.72 crore",
                         "11,499 rupees per square foot", "ab", 42, "x" * 80], CHUNK)
    assert got == ["Total 1,72,06,110", "total 1,72,06,110", "11,499 rupees per square foot"]
    assert valid_anchors("not a list", CHUNK) == []


def test_sample_chunks_round_robin_and_deterministic():
    rows = ([{"id": f"a{i}", "document_id": "A", "content": "x" * 200, "title": "A"}
             for i in range(20)]
            + [{"id": "b0", "document_id": "B", "content": "y" * 200, "title": "B"},
               {"id": "c0", "document_id": "C", "content": "z" * 200, "title": "C"},
               {"id": "short", "document_id": "D", "content": "too short", "title": "D"}])
    picked = sample_chunks(rows, 5, seed=1)
    docs = [p["document_id"] for p in picked]
    assert {"A", "B", "C"} <= set(docs) and "D" not in docs
    assert [p["id"] for p in picked] == [p["id"] for p in sample_chunks(rows, 5, seed=1)]


def _write(tmp_path, data):
    p = tmp_path / "q.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


def test_load_qrels_file_validates(tmp_path):
    ok = load_qrels_file(_write(tmp_path, {"items": [
        {"id": "G1", "question": "q?", "evidence": [["a", "b"]]},
        {"id": "G2", "question": "escalation?", "evidence": None}]}))
    assert ok["reviewed"] is False and ok["version"] == "q"
    for bad in ({"items": []},
                {"items": [{"id": "G1", "question": "q"}, {"id": "G1", "question": "r"}]},
                {"items": [{"id": "G1", "question": "q", "evidence": ["flat"]}]},
                {"items": [{"id": "G1", "question": "q", "evidence": [[]]}]},
                {"items": [{"question": "q"}]}):
        with pytest.raises(ValueError):
            load_qrels_file(_write(tmp_path, bad))
