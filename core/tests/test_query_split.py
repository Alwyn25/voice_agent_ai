"""Compound-question splitting (app/engine/rag/query_split.py) and the merge that keeps every
sub-question represented in the window (retrieval._merge_part_results). Pure, no DB.

    cd core && python -m pytest tests/test_query_split.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.rag.query_split import split_compound_query  # noqa: E402


def test_reported_compound_question_is_split():
    assert split_compound_query(
        "What is the per square foot rate for the 3 BHK, and how far is the project from Wipro?"
    ) == ["What is the per square foot rate for the 3 BHK", "how far is the project from Wipro?"]


def test_two_questions_in_one_breath_and_hinglish():
    assert split_compound_query("What's the price of the 3 BHK? Is there a gym?") == \
        ["What's the price of the 3 BHK?", "Is there a gym?"]
    assert split_compound_query("3 BHK ka price kya hai aur kitna door hai Wipro se") == \
        ["3 BHK ka price kya hai", "kitna door hai Wipro se"]


def test_one_question_with_two_subjects_is_not_split():
    # A wrong split strips the subject off a half — far worse than not splitting.
    for q in ("What is the price of the 2 BHK and the 3 BHK?",
              "Is there a swimming pool and a gym?",
              "What are the sizes and prices for 3 BHK?",
              "Can you send the floor plan and the brochure?",
              "I want a 3 BHK and I can pay up to 1.6 crore"):
        assert split_compound_query(q) == [q], q


def test_part_count_is_capped():
    q = "What is the rate? Is there a gym? How far is Wipro? When is possession?"
    assert len(split_compound_query(q, max_parts=3)) == 3


def test_merge_takes_turns_and_caps_per_document():
    from app.engine.rag.retrieval import (RetrievalResult, RetrievedChunk,
                                          _merge_part_results)

    def chunk(cid, doc):
        return RetrievedChunk(chunk_id=cid, document_id=doc, title=doc, doc_type="x",
                              heading=None, content=cid, score=1.0)

    # Part 1 (price) has eight pricing chunks; part 2 (distance) has the one Location chunk.
    price = RetrievalResult(query="rate", chunks=[chunk(f"p{i}", f"pricing{i % 4}")
                                                  for i in range(8)])
    where = RetrievalResult(query="wipro", chunks=[chunk("loc", "location"),
                                                   chunk("p0", "pricing0")])
    merged = _merge_part_results("rate and wipro", ["rate", "wipro"], [price, where],
                                 top_k=8, max_per_doc=2)
    ids = [c.chunk_id for c in merged.chunks]
    assert ids[:2] == ["p0", "loc"]          # each question's best evidence leads
    assert len(ids) == 8 and len(set(ids)) == 8
    assert merged.metadata["split"]["parts"] == ["rate", "wipro"]


def test_selftest_still_passes():
    from app.engine.rag import query_split
    query_split._selftest()
