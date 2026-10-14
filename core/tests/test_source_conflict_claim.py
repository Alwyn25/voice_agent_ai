"""Internal-document-conflict guard (app/engine/planner/source_conflict_claim.py) and the prompt
rule it backs. Pure regex logic, no DB. See the module docstring for the real call ("a slight
size-range discrepancy between the brochure and cost sheet") that motivated it.

    cd core && python -m pytest tests/test_source_conflict_claim.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.planner import source_conflict_claim as scc  # noqa: E402


def test_reported_call_first_turn_keeps_the_answer_drops_the_confession():
    reply = ("We have 3 BHK units ranging from about 1,239 to 1,560 sq ft, with prices starting "
             "at roughly 1.54 crore. There's a slight size-range discrepancy between the brochure "
             "and cost sheet, so our sales team can confirm the exact details.")
    assert scc.scrub(reply, asked="What 3 BHK sizes do you have?") == (
        "We have 3 BHK units ranging from about 1,239 to 1,560 sq ft, with prices starting "
        "at roughly 1.54 crore.")


def test_reported_call_follow_up_is_entirely_a_leak():
    reply = ("The brochure lists 3 BHK sizes between 1,239 and 1,422 sq ft, but the cost sheet "
             "shows a 1,560 sq ft unit, which falls outside that range. Our sales team can "
             "confirm the exact figures.")
    out = scc.scrub(reply, asked="sorry what is the discrepancy")
    assert "brochure" not in out and "1,422" not in out and "outside" not in out


def test_pushback_replay_keeps_the_correct_figure():
    assert scc.scrub(
        "The brochure might be outdated; the largest 3 BHK available is indeed 1,560 square feet.",
        asked="The brochure says 1,422, why 1,560?") == \
        "The largest 3 BHK available is indeed 1,560 square feet."


def test_confession_without_a_document_only_counts_when_the_caller_raised_it():
    reply = "Yes, there are differences. The largest 3 BHK is 1,560 square feet."
    assert scc.scrub(reply, asked="any difference between the brochure and cost sheet?") == \
        "The largest 3 BHK is 1,560 square feet."
    # Same words, ordinary context: a scheduling "conflict" is not a document leak.
    ok = "You already have a visit that day, so there's a conflict — shall I move it?"
    assert scc.scrub(ok, asked="book me for Saturday") == ok


def test_offering_a_document_is_not_a_leak():
    for ok in ("I can send you the brochure on WhatsApp right now.",
               "The cost sheet total includes car parking and clubhouse charges.",
               "Prices differ by floor, from the 4th floor onwards."):
        assert not scc.is_source_conflict_leak(ok, asked="send me the brochure"), ok


def test_fallback_is_language_aware():
    assert scc.honest_fallback(language="hinglish") == scc.FALLBACK["hinglish"]
    assert scc.honest_fallback(language="unknown") == scc.FALLBACK["english"]
    for line in scc.FALLBACK.values():
        assert not scc.is_source_conflict_leak(line)


def test_planner_and_responder_get_the_same_rule():
    """The first fix put this rule in the planner only; the responder writes what the caller
    hears and never saw it. Both prompts must carry the one shared string."""
    from app.engine.planner import prompts, responder
    assert scc.PROMPT_RULE in prompts.BASE_SYSTEM
    assert scc.PROMPT_RULE in responder.RESPONDER_SYSTEM
    # The engine prompt is shared by every tenant: no one tenant's figures in it.
    assert "1,560" not in prompts.BASE_SYSTEM and "1,422" not in prompts.BASE_SYSTEM


def test_selftest_still_passes():
    scc._selftest()
