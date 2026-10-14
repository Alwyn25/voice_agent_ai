"""Barge-in carry-over (app/engine/voice/barge_in.py): a turn cancelled by the caller
speaking again must not lose the caller's words. Pure logic -- no pipecat, no DB.

    cd core && python -m pytest tests/test_barge_in.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.voice import barge_in as bi  # noqa: E402


def test_nothing_carried_runs_text_unchanged():
    assert bi.merge(None, "tomorrow morning", 0.9, max_age_s=30, now=100.0) == (
        "tomorrow morning", 0.9, "none")


def test_cancelled_request_is_prepended_to_the_follow_up():
    # The core bug: "book a site visit" was cancelled by "tomorrow morning" and lost.
    c = bi.carry(None, "I want to book a site visit", 0.92, now=100.0)
    text, conf, outcome = bi.merge(c, "tomorrow morning", 0.88, max_age_s=30, now=105.0)
    assert outcome == "merged"
    assert text == "I want to book a site visit tomorrow morning"
    assert conf == 0.88  # the lower of the two, so the low-STT guardrail still sees it


def test_check_in_during_thinking_keeps_the_real_request():
    c = bi.carry(None, "when is the possession", 0.9, now=10.0)
    text, _, outcome = bi.merge(c, "hello?", 0.7, max_age_s=30, now=14.0)
    assert outcome == "merged" and text.startswith("when is the possession")


def test_repeat_of_the_same_words_is_not_duplicated():
    c = bi.carry(None, "book a visit tomorrow", 0.9, now=0.0)
    assert bi.merge(c, "Book a visit, tomorrow!", 0.9, max_age_s=30, now=3.0)[2] == "repeated"


def test_repeat_check_matches_whole_words_only():
    # A carried "no" must not count as repeated inside "i know".
    c = bi.carry(None, "no", 0.9, now=0.0)
    text, _, outcome = bi.merge(c, "i know the price", 0.9, max_age_s=30, now=1.0)
    assert outcome == "merged" and text == "no i know the price"


def test_stale_carry_over_is_dropped():
    c = bi.carry(None, "book a visit", 0.9, now=0.0)
    assert bi.merge(c, "what is the price", 0.9, max_age_s=30, now=31.0) == (
        "what is the price", 0.9, "expired")


def test_chained_cancellations_age_from_the_first():
    first = bi.carry(None, "book a visit", 0.9, now=0.0)
    merged_text, conf, _ = bi.merge(first, "tomorrow", 0.9, max_age_s=30, now=10.0)
    # The merged turn is cancelled too: its stamp stays at the FIRST cancellation.
    second = bi.carry(first, merged_text, conf, now=20.0)
    assert second.since == 0.0 and second.text == "book a visit tomorrow"
    assert bi.merge(second, "morning", 0.9, max_age_s=30, now=31.0)[2] == "expired"


def test_missing_confidence_on_either_side():
    c = bi.carry(None, "book a visit", None, now=0.0)
    assert bi.merge(c, "tomorrow", None, max_age_s=30, now=1.0)[1] is None
    assert bi.merge(c, "tomorrow", 0.8, max_age_s=30, now=1.0)[1] == 0.8
