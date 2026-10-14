"""Weekday/date consistency guard (app/engine/planner/date_claim.py): the pure
regex + calendar-math logic, which needs no DB. See the module docstring for the real
call ("Monday, 10 October" for an actual Saturday) that motivated it.

    cd core && python -m pytest tests/test_date_claim.py -q
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.planner import date_claim as dc  # noqa: E402


def test_wrong_weekday_is_corrected():
    fixed, notes = dc.correct_weekday_date_claims(
        "Would 5 PM on Monday, 10 October work for your evening visit?",
        today=date(2026, 10, 8))
    assert fixed == "Would 5 PM on Saturday, 10 October work for your evening visit?"
    assert len(notes) == 1


def test_correct_weekday_is_left_untouched():
    text = "Would you prefer 4 PM or 5 PM on Saturday, 10 October?"
    assert dc.correct_weekday_date_claims(text, today=date(2026, 10, 8)) == (text, [])


def test_no_date_pair_is_left_untouched():
    text = "Shall I book your visit for tomorrow at 10:00 AM?"
    assert dc.correct_weekday_date_claims(text, today=date(2026, 10, 8)) == (text, [])


def test_abbreviated_month_and_explicit_year_still_caught():
    fixed, notes = dc.correct_weekday_date_claims(
        "Your visit is confirmed for Tuesday 10 Oct 2026 at 5 PM.", today=date(2026, 10, 8))
    assert fixed == "Your visit is confirmed for Saturday 10 Oct 2026 at 5 PM."
    assert notes


def test_no_year_rolls_to_next_occurrence():
    # 10 October has already passed in the stated year by 1 November -- the next real
    # occurrence is the following year, and that year's weekday is what gets checked.
    fixed, _ = dc.correct_weekday_date_claims(
        "Would Wednesday, 10 October work?", today=date(2027, 11, 1))
    assert fixed == "Would Tuesday, 10 October work?"


def test_empty_and_none_text_are_safe():
    assert dc.correct_weekday_date_claims("") == ("", [])
    assert dc.correct_weekday_date_claims(None) == (None, [])


def test_selftest_still_passes():
    """The module's own self-test (python -m app.engine.planner.date_claim) covers the same
    ground; running it here means a regression fails pytest too, not just the manual
    invocation."""
    dc._selftest()  # raises AssertionError on failure
