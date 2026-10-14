"""Confidence banding (planner.banded_action): a read-only lookup is never demoted to a
"confirm" read-back. A noisy-audio voice turn (STT penalty -> "medium") used to turn the
locate_centre -> search_slots chain into "confirm" after one step, so the caller never heard
a slot before their next "okay" booked one.

    cd core && python -m pytest tests/test_confidence_banding.py -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.planner.planner import banded_action  # noqa: E402
from app.engine.tools.types import READ_ONLY_TOOLS  # noqa: E402


@pytest.mark.parametrize("tool", sorted(READ_ONLY_TOOLS))
def test_medium_band_never_demotes_a_lookup(tool):
    assert banded_action("invoke_tool", tool, "medium") == "invoke_tool"


@pytest.mark.parametrize("tool", ["book_appointment", "reschedule_appointment",
                                  "cancel_appointment", "schedule_callback",
                                  "record_do_not_contact"])
def test_medium_band_still_reads_back_a_write(tool):
    assert banded_action("invoke_tool", tool, "medium") == "confirm"


@pytest.mark.parametrize("tool", ["search_slots", "book_appointment"])
def test_low_band_escalates_any_tool(tool):
    assert banded_action("invoke_tool", tool, "low") == "escalate"


@pytest.mark.parametrize("tool", ["schedule_callback", "record_customer_fact",
                                  "book_appointment"])
def test_low_stt_reads_back_a_write_even_at_high_band(tool):
    # The engine-level replacement for DSR's low_speech_confidence rule: never ACT on words
    # that may have been misheard -- read the action back first.
    assert banded_action("invoke_tool", tool, "high", stt_confidence=0.45) == "confirm"
    assert banded_action("invoke_tool", tool, "high", stt_confidence=0.90) == "invoke_tool"
    assert banded_action("invoke_tool", tool, "high", stt_confidence=None) == "invoke_tool"


@pytest.mark.parametrize("tool", sorted(READ_ONLY_TOOLS))
def test_low_stt_never_holds_back_a_lookup(tool):
    # "11" heard correctly at 0.32 must still get its slot search -- what DSR's
    # very_low_speech_no_actions rule used to block.
    assert banded_action("invoke_tool", tool, "high", stt_confidence=0.32) == "invoke_tool"
    assert banded_action("invoke_tool", tool, "medium", stt_confidence=0.32) == "invoke_tool"


def test_low_stt_readback_can_be_disabled(monkeypatch):
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "conf_stt_readback_below", 0.0)
    assert banded_action("invoke_tool", "schedule_callback", "high",
                         stt_confidence=0.2) == "invoke_tool"


def test_high_band_and_non_tool_actions_are_untouched():
    assert banded_action("invoke_tool", "book_appointment", "high") == "invoke_tool"
    for action in ("respond", "clarify", "confirm", "escalate", "end_call"):
        assert banded_action(action, None, "medium") == action
        assert banded_action(action, None, "low") == action
