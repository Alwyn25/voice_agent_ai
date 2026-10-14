"""Chunked phone-number capture mode (app/engine/planner/phone_capture.py): pure state
machine, no DB or LLM. See the module docstring for what it owns vs. the real-time
audio/VAD behaviour it deliberately does not (and cannot, from here).

    cd core && python -m pytest tests/test_phone_capture.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.planner import phone_capture as pc  # noqa: E402


def test_three_chunk_number_confirmed():
    s = pc.start()
    r = pc.handle_turn(s, "98450")
    assert r.reply == "9-8-4-5-0"
    r = pc.handle_turn(r.state, "1 2")
    assert r.reply == "1-2"
    r = pc.handle_turn(r.state, "3 6 7")
    assert r.reply == "Is 98450 12367 correct — yes or no?"
    r = pc.handle_turn(r.state, "yes")
    assert r.done and r.phone == "9845012367"


def test_double_and_triple_normalisation():
    s = pc.start()
    r = pc.handle_turn(s, "nine eight four double five")
    assert r.state.digits == "98455"
    r = pc.handle_turn(r.state, "triple zero")
    assert r.state.digits == "98455000"


def test_hindi_digit_words():
    s = pc.start()
    r = pc.handle_turn(s, "nau aath char ek")
    assert r.state.digits == "9841"


def test_correction_replaces_last_chunk_not_appends():
    s = pc.start()
    r = pc.handle_turn(s, "98450")
    r = pc.handle_turn(r.state, "999")
    r = pc.handle_turn(r.state, "no, it's 12367")
    assert r.state.digits == "9845012367"
    assert r.reply == "Is 98450 12367 correct — yes or no?"


def test_correction_mid_capture_reads_back_corrected_chunk():
    s = pc.start()
    r = pc.handle_turn(s, "98450")
    r = pc.handle_turn(r.state, "999")
    r = pc.handle_turn(r.state, "no, it's 123")
    assert r.state.digits == "98450123"
    assert r.reply == "Got it, 1-2-3"


def test_bare_correction_with_no_digits_pops_and_waits():
    s = pc.start()
    r = pc.handle_turn(s, "98450")
    r = pc.handle_turn(r.state, "no")
    assert r.state.digits == ""
    assert "go ahead" in r.reply.lower()


def test_overflow_resets_cleanly():
    s = pc.start()
    r = pc.handle_turn(s, "984501236")
    assert not r.state.awaiting_confirm
    r = pc.handle_turn(r.state, "789")
    assert r.state.digits == ""
    assert "again from the start" in r.reply


def test_confirm_no_once_recollects_twice_hands_off():
    s = pc.start()
    r = pc.handle_turn(s, "9845012367")
    assert r.reply == "Is 98450 12367 correct — yes or no?"
    r = pc.handle_turn(r.state, "no")
    assert not r.done and not r.handed_off and r.state.digits == ""
    r = pc.handle_turn(r.state, "9845012367")
    r = pc.handle_turn(r.state, "no")
    assert r.handed_off and "WhatsApp or SMS" in r.reply


def test_same_number_im_calling_from_still_gets_confirmed():
    s = pc.start()
    r = pc.handle_turn(s, "same number I'm calling from", caller_id_phone="+919845012367")
    assert r.reply == "Is 98450 12367 correct — yes or no?"
    assert not r.done
    r = pc.handle_turn(r.state, "yes")
    assert r.done and r.phone == "9845012367"


def test_fillers_are_not_digits_or_corrections():
    s = pc.start()
    r = pc.handle_turn(s, "haan okay")
    assert r.state.digits == ""
    assert not r.done


def test_nudge_names_what_is_in_hand_so_far():
    s = pc.start()
    r = pc.handle_turn(s, "98450")
    r = pc.handle_turn(r.state, "umm okay")
    assert "98450" in r.reply and "so far" in r.reply


def test_natural_affirm_phrasing_confirms():
    s = pc.start()
    r = pc.handle_turn(s, "9845012367")
    r = pc.handle_turn(r.state, "yes it is correct")
    assert r.done and r.phone == "9845012367"


def test_natural_deny_phrasing_recollects():
    s = pc.start()
    r = pc.handle_turn(s, "9845012367")
    r = pc.handle_turn(r.state, "no it's wrong")
    assert not r.done and not r.handed_off and r.state.digits == ""


def test_deny_with_digits_uses_the_correction_not_a_bare_reset():
    s = pc.start()
    r = pc.handle_turn(s, "9845012367")
    r = pc.handle_turn(r.state, "no it's actually 9845012399")
    assert not r.done
    assert r.state.digits == "9845012399"
    assert r.reply == "Is 98450 12399 correct — yes or no?"


def test_selftest_still_passes():
    """The module's own self-test (python -m app.engine.planner.phone_capture) covers the
    same ground; running it here means a regression fails pytest too."""
    pc._selftest()
