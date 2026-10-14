"""Forward-pitch stripping on a policy-blocked turn (turn_runner._FORWARD_PITCH_RE,
_strip_forward_pitch). Used only when proposal.rule_blocked is set (e.g. DNC) -- nothing
should be pitched or re-contact offered however the model phrases it. Pure, no DB.

    cd core && python -m pytest tests/test_forward_pitch.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.planner import turn_runner as tr  # noqa: E402


def test_long_pitch_past_the_old_60_char_cap_is_caught():
    text = ("We only sell apartments, no rentals. Would you like me to share the brochure "
            "and floor plans for the 3 BHK units in Tower B on WhatsApp?")
    out = tr._strip_forward_pitch(text, language="english")
    assert "brochure" not in out and "floor plans" not in out
    assert out == "We only sell apartments, no rentals."


def test_hinglish_pitch_is_caught():
    text = "Hum sirf sale karte hain. Kya aap hamare naye project ke baare mein sunna chahenge?"
    out = tr._strip_forward_pitch(text, language="hinglish")
    assert "chahenge" not in out
    assert out == "Hum sirf sale karte hain."


def test_devanagari_pitch_is_caught():
    # _pitch_sentences now splits on the Devanagari danda too, so the two clauses are
    # separate sentences and only the pitch one is dropped -- the real answer survives.
    text = "हम केवल बिक्री करते हैं। क्या मैं आपको ब्रोशर भेज दूं?"
    out = tr._strip_forward_pitch(text, language="hindi")
    assert "भेज दूं" not in out
    assert out == "हम केवल बिक्री करते हैं।"


def test_whole_reply_being_the_pitch_uses_a_neutral_line_not_the_mangled_pitch():
    # THE BUG THIS FIXES: text.split("?")[0] + "." used to keep the pitch verbatim with the
    # question mark swapped for a period -- the exact thing this guard exists to remove.
    text = "Would you like to hear about the apartments we have for purchase?"
    out = tr._strip_forward_pitch(text, language="english")
    assert "apartments" not in out and "purchase" not in out
    assert out == tr._NEUTRAL_POLICY_BLOCKED_LINE["english"]


def test_whole_reply_pitch_fallback_is_language_aware():
    text = "Kya aap hamare naye project ke baare mein sunna chahenge?"
    out = tr._strip_forward_pitch(text, language="hinglish")
    assert out == tr._NEUTRAL_POLICY_BLOCKED_LINE["hinglish"]


def test_text_with_no_pitch_is_unchanged():
    text = "We only sell apartments at this project, no rentals or leases."
    assert tr._strip_forward_pitch(text, language="english") == text


def test_mixed_reply_keeps_the_honest_clause_drops_the_pitch():
    text = ("The largest 3 BHK is 1,560 square feet. Shall I send you the floor plan on "
            "WhatsApp?")
    out = tr._strip_forward_pitch(text, language="english")
    assert out == "The largest 3 BHK is 1,560 square feet."


def test_answer_and_pitch_sharing_one_sentence_keeps_the_answer():
    # THE BUG THIS FIXES: no sentence-ending punctuation separates the answer from the pitch
    # ("...no rentals, would you like...?" is ONE sentence), so the whole thing used to be
    # dropped and replaced with "I don't have that detail" -- even though the model had just
    # answered correctly.
    text = "We only sell, no rentals, would you like to see the 3 BHK?"
    out = tr._strip_forward_pitch(text, language="english")
    assert "would you like" not in out.lower() and "3 BHK" not in out
    assert out == "We only sell, no rentals."


def test_devanagari_answer_and_pitch_sharing_the_danda_keeps_the_answer():
    text = "हम केवल बिक्री करते हैं, किराये पर नहीं। क्या मैं आपको ब्रोशर भेज दूं?"
    out = tr._strip_forward_pitch(text, language="hindi")
    assert "भेज दूं" not in out
    assert out == "हम केवल बिक्री करते हैं, किराये पर नहीं।"
