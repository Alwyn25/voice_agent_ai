"""Abbreviations and symbols to spoken words (app/services/spoken_form.py). Pure, no DB.

    cd core && python -m pytest tests/test_spoken_form.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.spoken_form import PROMPT_RULE, to_spoken_form  # noqa: E402


def test_area_abbreviations_become_square_feet():
    for raw in ("1,560 sq ft", "1,560 sq. ft.", "1,560 sq.ft", "1,560 sqft", "1,560 sft",
                "1,560 SQ FT", "1,560 Sq.Ft.", "1,560 square ft", "1560sqft"):
        assert to_spoken_form(raw).endswith("square feet") or \
            to_spoken_form(raw).endswith("square feet."), raw


def test_rate_is_per_square_foot():
    assert to_spoken_form("₹11,499/sq ft") == "11,499 rupees per square foot"
    assert to_spoken_form("11,499 psf") == "11,499 per square foot"


def test_sentence_boundary_survives_an_abbreviation_dot():
    # The bridge splits on ". " — swallowing the dot would merge two sentences into one TTS
    # context and shift its replay dedup.
    out = to_spoken_form("The 3 BHK is 1,560 sq. ft. Shall I book a visit?")
    assert out == "The 3 BHK is 1,560 square feet. Shall I book a visit?"


def test_reported_reply_reads_cleanly():
    reply = ("The brochure lists 3 BHK sizes between 1,239 and 1,422 sq ft, but the cost sheet "
             "shows a 1,560 sq ft unit.")
    out = to_spoken_form(reply)
    assert "sq ft" not in out and out.count("square feet") == 2


def test_idempotent_on_already_spoken_text():
    spoken = ("The 3 BHK sizes range from 1,239 to 1,560 square feet, at 11,499 rupees per "
              "square foot, starting at 1.54 crore rupees.")
    assert to_spoken_form(spoken) == spoken
    assert to_spoken_form(to_spoken_form("₹1.54 Cr, 2.2 km, approx. 15 mins")) == \
        to_spoken_form("₹1.54 Cr, 2.2 km, approx. 15 mins")


def test_spoken_acronyms_are_left_alone():
    text = "It is a 3 BHK, RERA registered, GST extra, with AC points and CCTV."
    assert to_spoken_form(text) == text


def test_prompt_rule_reaches_both_models():
    from app.engine.planner import prompts, responder
    assert PROMPT_RULE in prompts.BASE_SYSTEM
    assert PROMPT_RULE in responder.RESPONDER_SYSTEM


def test_selftest_still_passes():
    from app.services import spoken_form
    spoken_form._selftest()
