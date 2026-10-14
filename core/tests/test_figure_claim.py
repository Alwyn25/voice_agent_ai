"""Unsupported-figure guard (app/engine/planner/figure_claim.py). Pure, no DB. See the module
docstring for the real call ("approximately 5 kilometers from Wipro"; the KB says 2.5 km).

    cd core && python -m pytest tests/test_figure_claim.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.planner import figure_claim as fc  # noqa: E402

KB = ("Location\nChikkanayakanahalli, off Sarjapur Road. RGA Tech Park 2.2 km. Wipro Corporate "
      "Office 2.5 km. Manipal Hospital 5.5 km.\nA visit takes about an hour.\n"
      "3 BHK — 1,239 to 1,560 square feet.")


def test_reported_invented_distances_are_caught():
    for reply in ("The project is approximately 5 kilometers from Wipro.",
                  "It's approximately seven kilometres from Wipro."):
        assert fc.unsupported(reply, KB), reply


def test_a_nearby_figure_about_something_else_does_not_vouch():
    # "Manipal Hospital 5.5 km" must not ground a rounded "about 5 km from Wipro".
    assert fc.unsupported("It's about 5 kilometres from Wipro.", KB)
    assert not fc.unsupported("It's about 5 kilometres from Manipal Hospital.", KB)


def test_true_figures_pass_including_conversions_and_hedged_rounding():
    for reply in ("It is 2.5 kilometres from Wipro Corporate Office.",
                  "It's 2,500 metres from Wipro.",
                  "It's about 2 kilometres from RGA Tech Park.",
                  "A site visit takes about an hour.",
                  "Sizes run from 1,239 to 1,560 square feet.",
                  "The rate is 11,499 rupees per square foot."):
        assert not fc.unsupported(reply, KB), reply


def test_scrub_keeps_the_rest_and_says_honestly_what_is_missing():
    out = fc.scrub("The 3 BHK rate is 11,499 rupees per square foot. The project is "
                   "approximately 5 kilometers from Wipro.", KB)
    assert out == ("The 3 BHK rate is 11,499 rupees per square foot. I don't have the exact "
                   "distance for that to hand.")


def test_streamed_and_final_scrub_agree():
    reply = ("It is about 6 kilometres from Wipro. The visit takes 3 hours. It is also 8 km "
             "from Bellandur. Shall I book a visit?")
    said: set = set()
    streamed = " ".join(x for x in (fc.scrub(s, KB, said=said) for s in fc._sentences(reply)) if x)
    assert streamed == fc.scrub(reply, KB)


def test_tool_json_and_the_callers_words_ground_figures():
    g = fc.grounding_text(tool_outputs=[{"sqft": 1560, "carpet_sqft": 1032}],
                          extra=["I live 12 km from the site"])
    assert not fc.unsupported("The 1,560 square foot unit has 1,032 square feet carpet.", g)
    assert not fc.unsupported("So you're about 12 kilometres away.", g)
    assert fc.unsupported("The 1,800 square foot unit is available.", g)


def test_selftest_still_passes():
    fc._selftest()
