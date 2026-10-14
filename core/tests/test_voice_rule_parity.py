"""Voice must not behave differently from chat on the same words because of tenant rules.

DSR's turn policy carried three voice-only rules that changed tool calling: two keyed on
STT confidence (one forced an unanswerable "what is your utterance?" ask, one blocked every
tool) and one recording-disclosure rule that could never fire. Their intents now live where
they can be met: low-confidence safety in the engine (planner.banded_action, see
test_confidence_banding.py) and the disclosure in the voice opening line.

    cd core && python -m pytest tests/test_voice_rule_parity.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.sessions import _default_greeting, with_recording_disclosure  # noqa: E402
from app.engine.planner import outbound_flow  # noqa: E402
from app.engine.rules.schema import compile_rules  # noqa: E402

_DSR_TURN_POLICY = (Path(__file__).resolve().parents[1]
                    / "scripts" / "dsr_rules" / "dsr_turn_policy.yaml")
_DISCLOSURE = "This call may be recorded for quality and training purposes."


def _tenant(disclosure=_DISCLOSURE):
    return SimpleNamespace(name="DSR Infra Projects",
                           settings={"recording_disclosure": disclosure} if disclosure else {})


# --- the DSR turn policy ---------------------------------------------------------------

def test_dsr_turn_policy_still_compiles():
    compiled = compile_rules(_DSR_TURN_POLICY.read_text(encoding="utf-8"))
    assert compiled["rules"], "rule set compiled empty"


def test_dsr_turn_policy_has_no_stt_or_first_turn_gates():
    compiled = compile_rules(_DSR_TURN_POLICY.read_text(encoding="utf-8"))
    ids = {r["id"] for r in compiled["rules"]}
    assert not ids & {"low_speech_confidence", "very_low_speech_no_actions",
                      "recording_disclosure_first"}
    # No rule may key on STT confidence: that is what made voice diverge from chat.
    for rule in compiled["rules"]:
        assert "stt_confidence" not in str(rule.get("when")), rule["id"]


def test_dsr_keeps_its_other_safety_rules():
    ids = {r["id"] for r in compile_rules(
        _DSR_TURN_POLICY.read_text(encoding="utf-8"))["rules"]}
    for kept in ("dnc_state_stops_everything", "identity_before_rescheduling",
                 "identity_before_cancelling", "going_in_circles", "hindi_response"):
        assert kept in ids


# --- the recording disclosure, in the opening line ----------------------------------------

def test_disclosure_goes_before_the_inbound_question():
    opening = with_recording_disclosure(_default_greeting("DSR Infra Projects", None), _tenant())
    assert opening == ("Thanks for calling DSR Infra Projects. " + _DISCLOSURE
                       + " How can I help you today?")


def test_disclosure_goes_before_the_outbound_question():
    opening = with_recording_disclosure(
        outbound_flow.identity_opener("DSR Infra Projects", "Rahul Sharma"), _tenant())
    assert _DISCLOSURE in opening
    assert opening.endswith("is now a good time for a quick word?")
    assert opening.index("am I speaking with Rahul?") < opening.index(_DISCLOSURE)


def test_opening_without_a_question_gets_it_appended():
    assert with_recording_disclosure("Hello there.", _tenant()) == f"Hello there. {_DISCLOSURE}"


def test_no_disclosure_configured_leaves_the_opening_alone():
    greeting = _default_greeting("Ajmera Realty & Infra", None)
    assert with_recording_disclosure(greeting, _tenant(None)) == greeting
    assert with_recording_disclosure(greeting, None) == greeting
    assert with_recording_disclosure(None, _tenant()) is None
