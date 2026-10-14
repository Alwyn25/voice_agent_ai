"""Consent gate (app/engine/planner/consent.py) — the reply to "Shall I book …?".

The bug this guards: a booking ran off "i'm sorry what time / ja por time again" because the
planner labelled the garble ``affirm`` and that label alone was read as consent. These tests
pin the two invariants that matter most:

  * nothing that is not a clear yes in the caller's OWN words can come out as ACCEPT;
  * a clear yes, in any common phrasing or language, still does (no re-ask loops).
"""
from datetime import date

import pytest

from app.engine.planner import consent, understanding
from app.engine.planner.consent import (ACCEPT, CHANGE, DECLINE, HOLD, REPEAT, SIDE_QUESTION,
                                        UNCLEAR, decide, read_lexical)

STAGED = "2026-10-13T10:00:00+05:30"     # Tuesday 13 October, 10:00 AM IST
TODAY = date(2026, 10, 13)


def lex(text, tool="book_appointment", staged=STAGED, today=TODAY):
    return read_lexical(text, pending_tool=tool, staged_start=staged, today=today).decision


def test_module_selftest():
    assert consent._selftest() == 0


# ------------------------------------------------------------------ the reported call

@pytest.mark.parametrize("reply", [
    "i'm sorry what time",
    "ja por time again",
    "i'm sorry what time ja por time again",
])
def test_reported_bug_is_a_repeat_request(reply):
    assert lex(reply) == REPEAT


@pytest.mark.parametrize("reply", ["ja por time again", "ja por", "mmm tre"])
def test_a_model_accept_never_books_a_garble(reply):
    # The planner/understanding model said "accept" — the exact failure. Without the words
    # of agreement in the transcript it must not come out as ACCEPT.
    for evidence in ("", "yes", "time again", "ja por"):
        r = decide(reply, semantic={"decision": "accept", "evidence": evidence},
                   pending_tool="book_appointment", staged_start=STAGED, today=TODAY)
        assert r.decision != ACCEPT, (reply, evidence, r)


def test_is_affirmative_ignores_model_labels():
    from app.engine.planner.turn_runner import is_affirmative
    # The second positional argument used to be the planner's intent label; a label can no
    # longer make an utterance consent.
    assert not is_affirmative("ja por time again")
    assert not is_affirmative("i'm sorry what time")
    assert is_affirmative("yes please")
    assert is_affirmative("haan kar do")


# ------------------------------------------------------------------ decision table

@pytest.mark.parametrize("reply", [
    "yes", "yeah sure", "ok", "okay go ahead", "go for it", "sure, let's do it",
    "let's do that", "sounds good", "that works for me", "please do", "book it",
    "no problem, go ahead", "why not", "i don't mind", "could you book it for me",
    "yes 10 am is fine", "ok today at 10",
    "haan", "haan ji", "theek hai", "chalega", "kar do", "haan bilkul book kar dijiye",
    "हाँ", "जी हाँ", "ठीक है", "हाँ बुक कर दीजिए",
])
def test_accept(reply):
    assert lex(reply) == ACCEPT


@pytest.mark.parametrize("reply", [
    "no", "no thanks", "not now", "I changed my mind", "on second thought, leave it",
    "forget it", "never mind", "don't book it", "maybe later", "I can't make it",
    "nahi", "rehne do", "abhi nahi", "baad mein", "mat karo", "nahi chahiye",
    "नहीं", "रहने दो", "मत कीजिए", "अभी नहीं",
])
def test_decline(reply):
    assert lex(reply) == DECLINE


@pytest.mark.parametrize("reply", [
    "make it 11 instead", "can we do tomorrow", "another day please", "how about 4 pm",
    "yes but at 11 am", "ok 11 am", "yes at 10 pm", "yes tomorrow", "kal ka time do",
    "I changed my mind, make it 11",
])
def test_change(reply):
    assert lex(reply) == CHANGE


@pytest.mark.parametrize("reply", [
    "what time", "okay what time?", "yes, which day was it", "say that again",
    "can you repeat that", "sorry?", "pardon", "I didn't catch that", "when is it",
    "phir se bolo", "kitne baje", "samajh nahi aaya", "फिर से बताइए", "कितने बजे",
])
def test_repeat(reply):
    assert lex(reply) == REPEAT


@pytest.mark.parametrize("reply", [
    "how much is parking", "is there parking", "ok and where is the site",
    "yes and is it ready to move", "parking kitna hai",
])
def test_side_question(reply):
    assert lex(reply) == SIDE_QUESTION


@pytest.mark.parametrize("reply", [
    "wait", "hold on", "let me think", "not sure", "maybe", "ruko", "soch ke batata hoon",
])
def test_hold(reply):
    assert lex(reply) == HOLD


# ------------------------------------------------------------------ context verbs

def test_cancel_means_yes_only_for_a_staged_cancellation():
    assert lex("yes cancel it", tool="cancel_appointment", staged=None) == ACCEPT
    assert lex("haan cancel kar do", tool="cancel_appointment", staged=None) == ACCEPT
    assert lex("no don't cancel", tool="cancel_appointment", staged=None) == DECLINE
    assert lex("cancel it", tool="book_appointment") == DECLINE


def test_move_means_yes_for_a_staged_reschedule():
    assert lex("yes move it", tool="reschedule_appointment") == ACCEPT


# ------------------------------------------------------------------ fusion

def test_mixed_polarity_needs_a_verified_semantic_read():
    assert read_lexical("nahi nahi, kar do").mixed
    assert decide("nahi nahi, kar do").decision == UNCLEAR
    assert decide("nahi nahi, kar do",
                  semantic={"decision": "accept", "evidence": "kar do"}).decision == ACCEPT


def test_semantic_accept_requires_quoted_words_and_clear_audio():
    sem = {"decision": "accept", "evidence": "carry on with this plan"}
    assert decide("let us carry on with this plan", semantic=sem).decision == ACCEPT
    assert decide("let us carry on with this plan", semantic=sem,
                  stt_confidence=0.3, stt_floor=0.55).decision == UNCLEAR
    assert decide("let us carry on with this plan",
                  semantic={"decision": "accept", "evidence": "yes book it"}
                  ).decision == UNCLEAR


def test_any_non_accept_reading_wins():
    assert decide("ok what time", semantic={"decision": "accept", "evidence": "ok"}
                  ).decision == REPEAT
    assert decide("yes", semantic={"decision": "decline", "evidence": "yes"}
                  ).decision == UNCLEAR
    assert decide("the clubhouse thing", semantic={"decision": "side_question",
                                                   "evidence": ""}).decision == SIDE_QUESTION


def test_off_list_semantic_value_is_ignored():
    # "affirm" is the planner's free-text intent label, not a consent decision.
    assert decide("ja por", semantic={"decision": "affirm", "evidence": "ja por"}
                  ).decision == UNCLEAR


def test_trace_shape():
    t = decide("yes").as_trace()
    assert t["decision"] == ACCEPT and t["source"] == "lexical" and "yes" in t["evidence"]


# ------------------------------------------------------------------ lines spoken

def test_readback_lines():
    when = "Tuesday 13 October at 10:00 AM"
    assert consent.readback("book_appointment", when, kind="repeat") == (
        "Sure — that's Tuesday 13 October at 10:00 AM. Shall I go ahead and book your visit?")
    assert "didn't quite catch" in consent.readback("book_appointment", when, kind="unclear")
    assert "yes or no" in consent.readback("book_appointment", when, kind="unclear",
                                           attempt=1)
    assert consent.readback("book_appointment", when, kind="reask").startswith(
        "And shall I go ahead and book your visit for Tuesday")
    assert consent.readback("book_appointment", when, kind="repeat",
                            language="hindi").startswith("जी, समय है")


# ------------------------------------------------------------------ understanding schema

def test_consent_field_only_when_awaiting():
    plain = understanding._schema_for(None)
    assert plain is understanding.UNDERSTANDING_SCHEMA
    assert "consent" not in plain["properties"]
    awaiting = understanding._schema_for("Shall I book your visit?")
    assert "consent" in awaiting["properties"] and "consent" in awaiting["required"]
    assert set(awaiting["properties"]["consent"]["properties"]["decision"]["enum"]) == set(
        consent.DECISIONS)
    # the base schema is not mutated
    assert "consent" not in understanding.UNDERSTANDING_SCHEMA["required"]


@pytest.mark.asyncio
async def test_understand_sends_consent_block_only_when_awaiting(monkeypatch):
    seen = {}

    class _Result:
        data = {"intent": "affirm", "entities_json": "{}", "needs_knowledge": False,
                "is_scheduling": True, "is_status_question": False,
                "conversational_signal": "none", "project_choice": None,
                "consent": {"decision": "repeat", "evidence": "time again"}}
        model, tokens = "fake", {}

    class _LLM:
        async def complete_json(self, **kw):
            seen.update(kw)
            return _Result()

    monkeypatch.setattr(understanding, "get_llm", lambda: _LLM())
    u = await understanding.understand("ja por time again",
                                       awaiting_confirmation="Shall I book your visit?")
    assert "AWAITING CONFIRMATION" in seen["user"] and "consent" in seen["system"]
    assert u.consent == {"decision": "repeat", "evidence": "time again"}
    assert u.as_trace()["consent"]["decision"] == "repeat"

    seen.clear()
    u = await understanding.understand("ja por time again")
    assert "AWAITING CONFIRMATION" not in seen["user"]
    assert "consent" not in seen["schema"]["properties"]
    assert u.consent is None and "consent" not in u.as_trace()
