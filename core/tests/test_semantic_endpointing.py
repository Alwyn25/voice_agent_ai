"""Unit tests for the P2.01 semantic-endpointing corroboration logic.

The decision the bridge makes on a Smart Turn INCOMPLETE verdict is driven by a PURE function,
entity_continuation.transcript_looks_unfinished, precisely so it can be tested without pipecat --
the TurnBridge that calls it pulls in the whole pipecat frame pipeline, which these tests never
touch. The gate's control flow (corroboration + the _grace_flushed loop-break) is exercised by a
small mirror of the branch, kept in lockstep with bridge.py and documented as the invariant the
bridge relies on.

Run: cd core && python -m pytest tests/test_semantic_endpointing.py -q
"""
import sys
from pathlib import Path

# core/ on the path so `import app...` resolves the same way the app does at runtime, without
# requiring the package to be pip-installed into the test environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.voice.entity_continuation import transcript_looks_unfinished


# --- transcript_looks_unfinished: the corroboration signal -----------------------------------
#
# Consulted ONLY when Smart Turn already said INCOMPLETE. True = the words corroborate the
# verdict, so HOLD for a continuation; False = the words read as finished, so the (misflagged)
# verdict is overruled and the turn RUNS. Cases are drawn from the real DSR Infra test traces.

def test_dangling_connector_is_unfinished():
    # "I want to" / "flat ka" ended a turn mid-thought and split from their continuation.
    assert transcript_looks_unfinished("I want to") is True
    assert transcript_looks_unfinished("flat ka") is True


def test_dangling_copula_is_unfinished():
    # The exact split from the trace: "My name is" ran as its own turn, "Rahul" as another.
    assert transcript_looks_unfinished("My name is") is True


def test_word_prefixed_number_mid_dictation_is_unfinished():
    # The phone-number re-ask bug: a number introduced by words and paused mid-dictation must
    # still be held, not overruled and run on a half-number. looks_like_partial_entity can't see
    # these (they contain words); the trailing-digit rule does.
    assert transcript_looks_unfinished("my phone number is 8 0 7 8") is True
    assert transcript_looks_unfinished("my phone number is 8078 0 2") is True


def test_bare_digits_including_single_are_unfinished():
    # A number dictated group by group -- a lone "9" must count (looks_like_partial_entity
    # excludes single digits; transcript_looks_unfinished does not, because here it rides on an
    # INCOMPLETE verdict, which is strong evidence the number isn't finished).
    assert transcript_looks_unfinished("9") is True
    assert transcript_looks_unfinished("4 3") is True
    assert transcript_looks_unfinished("98765 43210") is True


def test_finished_clauses_run_not_held():
    # These were wrongly held ~900 ms in the first run; corroboration lets them run immediately.
    assert transcript_looks_unfinished("Yes.") is False
    assert transcript_looks_unfinished("registered? Can you give that again?") is False
    assert transcript_looks_unfinished("book a visit.") is False


def test_bare_backchannel_runs():
    # A lone "yes"/"ok" answer is NOT unfinished -> it runs immediately, not delayed. The
    # "yes/ok false trigger" concern is handled upstream by MinWordsUserTurnStartStrategy on the
    # interruption path, not by holding genuine short answers at the turn boundary.
    assert transcript_looks_unfinished("yes") is False
    assert transcript_looks_unfinished("ok") is False
    assert transcript_looks_unfinished("okay") is False


def test_hindi_sentence_final_hai_is_not_a_continuation_cue():
    # "hai"/"hain" end complete Hindi sentences too often to treat as unfinished.
    assert transcript_looks_unfinished("theek hai") is False


def test_complete_short_answer_not_unfinished():
    # "2 BHK" is a complete answer to "what configuration?" -- not digits-only, no dangling tail.
    assert transcript_looks_unfinished("2 BHK") is False


def test_empty_is_not_unfinished():
    assert transcript_looks_unfinished("   ") is False


def test_trailing_punctuation_does_not_hide_a_dangling_word():
    # Deepgram sometimes appends a stray period; the tail check must see through it.
    assert transcript_looks_unfinished("my name is.") is True


# --- Gate-1 decision matrix (a faithful mirror of bridge.py's corroboration branch) ----------

def _gate1_holds(text, *, semantic_on, corroborate, incomplete, grace_flushed=False):
    """Mirror of TurnBridge gate 1's hold decision, kept in lockstep with bridge.py so the
    combined behaviour is testable without pipecat. Returns True if the turn is HELD by gate 1."""
    if not incomplete or grace_flushed:
        return False
    corr = semantic_on and corroborate
    return (not corr) or transcript_looks_unfinished(text)


def test_matrix_false_incomplete_complete_text_runs():
    # The core fix: model says INCOMPLETE, words read as finished -> overruled -> RUN.
    assert _gate1_holds("registered? Can you give that again?", semantic_on=True,
                        corroborate=True, incomplete=True) is False
    assert _gate1_holds("Yes.", semantic_on=True, corroborate=True, incomplete=True) is False


def test_matrix_dictation_holds():
    # Dangling / digits are corroborated -> HELD so the continuation can merge.
    assert _gate1_holds("My name is", semantic_on=True, corroborate=True, incomplete=True) is True
    assert _gate1_holds("9", semantic_on=True, corroborate=True, incomplete=True) is True


def test_matrix_complete_verdict_never_held_by_gate1():
    # A COMPLETE verdict skips gate 1 entirely (entity_continuation may still hold a digit run).
    assert _gate1_holds("My name is", semantic_on=True, corroborate=True, incomplete=False) is False


def test_matrix_grace_flush_breaks_the_hold_loop():
    # On the flush re-entry the gate must stand down, or a held "My name is" that never got a
    # continuation would re-hold every grace window forever.
    assert _gate1_holds("My name is", semantic_on=True, corroborate=True,
                        incomplete=True, grace_flushed=True) is False


def test_matrix_plain_smart_turn_mode_unchanged():
    # Semantic OFF: the raw INCOMPLETE verdict is always trusted -> HELD, exactly as before P2.01.
    assert _gate1_holds("registered? Can you give that again?", semantic_on=False,
                        corroborate=False, incomplete=True) is True
