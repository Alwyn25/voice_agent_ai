"""Unit tests for the P3.06 speculative-prefetch decision gate.

TurnBridge warms the retrieval embedding from a confident interim transcript during the STT
idle window. The decision of WHETHER a given interim is worth embedding is a pure function,
speculative_prefetch.should_prefetch, precisely so it can be tested without pipecat -- the
TurnBridge that calls it pulls in the whole pipecat frame pipeline, which these tests never
touch (same approach as test_semantic_endpointing.py).

Run: cd core && python -m pytest tests/test_speculative_prefetch.py -q
"""
import sys
from pathlib import Path

# core/ on the path so `import app...` resolves the same way the app does at runtime, without
# requiring the package to be pip-installed into the test environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.voice.speculative_prefetch import (
    FIRE,
    SKIP_BOT_SPEAKING,
    SKIP_EMPTY,
    SKIP_FEW_WORDS,
    SKIP_LOW_CONFIDENCE,
    SKIP_UNCHANGED,
    should_prefetch,
)

# Defaults mirroring config.py's voice_speculative_prefetch_* so the cases read like production.
MIN_WORDS = 3
MIN_CONF = 0.6


def _decide(text, conf, *, last_key=None, bot_speaking=False,
            min_words=MIN_WORDS, min_conf=MIN_CONF):
    """Thin helper: the key is just the lowercased text here -- normalize_query is exercised
    by retrieval's own tests; this gate only needs a stable key to dedup on."""
    return should_prefetch(
        text, conf, normalized_key=text.lower().strip(), last_prefetched_key=last_key,
        bot_speaking=bot_speaking, min_words=min_words, min_confidence=min_conf)


def test_confident_full_interim_fires():
    # The happy path: enough words, confident, bot silent, not seen before -> warm it.
    assert _decide("kya slots available hain is hafte", 0.92) == FIRE


def test_empty_interim_skipped():
    assert _decide("   ", 0.99) == SKIP_EMPTY


def test_backchannel_over_bot_skipped():
    # A caller talking over the bot is almost always a backchannel, not a question to retrieve
    # for -- skip regardless of how confident/long it is.
    assert _decide("okay sure go on then", 0.95, bot_speaking=True) == SKIP_BOT_SPEAKING


def test_too_short_interim_skipped():
    # Below MIN_WORDS the interim is too partial for its embedding to resemble the final turn.
    assert _decide("book me", 0.99) == SKIP_FEW_WORDS


def test_low_confidence_interim_skipped():
    # A shaky interim will likely be revised before it finalises -> its embedding would miss.
    assert _decide("nearest centre kahan hai powai", 0.40) == SKIP_LOW_CONFIDENCE


def test_missing_confidence_is_not_blocked():
    # Provider reported no confidence (None): fall back to the word-count gate, do not block.
    assert _decide("nearest centre kahan hai powai", None) == FIRE


def test_unchanged_key_deduped():
    # The same (normalised) interim already warmed -> don't re-embed identical text.
    text = "loan lene ki process kya hoti hai"
    assert _decide(text, 0.9, last_key=text.lower()) == SKIP_UNCHANGED


def test_grown_interim_after_previous_fires():
    # Interims grow word by word; a longer interim is a NEW key, so it fires even though a
    # shorter prefix was warmed a moment ago -- warming progressively closer to the final text.
    prev = "book me a slot"
    assert _decide("book me a slot saturday 10 am", 0.9, last_key=prev) == FIRE


def test_gate_order_bot_speaking_beats_word_count():
    # bot_speaking is checked before the word-count/confidence gates: a long confident
    # interim over the bot is still a backchannel skip, not a fire.
    assert _decide("wait no actually hold on a second", 0.99, bot_speaking=True) \
        == SKIP_BOT_SPEAKING
