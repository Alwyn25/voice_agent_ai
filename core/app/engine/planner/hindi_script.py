"""Normalise a small, fixed set of Devanagari closure/affirm/deny words to their Roman
transliteration before a guard built for Roman-script Hindi/Hinglish matches against them.

THE BUG THIS FIXES: every Hindi-aware deterministic guard in this codebase (call_closure.py's
wind-down loop, phone_capture.py's yes/no confirmation, unclear_input.py's gibberish check) was
written and tested against Roman-script transliteration only -- "nahi", "thik hai", "haan" --
because that is how test transcripts and most of this session's work were typed. The voice
pipeline's real STT config (DEEPGRAM_LANGUAGE=multi, Nova-3's code-switching mode) transcribes
genuine Hindi speech in actual Devanagari script by default, not Romanized. Confirmed live: a
caller saying "ठीक है धन्यवाद" (ok thanks) or "नहीं" (no) was not recognised as a closure/wind-down
signal AT ALL -- the deterministic loop-breaker never engaged, and the model, left without that
signal, improvised an unrelated reply instead of asking "anything else?" or ending the call.

Deliberately NOT a general Devanagari transliteration engine (IAST/ITRANS, full phonetic
mapping) -- that is a much bigger, riskier undertaking (ambiguous vowel length, conjuncts,
schwa deletion) for a problem that only needs a FIXED, SMALL vocabulary recognised: the exact
words the existing Roman-script patterns already key on. Word-boundary, case-insensitive-safe
(Devanagari has no case), applied once per guard's input text so every existing Roman pattern
keeps working unchanged against the result -- the same "normalize the input once" shape as
text_normalize.py's apostrophe fix, not a second copy of every pattern in Devanagari too.

Pure stdlib, self-tested: ``python -m app.engine.planner.hindi_script``.
"""
import re

# Devanagari word -> its Roman transliteration, keyed to EXACTLY the vocabulary the existing
# Roman-script patterns in call_closure.py / phone_capture.py / unclear_input.py already
# recognise -- this closes the script gap without claiming broader coverage than those
# patterns already have. Multi-word phrases are listed before their component words so the
# longer match wins (re.sub below applies them in this order).
_DEVANAGARI_TO_ROMAN = [
    ("कोई बात नहीं", "koi baat nahi"),
    ("ठीक है", "thik hai"),
    ("नहीं", "nahi"),
    ("नही", "nahi"),
    ("ठीक", "thik"),
    ("बस", "bas"),
    ("धन्यवाद", "dhanyavaad"),
    ("शुक्रिया", "shukriya"),
    ("हाँ", "haan"),
    ("हां", "haan"),
    ("अलविदा", "alvida"),
]

# Longer phrases first, so "ठीक है" is consumed whole before the bare "ठीक" entry could split
# it. re.escape because Devanagari combining marks would otherwise need no escaping, but the
# phrase entries contain a literal space that must stay literal.
#
# THE BUG THIS FIXES: \b is a transition between \w and non-\w, but Devanagari vowel signs
# (matras) are NOT \w -- so a plain alternation with no boundary at all replaced a short entry
# INSIDE a longer word too: "बसें" ("buses", bas + the plural matra) became "basें", and "bas"
# then matched the downstream Roman-script wind-down pattern as if the caller had said "bas"
# on its own. Guarded here with an explicit Devanagari-codepoint lookaround instead of \b, so a
# match is only replaced when it is NOT itself touching another Devanagari character on either
# side (ASCII/space/punctuation on either side is fine; another Devanagari letter is not).
_DEVANAGARI_RANGE = r"ऀ-ॿ"
_PATTERN = re.compile(
    rf"(?<![{_DEVANAGARI_RANGE}])(?:"
    + "|".join(re.escape(dev) for dev, _ in _DEVANAGARI_TO_ROMAN)
    + rf")(?![{_DEVANAGARI_RANGE}])")
_REPLACEMENTS = dict(_DEVANAGARI_TO_ROMAN)


def normalize_hindi_script(text: str) -> str:
    """Replace the recognised Devanagari words in ``text`` with their Roman transliteration,
    leaving everything else (English, Hinglish already in Roman script, untranslated
    Devanagari outside this vocabulary) untouched."""
    if not text:
        return text or ""
    return _PATTERN.sub(lambda m: _REPLACEMENTS[m.group(0)], text)


def _selftest() -> None:
    assert normalize_hindi_script("ठीक है धन्यवाद") == "thik hai dhanyavaad"
    assert normalize_hindi_script("नहीं") == "nahi"
    assert normalize_hindi_script("हाँ थीक है") == "haan थीक है"  # unmapped spelling untouched
    assert normalize_hindi_script("कोई बात नहीं") == "koi baat nahi"
    # mixed-script (real code-switched speech): only the Devanagari words change
    assert normalize_hindi_script("mujhe नहीं chahiye") == "mujhe nahi chahiye"
    assert normalize_hindi_script("plain english text") == "plain english text"
    assert normalize_hindi_script("") == ""
    assert normalize_hindi_script(None) == ""
    # longer phrase wins over the component word it contains
    assert "thik hai" in normalize_hindi_script("सब ठीक है शुक्रिया")
    # THE BUG THIS FIXES: "बसें" ("buses") contains "बस" (bas) as a bare substring, but the
    # matra right after it is still Devanagari -- must not be replaced, or "bas" counts as
    # the wind-down word downstream.
    assert normalize_hindi_script("बसें") == "बसें"
    assert normalize_hindi_script("बसें कहाँ से मिलती हैं") == "बसें कहाँ से मिलती हैं"
    # but a bare "बस" surrounded by spaces/punctuation (the actual word) still normalises
    assert normalize_hindi_script("ठीक है बस") == "thik hai bas"
    print("hindi_script selftest: all passed")


if __name__ == "__main__":
    _selftest()
