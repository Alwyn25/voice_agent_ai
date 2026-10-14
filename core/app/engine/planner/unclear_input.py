"""Guard: an unrecognizable utterance near an active booking must not be read as a decision
about that booking.

THE BUG THIS FIXES. Two real calls had the model read a meaningless, garbled word -- "noni",
"typique" -- as an instruction about the caller's existing visit. "noni" was classified as
"customer says no" with the model's own self-reported confidence at 0.99, and the reply
pivoted out of nowhere into "I've got you down for a visit tomorrow at 12 PM. Would you like
to keep it, cancel, or reschedule?" "typique" got the same treatment a few turns later, but
phrased completely differently ("I've noted you won't be coming to the visit tomorrow. Would
you like to reschedule it?") -- which is why the guard below does not try to recognise the
REPLY's phrasing (it varies too much to pattern-match reliably) and instead keys only on the
INPUT. Confidence scoring (confidence.py) does not catch this either: the model was
CONFIDENT, not uncertain, so a low-confidence gate would never have fired. The only reliable
signal left is the caller's own words -- if an utterance contains no recognizable yes/no/
booking-action/day/time/number at all, nothing it said actually supports any conclusion
about their booking, however the model chooses to phrase that conclusion.

THE RULE. When VERIFIED RESULTS establish a real appointment this turn (the same
``has_appointment_on_file`` ground truth booking_claim.py already uses) and the utterance
that prompted this reply is a short run of words none of which are recognizable, the reply is
replaced with a plain request to repeat -- regardless of what the model wrote. Deliberately
narrow on the input side: a longer utterance, or one containing any recognized word or a
digit, is left alone untouched -- this exists only to catch the stray garbled word, never to
second-guess a genuine reply.

Pure and self-tested (``python -m app.engine.planner.unclear_input``), same shape as
booking_claim.py and call_closure.py.
"""
import re

from app.engine.planner.hindi_script import normalize_hindi_script

# Words a caller could plausibly say in reply to anything about their booking: affirmations,
# negations, the three modification actions, and day/time vocabulary (naming a new slot while
# declining the current one, e.g. "move it to Friday").
#
# THE BUG THIS FIXES: "thank"/"thanks" were missing entirely, so a caller saying "thank you"
# right after a booking was confirmed -- a perfectly clear, extremely common two-word
# courtesy, not remotely in the same category as "noni"/"typique" -- was treated as
# unrecognizable gibberish and replaced with "Sorry, I didn't catch that," three times in one
# real call (decision_traces confirmed the planner's own reasoning was right every time:
# "customer expressed gratitude").
_RECOGNIZED_RE = re.compile(
    r"\b(?:yes|yeah|yep|yup|sure|ok|okay|fine|right|correct|definitely|please|go|ahead|"
    r"no|nope|nah|not|don'?t|do|cannot|can'?t|"
    r"keep|cancel|reschedule|postpone|move|change|shift|same|"
    r"today|tomorrow|morning|afternoon|evening|night|noon|midnight|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"am|pm|o'?clock|hour|minute|time|date|"
    r"thanks?|thank|"
    # Hindi/Hinglish (Roman script) -- was entirely missing: a caller saying a plain "haan"
    # or "nahi" near a booking was ALSO being read as unrecognizable gibberish, same failure
    # as "noni"/"typique" this guard exists to catch, just triggered by a real word instead
    # of a garbled one. normalize_hindi_script (called in looks_unrecognizable below) maps
    # the Devanagari spellings of these same words onto this Roman list before it runs.
    r"haan|nahi+n?|nhi+|thik|theek|bas|dhanyavaad|dhanyawad|shukriya|ji)\b", re.I)

ASK_TO_REPEAT = {
    "english": "Sorry, I didn't catch that — could you say that again?",
    "hindi": "माफ़ कीजिए, मैं समझ नहीं पायी — क्या आप फिर से बता सकते हैं?",
    "hinglish": "Sorry, samajh nahi aaya — kya aap phir se bata sakte hain?",
}


def looks_unrecognizable(text: str) -> bool:
    """True when a SHORT (1-2 word) utterance has no recognizable word and no digit in it.

    A longer utterance is left alone even with odd wording -- length itself is evidence of
    real content. A single short word that genuinely is "yes"/"cancel"/"tomorrow"/etc, or
    contains a number, is real content too.
    """
    low = normalize_hindi_script(text or "").strip()
    if not low:
        return False
    if len(low.split()) > 2:
        return False
    if any(ch.isdigit() for ch in low):
        return False
    return not _RECOGNIZED_RE.search(low)


def ask_to_repeat(language: str = "english") -> str:
    return ASK_TO_REPEAT.get(language, ASK_TO_REPEAT["english"])


def _selftest() -> None:
    # The exact two real-call utterances that must be caught.
    for garbled in ("noni", "typique"):
        assert looks_unrecognizable(garbled), garbled

    # Genuine replies to a booking question must NEVER be caught.
    recognized = ("yes", "no", "keep", "cancel", "reschedule", "keep it", "cancel it",
                  "yeah keep it", "move it", "tomorrow", "friday", "4 pm", "please cancel",
                  "thank you", "thanks", "thank you so much",
                  # Hindi/Hinglish -- Roman and, via normalize_hindi_script, Devanagari.
                  "haan", "nahi", "nhi", "thik hai", "dhanyavaad", "शुक्रिया", "हाँ", "नहीं")
    for ok in recognized:
        assert not looks_unrecognizable(ok), ok

    # A longer utterance is never flagged, however odd the words, since length alone is
    # evidence of real content.
    assert not looks_unrecognizable("no I actually wanted to ask something else")
    assert not looks_unrecognizable("")
    assert not looks_unrecognizable(None)

    assert ask_to_repeat() == ASK_TO_REPEAT["english"]
    assert ask_to_repeat("hindi") == ASK_TO_REPEAT["hindi"]
    assert ask_to_repeat("klingon") == ASK_TO_REPEAT["english"]
    print("unclear_input selftest: all passed")


if __name__ == "__main__":
    _selftest()
