"""Detects a finalized transcript that looks like a bare, still-incomplete number -- a
phone number or price cut short by a natural pause between digit groups.

Real testing this session (both a lowered VAD silence timer and pipecat's own bundled
"smart turn" completion model) proved this exact input shape is the one place a caller
gets cut off on a completely ordinary pause: Deepgram finalizes "five nine eight" as a
whole transcript the moment the gap before "five three two six five five" crosses the
silence threshold, and nothing downstream knows the caller was still going. See
TurnBridge's hold-and-merge logic (bridge.py), which uses this to hold a suspicious
fragment briefly rather than run the planner on half a phone number.

Deliberately liberal: the only cost of a false positive here is one extra grace-period
wait before the (correct, complete) turn still runs -- never a wrong cutoff. So this errs
toward flagging more short, bare-digit fragments than strictly necessary rather than
missing one. A single bare digit ("2", answering "how many bedrooms") is excluded -- real
phone numbers/prices break into groups of two or more, and a single digit is a common,
genuinely complete answer on its own; penalizing every one of those with a grace-period
wait would cost far more than it protects.
"""
import re

# Two or more digits, optionally grouped by spaces/dashes, and nothing else -- "59",
# "598", "5 9 8", "1-54". Deepgram's numerals=true setting (config.py) already converts
# spoken digits to actual digit characters, so this needs no spelled-out-number fallback.
# Capped at 16 characters -- long enough for a grouped 10-digit phone number ("98765
# 43210"), short enough to stay a "bare number", not a sentence that happens to end in one.
_BARE_NUMBER = re.compile(r"^\d[\d\s\-]{1,15}$")

# Nothing but digits and separators, of ANY length -- so a lone "9" matches here even though
# _BARE_NUMBER (which needs 2+ chars) deliberately does not. Used only by
# transcript_looks_unfinished below, and only in tandem with a Smart Turn INCOMPLETE verdict:
# a single digit on its own is usually a complete answer ("2 BHK"), but a single digit that
# the audio model ALSO flagged as unfinished is the first group of a number being dictated.
_ALL_DIGITS = re.compile(r"^[\d\s\-]+$")

# Tokens that, when they END an utterance, strongly imply more speech is coming -- dangling
# articles, prepositions, conjunctions, and possessives, in English and the Hinglish/Hindi a
# caller here mixes in. Kept deliberately TIGHT: every entry is a word that almost never ends
# a genuinely finished sentence, so a false "unfinished" (which only ever costs one grace-window
# wait, never a wrong action) stays rare. Ambiguous sentence-final words ("is", "hai", "so")
# are intentionally excluded. Tune per real-call evidence; matching is case-insensitive and
# ignores trailing punctuation.
_CONTINUATION_TAIL = frozenset({
    # English connectors, articles, prepositions, possessives
    "to", "and", "or", "the", "a", "an", "of", "for", "my", "your", "with", "at",
    # English copulas -- these only ever end a fragment mid-thought here ("My name is",
    # "the price is"). Safe to include because this set is consulted ONLY alongside a Smart
    # Turn INCOMPLETE verdict, so the model already believes more is coming; the copula just
    # corroborates it. A rare complete "...is" that the model misflags costs one grace wait.
    "is", "are", "am", "was", "were", "be",
    # Bare transitive verbs that virtually never end a complete utterance in a real-estate
    # call context. P2-F1 root cause: "I want" was finalized as its own turn and run through
    # the planner because "want" wasn't in this set. Alongside a Smart Turn INCOMPLETE verdict
    # these are safe: Smart Turn already believes more is coming; adding the word just
    # corroborates rather than overrules. The only cost of a false positive is one grace wait.
    "want", "need", "looking", "thinking", "planning", "asking", "calling",
    # Hinglish / Hindi connectors + possessives. Sentence-final "hai"/"hain" are deliberately
    # EXCLUDED -- they end complete Hindi sentences far too often ("mera naam Rahul hai") to
    # treat as a continuation cue; revisit only if real calls show a need.
    "aur", "ka", "ki", "ke", "mera", "meri", "mere", "ko", "se", "ek",
})


def looks_like_partial_entity(text: str) -> bool:
    """True if `text` is nothing but a short run of digits -- possibly a phone number,
    price, or unit number caught mid-recitation, not a genuine complete sentence."""
    stripped = text.strip()
    if not stripped:
        return False
    # Deepgram's smart_format (config.py) adds sentence-ending punctuation to a
    # transcript it judges complete -- "9 3 5 5 9." -- even when it's really just
    # another bare-digit fragment. A REAL confirmed bug: that trailing period alone
    # made this fail to match, so a second fragment arriving on its own (the first
    # already having been merged/flushed) ran immediately as its own five-digit
    # "phone number" instead of being held as suspiciously short. Only the TRAILING
    # mark is stripped -- an internal one (there shouldn't be any, given the character
    # class below) would still correctly fail the match.
    stripped = stripped.rstrip(".,!?")
    return bool(_BARE_NUMBER.match(stripped))


def transcript_looks_unfinished(text: str) -> bool:
    """True if the transcript itself reads as mid-utterance -- the linguistic corroboration
    behind task P2.01's fix for Smart Turn's noisy verdicts at a short VAD window.

    The problem this exists to solve: at a 120ms window Smart Turn is starved of trailing
    silence and hedges to INCOMPLETE on turns that are actually finished ("Yes.", "registered?
    Can you give that again?"), costing a full grace-window hold each time. TurnBridge uses this
    function so an INCOMPLETE verdict is only trusted enough to HOLD when the WORDS agree the
    utterance is unfinished; a complete-looking clause the model merely misflagged runs at once.

    Four shapes count as unfinished, in rising order of specificity:
      * a bare partial entity (looks_like_partial_entity) -- "59", "5 9 8";
      * an all-digit run of any length, including a lone "9" -- a number being dictated group by
        group (only meaningful alongside an INCOMPLETE verdict, which is the sole caller);
      * an utterance ENDING on a bare digit group -- "my phone number is 8 0 7 8" -- a number
        still being dictated, but introduced or interrupted by WORDS so the pure-digit checks
        above miss it. This was the real cause of the phone-number re-ask: a word-prefixed
        half-number was overruled and run instead of held;
      * an utterance ending on a dangling connector, possessive, or bare transitive verb
        (_CONTINUATION_TAIL) -- "My name is", "I want to", "flat ka", "I want", "I'm looking".

    Pure and pipecat-free so it is unit-testable in isolation. Conservative by construction: a
    miss (returning False on something that was unfinished) at worst allows a split the old
    behaviour already had; a false positive at worst adds one grace-window wait, never a wrong
    action.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if looks_like_partial_entity(stripped):
        return True
    if _ALL_DIGITS.match(stripped) and any(ch.isdigit() for ch in stripped):
        return True
    last_token = stripped.split()[-1].strip(".,!?;:—-")
    # Ends on a bare digit group -> a number still mid-dictation even when words precede it
    # ("my phone number is 8 0 7 8"). Verdict-gated by the sole caller (INCOMPLETE only), so a
    # genuinely complete numeric answer Smart Turn calls COMPLETE ("2 BHK", "flat 5") is never
    # delayed by this.
    if last_token.isdigit():
        return True
    return last_token.lower() in _CONTINUATION_TAIL
