"""A dedicated mode for collecting a phone number read aloud in chunks.

WHY THIS EXISTS. A caller reading a 10-digit number rarely says it in one breath --
"98450... 12... 367" with short pauses is normal phone etiquette, every chunk a separate
turn. Outside a dedicated mode, each of those turns gets the FULL planner/responder
pipeline run on it -- re-asked, re-explained, sometimes misread as a fresh intent rather
than a continuation. This module is the deterministic alternative for exactly that window:
once the agent has asked for a number and entered capture mode, every turn is handled here,
not by the LLM, until a complete, confirmed number comes out the other end.

WHAT THIS MODULE OWNS (pure, turn-by-turn, no DB/LLM):
  - normalising a chunk ("double five" -> "55", "triple zero" -> "000", Hindi digit words,
    stray fillers ignored) via the shared app.services.phone.extract_spoken_digits
  - reading back ONLY the new chunk, digit by digit ("9-8-4-5-0"), then waiting silently --
    no question, no re-reading earlier chunks
  - treating "no" / "not that" / "galat" (or a differently-spoken retry of the same chunk)
    as a correction that REPLACES the last chunk, not an addition to it
  - once 10 digits are in hand, a single grouped read-back ("98450 12367, is that right?")
    gated on an explicit yes before the number is ever used
  - "same number I'm calling from" -- resolved against caller ID, then still read back and
    confirmed the same way, never trusted silently
  - after two failed confirmations, handing off to a WhatsApp/SMS confirmation offer instead
    of looping a third time

WHAT THIS MODULE DOES NOT OWN, AND CANNOT FROM HERE. Two pieces of the original request are
real-time audio/VAD behaviour, not turn-level logic, and need voice-pipeline changes this
module has no reach into:
  - treating a mid-number PAUSE as "still their turn" rather than ending it (that is
    endpointing/VAD configuration, upstream of any text this module ever sees)
  - the "stops short of 10 digits, wait ~3 seconds, then prompt once" nudge is approximated
    here as "the NEXT turn has no new digits and no correction signal" rather than a true
    elapsed-silence timer, since a turn-based module has no clock of its own
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.engine.planner.hindi_script import normalize_hindi_script
from app.services.phone import _DIGIT_WORDS, _REPEAT_WORDS, extract_spoken_digits

TARGET_LENGTH = 10
MAX_FAILED_CONFIRMATIONS = 2


@dataclass
class PhoneCaptureState:
    """Session-held state for one in-progress capture. Callers persist this as plain data
    (e.g. session["phone_capture"] = dataclasses.asdict(state)) between turns; this module
    never touches a session dict directly, only this dataclass."""
    chunks: list[str] = field(default_factory=list)   # each confirmed chunk, in order
    awaiting_confirm: bool = False                     # grouped read-back is pending a yes/no
    failed_confirmations: int = 0

    @property
    def digits(self) -> str:
        return "".join(self.chunks)


@dataclass
class PhoneCaptureResult:
    state: PhoneCaptureState
    reply: str
    done: bool = False                  # a confirmed 10-digit number is ready
    phone: str | None = None            # the confirmed number, only set when done
    handed_off: bool = False            # gave up after repeated failures, offered WhatsApp/SMS


_SAME_NUMBER_RE = re.compile(
    r"\b(?:same number|this number|the number i'?m calling from|calling from this number|"
    r"number i'?m (?:on|using) (?:right )?now)\b", re.I)

_CORRECTION_RE = re.compile(
    r"^\s*(?:no|nope|not that|wrong|galat|galt|ghalat)\b", re.I)

#
# THE BUG THIS FIXES: both were anchored start-to-end (`^...$`), so only a BARE "yes" or
# "correct" matched -- a real caller answering a yes/no question almost never says just the
# bare word. "yes it is correct", "yeah that's right", "no it's wrong" all fell through to
# the generic "ask plainly" fallback below (no digits to extract either), which re-asked the
# IDENTICAL grouped read-back. A real call needed four repeats of "yes" before one happened
# to be bare enough to match. Anchored at the start only now -- the affirm/deny word leading
# the utterance is what carries the answer; what follows ("it is correct", "that's right")
# only reinforces it, so matching requires the word to START the reply, not BE the reply.
_AFFIRM_RE = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|correct|right|that'?s right|haan|ha|theek hai|thik hai)\b",
    re.I)
_DENY_RE = re.compile(
    r"^\s*(?:no|nope|wrong|incorrect|galat|galt|ghalat|nahi+n?|nahi+)\b", re.I)

# Words that carry NO topic of their own when they sit alongside digits in a genuine
# phone-number restatement -- correction fillers, not content. Anything else left over after
# removing digits/digit-words and these is real content about something ELSE, which means the
# digits found are incidental, not a restatement of the number.
_RESTATEMENT_FILLER_WORDS = frozenset({
    "no", "nope", "not", "that", "it", "its", "actually", "wait", "sorry", "um", "uh",
    "is", "the", "number", "its", "hai", "nahi", "galat", "galt", "ghalat",
})


def _looks_like_phone_restatement(text: str) -> bool:
    """True when ``text`` is plausibly the caller re-stating (part of) their phone number,
    rather than an unrelated utterance that merely happens to contain a digit.

    THE BUG THIS FIXES: "can I come this saturday around 11am" -- asked while a phone
    confirmation was pending -- has extract_spoken_digits pull "11" out of "11am" and the
    awaiting_confirm branch treated that as the caller re-stating their number, discarding
    the real, already-given, correct one and reading back a nonsense "1-1" digit-by-digit.
    The number genuinely given moments earlier was lost over an unrelated scheduling
    question that happened to contain a number.

    A real restatement is almost ALL digits/digit-words/correction fillers ("no", "it's",
    "actually") with little or nothing else; this utterance has "come"/"saturday"/"around" --
    substantial content about a different topic entirely. Checked by removing every digit
    character, digit word, repeat word ("double"/"triple") and filler from the token stream
    and seeing what's left: two or more real words left over means this turn is about
    something else, whatever digits happen to be embedded in it.

    Two edge cases in that word-count check alone, found on review:
      * A Hindi aside counts every Devanagari word as part of the SAME tokenizer run, but
        `[a-zA-Z]+|\\d+` cannot see them at all -- "क्या मैं शनिवार 11 बजे आ सकता हूँ" ("can I
        come Saturday at 11") tokenizes to just "11", leftover 0, and the real number is
        discarded over this Hindi scheduling aside exactly as it was over the English one.
        Fixed by tokenizing Devanagari words too, so they count as leftover same as English.
      * A genuine correction phrased with words outside the fixed filler list ("no wait, it
        should be 98450 12368", "let me repeat", "my number is...", "the correct one is...")
        now has leftover >= 2 and is wrongly rejected, making the caller repeat a number they
        already gave correctly.

        THE BUG THIS RE-INTRODUCED, found on a second review: a plain "5+ extracted digits"
        escape brought P7-F8 straight back for any OTHER kind of numeric aside -- a date
        ("can I come on 25th december 2026 at 11am"), a size+price question ("is the 2 bhk
        1200 sq ft for 85 lakh"), a time range ("can we do saturday 11:30 to 12:30") -- each
        has 5+ digits once concatenated, so the real pending number was discarded and replaced
        by a run-on of unrelated digits every time. A genuine restatement is not just "enough
        digits somewhere in the sentence" -- it is a number that actually LOOKS like the
        10-digit mobile number being captured. Checked instead by shape: exactly 10 digits
        (after stripping a leading country/trunk prefix) starting 6-9, same as a real Indian
        mobile always does and a date/size/price/range concatenation essentially never does.
    """
    tokens = re.findall(r"[a-zA-Zऀ-ॿ]+|\d+", text or "")
    leftover = 0
    for tok in tokens:
        low = tok.lower()
        if tok.isdigit() or low in _DIGIT_WORDS or low in _REPEAT_WORDS:
            continue
        if low in _RESTATEMENT_FILLER_WORDS:
            continue
        leftover += 1
    if leftover < 2:
        return True
    digits = _strip_country_prefix(extract_spoken_digits(text))
    return len(digits) == TARGET_LENGTH and digits[0] in "6789"


def _digit_by_digit(digits: str) -> str:
    """'98450' -> '9-8-4-5-0'. A short, neutral read-back of one chunk."""
    return "-".join(digits)


def _grouped(digits: str) -> str:
    """A 10-digit string -> '98450 12367', the standard spoken grouping for an Indian
    mobile number. Only meaningful at exactly TARGET_LENGTH; callers check length first."""
    return f"{digits[:5]} {digits[5:]}"


def _strip_country_prefix(digits: str) -> str:
    """A caller who includes the country code or a leading trunk 0 shouldn't have it eat
    into the 10-digit target. Stripped once, only when doing so leaves a sane length."""
    if digits.startswith("91") and len(digits) == 12:
        return digits[2:]
    if digits.startswith("0") and len(digits) == 11:
        return digits[1:]
    return digits


def start() -> PhoneCaptureState:
    return PhoneCaptureState()


def handle_turn(state: PhoneCaptureState, user_input: str, *,
                caller_id_phone: str | None = None) -> PhoneCaptureResult:
    """Advance the capture by one turn. Pure -- same inputs, same outputs, no I/O."""
    # _AFFIRM_RE/_DENY_RE below are Roman-script only ("haan", "nahi") -- a caller confirming
    # their number in genuine Devanagari ("हाँ"/"नहीं", what real Hindi STT actually produces)
    # was not recognised at all. Normalized once here, at the single entry point every branch
    # below reads from, same shape as call_closure.py's fix for the identical gap.
    text = normalize_hindi_script((user_input or "")).strip()

    # --- confirming the finished grouped number -------------------------------------------
    if state.awaiting_confirm:
        if _AFFIRM_RE.match(text):
            return PhoneCaptureResult(state=state, reply="", done=True, phone=state.digits)
        if _DENY_RE.match(text):
            # "no" on its own means start over; "no, it's actually 98450123" carries the
            # real correction in the same breath. Now that _DENY_RE matches the LEADING
            # word rather than the whole utterance (see the regex's own comment), both
            # phrasings match here -- check for digits first so a caller who corrects
            # themselves in one go isn't made to repeat AGAIN just because the sentence
            # also happened to start with "no". Same _looks_like_phone_restatement gate as
            # the plain fallthrough below -- "no, can I come at 11am instead" also starts
            # with "no" and also has an incidental digit that is not a phone restatement.
            fresh = extract_spoken_digits(text) if _looks_like_phone_restatement(text) else ""
            if fresh:
                state.chunks = [_strip_country_prefix(fresh)]
                state.awaiting_confirm = False
                return _after_chunk_added(state)
            state.failed_confirmations += 1
            if state.failed_confirmations >= MAX_FAILED_CONFIRMATIONS:
                return PhoneCaptureResult(
                    state=state, handed_off=True,
                    reply=("I'm having trouble getting that number right over the call -- "
                           "could I confirm it with you over WhatsApp or SMS instead?"))
            state.chunks = []
            state.awaiting_confirm = False
            return PhoneCaptureResult(
                state=state,
                reply="No problem — let's take the number again, whenever you're ready.")
        # Caller re-stated digits directly instead of a bare yes/no -- treat as a fresh
        # number, not a continuation of the old one (the old one was just rejected). Gated
        # on _looks_like_phone_restatement: see that function's own docstring for the real
        # call this guards against -- an unrelated aside that merely contains a number
        # ("...around 11am") must never be read as the caller re-stating their phone number.
        fresh = extract_spoken_digits(text) if _looks_like_phone_restatement(text) else ""
        if fresh:
            state.chunks = [_strip_country_prefix(fresh)]
            state.awaiting_confirm = False
            return _after_chunk_added(state)
        # Anything else while awaiting confirm -- ask plainly, don't guess.
        return PhoneCaptureResult(
            state=state, reply=f"Is {_grouped(state.digits)} correct — yes or no?")

    # --- "same number I'm calling from" -----------------------------------------------------
    if _SAME_NUMBER_RE.search(text) and caller_id_phone:
        caller_digits = _strip_country_prefix(re.sub(r"\D", "", caller_id_phone))
        if len(caller_digits) == TARGET_LENGTH:
            state.chunks = [caller_digits]
            state.awaiting_confirm = True
            return PhoneCaptureResult(
                state=state,
                reply=f"Is {_grouped(caller_digits)} correct — yes or no?")

    # --- a correction to the last chunk -------------------------------------------------
    if _CORRECTION_RE.match(text) and state.chunks:
        replacement = extract_spoken_digits(text)
        state.chunks.pop()
        if replacement:
            state.chunks.append(replacement)
            return _after_chunk_added(state, corrected=True)
        return PhoneCaptureResult(
            state=state,
            reply=(f"Sorry about that — go ahead with the correct part"
                   f"{f' (I have {_grouped_partial(state.digits)} so far)' if state.digits else ''}."))

    # --- a plain new chunk -----------------------------------------------------------------
    fresh = extract_spoken_digits(text)
    if fresh:
        if not state.chunks:
            fresh = _strip_country_prefix(fresh)
        state.chunks.append(fresh)
        return _after_chunk_added(state)

    # --- nothing usable this turn: filler, silence, or an unrelated aside ------------------
    # Approximates the "stopped short, nudge once" behaviour -- see the module docstring for
    # why this is turn-based rather than a true elapsed-silence timer.
    if state.chunks:
        return PhoneCaptureResult(
            state=state,
            reply=f"I have {_grouped_partial(state.digits)} so far — please go on.")
    return PhoneCaptureResult(state=state, reply="Whenever you're ready, go ahead with the number.")


def _grouped_partial(digits: str) -> str:
    """A spoken form for a not-yet-complete digit string: group in fives, no trailing
    space/hyphen. '98450123' -> '98450 123'."""
    return f"{digits[:5]} {digits[5:]}".strip() if len(digits) > 5 else digits


def _after_chunk_added(state: PhoneCaptureState, *, corrected: bool = False) -> PhoneCaptureResult:
    digits = state.digits
    if len(digits) > TARGET_LENGTH:
        # Overflow -- more than 10 digits on the table makes the whole thing unreliable.
        # Start clean rather than guess which digits belong.
        state.chunks = []
        state.awaiting_confirm = False
        return PhoneCaptureResult(
            state=state,
            reply="That's more digits than a mobile number — let's take it again from the start.")
    if len(digits) == TARGET_LENGTH:
        state.awaiting_confirm = True
        return PhoneCaptureResult(
            state=state, reply=f"Is {_grouped(digits)} correct — yes or no?")
    last_chunk = state.chunks[-1]
    prefix = "Got it, " if corrected else ""
    return PhoneCaptureResult(state=state, reply=f"{prefix}{_digit_by_digit(last_chunk)}")


def _selftest() -> None:
    # A number read in three natural chunks, confirmed on the first try.
    s = start()
    r = handle_turn(s, "98450")
    assert r.reply == "9-8-4-5-0", r.reply
    assert not r.done
    r = handle_turn(r.state, "1 2")
    assert r.reply == "1-2", r.reply
    r = handle_turn(r.state, "3 6 7")
    assert r.reply == "Is 98450 12367 correct — yes or no?", r.reply
    assert not r.done
    r = handle_turn(r.state, "yes")
    assert r.done and r.phone == "9845012367", r

    # "double five"/"triple zero" normalisation (via the shared extract_spoken_digits).
    s = start()
    r = handle_turn(s, "nine eight four double five")
    assert r.state.digits == "98455", r.state.digits
    r = handle_turn(r.state, "triple zero")
    assert r.state.digits == "98455000", r.state.digits

    # Hindi digit words mixed with English mid-number.
    s = start()
    r = handle_turn(s, "nau aath char ek")
    assert r.state.digits == "9841", r.state.digits

    # A correction: "no, it's actually X" replaces the LAST chunk, not appended to it.
    # (chunk2 deliberately stays short so the total stays under 10 and the correction is
    # tested against the plain "collecting" branch, not the grouped-confirm one.)
    s = start()
    r = handle_turn(s, "98450")
    r = handle_turn(r.state, "999")             # wrong chunk
    assert r.state.digits == "98450999", r.state.digits
    r = handle_turn(r.state, "no, it's 12367")  # correction, and it completes the number
    assert r.state.digits == "9845012367", r.state.digits
    # Completing the number via a correction still goes straight to the grouped confirm,
    # same as reaching 10 digits any other way -- not a bare chunk read-back.
    assert r.reply == "Is 98450 12367 correct — yes or no?", r.reply

    # A correction that does NOT complete the number yet -- "Got it" + the corrected chunk's
    # digit-by-digit read-back, same as any other mid-capture chunk.
    s = start()
    r = handle_turn(s, "98450")
    r = handle_turn(r.state, "999")
    r = handle_turn(r.state, "no, it's 123")
    assert r.state.digits == "98450123", r.state.digits
    assert r.reply == "Got it, 1-2-3", r.reply

    # A bare correction word with no replacement digits yet -- pop and wait, no guess.
    s = start()
    r = handle_turn(s, "98450")
    r = handle_turn(r.state, "no")
    assert r.state.digits == "", r.state.digits
    assert "go ahead" in r.reply.lower()

    # Overflow -- more than 10 digits resets cleanly rather than guessing which belong.
    # (first chunk deliberately stays under 10 so the overflow is hit while still in the
    # plain collecting branch, not after already reaching the grouped-confirm gate.)
    s = start()
    r = handle_turn(s, "984501236")   # 9 digits
    assert not r.state.awaiting_confirm
    r = handle_turn(r.state, "789")   # pushes total to 12
    assert r.state.digits == "", r.state.digits
    assert "again from the start" in r.reply

    # Grouped confirm: "no" once re-collects; twice hands off to WhatsApp/SMS.
    s = start()
    r = handle_turn(s, "9845012367")
    assert r.reply == "Is 98450 12367 correct — yes or no?"
    r = handle_turn(r.state, "no")
    assert not r.done and not r.handed_off and r.state.digits == ""
    r = handle_turn(r.state, "9845012367")
    assert r.reply == "Is 98450 12367 correct — yes or no?"
    r = handle_turn(r.state, "no")
    assert r.handed_off and "WhatsApp or SMS" in r.reply, r.reply

    # THE BUG THIS FIXES: "can I come this saturday around 11am", said while awaiting
    # confirmation, has extract_spoken_digits pull "11" out of "11am" -- an unrelated aside
    # must never be read as the caller re-stating their number. The real, already-confirmed-
    # pending number must survive untouched and the plain re-ask must fire instead.
    s = start()
    r = handle_turn(s, "9845012367")
    assert r.reply == "Is 98450 12367 correct — yes or no?"
    r = handle_turn(r.state, "can I come this saturday around 11am")
    assert r.state.digits == "9845012367", r.state.digits
    assert r.state.awaiting_confirm, "must still be awaiting confirmation"
    assert r.reply == "Is 98450 12367 correct — yes or no?", r.reply
    # The same aside prefixed with "no" (the DENY branch) must be guarded identically.
    s = start()
    r = handle_turn(s, "9845012367")
    r = handle_turn(r.state, "no, can I come at 11am instead")
    assert r.state.digits == "", r.state.digits  # genuinely denied, not hijacked by "11"
    assert not r.state.awaiting_confirm

    # THE BUG THIS FIXES (Devanagari variant of the same P7-F8 bug): the tokenizer that
    # counts "leftover" words could not see Devanagari at all, so a Hindi scheduling aside
    # with an embedded number was still misread as a restatement.
    s = start()
    r = handle_turn(s, "9845012367")
    assert r.reply == "Is 98450 12367 correct — yes or no?"
    r = handle_turn(r.state, "क्या मैं शनिवार 11 बजे आ सकता हूँ")
    assert r.state.digits == "9845012367", r.state.digits
    assert r.state.awaiting_confirm, "must still be awaiting confirmation"

    # THE BUG THIS FIXES: a genuine correction phrased with words outside the fixed filler
    # list ("should", "be") used to push leftover to 2+ and get wrongly rejected, making the
    # caller repeat a number they already gave correctly. Extracted digits that are EXACTLY
    # the shape of a real mobile number (10 digits, starting 6-9) count as a restatement
    # regardless of how many other words surround them.
    s = start()
    r = handle_turn(s, "9845012367")
    r = handle_turn(r.state, "no wait, it should be 98450 12368")
    assert r.state.digits == "9845012368", r.state.digits
    assert r.reply == "Is 98450 12368 correct — yes or no?", r.reply

    # THE BUG THIS RE-INTRODUCED (found on a second review): a plain "5+ digits extracted"
    # escape brought the original P7-F8 bug straight back for any OTHER numeric aside -- a
    # date, a size+price question, a time range -- each concatenates to 5+ digits that are
    # not a phone number at all. The real pending number must survive every one of these.
    for aside in ("can I come on 25th december 2026 at 11am",
                  "is the 2 bhk 1200 sq ft for 85 lakh",
                  "can we do saturday 11:30 to 12:30"):
        s = start()
        r = handle_turn(s, "9845012367")
        assert r.reply == "Is 98450 12367 correct — yes or no?"
        r = handle_turn(r.state, aside)
        assert r.state.digits == "9845012367", (aside, r.state.digits)
        assert r.state.awaiting_confirm, (aside, "must still be awaiting confirmation")

    # "same number I'm calling from" -- resolved from caller ID, but STILL read back and
    # confirmed, never trusted silently.
    s = start()
    r = handle_turn(s, "same number I'm calling from", caller_id_phone="+919845012367")
    assert r.reply == "Is 98450 12367 correct — yes or no?", r.reply
    assert not r.done
    r = handle_turn(r.state, "yes")
    assert r.done and r.phone == "9845012367"

    # Filler words never count as digits or as a correction.
    s = start()
    r = handle_turn(s, "haan okay")
    assert r.state.digits == "", r.state.digits
    assert not r.done

    # Nothing usable mid-capture -- nudge once, naming what's in hand so far.
    s = start()
    r = handle_turn(s, "98450")
    r = handle_turn(r.state, "umm okay")
    assert "98450" in r.reply and "so far" in r.reply, r.reply

    # Natural affirm/deny phrasing at the grouped confirm, not just a bare "yes"/"no" --
    # THE BUG THIS FIXES: _AFFIRM_RE/_DENY_RE were anchored start-to-end, so "yes it is
    # correct" and "no it's wrong" matched NEITHER and fell through to the generic
    # "ask plainly" fallback, re-asking the identical question. A real call needed four
    # repeats of bare "yes" before one happened to match.
    s = start()
    r = handle_turn(s, "9845012367")
    r = handle_turn(r.state, "yes it is correct")
    assert r.done and r.phone == "9845012367", r
    s = start()
    r = handle_turn(s, "9845012367")
    r = handle_turn(r.state, "yeah that's right")
    assert r.done and r.phone == "9845012367", r
    s = start()
    r = handle_turn(s, "9845012367")
    r = handle_turn(r.state, "no it's wrong")
    assert not r.done and not r.handed_off and r.state.digits == "", r

    # "no, it's actually X" at the grouped confirm carries a real correction, not just a
    # bare rejection -- the digits in the same breath must be used, not thrown away.
    s = start()
    r = handle_turn(s, "9845012367")
    r = handle_turn(r.state, "no it's actually 9845012399")
    assert not r.done and r.state.digits == "9845012399", r
    assert r.reply == "Is 98450 12399 correct — yes or no?", r.reply

    print("phone_capture selftest: all passed")


if __name__ == "__main__":
    _selftest()
