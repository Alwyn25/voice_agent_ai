"""Conversational signals — the short caller turns that are ABOUT the conversation itself,
not about property, booking or the caller's records.

WHY THIS EXISTS. On a live phone call a large share of a caller's turns are not content at
all: they keep the channel open ("mm-hmm", "go on"), they check the line is alive
("Hello?", "you there?", "can you hear me?"), or they nod agreement while you talk. The
engine had no concept of these. Its trivial-turn prefilter (``rag.is_trivial_turn``) folds
every one of them into the same bucket as an opening greeting — ``"Hello?"`` matches the
same ``h+e+l+o+`` pattern as a first "hello" — so the strong planner, handed a turn with no
subject and no history signal, labelled it ``greeting`` and answered with a fresh "Hello!"
and a canned re-introduction. To the caller who said "Hello?" because they had lost the
agent's voice, that is the agent cheerfully greeting them again while never repeating the
thing they missed. To the caller who hummed "mm-hmm" to say "I'm with you, keep going", it
is the agent forgetting what it was mid-way through saying and starting over.

WHAT IT DOES. A pure, dependency-free reader of the LATEST caller utterance that separates
three things the planner cannot tell apart from words alone, because the difference is in
the CALL POSITION and in what the AGENT just said, not in the caller's five characters:

  * GREETING     — a genuine opening "hi/hello" on the caller's FIRST turn. Left to the
                   normal engine; nothing here overrides it.
  * CHECK_IN     — a presence / audibility probe MID-call: "Hello?", "are you there?",
                   "can you hear me?", "you still there?". The caller heard silence or a
                   dropout. The right answer is to confirm we are here AND repeat what they
                   missed — never to greet them as if the call just started.
  * BACKCHANNEL  — a continuer: "mm-hmm", "uh-huh", "go on", "right", "I see". The caller is
                   holding the floor open for us. The right answer is to carry on from the
                   last point, not to treat it as a new question or a greeting.
  * NONE         — anything else, including every utterance that carries real content or
                   consent. The normal engine owns it untouched.

SAFETY (the guarded boundary). The words a nod shares with a "yes" — "ok", "yeah", "sure",
"right" — are also how a caller CONSENTS to a booking, a reschedule, a handoff. Reading one
of those as a backchannel when a "yes" was expected would silently swallow consent. So the
ambiguous-agreement words are classified BACKCHANNEL only when the caller supplies the fact
that nothing is pending: ``agent_asked_question=False`` AND the call site confirms no staged
action / offer / awaited answer (``pending=False``). The UNAMBIGUOUS continuers ("mm-hmm",
"go on", "uh-huh", "keep going", …) never mean "yes" and are always safe, pending or not.
A word that is not purely one of these signals — anything with real content beside it —
falls through to NONE, so "yes, book the 3 PM" is never mistaken for a hum.

Pure and self-tested: ``python -m app.engine.planner.conversational_signal``.
"""
from __future__ import annotations

import re

# ------------------------------------------------------------------ signal categories
SIGNAL_NONE = "none"            # not a conversational signal — normal engine owns it
SIGNAL_GREETING = "greeting"    # a genuine opening greeting (first caller turn)
SIGNAL_CHECK_IN = "check_in"    # presence / audibility probe mid-call
SIGNAL_BACKCHANNEL = "backchannel"  # a continuer / nod that holds the floor open


# A bare greeting word, possibly stretched by STT ("helloo") or doubled ("hello hello"),
# with nothing else of substance. This is the SHAPE both a first-turn greeting and a
# mid-call "Hello?" share; call position tells them apart, not these characters.
_GREETING_WORD = r"(?:h+e+l+o+|h+[ea]+y+|h+i+|yo+|hiya|heya|namaste|namaskar)"
# A greeting is one or more greeting words, optionally trailed by a bare "there"/"ya" the way
# people say "hi there" / "hey ya" — still just a greeting, nothing of substance beside it.
_GREETING_RE = re.compile(
    rf"^\s*(?:{_GREETING_WORD})(?:[\s,.!?-]+(?:{_GREETING_WORD}|there|ya))*[\s.,!?-]*$", re.I)

# An explicit presence / audibility probe — the caller telling us the line went quiet. These
# are check-ins WHOEVER speaks first and WHENEVER they land, because they name the channel
# ("hear", "there", "the line", "cutting out"), so they never read as a greeting or a nod.
_CHECK_IN_RE = re.compile(
    r"\bare you (?:still )?(?:there|here|with me|on|around)\b"
    r"|\byou (?:still )?there\b|\bstill (?:there|with me|on the line)\b"
    r"|\b(?:can|could|do) you hear me\b|\bcan you hear\b"
    r"|\bam i (?:audible|coming through|breaking up)\b"
    r"|\bare you able to hear\b|\bis anyone (?:there|on the line)\b"
    r"|\banybody there\b|\bhello,? (?:are you|you) (?:there|hear)\b"
    r"|\byou'?re? (?:cutting|breaking) (?:out|up)\b|\bi (?:can'?t|cannot) hear (?:you|anything)\b"
    r"|\bare you there\b|\bthe line (?:is |went |has gone )?(?:dead|quiet|silent|cut|cutting)\b",
    re.I)

# Continuers that CANNOT mean "yes" — non-lexical hums and explicit "keep going" prompts.
# A caller never answers "shall I book it?" with "mm-hmm" or "go on", so these are safe to
# read as a backchannel with no regard for what is pending.
_CONTINUER_RE = re.compile(
    r"^\s*(?:"
    r"(?:m+|h+m+|u+h+|a+h+|o+h+|e+r+|u+m+)[- ]?(?:hm+|huh|hmm+|m+)?"   # mm, hmm, mhm, uh-huh, aha, oh
    r"|mm+[- ]?h+m+|uh[- ]?huh|hm+|hmm+|mhm+|aha+|uh[- ]?hm+|mkay"
    # "go ahead" is NOT here: it is how callers say yes to "shall I book it?". It lives in the
    # guarded agreement set below, a nod only when nothing is waiting on a yes.
    r"|go on|carry on|keep going|continue|i'?m listening|i am listening"
    r"|and (?:then|\?)|and\??|go on then|please (?:go on|continue)|carry on then"
    r")[\s.,!?-]*$", re.I)

# Agreement words that DOUBLE as consent — a nod when nothing is pending, a "yes" when a
# question or a staged action is. Only ever read as a backchannel under the guarded branch
# (agent made a statement AND nothing is pending); otherwise left entirely to the normal
# affirmation path, so booking / handoff consent is never swallowed.
_AGREEMENT_RE = re.compile(
    r"^\s*(?:(?:ok(?:ay)?|okey|kay|alright|all right|right|sure|yeah|yep|yup|yes|ya|"
    r"i see|i understand|understood|got it|gotcha|fair enough|makes sense|true|indeed|"
    r"cool|nice|great|good|fine|gotcha|noted|go ahead)"
    r"(?:[\s,]+(?:then|now|sure|cool|great|fine|ok(?:ay)?))*)[\s.,!?-]*$", re.I)


def classify(text: str, *, agent_asked_question: bool, is_opening_turn: bool,
             pending: bool = False) -> str:
    """Which conversational signal the caller's LATEST utterance is, or SIGNAL_NONE.

    ``agent_asked_question`` — did the agent's immediately preceding line end in a question?
      A bare agreement answering a question is consent, not a nod, so this steers the
      ambiguous-agreement branch away from BACKCHANNEL.
    ``is_opening_turn`` — is this the caller's FIRST utterance of the call? A greeting word
      here is a real greeting (SIGNAL_GREETING); the same word later is a presence check.
    ``pending`` — the call site's verdict that a staged action / offer / awaited answer is
      outstanding. Combined with ``agent_asked_question`` it guards the ambiguous-agreement
      words: those become BACKCHANNEL only when BOTH say nothing is waiting on a "yes".

    Pure: no I/O, no engine types. The reply for a CHECK_IN / BACKCHANNEL is built by the
    companion functions below from the agent's actual last line.
    """
    t = (text or "").strip()
    if not t:
        return SIGNAL_NONE

    # An explicit audibility probe is a check-in no matter where it lands or who spoke first —
    # it names the channel, so it can never be a greeting or a nod. Checked before greeting so
    # "hello, are you there?" is a check-in, not a greeting.
    if _CHECK_IN_RE.search(t):
        return SIGNAL_CHECK_IN

    # A bare greeting word: a real greeting on the opening turn, a "did the line drop?" probe
    # once the conversation is already underway (we have spoken, they are answering silence).
    if _GREETING_RE.match(t):
        return SIGNAL_GREETING if is_opening_turn else SIGNAL_CHECK_IN

    # An unambiguous continuer — a hum or an explicit "go on". Never consent, so always safe.
    if _CONTINUER_RE.match(t):
        return SIGNAL_BACKCHANNEL

    # An agreement word that could be a nod OR a "yes". Read it as a nod ONLY when nothing is
    # waiting on a yes: the agent made a statement (not a question) and no action/offer/answer
    # is pending. Otherwise leave it to the normal affirmation path — consent must win.
    if _AGREEMENT_RE.match(t):
        if not agent_asked_question and not pending:
            return SIGNAL_BACKCHANNEL
        return SIGNAL_NONE

    return SIGNAL_NONE


# ------------------------------------------------------------------ the lines the caller hears

# Prefixes that acknowledge the line issue before repeating what was missed. Rotated by a
# cheap, stateless index so a caller who probes twice does not hear the identical opener both
# times (responder Rule 7 in spirit) — the REPEATED CONTENT is the point, the wrapper varies.
_CHECK_IN_PREFIX = (
    "Yes, I'm here",
    "I'm still here",
    "Sorry about that — I'm here",
    "Yes, still with you",
)
_CHECK_IN_NO_CONTENT = (
    "Yes, I'm here — can you hear me okay?",
    "I'm still here — are you able to hear me?",
    "Yes, I'm on the line — is the audio clear on your end?",
)


def _condense(reply: str | None) -> str:
    """The agent's last line, tidied for a spoken repeat: whitespace collapsed and any
    trailing pleasantry left intact. Empty string when there is nothing to repeat."""
    return re.sub(r"\s+", " ", (reply or "").strip())


def check_in_reply(agent_last_reply: str | None, *, variant: int = 0) -> str:
    """What to say to a presence / audibility probe: confirm we are here, then REPEAT the
    thing the caller missed — the agent's own last line — verbatim in substance.

    ``agent_last_reply`` is the agent's immediately preceding utterance (fetched from the
    transcript by the call site). When we have it, the caller hears reassurance plus the
    missed content, which is the whole point — "Hello?" means "say that again", not "hi".
    When we have nothing to repeat (rare — the agent always speaks first), we fall back to a
    plain audibility check. ``variant`` rotates the wording so a second probe differs.
    """
    last = _condense(agent_last_reply)
    if not last:
        return _CHECK_IN_NO_CONTENT[variant % len(_CHECK_IN_NO_CONTENT)]
    prefix = _CHECK_IN_PREFIX[variant % len(_CHECK_IN_PREFIX)]
    # If the last line was itself a question, "I'm here — <question>?" reads naturally; if a
    # statement, the same shape still works ("I'm here — <statement>"). One dash join keeps it
    # to one spoken breath.
    return f"{prefix} — {last}"


_BACKCHANNEL_STATEMENT_PREFIX = (
    "",                       # re-deliver the point plainly, no filler
    "Sure — ",
    "Right — ",
)


def backchannel_reply(agent_last_reply: str | None, *, agent_asked_question: bool,
                      variant: int = 0) -> str:
    """What to say to a continuer / nod: carry on from where we were, do NOT greet or restart.

    A nod holds the floor open for us, so the substance of the reply is the agent's OWN last
    line moved forward one step:

      * the last line was a QUESTION the caller has not answered ("…is that right?", "shall I
        share them?") → they hummed instead of answering, so re-put the question, lightly, so
        the call does not stall waiting on an answer neither side is giving.
      * the last line was a STATEMENT/offer → acknowledge and re-offer the hook it ended on,
        so the caller has the same thing to respond to without the agent starting over.

    When there is no last line to build on (rare), a gentle, forward invitation. Never a
    greeting — that is the exact failure this replaces.
    """
    last = _condense(agent_last_reply)
    if not last:
        return "Sure — where would you like me to start?"
    if agent_asked_question:
        # They acknowledged but did not answer — surface the question again, unchanged, so the
        # decision in front of them stays in front of them.
        return last
    prefix = _BACKCHANNEL_STATEMENT_PREFIX[variant % len(_BACKCHANNEL_STATEMENT_PREFIX)]
    return f"{prefix}{last}"


def ends_with_question(text: str | None) -> bool:
    """Whether the agent's last line was a question — a spoken "?" or a wh-/aux-lead. Used by
    the call site to set ``agent_asked_question`` from the transcript without a stored flag."""
    t = (text or "").strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    # A question mark is not always transcribed; catch the common spoken openers on the LAST
    # sentence so "shall I book it" / "would you like to see it" still read as questions.
    tail = re.split(r"(?<=[.!?])\s+", t)[-1].strip().lower()
    return bool(re.match(
        r"^(?:shall|should|would|could|can|may|do|does|did|are|is|was|were|will|"
        r"have|has|which|what|when|where|who|how|why)\b", tail))


# ---------------------------------------------------------------------------------------
# Built-in self-test — runs without pytest.
#   python -m app.engine.planner.conversational_signal
# ---------------------------------------------------------------------------------------
def _selftest() -> int:
    failures: list[str] = []

    def check(name, cond):
        if not cond:
            failures.append(name)

    c = classify

    # --- opening greeting stays a greeting; the same word mid-call is a presence check ------
    for g in ("hi", "hello", "hey", "hey there", "helloo", "hiya", "namaste", "yo"):
        check(f"open_greeting::{g}",
              c(g, agent_asked_question=False, is_opening_turn=True) == SIGNAL_GREETING)
        check(f"midcall_greeting_is_checkin::{g}",
              c(g, agent_asked_question=False, is_opening_turn=False) == SIGNAL_CHECK_IN)

    # --- the reported bug: "Hello?" mid-call is a CHECK_IN, never a greeting ----------------
    for probe in (
            "Hello?", "hello hello", "hello, are you there?", "are you there",
            "are you still there", "you there?", "you still there", "can you hear me",
            "can you hear me?", "am I audible", "is anyone there", "anybody there?",
            "you're breaking up", "I can't hear you", "the line went quiet"):
        check(f"checkin::{probe}",
              c(probe, agent_asked_question=False, is_opening_turn=False) == SIGNAL_CHECK_IN)
    # a check-in is a check-in even on the opening turn (they answered our opener with silence)
    check("checkin_on_open",
          c("can you hear me?", agent_asked_question=True, is_opening_turn=True) == SIGNAL_CHECK_IN)

    # --- unambiguous continuers are ALWAYS backchannels, pending or not ---------------------
    for cont in ("mm-hmm", "mmhmm", "mhm", "uh huh", "uh-huh", "hmm", "hm", "aha", "oh",
                 "go on", "carry on", "keep going", "i'm listening",
                 "and then?", "go on then", "please continue"):
        check(f"continuer_no_pending::{cont}",
              c(cont, agent_asked_question=False, is_opening_turn=False,
                pending=False) == SIGNAL_BACKCHANNEL)
        check(f"continuer_even_pending::{cont}",
              c(cont, agent_asked_question=True, is_opening_turn=False,
                pending=True) == SIGNAL_BACKCHANNEL)

    # --- GUARDED agreement words: a nod only when nothing waits on a "yes" ------------------
    # "go ahead" is consent to a staged action, so it is guarded like the other yes-words.
    for ack in ("ok", "okay", "alright", "right", "sure", "yeah", "yep", "i see",
                "got it", "understood", "makes sense", "cool", "fine", "go ahead"):
        # statement + nothing pending → a nod (this is the case the engine used to greet)
        check(f"ack_nod::{ack}",
              c(ack, agent_asked_question=False, is_opening_turn=False,
                pending=False) == SIGNAL_BACKCHANNEL)
        # a question was asked → this is an ANSWER, leave it to the affirmation path
        check(f"ack_answering_question::{ack}",
              c(ack, agent_asked_question=True, is_opening_turn=False,
                pending=False) == SIGNAL_NONE)
        # something is pending (staged booking, offer) → leave it; consent must win
        check(f"ack_pending::{ack}",
              c(ack, agent_asked_question=False, is_opening_turn=False,
                pending=True) == SIGNAL_NONE)

    # --- anything with real content is NONE — never a nod, never swallowed -----------------
    for content in (
            "yes book the 3 pm slot", "yes please book it", "ok what about the 2 bhk",
            "sure, is there parking", "right so how much is it", "yeah I want to reschedule",
            "no", "no thanks", "not interested", "what is the price",
            "tell me about the amenities", "my name is Rohan", "9 8 7 6"):
        check(f"content_is_none::{content}",
              c(content, agent_asked_question=False, is_opening_turn=False,
                pending=False) == SIGNAL_NONE)
    # the opening turn: real content is still NONE (not a greeting)
    check("open_content_none",
          c("do you have a 3 bhk", agent_asked_question=False, is_opening_turn=True) == SIGNAL_NONE)

    # --- empty / whitespace is NONE --------------------------------------------------------
    check("empty_none", c("", agent_asked_question=False, is_opening_turn=False) == SIGNAL_NONE)
    check("blank_none", c("   ", agent_asked_question=False, is_opening_turn=True) == SIGNAL_NONE)

    # --- check_in_reply: confirm presence AND repeat the missed line -----------------------
    last = "The 3 BHK ranges from 1,239 to 1,560 square feet. Would you like to see it?"
    r0 = check_in_reply(last, variant=0)
    check("checkin_reply_reassures", r0.lower().startswith("yes, i'm here"))
    check("checkin_reply_repeats", "1,239 to 1,560 square feet" in r0)
    # a second probe varies the wrapper but still repeats the content
    r1 = check_in_reply(last, variant=1)
    check("checkin_reply_varies", r1 != r0 and "1,239 to 1,560 square feet" in r1)
    # nothing to repeat → a plain audibility check, never a greeting
    rn = check_in_reply(None)
    check("checkin_reply_nocontent", "hear me" in rn.lower() and "hello" not in rn.lower())
    check("checkin_reply_nocontent_blank", "hear me" in check_in_reply("   ").lower())

    # --- backchannel_reply: carry on, never greet or restart -------------------------------
    stmt = "A few options have come up along those lines, and I'd love to share them."
    b = backchannel_reply(stmt, agent_asked_question=False, variant=0)
    check("backchannel_carries_on", "share them" in b and "hello" not in b.lower())
    # the last line was a question they hummed at → re-put the question unchanged
    q = "You'd enquired about a 2 BHK at DSR Highland Greenz, is that right?"
    bq = backchannel_reply(q, agent_asked_question=True)
    check("backchannel_requestions", bq == q)
    # nothing to build on → a forward invitation, never a greeting
    bn = backchannel_reply(None, agent_asked_question=False)
    check("backchannel_nocontent", "start" in bn.lower() and "hello" not in bn.lower())

    # --- ends_with_question --------------------------------------------------------------
    for q in ("Would you like to see it?", "shall I book it", "which one suits you",
              "The 2 BHK is 1,050 sq ft. Do you want to visit"):
        check(f"is_question::{q}", ends_with_question(q))
    for s in ("The 2 BHK is 1,050 square feet.", "I've booked your visit for 3 PM.",
              "Thanks, have a great day."):
        check(f"not_question::{s}", not ends_with_question(s))
    check("q_empty", not ends_with_question(""))

    if failures:
        print(f"conversational_signal self-test: {len(failures)} FAILED -> {failures}")
        return 1
    print("conversational_signal self-test: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
