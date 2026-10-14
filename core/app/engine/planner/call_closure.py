"""Deterministic guard for the ONE action a phone agent can never take back: hanging up.

WHY THIS EXISTS. ``end_call`` is decided by the model alone (``proposal.action``), and the
prompt's rule "never end the call yourself because they said no to one [offer]"
(prompts.py) is guidance the model does not reliably follow. In a real call it read a plain
"no" to "shall I book a site visit?" as the conversation being over and hung up on a caller
who was still full of questions — a live lead, dropped. Prompt-only guardrails do not hold;
a dropped call is unrecoverable, so this backstops the decision in code.

THE RULE (chosen: require an explicit closure signal). A hang-up is honoured only when the
CALLER'S OWN WORDS carry positive evidence that they are done — a farewell ("bye", "take
care"), a completion ("that's all", "nothing else", "I'm done"), or an explicit request to
end ("you can hang up"). A bare "no" / "no thanks" / "nope" / "ok" is NOT closure: it
declines whatever was just offered and the call stays open. This is deliberately biased to
the SAFE direction — the cost of a false negative is one extra "anything else?" turn, which
is recoverable; the cost of a false positive is hanging up on an engaged caller, which is
not. Phrases that double as an offer-decline ("I'm good", "that's fine", "all good") are
therefore treated as NON-closure on purpose, even though they can sometimes mean goodbye.

Pure and dependency-free (stdlib ``re`` only), so it unit-tests without a database or any
engine types — run ``python -m app.engine.planner.call_closure`` for the built-in self-test.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.engine.planner.hindi_script import normalize_hindi_script

# Positive closure signals in the CALLER's utterance. Matched on lowercased text with word
# boundaries so a signal word inside another word ("bye" in "byline") cannot fire. Only
# UNAMBIGUOUS terminal phrases live here; anything that commonly means "no thanks to that
# offer" is deliberately excluded so it keeps the call open.
_CLOSURE_PATTERNS = (
    # farewells
    r"\bgood\s?bye\b", r"\bbye\b", r"\bbye[-\s]?bye\b",
    r"\bsee (?:you|ya)\b", r"\btalk (?:to you )?(?:later|soon)\b",
    r"\bcatch you later\b", r"\btake care\b",
    r"\bhave a (?:good|nice|great|lovely|wonderful) (?:day|night|evening|one|weekend)\b",
    r"\bgood night\b",
    # Hindi farewell -- transliterated from Devanagari अलविदा by normalize_hindi_script
    # (hindi_script.py) before this pattern ever runs; a bare "bye" said in English mid-Hindi
    # sentence already matches the English pattern above, so this covers the genuinely
    # Hindi-only farewell that pattern would miss.
    r"\balvida\b",
    # completion — "that's all / it / everything", "that will/would be all"
    r"\bthat'?s (?:all|it|everything)\b",
    r"\bthat (?:will|would|'?ll|'?d) be all\b",
    r"\bnothing (?:else|more|further)\b",
    # "I don't want/need anything else" -- the same completion as "nothing else", just phrased
    # the other way round. A real call said exactly this ("no thank you i don't want anything
    # else") and it matched NEITHER this function NOR reads_like_no_further_need (8 words, over
    # that function's 6-word cap) -- so the turn fell all the way through to "no closure signal,
    # keep the call open," and the agent kept restating the booking and asking a fresh question
    # on every subsequent turn instead of winding down, however plainly the caller had said they
    # were done.
    r"\b(?:don'?t|do not) (?:want|need) anything (?:else|more)\b",
    r"\bno (?:more|further) (?:questions|queries|help|thanks needed)\b",
    r"\b(?:i'?m|we'?re|i am|we are) (?:all )?done\b",
    r"\bthat'?s (?:all|it) for now\b",
    # explicit end-the-call requests
    r"\byou can (?:hang up|disconnect|end (?:the|this) call)\b",
    r"\bhang up\b", r"\bend (?:the|this) call\b", r"\bcut the call\b",
    r"\bdisconnect (?:the|this|now|please)\b", r"\bdisconnect\b",
    r"\bi'?ll let you go\b",
    r"\b(?:i|we) (?:have to|need to|gotta|have got to|got to|must) go\b",
)
_CLOSURE_RE = re.compile("|".join(_CLOSURE_PATTERNS), re.I)

# Sign-off phrasing in the AGENT's drafted reply. When end_call is suppressed but the reply
# the responder wrote was itself a goodbye, it is swapped for a forward question so the agent
# does not bid farewell and then stay on the line. Kept separate from the caller set because
# agent sign-offs are phrased differently ("thank you for calling", "feel free to reach out").
_SIGNOFF_PATTERNS = (
    r"\bgood\s?bye\b",
    r"\bhave a (?:good|nice|great|lovely|wonderful) (?:day|night|evening|one|weekend)\b",
    r"\btake care\b", r"\bgood night\b",
    r"\bthank you for calling\b", r"\bthanks for calling\b",
    r"\bfeel free to (?:call|reach out|get in touch|contact us)\b",
    r"\bwe look forward to\b", r"\bhave a great rest of your\b",
    r"\bhope you have a\b",
)
_SIGNOFF_RE = re.compile("|".join(_SIGNOFF_PATTERNS), re.I)

# Any "anything else?"-style continued-engagement check in the AGENT's drafted text, in
# whatever words it used -- not just the three canned strings below (REENGAGE,
# OUTBOUND_REENGAGE, OUTBOUND_MISTAKE_REENGAGE). A real call had the model append its OWN
# "Is there anything else I can do for you right now?" straight onto a record_do_not_contact
# confirmation, in a turn whose action was never "end_call" -- so reengage_count stayed at 0
# and the counter below had no idea an invitation had already gone out. Used by turn_runner
# to catch that ask regardless of which action carried it, see reads_like_reengage_offer.
_REENGAGE_OFFER_RE = re.compile(
    r"\banything else\b|\bsomething else\b|\banything (?:more|further)\b|"
    r"\bshall i let you go\b|\bwas there anything\b|\bis there anything\b",
    re.I)

# The reply used in place of a suppressed sign-off: forward-moving, so the call keeps going.
REENGAGE = "Is there anything else I can help you with?"

# Announced when the call is wound down deliberately (see reads_like_no_further_need). Unlike
# REENGAGE this is a real goodbye AND the hang-up stands — it is only ever reached on the
# SECOND winding-down turn, after "anything else?" has already been asked and declined, so the
# caller has had their chance to raise anything. Warm and tenant-neutral.
SIGNOFF = "Thank you for calling — have a great day!"

# The outbound counterpart of SIGNOFF. WE placed the call, so "thank you for calling" is the
# wrong register; this accepts the decline gracefully and thanks them for their time instead.
# Used only when the drafted reply was not already a clean sign-off of its own.
OUTBOUND_SIGNOFF = "No problem at all — thank you for your time, and have a great day!"

# Acknowledging re-ask for an outbound caller who says the enquiry was a MISTAKE / accidental.
# It ACKNOWLEDGES what they said ("I understand") and, since we have them on the line, offers ONE
# soft chance to engage before letting them go — rather than either a blank "anything else?" or a
# cold hang-up. If they then decline, the bounded loop ends the call on the next turn.
OUTBOUND_MISTAKE_REENGAGE = ("No problem at all — I understand. Since I've got you, was there "
                             "anything about it you'd like me to help with, or shall I let you go?")

# The outbound counterpart of REENGAGE. WE called them, so the bounded single re-ask should
# still sound like a real person on an outbound call: acknowledge first, offer help once, and
# give a graceful way out rather than the bare inbound "anything else?". Inbound keeps REENGAGE
# byte-for-byte (the caller dialled US there), so inbound behaviour is unchanged.
OUTBOUND_REENGAGE = ("No problem at all — was there anything about it you'd like me to help "
                     "with, or shall I let you go?")

# "Winding-down" utterances: a caller who is done but did NOT say an unambiguous farewell.
# Bare acknowledgements ("ok", "thanks", "great") and soft declines ("no", "nope", "no need").
# These are NOT hard closures (a single one keeps the call open) — they only end the call on
# the SECOND consecutive pass, which is what caps the "anything else?" loop at two turns.
_NO_FURTHER_NEED_PATTERNS = (
    r"\bno\b", r"\bnope\b", r"\bnah\b", r"\bnothing\b", r"\bno need\b",
    r"\bnot (?:really|now|right now)\b",
    r"\bok(?:ay)?\b", r"\bthanks?\b", r"\bthank you\b", r"\bcheers\b",
    r"\bthat'?s (?:fine|ok|okay|alright|great|good|all|it)\b",
    r"\bthat (?:should|will|would) be (?:it|all)\b",
    r"\ball good\b", r"\b(?:i'?m|we'?re) good\b",
    r"\bcool\b", r"\bgreat\b", r"\bsure\b", r"\bfine\b", r"\balright\b", r"\bperfect\b",
    r"\bthat(?:'s| is)? (?:works?|sorted|done|covered|helpful)\b",
    # Hindi/Hinglish equivalents (Roman script — a real call's "anything else?" loop never
    # ended on a Hindi "nhi" because every pattern here was English-only; the caller's own
    # negation/acknowledgement never matched, so the bounded wind-down loop never engaged
    # and the model was left to freely improvise an interpretation instead — which is also
    # how a plain "nhi" (declining a callback) got expanded into "not interested in the
    # 2 BHK" on a real call).
    r"\bnahi+n?\b", r"\bnhi+\b", r"\bnako\b",
    r"\bthik\s?hai\b", r"\btheek\s?hai\b", r"\bbas\b", r"\bbas\s?(?:hai|hogaya|ho gaya)\b",
    r"\bkoi\s?baat\s?nahi+n?\b", r"\bshukriya\b", r"\bdhanyavaad\b", r"\bdhanyawad\b",
    r"\bhaan\s?thik\s?hai\b",
)
_NO_FURTHER_NEED_RE = re.compile("|".join(_NO_FURTHER_NEED_PATTERNS), re.I)

# A question hiding inside an otherwise short reply means the caller is NOT done — never wind
# the call down on it. Catches wh-questions, "can you / is there / do you"-style openers, and
# any literal question mark.
_INTERROGATIVE_RE = re.compile(
    r"\?|\b(?:what|why|how|when|where|which|who|whose)\b|"
    r"\b(?:can|could|would|will|do|does|did|is|are|have|any|another|more)\s+"
    r"(?:you|there|i|we|it|that|the)\b|"
    r"\btell me\b|\bwhat about\b",
    re.I,
)


# UNAMBIGUOUS disinterest in the WHOLE call — "not interested", "not looking to buy", "already
# bought". On an OUTBOUND pitch these mean the operator policy applies: accept it gracefully and
# close, never push (outbound_flow marketing guidance). Deliberately narrow: it must express
# disinterest in the offer/call itself, NOT a decline of one option ("no, not that flat"), so an
# engaged lead is never dropped. A bare "no"/"no thanks" is intentionally NOT here — that stays a
# soft wind-down handled by the bounded two-turn loop, which keeps the call open one more turn.
_NOT_INTERESTED_PATTERNS = (
    r"\bnot (?:really |at all |very )?interested\b",
    r"\bno longer (?:interested|looking)\b",
    r"\bnot looking (?:to (?:buy|purchase|invest)|for (?:anything|a|any|any ?more)|"
    r"right now|at the moment|currently)\b",
    r"\bnot (?:keen|planning to buy|in the market)\b",
    r"\balready (?:bought|purchased|booked|sorted|invested)\b",
)
_NOT_INTERESTED_RE = re.compile("|".join(_NOT_INTERESTED_PATTERNS), re.I)


def reads_like_not_interested(text: str | None) -> bool:
    """True when the caller expresses unambiguous disinterest in the whole offer/call.

    Distinct from ``reads_like_no_further_need`` (soft wind-downs kept open one more turn): a
    match here is positive evidence the pitch is declined outright, which on an outbound call the
    operator policy answers by closing gracefully rather than re-engaging. Not consulted on
    inbound calls — the caller dialled US there, so "not interested" carries no such instruction.
    """
    return bool(_NOT_INTERESTED_RE.search(text or ""))


# Accidental / mistaken enquiry — "I enquired by mistake", "sorry, that was accidental", "I
# didn't mean to enquire". On an OUTBOUND call this is disengagement just like "not interested":
# the person is telling us there is no live intent, so the right move is to acknowledge, reassure
# and close gracefully — never to ask "anything else?" as if they had a need. Spelling variants
# ("accidentally"/"accidently") are both matched because STT and typos produce both. Narrow on
# purpose (mistake/accident wording); rule 2's has_live_question guard still keeps a genuine
# question open, so "sorry, didn't mean to interrupt — what's the price?" never closes.
_MISTAKEN_ENQUIRY_PATTERNS = (
    r"\bby (?:mistake|accident)\b",
    r"\baccident(?:al)?ly\b",            # accidentally / accidently
    r"\baccidental\b",
    r"\b(?:did|do)(?:n'?t| not) (?:mean|intend) to\b",
    r"\bmust (?:have|'?ve) been (?:a|an) (?:mistake|error|accident)\b",
    r"\bwrong(?:ly)? (?:enquir|inquir|number|contact)",
    r"\bnot sure why i (?:enquir|inquir|call|contact|reach|register)",
)
_MISTAKEN_ENQUIRY_RE = re.compile("|".join(_MISTAKEN_ENQUIRY_PATTERNS), re.I)


def reads_like_mistaken_enquiry(text: str | None) -> bool:
    """True when the caller says their enquiry was a mistake / accidental.

    On an outbound call this is disengagement (handled like ``reads_like_not_interested``): the
    caller is telling us there is no live intent behind the call, which the operator policy
    answers by acknowledging and closing gracefully rather than re-engaging. Not consulted on
    inbound calls. Guarded by has_live_question at the call site so a real question still wins.
    """
    return bool(_MISTAKEN_ENQUIRY_RE.search(text or ""))


def has_live_question(text: str | None) -> bool:
    """True when the caller's utterance contains a question / a request to continue.

    A safety guard for the deterministic closers: whatever else the sentence contains, an open
    question means the caller is NOT done, so the call must never be wound down or closed on it.
    Shares the interrogative pattern used by ``reads_like_no_further_need`` so the two agree.
    """
    return bool(_INTERROGATIVE_RE.search((text or "").strip().lower()))


def reads_like_no_further_need(text: str | None) -> bool:
    """True when the caller is winding down — a short acknowledgement or soft decline with no
    question in it — WITHOUT an unambiguous farewell.

    Distinct from ``caller_signalled_closure`` (hard farewells that end the call on turn one):
    a single "no"/"ok"/"thanks" is deliberately kept open by that function so declining an
    offer never hangs up. This one is consulted only after "anything else?" has already been
    asked, to decide whether a SECOND winding-down turn should end the call. Guarded three
    ways so it never swallows a real request: it fires only on a SHORT utterance, only when
    no interrogative is present, and only when a positive winding-down phrase matches.
    """
    low = normalize_hindi_script(text or "").strip().lower()
    if not low or len(low.split()) > 6:
        return False
    if _INTERROGATIVE_RE.search(low):
        return False
    return bool(_NO_FURTHER_NEED_RE.search(low))


def caller_signalled_closure(text: str | None) -> bool:
    """True when the caller's own words positively signal they want to end the call.

    Positive-evidence only: absent a match, the call is kept open. A bare "no"/"no thanks"/
    "nope"/"ok" never matches, which is the whole point — declining an offer is not goodbye.
    """
    return bool(_CLOSURE_RE.search(normalize_hindi_script(text or "")))


def reads_like_signoff(text: str | None) -> bool:
    """True when the AGENT's drafted reply is a farewell/sign-off."""
    return bool(_SIGNOFF_RE.search(text or ""))


def reads_like_reengage_offer(text: str | None) -> bool:
    """True when the AGENT's drafted reply asks an "anything else?"-style continued-
    engagement question, in whatever wording -- used to mark that an invitation to wind
    down has already gone out even on a turn that never goes through decide_end_call (e.g.
    one where the model appends it to a tool-confirmation sentence of its own accord)."""
    return bool(_REENGAGE_OFFER_RE.search(text or ""))


# ---------------------------------------------------------------------------------------
# The one place that decides what happens when the model proposes end_call.
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ClosureDecision:
    """The resolved outcome of a proposed ``end_call``.

    ``end``            – True to hang up; False to keep the call open (respond instead).
    ``text``           – replacement reply, or None to keep the responder's drafted text.
    ``reengage_count`` – the new value for ``session['reengage_count']``; None clears it.
    ``note``           – a trace note recording which rule fired and why.
    """
    end: bool
    text: str | None
    reengage_count: int | None
    note: str


def decide_end_call(
    user_input: str | None,
    agent_text: str | None,
    *,
    outbound: bool = False,
    identity_confirmed: bool = False,
    reengage_count: int = 0,
) -> ClosureDecision:
    """Resolve a model-proposed ``end_call`` into a concrete, bounded outcome.

    Hanging up cannot be undone, so this is positive-evidence-only and biased to the safe
    (keep-open) direction. It unifies every path that could ask "anything else?" behind ONE
    counter so the loop is bounded to a single re-engage however it is reached — closing the
    leak where a suppressed sign-off asked once WITHOUT counting, letting a later soft "no" ask
    again. Precedence, highest first:

      1. Explicit farewell in the caller's words  -> end now (any channel, turn one).
      2. OUTBOUND, right person reached, no live question, unambiguous disinterest
         ("not interested", "not looking to buy") -> close gracefully, turn one. This is the
         outbound operator policy — accept the decline, never push — and it is why the call no
         longer pesters a caller who already said they are not interested (and, often, driving).
      3. Soft wind-down (short ack / bare "no", no question) -> bounded two-turn loop: ask
         "anything else?" once, then sign off and let the hang-up stand on the next one.
      4. No closure signal at all -> keep the call open; if the drafted reply was itself a
         farewell, swap it for the forward question AND count it, so the single-ask bound holds.

    Rules 2 and its disinterest test apply ONLY when ``outbound`` — an inbound caller dialled US,
    so "not interested" there carries no close-the-call instruction and behaviour is unchanged.
    """
    prior = int(reengage_count or 0)

    # 1. The caller's own words end the call — honoured on every channel, on turn one.
    if caller_signalled_closure(user_input):
        return ClosureDecision(
            end=True, text=None, reengage_count=None,
            note=f"end_call honoured: explicit farewell ({user_input!r}); reset.")

    # 2. Outbound pitch, identity confirmed, no live question, unambiguous disinterest ("not
    #    interested", "not looking to buy"): accept the decline and close gracefully, turn one —
    #    never push. A live question or a mere offer-decline ("no") does NOT reach here.
    if (outbound and identity_confirmed
            and not has_live_question(user_input)
            and reads_like_not_interested(user_input)):
        text = agent_text if reads_like_signoff(agent_text) else OUTBOUND_SIGNOFF
        return ClosureDecision(
            end=True, text=text, reengage_count=None,
            note=f"end_call honoured: outbound decline ({user_input!r}); graceful close, turn one.")

    # 2b. Outbound, identity confirmed, no live question, a MISTAKEN / accidental enquiry ("sorry,
    #     I enquired by accident"): the caller says it was a mistake — do NOT hang up cold and do
    #     NOT interrogate. ACKNOWLEDGE it and, since we have them on the line, offer ONE soft
    #     re-ask. This is folded into the SAME bounded re-engage counter as the soft wind-down, so
    #     a following "no" ends the call on the next turn (ask once, then let go); on the second
    #     pass here, sign off and end.
    if (outbound and identity_confirmed
            and not has_live_question(user_input)
            and reads_like_mistaken_enquiry(user_input)):
        if prior >= 1:
            text = agent_text if reads_like_signoff(agent_text) else OUTBOUND_SIGNOFF
            return ClosureDecision(
                end=True, text=text, reengage_count=None,
                note=f"end_call honoured: 2nd pass after mistaken enquiry ({user_input!r}); signed off.")
        return ClosureDecision(
            end=False, text=OUTBOUND_MISTAKE_REENGAGE, reengage_count=1,
            note=f"outbound mistaken enquiry ({user_input!r}); acknowledged and re-asked once.")

    # The bounded single re-ask, phrased for the channel: an outbound caller hears an
    # acknowledging, real-person re-engage; inbound keeps the original line unchanged.
    _reengage = OUTBOUND_REENGAGE if outbound else REENGAGE

    # 3. Soft wind-down: ask once, end on the second.
    if reads_like_no_further_need(user_input):
        if prior >= 1:
            text = agent_text if reads_like_signoff(agent_text) else (
                OUTBOUND_SIGNOFF if outbound else SIGNOFF)
            return ClosureDecision(
                end=True, text=text, reengage_count=None,
                note=f"end_call honoured on 2nd wind-down ({user_input!r}); signed off and ended.")
        return ClosureDecision(
            end=False, text=_reengage, reengage_count=1,
            note=f"end_call held on 1st wind-down ({user_input!r}); asked once.")

    # 4. No closure signal — keep the call open. Swap a drafted farewell for the forward question
    #    and COUNT it (the leak fix), so a following soft "no" ends rather than re-asking; if the
    #    drafted reply was substantive, keep it and treat the turn as engagement (reset).
    if reads_like_signoff(agent_text):
        return ClosureDecision(
            end=False, text=_reengage, reengage_count=max(prior, 1),
            note=f"end_call suppressed: no closure signal ({user_input!r}); asked once.")
    return ClosureDecision(
        end=False, text=None, reengage_count=None,
        note=f"end_call suppressed: no closure signal ({user_input!r}); kept the call open.")


# ---------------------------------------------------------------------------------------
# Built-in self-test — runs without pytest.
#   python -m app.engine.planner.call_closure
# ---------------------------------------------------------------------------------------
def _selftest() -> int:
    failures: list[str] = []

    def check(name, cond):
        if not cond:
            failures.append(name)

    # --- genuine closures MUST be honoured ---
    for phrase in (
            "bye", "goodbye", "ok bye", "okay, goodbye then", "bye bye",
            "no that's all, thanks", "that's all for now", "that's it thanks",
            "that's everything, thank you", "that will be all", "that would be all",
            "nothing else thanks", "nothing more for now", "no more questions",
            "i'm done", "we're done here", "i am all done",
            "you can hang up now", "please hang up", "end the call",
            "i have to go", "i gotta go", "i'll let you go", "take care, bye",
            "have a good day", "see you", "talk to you later",
            # the exact reported call: an 8-word explicit decline, missed before this fix
            "no thank you i don't want anything else",
            "no i don't need anything else", "i don't want anything more"):
        check(f"closure::{phrase}", caller_signalled_closure(phrase))

    # --- offer-declines and mid-call turns MUST NOT be treated as closure ---
    for phrase in (
            "no", "no thanks", "no thank you", "nope", "not now", "not really",
            "not interested", "ok", "okay", "alright", "fine", "sure",
            "i'm good", "i'm good thanks", "that's fine", "all good", "no need",
            "maybe later", "not right now",
            # the exact bug: declining the visit offer is not a hang-up
            "no i don't want to schedule a visit",
            "no i don't need a site visit",
            # substantive questions must never read as closure
            "what is the price of the 3 bhk", "is there a park facing flat",
            "can you tell me about the amenities"):
        check(f"open::{phrase}", not caller_signalled_closure(phrase))

    # --- word-boundary safety: signal words embedded in other words must not fire ---
    check("no_substring_bye", not caller_signalled_closure("what's the byelaw on parking"))
    check("empty_is_open", not caller_signalled_closure(""))
    check("none_is_open", not caller_signalled_closure(None))

    # --- winding-down utterances (soft closers used only on the 2nd pass) ---
    for phrase in (
            "no", "nope", "nah", "no thanks", "no thank you", "no need",
            "nothing", "ok", "okay", "ok thanks", "thanks", "thank you", "cheers",
            "that's fine", "that's ok", "that's great", "all good", "i'm good",
            "we're good", "cool", "great", "sure", "fine", "alright", "perfect",
            "not now", "not really",
            "that's it", "that's all", "that should be it", "that will be it",
            "that works", "that is sorted", "that's helpful", "that is done",
            # the exact reported call: "nhi" on the second wind-down pass, after "ok thank
            # you" on the first, never registered — the model was left to improvise and
            # read it as "not interested in the 2 BHK" instead of ending the call
            "nhi", "nahi", "nahin", "thik hai", "theek hai", "bas", "shukriya",
            "dhanyavaad", "koi baat nahi",
            # Devanagari script -- real Hindi STT output (DEEPGRAM_LANGUAGE=multi transcribes
            # Hindi speech in its native script, not Romanized), confirmed live to fail every
            # pattern above until normalize_hindi_script runs first.
            "नहीं", "ठीक है धन्यवाद", "शुक्रिया", "कोई बात नहीं"):
        check(f"winddown::{phrase}", reads_like_no_further_need(phrase))
    # a question — even a short one — is NOT winding down
    for phrase in (
            "what about parking?", "is there a gym", "can you tell me the price",
            "do you have 3 bhk", "how much is it", "what floor",
            "no what about the 2 bhk", "another question", "tell me more"):
        check(f"not_winddown::{phrase}", not reads_like_no_further_need(phrase))
    # a hard farewell is handled by caller_signalled_closure, and long substantive turns
    # must not read as winding down either
    check("winddown_empty", not reads_like_no_further_need(""))
    check("winddown_long",
          not reads_like_no_further_need("no I actually wanted to ask about the club house"))

    # --- agent sign-off detection ---
    check("signoff_have_a_day", reads_like_signoff("Thank you, have a great day!"))
    check("signoff_take_care", reads_like_signoff("Take care and goodbye."))
    check("signoff_thanks_for_calling", reads_like_signoff("Thanks for calling, bye."))
    check("not_signoff_question",
          not reads_like_signoff("The 3 BHK ranges from 1,239 to 1,560 square feet. "
                                 "Would you like to see it?"))
    check("not_signoff_answer",
          not reads_like_signoff("Yes, we have 2 BHK units in Towers A, B and C."))

    # --- unambiguous disinterest (outbound close signal) ---
    for phrase in (
            "not interested", "i'm not interested", "no i am not interested",
            "ok, I am not interested for now. I am driving",   # the exact reported turn
            "not really interested", "not at all interested",
            "no longer interested", "not looking to buy right now",
            "not looking for anything at the moment", "not in the market",
            "already bought a flat", "we already booked somewhere"):
        check(f"not_interested::{phrase}", reads_like_not_interested(phrase))
    # a mere offer-decline / a question must NOT read as whole-call disinterest
    for phrase in (
            "no", "no thanks", "not that flat", "no not the 3 bhk",
            "i'm interested", "yes i'm interested in the 2 bhk",
            "what else do you have", "are you interested in my feedback"):
        check(f"not_disinterest::{phrase}", not reads_like_not_interested(phrase))

    # --- mistaken / accidental enquiry (outbound close signal) ---
    for phrase in (
            "sorry, I enquired to you accidently",     # the exact reported turn (STT spelling)
            "sorry I enquired accidentally",
            "I think I enquired by mistake",
            "that was by accident",
            "I didn't mean to enquire",
            "I don't mean to waste your time, it was a mistake",
            "you must have got the wrong number",
            "not sure why I registered, sorry"):
        check(f"mistaken::{phrase}", reads_like_mistaken_enquiry(phrase))
    # a real question or plain engagement must NOT read as a mistaken enquiry
    for phrase in (
            "what is the price", "yes tell me more", "no thanks",
            "I enquired last week about the 3 bhk", "can you remind me what I wanted"):
        check(f"not_mistaken::{phrase}", not reads_like_mistaken_enquiry(phrase))

    # --- has_live_question ---
    for phrase in ("what's the price?", "do you have 3 bhk", "tell me more",
                   "is there parking", "how about the 2 bhk", "what about amenities"):
        check(f"live_q::{phrase}", has_live_question(phrase))
    for phrase in ("no thanks", "ok", "not interested", "i am driving", "that's all"):
        check(f"no_live_q::{phrase}", not has_live_question(phrase))

    # --- decide_end_call: the unified policy ---
    d = decide_end_call

    # inbound behaviour is UNCHANGED (outbound defaults False) --------------------------------
    # explicit farewell ends on turn one
    r = d("ok bye", "Take care!", outbound=False, reengage_count=0)
    check("dec_inbound_farewell_end", r.end and r.reengage_count is None)
    # soft "no" -> ask once (held), keep open
    r = d("no thanks", "Goodbye then.", outbound=False, reengage_count=0)
    check("dec_inbound_wind1_hold", (not r.end) and r.text == REENGAGE and r.reengage_count == 1)
    # second soft "no" -> sign off and end
    r = d("no thanks", "Sure.", outbound=False, reengage_count=1)
    check("dec_inbound_wind2_end", r.end and r.text == SIGNOFF and r.reengage_count is None)
    # a real question the model mis-planned as end_call -> keep open, keep the substantive answer
    r = d("what about the 2 bhk", "The 2 BHK is 1,050 sq ft.", outbound=False, reengage_count=0)
    check("dec_inbound_question_keepopen", (not r.end) and r.text is None)

    # THE COUNTER-LEAK FIX: a suppressed sign-off now counts, so the loop is bounded to ONE ask.
    #   turn A: long decline, no closure signal, agent drafted a farewell -> REENGAGE + count=1
    rA = d("ok I am not interested for now, I am driving", "Have a great day!",
           outbound=False, reengage_count=0)
    check("dec_leak_A_reengage_counts", (not rA.end) and rA.text == REENGAGE
          and rA.reengage_count == 1)
    #   turn B: a following soft "no" now ENDS (previously it asked 'anything else?' a 2nd time)
    rB = d("no thanks", "Bye.", outbound=False, reengage_count=1)
    check("dec_leak_B_ends", rB.end and rB.reengage_count is None)

    # OUTBOUND graceful close, turn one — the exact reported call --------------------------------
    r = d("ok, I am not interested for now. I am driving", "Alright then, goodbye.",
          outbound=True, identity_confirmed=True, reengage_count=0)
    check("dec_outbound_decline_end", r.end and r.reengage_count is None)
    check("dec_outbound_decline_keeps_signoff", r.text == "Alright then, goodbye.")
    # when the model did NOT draft a clean sign-off, the outbound sign-off is supplied
    r = d("not interested", "action was end_call", outbound=True, identity_confirmed=True)
    check("dec_outbound_decline_supplies_signoff", r.end and r.text == OUTBOUND_SIGNOFF)
    # outbound but a LIVE QUESTION alongside "not interested" -> must NOT close (still engaged)
    r = d("not interested in the 3 bhk, what about the 2 bhk?", "The 2 BHK is 1,050 sq ft.",
          outbound=True, identity_confirmed=True, reengage_count=0)
    check("dec_outbound_question_keepopen", not r.end)

    # OUTBOUND mistaken/accidental enquiry — the exact reported call: turn one ACKNOWLEDGES and
    # RE-ASKS once (kept open, counted), never a blank "anything else?" and never a cold hang-up.
    r = d("sorry, I enquired to you accidently", "Is there anything else I can help you with?",
          outbound=True, identity_confirmed=True, reengage_count=0)
    check("dec_outbound_mistake_reask", (not r.end) and r.text == OUTBOUND_MISTAKE_REENGAGE
          and r.reengage_count == 1)
    # ...then a following "no" ENDS the call on the next turn (bounded loop), with a sign-off.
    r = d("no", "Sure.", outbound=True, identity_confirmed=True, reengage_count=1)
    check("dec_outbound_mistake_then_no_ends", r.end and r.text == OUTBOUND_SIGNOFF
          and r.reengage_count is None)
    # a SECOND mistaken-enquiry pass (already re-asked once) also ends rather than re-asking again
    r = d("no it really was a mistake", "Okay.", outbound=True, identity_confirmed=True,
          reengage_count=1)
    check("dec_outbound_mistake_2nd_pass_ends", r.end and r.reengage_count is None)
    # a mistaken enquiry that ALSO carries a live question stays open (still engaged)
    r = d("I enquired by mistake, but what do you have in 2 bhk?", "We have 2 BHK in Tower A.",
          outbound=True, identity_confirmed=True, reengage_count=0)
    check("dec_outbound_mistake_question_keepopen", not r.end)
    # inbound: the mistaken-enquiry re-ask path is outbound-only; inbound keeps the call open
    r = d("sorry I enquired by mistake", "Okay.", outbound=False, reengage_count=0)
    check("dec_inbound_mistake_keepopen", not r.end)

    # OUTBOUND re-engage is the acknowledging real-person line; inbound keeps the bare REENGAGE.
    r = d("no", "Goodbye then.", outbound=True, identity_confirmed=True, reengage_count=0)
    check("dec_outbound_reengage_ack", (not r.end) and r.text == OUTBOUND_REENGAGE
          and r.reengage_count == 1)
    r = d("no", "Goodbye then.", outbound=False, reengage_count=0)
    check("dec_inbound_reengage_unchanged", (not r.end) and r.text == REENGAGE)
    # outbound but identity NOT yet confirmed -> the turn-one close does not apply; the call is
    # kept open (the outbound identity gate, which runs afterwards, drives this phase).
    r = d("not interested", "Goodbye.", outbound=True, identity_confirmed=False, reengage_count=0)
    check("dec_outbound_unconfirmed_no_fastclose", not r.end)
    # inbound "not interested" gets NO turn-one close — the disinterest fast path is outbound-only;
    # inbound keeps the call open (the caller dialled us and may still have questions).
    r = d("not interested", "Okay.", outbound=False, reengage_count=0)
    check("dec_inbound_not_interested_keepopen", not r.end)

    # --- reads_like_reengage_offer: catches an "anything else?"-style ask in the AGENT's
    # own drafted text regardless of wording, used to mark the wind-down counter on a turn
    # that never goes through decide_end_call at all (see P6-F3 in turn_runner.py) ---
    for phrase in (
            "I've marked your number as do-not-call, so you won't receive any more calls "
            "from us. Is there anything else I can do for you right now?",
            "Is there anything else I can help you with?",
            "No problem at all — was there anything about it you'd like me to help with, "
            "or shall I let you go?",
            "Anything else before I let you go?",
            "Is there something else you need?",
            "Was there anything further you wanted to ask?"):
        check(f"reengage_offer::{phrase[:40]}", reads_like_reengage_offer(phrase))
    for phrase in (
            "Your number has been removed from our call list.",
            "The 2 BHK is 1,050 square feet.",
            "Would you like to book a site visit?",
            ""):
        check(f"not_reengage_offer::{phrase[:40]}", not reads_like_reengage_offer(phrase))
    check("reengage_offer_none", not reads_like_reengage_offer(None))

    if failures:
        print(f"call_closure self-test: {len(failures)} FAILED -> {failures}")
        return 1
    print("call_closure self-test: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
