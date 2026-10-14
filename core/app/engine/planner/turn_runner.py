"""process_turn — the complete cycle. Steps 1-7 come from planner.plan(); this module
adds steps 8-12: validate, execute, verify, respond (grounded), and persist the trace.

The confirmation gate lives here: a tool marked requires_confirmation is never executed
on the turn it is proposed. It is parked as pending_action, read back to the customer,
and only released when the next turn confirms it.
"""
import asyncio
import difflib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import (Call, Conversation, Customer, DecisionTrace, Escalation, Message,
                           PromptTemplate, Tenant)
from app.engine.memory import working
from app.engine.planner import booking_claim
from app.engine.planner.text_normalize import normalize_quotes
from app.engine.planner import email_claim
from app.engine.planner import figure_claim
from app.engine.planner import source_conflict_claim
from app.engine.planner import booking_task as bt
from app.engine.planner import call_closure as closure
from app.engine.planner import consent
from app.engine.planner import conversational_signal
from app.engine.planner import date_claim
from app.engine.planner import feature_grounding as fg
from app.engine.planner import grounding_check
from app.engine.planner import outbound_flow
from app.engine.planner import phone_capture
from app.engine.planner import responder
from app.engine.planner import preference
from app.engine.planner import scheduling_progress as sp
from app.engine.planner import scope_claim
from app.engine.planner import unclear_input
from app.engine.planner import visit_identity
from app.engine.planner.focus import merge_entities
from app.engine.planner.planner import Proposal, plan
from app.engine.planner.turn_context import load_upcoming_appointments
from app.engine.tools import registry
from app.engine.tools.executor import execute
from app.engine.tools.types import READ_ONLY_TOOLS, ToolContext, ToolOutcome
from app.engine.voice import asr_hints as asr_hints_mod
from app.engine.voice import templated_reply
from app.observability import StageTimer, pricing
from app.services.events import publish
from app.services.money import crore_phrase, fix_lakh_crore_mismatch, round_spoken_prices
from app.services.spoken_form import to_spoken_form
from app.services.phone import extract_spoken_digits, normalize_phone


# A tool "worked" whenever its status is either of these. executor.py's idempotency
# check returns "skipped_duplicate" — carrying the SAME real output.output as the
# original "success", just deduped against a prior run with the same key — for any
# write tool called again with the same effective params. Several checks in this file
# tested only for the literal string "success" and silently missed a tool that had, in
# every way that matters, worked: book_appointment succeeding this way meant
# session["appointment_id"] never got remembered (see _remember_ids), and a plain
# "Thanks" one turn after a genuine booking drew "that is not booked yet" — flatly
# contradicting a booking that had gone through, because the one place checking for
# success used a narrower definition of it than the tool executor itself does.
logger = logging.getLogger(__name__)

TOOL_SUCCEEDED = ("success", "skipped_duplicate")


async def _tenant_vocabulary(db, tenant_id):
    """Best-effort pack vocabulary lookup for this file's pack-aware call sites (Vocal
    Backend Tasksheet P6 Generic Sprint 3). A lookup failure — a DB hiccup, or migration
    0019 not yet applied so tenant_pack_config doesn't exist — must degrade to None (each
    call site's own fallback vocabulary, reproducing today's hardcoded wording) rather than
    raise and fail the whole turn, the same guarantee state_machine/machine.py's identical
    guard makes for the state machine lookup."""
    from app.packs import PackRegistry
    try:
        return await PackRegistry.vocabulary_for_tenant(db, tenant_id)
    except Exception:
        return None

# Tools that only READ state — safe to pipeline TTS on the first sentence while the
# responder LLM is still generating the rest. Write tools (book, reschedule, cancel)
# are intentionally excluded: the full grounding check must settle before any word
# about a completed action reaches the caller. Shared with the planner's confidence
# banding (app/engine/tools/types.py), so the two can never disagree on what is read-only.
_READ_ONLY_PIPELINE_TOOLS = READ_ONLY_TOOLS

def is_affirmative(text: str, pending_tool: str | None = None) -> bool:
    """Whether the caller's own words agree — the deterministic consent read (consent.py).

    THE BUG THIS FIXES: this used to fall back to the planner's intent label when neither
    its yes-list nor its no-list matched. "Shall I book 10 AM?" → "i'm sorry what time / ja
    por time again" (a garbled "just repeat the time") matched neither, the model labelled
    the garble ``affirm``, and the label alone booked a visit the caller never agreed to. A
    model label is no longer consent anywhere; only the caller's words are. The confirmation
    gate itself uses the full two-layer ``consent.decide``; this lexical-only form serves the
    pre-plan checks (an offered handoff) where no semantic read exists yet.
    """
    return consent.read_lexical(text, pending_tool=pending_tool).decision == consent.ACCEPT



_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def _looks_like_uuid(value) -> bool:
    """Only a real id is worth remembering. The model also sends names, short references
    and text lifted from its own instructions; none of those should be stored as state."""
    return bool(_UUID_RE.match(str(value or "").strip()))


# How each staged write reads back in a "Shall I …?" question, so the grounded responder
# describes the parked action in the future tense instead of the model's past-tense draft.
_STAGED_VERB = {
    "book_appointment": "book your visit",
    "reschedule_appointment": "move your visit",
    "cancel_appointment": "cancel your visit",
    "create_ticket": "log that for you",
    "record_do_not_contact": "take you off our contact list",
    "schedule_callback": "arrange the callback",
    "update_customer_profile": "update your details",
    "create_customer": "get you set up",
}


def _staged_when(session: dict, params: dict | None) -> str | None:
    """The human-readable time of a staged slot, copied from what search_slots offered.

    So the read-back quotes "Tuesday 18 August at 5:00 PM" exactly as the customer heard
    it, rather than the model reconstructing it from a raw ISO string.
    """
    start = (params or {}).get("slot_start")
    if not start:
        return None
    for s in (session.get("offered_slots") or []):
        if s.get("starts_at") == start and s.get("when"):
            return s["when"]
    return None


def _call_today(as_of_date: date | None) -> date:
    """The call's "today", in the tenant's timezone — what "today"/"tomorrow" in a caller's
    reply are measured against when checked against a staged slot. The server clock is UTC,
    so a bare date.today() is the wrong day for the first 5½ hours of an IST morning."""
    if as_of_date is not None:
        return as_of_date
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(get_settings().tenant_timezone)).date()
    except Exception:
        return date.today()


def _consent_line(session: dict, pending: dict, *, kind: str, user_input: str,
                  attempt: int = 0) -> str:
    """The deterministic line for a turn that keeps a parked action waiting (consent.py):
    a repeat of the staged details, an "I didn't catch that", or the re-ask after a side
    question. The time comes from the offered-slot list (``_staged_when``), never from model
    text, so it can only ever state a slot the caller was actually offered."""
    tool = pending.get("tool")
    when = _staged_when(session, pending.get("params"))
    # Same rule as the staged read-back: a reschedule parked on the CURRENT time has no new
    # time to state.
    if (tool == "reschedule_appointment"
            and bt._same_instant((pending.get("params") or {}).get("slot_start"),
                                 session.get("appointment_current_start"))):
        when = None
    return consent.readback(tool, when, kind=kind, attempt=attempt,
                            language=responder.detect_language(user_input),
                            english_verb=_STAGED_VERB.get(tool))


# round_spoken_prices (app.services.money) is the very last thing done to a reply before
# it is spoken: any raw Indian-grouped rupee figure still in the text gets rewritten as the
# rounded crore/lakh phrase. Shared with responder.py's streaming path -- see that
# function's docstring for why both call sites must use the exact same one.


# A trailing "is there anything else" -style closer, in its common phrasings. Deliberately
# only matched at the END of a sentence (checked against the LAST split sentence, not
# anywhere in the text) -- this is specifically about the throwaway closing question, not
# a genuine mid-reply mention of "additional information" as a real answer.
_ANYTHING_ELSE_RE = re.compile(
    r"\b(anything else|something else|anything more|anything further|"
    r"any(?:thing)? (?:other|additional) (?:info|information|details?|help|questions?)|"
    r"need (?:any )?(?:additional|more) (?:info|information|details?)|"
    r"(?:is there|do you have) any(?:thing)? (?:else|more))\b", re.I)


# "It will end at 1:00 PM" — the specific construction booking results use for the
# appointment end-time. Deliberately narrow: only "it will end at <time>" to avoid matching
# legitimate "ends at [date/event]" phrases ("the sale ends at midnight"). The match must be
# anchored at start or preceded by sentence-boundary whitespace/punctuation so "and will end
# at" mid-sentence is also caught; the full suffix from the match point is dropped by
# _strip_booking_noise so no dangling "and will" fragment is left behind.
_ENDS_AT_RE = re.compile(
    r"(?:(?:^|(?<=[\s,]))\s*(?:and\s+)?(?:it\s+)?will\s+end\s+at\s+\d{1,2}(?::\d{2})?"
    r"\s*(?:am|pm|a\.m\.|p\.m\.)?)",
    re.I)

# Indian 6-digit PIN codes after a comma — the canonical address separator ("Mumbai, 400053").
# Requiring a preceding comma is the only safe discriminator: booking reference numbers,
# unformatted prices, and flat numbers don't have a comma before them, so they are never
# mistakenly stripped. Space-only "City 400053" cases (no comma) are left to Rule 8's prompt
# instruction rather than risking false positives against other 6-digit numbers.
_PINCODE_RE = re.compile(r",\s*\d{6}\b")


def _strip_booking_noise(text: str) -> str:
    """Deterministically remove booking-result noise the model sometimes reads aloud
    despite Rule 8 in responder.py prohibiting both.

    - ends_at: booking results include an end-time for calendar apps; callers never asked
      how long the visit runs. Drop everything from "and will end at X" to sentence end so
      no dangling fragment ("and will ") is left behind.
    - pincode: stored addresses end in a 6-digit PIN; only strip when the 6-digit run is the
      last token before punctuation/end (address-tail position), never in the middle of a
      sentence where it would be a booking ID or price.
    Sentence-level: if stripping empties a sentence it is dropped entirely.
    """
    sentences = responder.split_sentences(text)
    out = []
    for s in sentences:
        # For ends_at: drop from the match point to end-of-sentence to avoid orphaned
        # connectors ("and will end at 12 PM" → drop "and will end at 12 PM", not just
        # "end at 12 PM").
        m = _ENDS_AT_RE.search(s)
        if m:
            s2 = s[:m.start()].rstrip(", ").rstrip()
        else:
            s2 = s
        # For pincode: only the trailing 6-digit run in an address context.
        s2 = _PINCODE_RE.sub("", s2)
        s2 = re.sub(r"\s{2,}", " ", s2).strip().rstrip(",").strip()
        if s2:
            out.append(s2)
    return " ".join(out) if out else text


def _trim_repeated_anything_else(text: str, session: dict) -> str:
    """Drop a repeated "anything else?"-style closer, the same way _round_spoken_prices
    guarantees the price-rounding rule instead of just requesting it.

    Rule 6 (responder.py) already discourages ending replies this way — "a way out, not a
    way forward" — but the model doesn't reliably follow it every turn. On a real call it
    was asked twice a few turns apart, in different wording each time ("anything else you
    need?", then "anything additional before your visit?"), so a plain repeated-sentence
    check (Rule 7) never caught it either. The first time this call, the question stands —
    it is a real, useful close. Every time after, the trailing sentence asking it is
    dropped rather than repeated; never dropped if it is the ONLY sentence in the reply,
    so this can never leave nothing to say.
    """
    segments = responder.split_sentences(text)
    if not segments or not _ANYTHING_ELSE_RE.search(segments[-1]):
        return text
    if session.get("asked_anything_else_this_call") and len(segments) > 1:
        segments = segments[:-1]
    else:
        session["asked_anything_else_this_call"] = True
    return " ".join(segments)


# A trailing "would you like to set up a visit?" -style push question. Matched at the END
# of a sentence to distinguish it from an honest scheduling question mid-reply. Suppressed
# after the first time this call so it does not appear on every single response turn.
_VISIT_OFFER_RE = re.compile(
    r"\b(?:would you (?:like|be interested in)|shall i|can i|may i)\b[^.?!]{0,60}"
    r"\b(?:set up|schedule|arrange|book|organise)\b[^.?!]{0,30}"
    r"\b(?:a (?:site )?visit|an? (?:viewing|appointment|tour))\b"
    r"|\b(?:would you like (?:to|me) (?:to )?book|shall i (?:set up|schedule|arrange))"
    r"[^.?!]{0,40}\b(?:visit|appointment|viewing|tour)\b", re.I)


def _trim_repeated_visit_offer(text: str, session: dict) -> str:
    """Drop a repeated visit-offer closing question after the first time this call.

    The responder legitimately ends replies with "Would you like to set up a visit?" during
    the discovery phase — once. Repeating it on every turn is the P1-F15 bug: the caller
    heard it, they're still talking, and being pushed again every response reads as robotic.
    The first occurrence stands; every later one is dropped from the trailing sentence only,
    so a genuine mid-reply scheduling context (a confirmed slot read back alongside the offer)
    is never stripped.
    """
    segments = responder.split_sentences(text)
    if not segments or not _VISIT_OFFER_RE.search(segments[-1]):
        return text
    if session.get("offered_visit_this_call") and len(segments) > 1:
        segments = segments[:-1]
    else:
        session["offered_visit_this_call"] = True
    return " ".join(segments)


# "I can send you the location pin on WhatsApp if that helps" -- the post-booking pin offer
# responder.py's Rule 8 adds. Same repetition risk as the visit offer above, just for a
# different closer: a real call had it offered once, declined, then offered again on a later
# vague "sure" as though the decline never happened. Matched loosely (pin/location + WhatsApp)
# since the model's exact phrasing varies turn to turn.
_PIN_OFFER_RE = re.compile(
    r"\b(?:location )?pin\b[^.?!]{0,40}\bwhatsapp\b|\bwhatsapp\b[^.?!]{0,40}\bpin\b", re.I)


def _trim_repeated_pin_offer(text: str, session: dict) -> str:
    """Drop a repeated WhatsApp-pin offer after the first time this call, same pattern as
    _trim_repeated_visit_offer -- that fix covers the visit-offer closer; this is the
    post-booking pin offer PR #85 added, which never got the equivalent guard."""
    segments = responder.split_sentences(text)
    if not segments or not any(_PIN_OFFER_RE.search(s) for s in segments):
        return text
    if session.get("pin_offered_this_call"):
        segments = [s for s in segments if not _PIN_OFFER_RE.search(s)]
    else:
        session["pin_offered_this_call"] = True
    return " ".join(segments) if segments else text


# THE BUG THIS FIXES. A real call had the responder volunteer "let me know if you'd like to
# cancel it or reschedule" / "...move it" repeatedly, after turns where the caller had
# answered something else entirely -- declining a WhatsApp pin, saying "thanks", a bare "no"
# to "anything else?". decision_traces confirmed the planner's own reasoning was correct
# every single time ("customer declined location pin", "customer expressed gratitude"); the
# responder surfaced the offer on its own. A prompts.py rule asks the model not to do this,
# and reduces it, but doesn't eliminate it -- the model keeps finding new phrasing ("let me
# know if you'd like to move it" slipped past the rule's own examples). Deterministic here,
# same reasoning as the other _trim_repeated_* guards: catch the SHAPE regardless of exact
# wording, conditioned narrowly so a GENUINE reschedule/cancel conversation is never touched.
#
# Scoped deliberately narrow on the trigger side: only when THIS turn's utterance is a bare
# decline/closing reply (no, thanks, ok, leave it...) with nothing in it about changing the
# visit -- a caller already mid-reschedule says something with real content (a day, a time, a
# place), which never matches this pattern, so a legitimate flow is left completely alone.
_SHORT_CLOSING_RE = re.compile(
    r"^\s*(?:no|nope|nah|not now|no\s+thanks?|thanks?|thank\s*you(?:\s+so\s+much)?|"
    r"ok(?:ay)?|sure|fine|alright|that'?s\s+(?:it|all)|i'?m\s+done|leave\s+it(?:\s+like\s+"
    r"this)?|let\s+it\s+be|no\s+leave\s+it(?:\s+like\s+this)?)\s*[.!]?\s*$", re.I)

_UNSOLICITED_RESCHEDULE_OFFER_RE = re.compile(
    r"\b(?:would you like|let me know if you'?d? (?:actually )?(?:like|want)|shall i|"
    r"should (?:i|we)|do you want)\b[^.?!;]{0,60}\b(?:cancel\w*|resched\w*|\bmove\b|"
    r"\bshift\b)[^.?!]*[.?!]", re.I)


def _trim_unsolicited_reschedule_offer(text: str, user_input: str) -> str:
    """Drop the clause offering to cancel/move/reschedule the visit when nothing the caller
    just said touches the booking at all -- see the module-level comment above.

    Removes by direct substitution rather than sentence-splitting: the real observed case
    had the offer joined onto the preceding fact with a semicolon in the SAME sentence
    ("...is still confirmed; let me know if you'd like to move it."), which
    responder.split_sentences (period/!/? only) cannot separate. Whatever separator is left
    dangling where the clause used to be gets folded into a clean trailing period.
    """
    if not _SHORT_CLOSING_RE.match((user_input or "").strip()):
        return text
    # The agent's own drafted text can carry a curly apostrophe ("you'd" as U+2019) that this
    # regex's straight one never matches -- normalize before matching, same fix as
    # email_claim.py/booking_claim.py (see text_normalize.py's module docstring for the real
    # production case this was caught from).
    text = normalize_quotes(text)
    stripped = _UNSOLICITED_RESCHEDULE_OFFER_RE.sub("", text)
    if stripped == text:
        return text
    stripped = re.sub(r"\s*[;,]\s*$", ".", stripped.strip())
    stripped = re.sub(r"\s{2,}", " ", stripped).strip()
    if not stripped:
        # The whole reply WAS the unsolicited offer -- nothing legitimate left to keep.
        # Returning the original here would just speak the offer anyway; a plain
        # acknowledgment is the honest, neutral thing to say instead.
        return "Alright, noted."
    if stripped[-1] not in ".!?":
        stripped += "."
    return stripped


# P6-F1: an outbound caller comparing other options gets misread as declining, and the
# model then offers an unprompted callback and asks for a callback time -- a different
# shape of unprompted offer than the reschedule one above (that one triggers on a SHORT
# closing reply; this one triggers on SUBSTANTIVE content -- "I'm also looking at other
# alternatives" is real engagement, not a decline, and classify_pitch_reply (outbound_flow.py)
# correctly returns 'other' for it, so this is purely the main planner/responder's own
# judgment call going wrong, with no deterministic state-machine path to intercept.
#
# THE BUG THIS FIXES: a real outbound call had the caller say "yeah i actually was looking
# at other alternate as well" -- the model labelled the turn 'decline' and replied "...I can
# arrange a callback to share more details — does that work for you?", then (unprompted)
# "could you let me know a convenient time for us to call you back?" The caller never asked
# to be called back; comparing options is a live lead the call should keep pitching, not a
# reason to end this call and defer to a future one. prompts.py now says this directly
# ("COMPARING OTHER OPTIONS IS NOT A DECLINE...") but a prompt instruction is a request, not
# a guarantee, the same reasoning as every other guard in this module.
#
# Scoped on the CALLER's words, not the call direction: the comparison language is the
# trigger either way, and an explicit callback request in the SAME utterance ("I'm looking
# at others too, can you call me back next week") must never be stripped -- that caller DID
# ask, so _CALLBACK_REQUESTED_RE excludes it.
_COMPARING_ALTERNATIVES_RE = re.compile(
    r"\b(?:also |already )?(?:looking at|checking out|considering|comparing|exploring)\b"
    r"[^.?!]{0,40}\b(?:other|different|alternate|alternative)s?\b"
    r"[^.?!]{0,20}\b(?:option|project|builder|property|properties|place)s?\b"
    r"|\bcomparing (?:my )?options\b"
    r"|\b(?:also |already )?(?:looking at|checking out|considering)\b[^.?!]{0,10}"
    r"\b(?:other|different|alternate|alternative)s?\b\s*(?:as well|too)?\s*$",
    re.I)
_CALLBACK_REQUESTED_RE = re.compile(
    r"\b(?:call|ring|phone) (?:me |us )?back\b|\bcallback\b|\bcall (?:me |us )?(?:later|"
    r"tomorrow|next week|some other time)\b", re.I)
_UNPROMPTED_CALLBACK_OFFER_RE = re.compile(
    r"\b(?:i can (?:arrange|set up|schedule)|shall i (?:arrange|set up|schedule)|would you "
    r"like (?:me to|a)|can i (?:arrange|set up|schedule))\b[^.?!]{0,40}\bcallback\b"
    r"[^.?!]*[.?!]"
    r"|\b(?:could|can) you (?:let me know|tell me|share|give me)\b[^.?!]{0,40}\b(?:a "
    r"(?:convenient|good) time|time (?:that|which) (?:works|suits))\b[^.?!]{0,40}\b(?:call"
    r"(?:ed|ing)? (?:you |back)|callback)\b[^.?!]*[.?!]",
    re.I)

# Any clause that pitches further or offers re-contact -- used ONLY when proposal.rule_blocked
# is set (a policy/compliance rule, e.g. DNC, blocked this turn), where NOTHING should be
# offered going forward, however the model phrases it. Deliberately broad: the exact wording
# of an unwanted pitch is as unpredictable as everything else a model improvises, and the
# trigger (rule_blocked being set) already scopes this tightly enough that catching too much is
# the safe failure mode here, not catching too little.
#
# THE BUG THIS FIXES: the `{0,60}` cap on every English pattern meant a pitch whose clause ran
# past 60 characters -- "Would you like me to share the brochure and floor plans for the 3 BHK
# units in Tower B on WhatsApp?" -- never matched at all, and every pattern was English-only,
# so a Hindi/Hinglish pitch ("Kya aap hamare naye project ke baare mein sunna chahenge?",
# "क्या मैं आपको ब्रोशर भेज दूं?") went straight through unblocked on a PR largely about Hindi
# callers. No length cap now (the sentence-end punctuation already bounds the match); Hinglish
# and Devanagari offer verbs added alongside the English ones.
#
# The English openers ("would you like", "shall i", ...) sit at the START of the pitch clause,
# so the match's own start point is exactly where a genuine answer sharing the sentence ends
# ("We only sell, no rentals, would you like...?") -- _strip_forward_pitch below cuts there. The
# Hinglish/Devanagari verb ("chahenge", "भेज दूं") sits at the END of its clause instead, so
# those two alternatives carry their own `[^.?!]*` prefix, forcing the match back to the start
# of whatever contains them -- cutting mid-clause there would keep a dangling sentence fragment
# with the verb itself removed ("Kya aap hamare naye project ke baare mein sunna" / "क्या मैं
# आपको ब्रोशर"), not a real answer.
_FORWARD_PITCH_RE = re.compile(
    r"\bwould you like\b[^.?!]*[.?!]"
    r"|\bshall i\b[^.?!]*[.?!]"
    r"|\bcan i (?:arrange|set up|help you with|tell you about|share|send)\b[^.?!]*[.?!]"
    r"|\bi can have\b[^.?!]*\bcall you\b[^.?!]*[.?!]"
    r"|\bis there anything else\b[^.?!]*[.?!]"
    r"|[^.?!]*\b(?:chahenge|chahte hain|chahogi|bhej doon|bhej du|bata doon|bata du|"
    r"arrange kar doon|arrange kar du)\b[^.?!]*[.?!]"
    r"|[^.?!]*(?:चाहेंगे|चाहते हैं|चाहती हैं|भेज दूं|भेजूं|बता दूं|बताऊं|अरेंज कर दूं)[^.?!]*[.?!]",
    re.I)

# Acknowledges and moves on without claiming "I don't have that information" -- which would be
# a second lie when the model's answer and the pitch shared one sentence (see
# _strip_forward_pitch) and the real, correct answer is what got cut along with the pitch.
_NEUTRAL_POLICY_BLOCKED_LINE = {
    "english": "Understood.",
    "hindi": "ठीक है।",
    "hinglish": "Theek hai.",
}


def _pitch_sentences(text: str) -> list[str]:
    """Sentence split that also treats the Devanagari danda ("।") as a sentence end --
    responder.split_sentences only recognises ASCII .!?, so a Hindi reply's two clauses
    ("हम केवल बिक्री करते हैं। क्या मैं भेज दूं?") stayed one unit and a pitch sharing a
    "sentence" with a real answer took the answer down with it."""
    return [s for s in re.split(r"(?<=[.!?।])\s+", text or "") if s.strip()]


def _strip_forward_pitch(text: str, *, language: str) -> str:
    """`text` with every forward pitch / re-contact offer removed -- used ONLY when
    proposal.rule_blocked is set (see _FORWARD_PITCH_RE).

    THE BUG THIS FIXES: a pitch and a genuine answer can share one sentence ("We only sell,
    no rentals, would you like to see the 3 BHK?") -- dropping the whole sentence lost the
    answer along with the pitch, even though the model had answered correctly. The text
    before the pitch is kept when it says something on its own; only when nothing survives
    at all does this fall back to a neutral acknowledgement -- never a claim that the
    information isn't available, which would itself be false when the dropped sentence HAD
    a real answer in it. Returns `text` unchanged when nothing in it pitches at all."""
    if not text or not _FORWARD_PITCH_RE.search(text):
        return text
    kept: list[str] = []
    for s in _pitch_sentences(text):
        m = _FORWARD_PITCH_RE.search(s)
        if not m:
            kept.append(s)
            continue
        prefix = s[:m.start()].strip().rstrip(",;")
        if len(prefix.split()) >= 3:
            kept.append(prefix if prefix[-1:] in ".!?।" else prefix + ".")
    if kept:
        out = " ".join(kept)
        if out and out[-1] not in ".!?।":
            out += "."
        return out
    return _NEUTRAL_POLICY_BLOCKED_LINE.get(language, _NEUTRAL_POLICY_BLOCKED_LINE["english"])


def _trim_unprompted_callback_offer(text: str, user_input: str) -> str:
    """Drop a callback offer/callback-time ask when the caller was comparing other options
    and never asked to be called back -- see the module-level comment above."""
    low = (user_input or "").strip()
    if not _COMPARING_ALTERNATIVES_RE.search(low) or _CALLBACK_REQUESTED_RE.search(low):
        return text
    stripped = _UNPROMPTED_CALLBACK_OFFER_RE.sub("", text)
    if stripped == text:
        return text
    # The real observed case joins the offer onto the preceding clause with "...Elixir, and
    # I can arrange a callback..." -- stripping just the offer leaves a dangling conjunction
    # ("...Elixir, and") that a plain trailing-punctuation fix would turn into "...and.".
    # Drop a trailing connector (with its own leading comma/semicolon) before the punctuation
    # cleanup, not after, so this never fires on a conjunction that was never attached to the
    # removed clause in the first place.
    stripped = re.sub(r"\s*[;,]?\s*\b(?:and|but|so|or)\s*$", "", stripped.strip(), flags=re.I)
    stripped = re.sub(r"\s*[;,]\s*$", ".", stripped.strip())
    stripped = re.sub(r"\s{2,}", " ", stripped).strip()
    if not stripped:
        # The whole reply WAS the unprompted callback push -- a live lead deserves an
        # engaged follow-up question, never a bare acknowledgment that drops the thread.
        return "No problem — what are you comparing it against, or what matters most to you?"
    if stripped[-1] not in ".!?":
        stripped += "."
    return stripped


# A fact restated twice in the same reply, in different phrasing -- "priced at 1,92,53,440
# rupees... priced at 1.93 crore rupees" or "set at 4 PM in X... for Monday at 4 PM in X".
# No deterministic guard existed for this: _strip_booking_noise only removes known noise
# (ends_at, PIN codes), and the only defence is a free-form responder.py instruction ("one
# sentence of confirmation... nothing else") the model does not reliably follow. Caught via
# plain textual similarity between adjacent sentences rather than parsing out specific facts,
# since the repeated content varies (a price once, a day/time/address another time) -- two
# sentences saying substantially the same thing read as near-identical text either way.
_DUPLICATE_SENTENCE_RATIO = 0.6


def _collapse_duplicate_facts(text: str) -> str:
    """Drop a later sentence that substantially restates an earlier one in the same reply.

    Adjacent sentences only, keeping the first occurrence -- same "first stands, repeats
    drop" convention as _trim_repeated_visit_offer/_trim_repeated_pin_offer. A plain
    difflib ratio on lowercased, whitespace-collapsed text catches both observed shapes
    (same number restated in a different format; same sentence restated with a dangling
    connective) without needing to parse out what KIND of fact is being repeated.
    """
    segments = responder.split_sentences(text)
    if len(segments) < 2:
        return text
    kept = [segments[0]]
    for seg in segments[1:]:
        norm_new = re.sub(r"\s+", " ", seg.lower()).strip()
        is_dup = False
        for prior in kept:
            norm_prior = re.sub(r"\s+", " ", prior.lower()).strip()
            if not norm_new or not norm_prior:
                continue
            ratio = difflib.SequenceMatcher(None, norm_prior, norm_new).ratio()
            if ratio >= _DUPLICATE_SENTENCE_RATIO:
                is_dup = True
                break
        if not is_dup:
            kept.append(seg)
    return " ".join(kept) if kept else text


# The "you already have a visit on the books" pointer prompts.py's YOUR VISIT ON FILE block
# asks the model to add, at most once, on a turn about the property this visit is for. Unlike
# the closer above it isn't anchored to sentence position -- it can land as the opener, the
# closer, or the whole reply -- so every sentence is checked, not just the last.
_VISIT_REMINDER_RE = re.compile(
    r"\b(you(?:'re| are) (?:already |all )?set (?:for|to see|to visit)|"
    r"you (?:already )?have (?:a|an) (?:visit|appointment) (?:scheduled|booked)|"
    r"already (?:have|booked) (?:a|the|your) (?:visit|appointment)|"
    r"your (?:visit|appointment) (?:is|for) .{0,60}\b(?:scheduled|confirmed|booked)\b|"
    r"you(?:'ll| will) be visiting|"
    r"just a reminder,? your (?:visit|appointment))\b", re.I)


def _trim_repeated_visit_reminder(text: str, session: dict, user_input: str | None = None, *,
                                   already_given_at_turn_start: bool | None = None) -> str:
    """Drop a repeated "you're already set to see it on Wednesday" -style pointer, the same
    way _trim_repeated_anything_else guarantees its rule instead of just requesting it.

    prompts.py already says to add this ONCE where it fits, never as a reminder on every
    turn -- but on a real call it was said on nearly every reply, including a bare "hello"
    with nothing to do with the visit, in different wording each time ("you're all set...",
    "just a reminder, your visit is...", "you'll be visiting..."), so a plain repeated-text
    check never caught it. The first time this call, the pointer stands. Every time after, a
    matching sentence is dropped rather than repeated -- EXCEPT at index 0 when the CALLER
    actually asked about their visit this turn (mentions_existing_visit(user_input)): then
    that position is the direct answer, not a tacked-on reminder, and must survive. Without
    that condition, a goodbye turn that happens to lead with an unprompted "I see you have a
    visit booked..." (a real example: the caller said nothing about the visit, the model just
    volunteered it first) was protected by the same index-0 rule meant for a direct question,
    and the exact repeat this function exists to remove sailed through. Never dropped down to
    nothing, so this can never leave an empty reply.

    already_given_at_turn_start OVERRIDES session's live value for the "is this a repeat?"
    decision -- pass the flag's value from BEFORE this turn ran, not after. Booking/reschedule
    success sets session["visit_reminder_given_this_call"] = True as soon as the write
    succeeds, which is earlier in the SAME turn this function runs for. Reading the live
    (already-True) value here treated that turn's own fresh confirmation as a repeat and
    deleted it outright -- the caller booked a visit and was never actually told it was
    booked. The turn-start snapshot fixes that: this turn's own confirmation is always judged
    as the first mention, while a genuinely later turn still sees it as a repeat.
    """
    was_given = (session.get("visit_reminder_given_this_call")
                if already_given_at_turn_start is None else already_given_at_turn_start)
    # _VISIT_REMINDER_RE matches "you're"/"you'll" with a straight apostrophe; the agent's own
    # drafted text can carry a curly one instead -- same fix as text_normalize.py's other call
    # sites, see its module docstring for the real production case this was caught from.
    segments = responder.split_sentences(normalize_quotes(text))
    hits = [i for i, s in enumerate(segments) if _VISIT_REMINDER_RE.search(s)]
    if not hits:
        return text
    if was_given:
        protect_first = visit_identity.mentions_existing_visit(user_input)
        drop = {i for i in hits if not (protect_first and i == 0)}
        keep = [s for i, s in enumerate(segments) if i not in drop]
        segments = keep or segments
    else:
        # Multiple mentions in the same reply collapse to the first -- still "given" once.
        first = hits[0]
        segments = [s for i, s in enumerate(segments) if i == first or i not in hits]
    session["visit_reminder_given_this_call"] = True
    return " ".join(segments)


_HIDDEN_FROM_RESPONDER = ("price_exact", "id")


def _hide_from_responder(value, hidden_keys=_HIDDEN_FROM_RESPONDER):
    """A deep copy of a tool's output with internal, correctness-check-only fields removed —
    the one thing the responder LLM is allowed to see.

    inventory_query's `_shape()` (inventory_tools.py) documents `id`/`price_exact` as "never
    spoken" — `id` is read by contracts.py's inventory_units_match_query postcondition (it
    re-fetches the row and checks the tool didn't report a stale price), `price_exact` is
    what that check compares against. Both are only for that verification, but a docstring
    saying "never spoken" is a comment, not an enforcement — nothing actually stopped either
    from being right there in the same dict handed to the responder's prompt. On a real
    Hindi call the model read price_exact straight out as "1,54,32,011 रुपये" instead of the
    rounded price_phrase ("1.5 crore") sitting right next to it in the same unit dict. This
    is the actual enforcement: it returns a NEW structure and never mutates the original,
    which the postcondition check and session state still need untouched.
    """
    if isinstance(value, dict):
        return {k: _hide_from_responder(v, hidden_keys) for k, v in value.items()
                if k not in hidden_keys}
    if isinstance(value, list):
        return [_hide_from_responder(v, hidden_keys) for v in value]
    return value


def _action_state(action: str, outcome, session: dict) -> str:
    """What actually happened to the turn's action: done|failed|blocked|staged|none.

    Ground truth for the grounded responder (RC1). A write parked awaiting the customer's
    yes is ``staged`` even when a read (search_slots) succeeded in the same chain — the
    meaningful action is the parked write, not the lookup — so the responder is told to ask,
    never to claim. ``blocked``/``failed`` come straight off the executor's status.
    """
    if action == "confirm" and session.get("pending_action"):
        return "staged"
    if outcome is None:
        return "none"
    if outcome.status in TOOL_SUCCEEDED:
        return "done"
    if outcome.status == "blocked":
        return "blocked"
    if outcome.status == "failed":
        return "failed"
    return "none"


# The same apology, phrased for what they actually asked.
#
# The generic line quietly reframes the call. Someone who said "move my site visit to
# Friday afternoon" was answered with "which day and time would suit you? I'll pull up
# what's free" — which is a NEW booking question, and the rest of the call drifted into
# booking a second visit rather than moving the one she had. The guard was right to stop
# the promise; it was wrong to change the subject while doing it.
SAFE_PROMPT_BY_INTENT = {
    "reschedule_appointment": ("Sorry — what day and time would you like to move it to? "
                               "I'll check what's free."),
    "cancel_appointment": ("Sorry — I have not cancelled anything yet. Do you want me to "
                           "cancel that visit?"),
    "book_appointment": ("Sorry — which day and roughly what time would suit you? "
                         "I'll pull up what's free."),
    "schedule_callback": "Sorry — when would you like us to call you back?",
}

# Said when a RULE forces the handover and the model's own reply was about something else.
# A refund request escalates by policy — the agent may not settle money — but the model
# answered "I'll create a ticket for this issue, shall I proceed?" and the call was
# transferred mid-sentence. The customer said yes to a ticket that policy forbade the agent
# to raise, and then heard that nothing had been logged. Whatever else happens, the reply
# has to match the action taken.
FORCED_HANDOVER = ("That one needs a colleague rather than me — I'm putting you through "
                   "to someone who can sort it out properly.")


def _forced_handover_line(target_display: str | None) -> str:
    """The line spoken when a RULE forces a handover and the model's own reply was about
    something else. When the routing instruction named a transfer target (resolved to an
    active registry row), SAY where the caller is going — "connect loan enquiries to the
    Financial Desk" should be heard as exactly that, not as a generic brush-off. Falls back to
    the generic line when there is no named target (a bare escalation, or one that degraded)."""
    if target_display:
        return (f"Let me put you through to our {target_display} — they'll be able to help you "
                f"with that.")
    return FORCED_HANDOVER


def _required_info_line(req: dict | None) -> str:
    """The line spoken when a `require` rule ("collect X before doing Y") has fired and the
    slot is still missing — so the agent must ASK for it before proceeding. Uses the compiler's
    own natural question (`ask`) when present, else a plain fallback naming the slot."""
    ask = (req or {}).get("ask")
    if isinstance(ask, str) and ask.strip():
        return ask.strip()
    slot = str((req or {}).get("slot") or "").replace("_", " ").strip()
    if slot:
        # Name the information explicitly so the caller knows what to answer — a vague ask is
        # why an earlier version got "ok"/a name back instead of the actual value.
        return (f"Before I set that up, may I ask what your {slot} is? "
                f"It just helps me point you to the right options.")
    return "Before I set that up, could I get a couple of details from you first?"

# Used only when the triage re-plan refuses to ask anything. Told to find out what the call
# is about, the model kept writing "I'll connect you to someone" — obeying the customer
# rather than the instruction, which is defensible but means the triage step never happens.
# One written line is worth more here than a fourth attempt at wording the prompt, and it
# costs the customer a single turn: ask again and they go straight through.
TRIAGE_QUESTION = ("Of course — so I can put you through to the right person, what is it "
                   "about? If you would rather not say, that is completely fine and I will "
                   "transfer you now.")

# Read-only lookups, after which the turn may CONTINUE rather than stopping to speak.
# Every one of these produces information the agent needs before it can do the real thing,
# and none of them changes a record — so chaining them commits the customer to nothing.
# Write tools are excluded on purpose: they end the turn and go through the confirmation
# gate, so no booking, reschedule, cancellation or profile change can ever happen inside a
# chain without the customer hearing it read back first.
CHAINABLE_TOOLS = ("locate_centre", "search_slots", "my_appointments", "my_tickets",
                   "knowledge_lookup", "verify_identity",
                   # create_customer writes a row, so it sits oddly in a list of read-only
                   # lookups — but it is the same kind of step: something that must happen
                   # BEFORE the thing the caller actually asked for. Without it here, the
                   # turn stops after registering and she has to ask to book all over
                   # again. It creates nothing she did not just ask for, and the booking it
                   # unblocks still goes through the confirmation gate.
                   "create_customer")
# The irreversible scheduling WRITES. A bare "yes/ok" that confirms a read-back must release
# the action that was READ BACK, never let the model swap in a DIFFERENT one of these on the
# same turn — "shall I book?" + "ok" can only mean book, and an actual change of mind ("no,
# cancel it") is not affirmative, so it never reaches this guard.
_SCHEDULING_WRITES = ("book_appointment", "reschedule_appointment", "cancel_appointment")
# Raised from 2 once the chain stopped spending a planner call on every step. The longest
# legitimate deterministic sequence is three — a reschedule needs my_appointments (which
# visit?) then locate_centre (where?) then search_slots (when?) — and at 2 the last one was
# always cut off, so the caller was asked to say "ok" one more time to get the answer they
# had already asked for. A step that comes from `_next_scheduling_lookup` now costs only the
# tool's own time (no model round trip), so the extra step is cheap; CHAIN_BUDGET_MS below
# still bounds the wall clock, and a step that DOES re-plan is still paying full price
# against it.
MAX_TOOL_STEPS = 3
# Chaining trades per-turn latency for total call length: three steps in one turn is
# slower than one, but far faster than the three turns and two customer utterances it
# replaced. The budget stops a slow chain from leaving the caller in silence.
CHAIN_BUDGET_MS = 6000


# An escalation turn says so out loud. Used to tell a genuine handover reply ("let me get a
# colleague for you") apart from an escalate-labelled turn whose reply is about something
# else entirely — asked to book a visit, the model tagged the turn escalate and wrote "Let
# me find available slots for Friday afternoon": no colleague was fetched, yet the label
# claimed a transfer. The forced-handover check below uses this to replace such a reply with
# an honest handover line.
#
# The label is the model's claim. This is the check on it.
HANDOVER_PATTERNS = (
    r"\b(?:colleague|human|someone|a person|team member|advisor|executive|specialist)\b",
    r"\b(?:transfer\w*|connect\w*|put(?:ting)? you through|hand(?:ing)? (?:you )?over)\b",
)


# A REFUSAL to transfer contains all the same words as an offer to transfer.
#
# "I'm not able to transfer calls, but I can help with some information" was accepted as a
# handover reply on a turn where the call WAS being handed to a person. The customer was
# told transfers are impossible while one was happening, and carried on talking to an
# agent that had already let go of her. The triage gate learned this a week ago; this
# second, separate check never did.
_REFUSES_HANDOVER = re.compile(
    r"\b(?:can'?t|cannot|can not|unable to|not able to|do not|don'?t|"
    r"am not able|no way to)\b[^.?!]{0,40}"
    r"\b(?:transfer|put you through|connect|hand (?:you )?over|pass you)\b", re.I)


def _reads_like_handover(text: str) -> bool:
    low = (text or "").lower()
    if _REFUSES_HANDOVER.search(low):
        return False
    return any(re.search(pat, low) for pat in HANDOVER_PATTERNS)


# The agent's OWN offer to hand the caller to a person — "shall I connect you with the
# sales team?", "would you like their contact details?". This is the outbound mirror of
# _asks_for_human: the caller never said "human", the agent did, and the caller's next
# word is a yes/no on THAT offer.
#
# THE BUG THIS ARMS THE FIX FOR: the agent failed a question, offered the sales team, the
# caller said "yes" — and the bare "yes" was re-classified from scratch as a fresh product
# question (human_handoff_requested=False), so it searched again, found nothing, and offered
# the very same handoff a second time. The caller had to say yes twice to be transferred.
# Detecting the offer here lets the next turn read that "yes" as accepting the transfer.
_HANDOFF_OFFER_CUES = (
    r"\bsales (?:team|executive|colleague|advisor|agent|rep\w*|person|department)\b",
    r"\b(?:connect|put) you (?:with|through|in touch|to)\b",
    r"\b(?:transfer|hand(?:ing)? (?:you )?over|pass you)\b",
    r"\bget (?:you )?(?:in touch|connected|through)\b",
    r"\b(?:their|the|our) contact details\b",
    r"\breach out to (?:the|our)[^.?!]{0,15}\bteam\b",
)
# A mixed offer whose "yes" belongs to a tool the confirmation gate already handles — a
# booking, callback or ticket. Those are not a pure handoff, so leave them to normal
# planning rather than escalating a "yes" that meant "book it".
_HANDOFF_OFFER_EXCLUDE = re.compile(
    r"\b(?:book|schedule|reschedul\w*|slot|appointment|site visit|viewing|tour|"
    r"call you back|callback|call back|raise (?:a )?(?:ticket|complaint)|"
    r"log (?:a )?(?:ticket|complaint))\b", re.I)


def _offers_human_handoff(text: str) -> bool:
    """Did the agent's reply OFFER (as a question) to put the caller through to a person?

    Deliberately narrow: only a question arms the accept, a refusal to transfer never
    counts, and an offer entangled with a bookable action is left to the confirmation gate.
    """
    low = (text or "").lower()
    if "?" not in low:
        return False
    if _REFUSES_HANDOVER.search(low):
        return False
    if _HANDOFF_OFFER_EXCLUDE.search(low):
        return False
    return any(re.search(p, low) for p in _HANDOFF_OFFER_CUES)


# Honest reply when the model affirmed a feature it cannot cite in the material (Rule 13).
# Deliberately promises no callback — the phantom "I'll find out and call you back" was its
# own bug; this just declines to guess and keeps the conversation moving. The verification
# itself (synonym-aware, so a real "yes" under a different word is not rejected) lives in the
# pure, self-tested app.engine.planner.feature_grounding module.
#
# Keyed by language and written by hand, not translated by the LLM at call time: this text
# replaces something the model got WRONG (an ungrounded claim), so the replacement must never
# go anywhere near a model that could get it wrong again. This bypasses the responder LLM
# entirely by design — a caller speaking Hindi all call got this fixed English sentence
# regardless, because there was only ever the one English string to substitute in.
UNSUPPORTED_FEATURE_FALLBACK = {
    "english": ("I'm not able to confirm that from the project details I have in front of me, "
               "so I don't want to give you a wrong answer. Is there something else about the "
               "flats I can help with?"),
    "hindi": ("मेरे पास मौजूद जानकारी से मैं इसकी पुष्टि नहीं कर पा रहा, इसलिए गलत जवाब नहीं देना चाहूंगा। "
             "क्या फ्लैट्स के बारे में कुछ और जानकारी चाहिए?"),
    "hinglish": ("Mere paas jo details hain unse main ye confirm nahi kar pa raha, isliye galat "
                "jawab nahi dena chahunga. Flats ke baare mein aur kuch jaanna hai?"),
}

# Honest reply when the model asserted a SCOPE/POLICY boundary it could not ground (Rule 14) —
# "we don't offer rentals", "we only sell". Not finding a passage is not proof of the opposite,
# so rather than let a fabricated policy stand we admit we don't have the detail and OFFER (not
# promise) the real next step: a sales-team callback the caller can accept on the next turn.
# Same language-bypass reasoning as UNSUPPORTED_FEATURE_FALLBACK above.
UNGROUNDED_SCOPE_FALLBACK = {
    "english": ("I don't have that detail in front of me, so I wouldn't want to give you the "
               "wrong answer. I can have our sales team call you back to confirm — would you "
               "like that?"),
    "hindi": ("यह जानकारी मेरे पास अभी उपलब्ध नहीं है, इसलिए गलत जवाब नहीं देना चाहूंगा। मैं हमारी सेल्स टीम "
             "से आपको कॉलबैक करवा सकता हूं ताकि वे पुष्टि कर सकें — क्या यह ठीक रहेगा?"),
    "hinglish": ("Ye detail abhi mere paas nahi hai, isliye galat jawab nahi dena chahunga. "
                "Main hamari sales team se aapko callback karva sakta hoon confirm karne ke "
                "liye — kya ye thik rahega?"),
}

# The DNC-safe counterpart of UNGROUNDED_SCOPE_FALLBACK above: the normal fallback OFFERS a
# callback as the honest next step -- which is itself re-contacting someone who, this same
# call, asked to be left alone. Used instead whenever proposal.rule_blocked is set (a
# policy/compliance rule blocked this turn), so the honest "I don't have that" admission
# never comes bundled with the one thing this caller explicitly opted out of.
UNGROUNDED_SCOPE_FALLBACK_POLICY_BLOCKED = {
    "english": "I don't have that detail in front of me, so I wouldn't want to give you the wrong answer.",
    "hindi": "यह जानकारी मेरे पास अभी उपलब्ध नहीं है, इसलिए गलत जवाब नहीं देना चाहूंगा।",
    "hinglish": "Ye detail abhi mere paas nahi hai, isliye galat jawab nahi dena chahunga.",
}


def _turn_needs_project(proposal, user_input: str) -> bool:
    """Does answering THIS turn require knowing which project the caller means?

    True for a product / price / configuration / amenity / policy / availability / booking
    turn — anything answered from a project's documents or that books against a project.
    False for a greeting, small talk, a stop-contacting request or a plain human-handoff, none
    of which depend on the project. The signal is the same one retrieval uses: the fast
    understanding classifier decided a document could answer this turn (``used``), or it is a
    scheduling turn, or the intent label is one of the project-specific kinds. This is a
    MEANING judgement taken from the classifier, not a keyword scan of the words.
    """
    meta = proposal.retrieval or {}
    if meta.get("used"):                       # a document was fetched → product/price/policy
        return True
    routing = meta.get("routing") or {}
    if routing.get("is_scheduling"):
        return True
    intent = str((proposal.understood or {}).get("intent") or "").lower()
    return any(k in intent for k in ("pric", "product", "policy", "book", "reschedul",
                                     "cancel", "availab", "config", "amenit", "unit"))


def _project_clarify_question(candidates: list[str]) -> str:
    """The deterministic 'which project?' question, naming each project with its place so the
    caller can tell them apart. ``candidates`` are already 'Name (location)' strings."""
    if len(candidates) == 2:
        listed = f"{candidates[0]} and {candidates[1]}"
    else:
        listed = ", ".join(candidates[:-1]) + f", and {candidates[-1]}"
    return (f"We actually have more than one project — {listed}. "
            f"Which one would you like to know about?")


def _gate_slots(session, proposal) -> dict:
    """The captured information a slot-based `require` gate is checked against, up to date for
    THIS turn. session['active_entities'] is only merged at the END of the turn, so a value the
    caller gives on the very turn they are asked for it (the city they were just asked for) is
    not yet in it — folding this turn's freshly extracted entities in is what stops the gate
    asking a second time for something already provided."""
    slots = {k: v for k, v in (session.get("active_entities") or {}).items()
             if v not in (None, "", [], {})}
    te = (proposal.understood or {}).get("entities")
    if not isinstance(te, dict):
        try:
            _ej = (proposal.understood or {}).get("entities_json")
            te = json.loads(_ej) if isinstance(_ej, str) and _ej.strip() else {}
        except Exception:
            te = {}
    for k, v in (te or {}).items():
        if v not in (None, "", [], {}):
            slots[str(k).strip().lower()[:40]] = v
    return slots


def _tool_context(db, tenant_id, actor, customer, conversation, session, as_of_date,
                  proposal) -> ToolContext:
    """Built fresh for every tool run, because the ids on the session change as the chain
    proceeds — locate_centre establishes the centre that search_slots then needs."""
    return ToolContext(
        db=db, tenant_id=tenant_id, actor=actor, customer=customer,
        conversation_id=conversation.id,
        identity_verified=bool(session.get("identity_verified")),
        # The project this call is pinned to (multi-project-per-tenant), so a tool's own
        # retrieval or a booking scopes to the same project the planner resolved. Set on the
        # conversation at context assembly; None for tenant-wide tenants or while ambiguous.
        project_id=getattr(conversation, "project_id", None),
        extra={"channel": conversation.channel,
               "direction": conversation.direction,
               "as_of_date": as_of_date,
               # The centre established earlier in this call. The model reliably loses
               # the id across turns and sends a name, a description, or something
               # copied out of the prompt; the conversation itself is the only
               # dependable place to keep it.
               "known_centre_id": session.get("centre_id"),
               # The unit the caller chose, so book_appointment can bind it to the visit.
               # The pinned session value (set once a unit is first stated) is preferred over
               # the live entity bag, which drifts as the booking progresses.
               "desired_unit": (session.get("desired_unit")
                                or preference.desired_unit(
                                    session.get("active_entities") or {})),
               # This call's booking first; otherwise the visit already on file (a returning
               # caller moving/cancelling a visit booked on an earlier call). _resolve_appointment
               # reads this only when the model supplies no usable id of its own.
               "known_appointment_id": (session.get("appointment_id")
                                        or session.get("on_file_appointment_id")),
               # The times a slot search returned on this call. Passed so the contract
               # can refuse a booking for a time nobody looked up — checking it here, once,
               # is what makes the pattern-matching downstream unnecessary.
               "offered_slots": session.get("offered_slots") or [],
               # The number the caller is reaching us on, so send_document can deliver to
               # WhatsApp WITHOUT asking them to recite it (never-ask-caller-for-own-number).
               # An identified customer's own number is preferred inside the tool; this is the
               # unknown-caller fallback — the session cache first, then the durable column.
               "caller_phone": (session.get("caller_phone")
                                or getattr(conversation, "caller_phone", None)),
               "turn": {"intent": (proposal.understood or {}).get("intent"),
                        "sentiment": (proposal.understood or {}).get("sentiment")},
               # Captured conversation entities + the per-condition require-ask counter, so a
               # slot-based `require` rule ("collect budget before offering a visit") is
               # satisfied by real data at the tool gate and stands aside after a couple of
               # asks. Generic: empty when no such rule exists, so it is inert by default.
               "slots": _gate_slots(session, proposal),
               "require_asks": dict(session.get("__require_asks__") or {})})


async def _project_default_centre(db, tenant_id, project_id) -> str | None:
    """The centre a visit to the call's resolved PROJECT is booked at — ``project.default_centre_id``,
    but only when it still names an ACTIVE centre in this tenant.

    This is the piece the scheduling flow was missing. Centre resolution otherwise falls back to
    "the tenant's one active centre", which is wrong the moment a project has a site office AND a
    sales gallery, or a tenant runs several projects — and fragile even for a single project,
    because it breaks the instant that lone-centre assumption stops holding (a second centre is
    added, or a reseed leaves zero). Reading the centre FROM the resolved project makes a booking
    resolve the same deterministic way whether the tenant has one project or many:
    ``project_resolution`` already pins the project (the only project, the dialled number, the
    caller's history, an explicit mention), and this turns that decision into a centre.

    Returns the centre id as a string, or None when there is no resolved project, it has no
    default centre configured, or that centre is gone/inactive — in which case the existing
    tenant-level fallbacks still run, so behaviour for an unconfigured tenant is unchanged. Never
    raises on the hot path.
    """
    if not project_id:
        return None
    try:
        from app.db.models import Location as _C
        from app.db.models import Project as _P
        from sqlalchemy import select as _sel
        _pid = project_id if isinstance(project_id, uuid.UUID) else uuid.UUID(str(project_id))
        proj = (await db.execute(_sel(_P).where(
            _P.tenant_id == tenant_id, _P.id == _pid))).scalar_one_or_none()
        if proj is None or not proj.default_centre_id:
            return None
        centre = (await db.execute(_sel(_C).where(
            _C.tenant_id == tenant_id, _C.id == proj.default_centre_id,
            _C.is_active.is_(True)))).scalar_one_or_none()
        return str(centre.id) if centre is not None else None
    except Exception:
        return None


async def _fit_params(db, tenant_id, tool_name: str, params: dict | None,
                      session: dict, user_input: str, draft: str,
                      project_id=None) -> tuple[dict, list[str]]:
    """Params the tool will actually accept, with what we already know filled in.

    Called at EVERY point a tool executes — there are two, and only one of them had this.
    The chain path ran `execute(ctx, proposal.tool, proposal.params)` directly, so a
    booking that reached the executor through a chain step carried whatever the last
    rewrite left behind: a customer's name and phone, and no centre. It failed with
    "Centre not found or inactive" on a call where the centre had been in the session
    since turn four.

    Fixing one call site and not the other is how a fix looks live and behaves as though
    it is not.
    """
    notes: list[str] = []
    tool_def = await registry.get_tool(db, tenant_id, tool_name)
    if tool_def is None:
        return dict(params or {}), notes

    allowed = set(((tool_def.input_schema or {}).get("properties") or {}).keys())
    out = dict(params or {})

    # Drop keys the model set to null. A typed schema field (city: string, date_from:
    # string) rejects an explicit null before the tool ever runs — "city: None is not of
    # type 'string'" — even though the tool treats a missing value as "not provided" and
    # copes fine. The model routinely emits {"city": null} when it has no city, so a null
    # is "absent", never real data; stripping it here turns a hard schema failure into the
    # tool's own graceful no-value path.
    _nulls = sorted(k for k, v in out.items() if v is None)
    if _nulls:
        out = {k: v for k, v in out.items() if v is not None}
        notes.append(f"dropped null-valued {_nulls} — treated as not provided")

    # Drop keys whose value is a stringly-typed null. The model does not only emit a JSON
    # null when it has nothing for a field — it emits the WORD: {"centre_id": "null"},
    # "none", "undefined", "nil", or an empty string. Unlike a real null that string is
    # TRUTHY, so it survives every `not params.get(key)` guard downstream and is handed to
    # the tool as though it were data the caller supplied. That is the whole outbound bug:
    # search_slots received centre_id "null", was refused with "No active centre matches
    # 'null'", and — because a truthy value bypassed both the single-centre auto-fill and the
    # locate_centre force below — the agent, with no centre and no step forced, improvised by
    # asking the CALLER to "provide the centre details", i.e. a database id no caller has.
    # Collapsing these placeholders to "not provided", exactly like a JSON null, hands the
    # gap to the centre-resolution and locate_centre-forcing paths that already handle a
    # missing centre. Safe for these tools' fields (ids, dates, cities, enums): none of them
    # legitimately equals the word "null" or "none".
    _sentinels = {"", "null", "none", "nil", "undefined", "n/a", "na"}
    _placeholder = sorted(k for k, v in out.items()
                          if isinstance(v, str) and v.strip().lower() in _sentinels)
    if _placeholder:
        out = {k: v for k, v in out.items() if k not in _placeholder}
        notes.append(f"dropped placeholder-valued {_placeholder} — the model sent a literal "
                     f"'null'/'none' string, treated as not provided")

    if allowed:
        dropped = sorted(k for k in out if k not in allowed)
        if dropped:
            out = {k: v for k, v in out.items() if k in allowed}
            notes.append(f"dropped {dropped} — not parameters of {tool_name}")

    if tool_name in ("book_appointment", "reschedule_appointment",
                     "cancel_appointment", "search_slots"):
        filled = []
        # The centre the call's resolved PROJECT is booked at (project.default_centre_id),
        # validated as an active centre. Computed once and used as a FALLBACK two ways below:
        # to fill a centre when none is set, and to REPLACE one that no longer resolves (which
        # is how a model-invented or reseed-stale id gets corrected). A centre the call actually
        # established — from an appointment or an earlier locate_centre/search — is left as is.
        # This is what lets a booking work with one project or many; None when there is no
        # resolved project or it has no default centre, leaving the tenant-level fallbacks in
        # place, so an unconfigured tenant behaves exactly as before.
        _proj_centre = (await _project_default_centre(db, tenant_id, project_id)
                        if "centre_id" in allowed else None)
        # A reschedule's centre is on the appointment. Look it up rather than guessing.
        #
        # The session's centre can be stale or absent; the appointment's cannot — it is
        # where the visit actually is. A customer moving a booking should never be asked
        # which branch she means, and a search for her new time should run against the
        # centre she is already going to.
        # This call's booking wins; otherwise the visit already on file, so a returning
        # caller's reschedule searches against the centre their existing visit is at.
        _known_appt = session.get("appointment_id") or session.get("on_file_appointment_id")
        _appt_for_centre = (params or {}).get("appointment_id") or _known_appt
        _centre_from_appt = None
        if _appt_for_centre and "centre_id" in allowed and not (params or {}).get("centre_id"):
            from app.db.models import Appointment as _A
            from sqlalchemy import select as _s
            try:
                _a = (await db.execute(_s(_A).where(
                    _A.id == uuid.UUID(str(_appt_for_centre))))).scalar_one_or_none()
                if _a is not None and _a.centre_id:
                    _centre_from_appt = str(_a.centre_id)
            except (ValueError, TypeError):
                pass

        # A value WE looked up beats one the model produced.
        #
        # This only filled a gap — `not out.get(key)` — so a centre the model invented was
        # left in place and the real one ignored. It invented a well-formed UUID,
        # 6ff20d6c-c88a-4d48-b819-877b2312e422, which belongs to no centre in any tenant;
        # the booking and then the search were both refused as "Centre not found or
        # inactive" while locate_centre's answer sat unused in the session.
        #
        # A model cannot know a database id. Anything it supplies for one is a guess, and
        # a guess should never outrank a lookup.
        for key, val in (("centre_id", _centre_from_appt or session.get("centre_id")
                          or session.get("offered_centre_id")),
                         ("appointment_id", _known_appt)):
            if key in allowed and val and out.get(key) != val:
                if out.get(key):
                    notes.append(f"replaced {key} {out[key]!r} with the one from this call")
                out[key] = val
                filled.append(key)
        if "slot_start" in allowed:
            # A model cannot know what is actually free either — same principle as
            # centre_id and appointment_id above, just never applied to this one.
            #
            # THE BUG THIS FIXES: nothing here ever checked a SUPPLIED slot_start
            # against a time that had genuinely been searched and offered. The model
            # (or the engine's own single-offered-slot fallback) could hand over any
            # timestamp and it went straight to book_appointment — which is exactly how
            # a customer who explicitly rejected the 15th ended up booked onto the
            # 15th's only offered time, relabelled with the 18th's date, having never
            # heard "ten in the morning" spoken back to her before it was booked.
            _offered_starts = {s.get("starts_at") for s in (session.get("offered_slots") or [])
                               if s.get("starts_at")}
            if out.get("slot_start") and out["slot_start"] not in _offered_starts:
                notes.append(f"slot_start {out['slot_start']!r} was not among the times "
                             f"actually offered this call; dropped rather than book an "
                             f"unverified time")
                out.pop("slot_start", None)
            if not out.get("slot_start") and session.get("offered_slots"):
                picked = _pick_offered_slot(session, user_input, draft or "")
                if picked:
                    out["slot_start"] = picked
                    filled.append("slot_start")
        # No centre yet? Take the one the call's resolved PROJECT is booked at, before any
        # tenant-wide guessing. This is the deterministic answer for a single-project tenant
        # (its one project) AND a multi-project one (the project this call resolved to), and —
        # crucially — it works when a project has MORE THAN ONE active centre, where the
        # lone-active-centre fallback below cannot, and it is not defeated by a stray extra
        # centre in the tenant. When it fills, the "ask which centre" and single-active paths
        # below are skipped (both guard on the centre still being empty).
        if "centre_id" in allowed and not out.get("centre_id") and _proj_centre:
            out["centre_id"] = _proj_centre
            session["centre_id"] = _proj_centre
            filled.append("centre_id")

        # Several centres and none chosen — ask, do not guess.
        #
        # locate_centre returned two, so nothing was recorded, and the model filled the gap
        # by inventing a UUID. Every subsequent call failed on it. With more than one real
        # centre the honest move is a question the customer can answer — their names, not a
        # database id, which is what the agent asked for once before.
        if ("centre_id" in allowed and not out.get("centre_id")
                and tool_name in ("book_appointment", "search_slots")):
            from app.db.models import Location as _CC
            from sqlalchemy import select as _ss
            _many = (await db.execute(_ss(_CC).where(
                _CC.tenant_id == tenant_id, _CC.is_active.is_(True)))).scalars().all()
            if len(_many) > 1:
                notes.append("several centres and none chosen — ask which, by name: "
                             + "; ".join(c.name for c in _many[:4]))

        # Still no centre, and the tool needs one.
        #
        # A booking reached the executor with no centre at all and was refused with
        # "Centre not found or inactive" — which reads as though a centre was named and
        # rejected. Nothing had been named. With one active centre there is nothing to
        # choose between, and asking the customer which of one branch she means is not a
        # question anybody should be asked.
        if "centre_id" in allowed and not out.get("centre_id"):
            from app.db.models import Location as _C
            from sqlalchemy import select as _sel
            _active = (await db.execute(_sel(_C).where(
                _C.tenant_id == tenant_id, _C.is_active.is_(True)))).scalars().all()
            if len(_active) == 1:
                out["centre_id"] = str(_active[0].id)
                session["centre_id"] = str(_active[0].id)
                filled.append("centre_id")

        if filled:
            notes.append(f"filled {filled} from the call rather than asking again")

        # P1-F9: reject past dates in search_slots. The model occasionally generates a
        # date_from from its training corpus (e.g. August when today is October) instead of
        # reading the ## TODAY block. A past date can never produce a bookable slot; clamp
        # it to today so the search returns real availability rather than nothing.
        if tool_name == "search_slots":
            _today_ist = datetime.now(IST).date()
            for _df_key in ("date_from", "date_to"):
                _df_raw = out.get(_df_key)
                if _df_raw:
                    try:
                        _df_parsed = date.fromisoformat(str(_df_raw)[:10])
                        if _df_parsed < _today_ist:
                            out[_df_key] = _today_ist.isoformat()
                            notes.append(f"{_df_key} {_df_raw!r} is in the past; "
                                         f"clamped to today ({_today_ist.isoformat()})")
                    except (ValueError, TypeError):
                        pass

        # Verify the centre still resolves — but NEVER drop it without a replacement.
        #
        # The first version dropped an unresolvable id and only refilled when the tenant
        # had exactly one active centre. This tenant has two, so a reschedule lost its
        # centre entirely and search_slots was refused for a missing required property —
        # a worse outcome than passing an id that might have been fine. Removing a value
        # and leaving nothing in its place cannot help anybody.
        #
        # The rejected value is logged, because "no longer resolves" was a guess about why
        # and there was no way to check it.
        if out.get("centre_id"):
            from app.db.models import Appointment as _Appt
            from app.db.models import Location as _Ctr
            from sqlalchemy import select as _sel
            try:
                _cid = uuid.UUID(str(out["centre_id"]))
            except (ValueError, TypeError):
                _cid = None

            _row = None
            if _cid is not None:
                _row = (await db.execute(_sel(_Ctr).where(
                    _Ctr.tenant_id == tenant_id, _Ctr.id == _cid,
                    _Ctr.is_active.is_(True)))).scalar_one_or_none()

            if _row is None:
                _rejected = out.get("centre_id")
                _replacement, _repl_source = None, None
                # The centre the customer's own appointment is at beats any guess.
                _appt_id = out.get("appointment_id") or session.get("appointment_id")
                if _appt_id:
                    try:
                        _a = (await db.execute(_sel(_Appt).where(
                            _Appt.id == uuid.UUID(str(_appt_id))))).scalar_one_or_none()
                        if _a is not None and _a.centre_id:
                            _replacement, _repl_source = str(_a.centre_id), "the customer's appointment"
                    except (ValueError, TypeError):
                        pass
                # Then the resolved project's default centre — deterministic whether the tenant
                # has one project or several, and the case a lone-active-centre fallback cannot
                # cover (a project with a site office AND a sales gallery).
                if _replacement is None and _proj_centre:
                    _replacement, _repl_source = _proj_centre, "the project's default site office"
                if _replacement is None:
                    _active = (await db.execute(_sel(_Ctr).where(
                        _Ctr.tenant_id == tenant_id,
                        _Ctr.is_active.is_(True)))).scalars().all()
                    if len(_active) == 1:
                        _replacement, _repl_source = str(_active[0].id), "the tenant's only active centre"

                if _replacement:
                    out["centre_id"] = _replacement
                    session["centre_id"] = _replacement
                    notes.append(f"centre_id {_rejected!r} did not resolve; used {_repl_source} instead")
                else:
                    # Keep it. An id that may be wrong still gives the tool something to
                    # reject with a clear message; no id at all is a validation error the
                    # customer hears as "I couldn't access the schedule".
                    notes.append(f"centre_id {out['centre_id']!r} did not resolve and there "
                                 f"is no unambiguous replacement — passing it through")

    return out, notes



# Mid-booking, a short remark about time is for the SCHEDULER, not the library.
#
# Two failures in one call, both from documents answering a scheduling question:
#
#   "i want for tomorrow at 5 pm"
#     -> "site visits are available until six thirty, can I schedule five thirty?"
#        Five o'clock is before half past six. It read the opening hours out of a brochure
#        and reasoned about availability from them instead of searching for slots.
#
#   "why not 5"
#     -> "we don't have a 5 BHK at DSR Highland Greenz"
#        She was talking about five o'clock, and had said so in the previous sentence.
#
# Whether a booking / reschedule / slot search is the active task on THIS call — read from
# session state, not from the customer's words. Distinct from the deleted scheduling/status
# REGEXES (P1): those tried to read the MEANING of an utterance ("is this a time?", "is this
# a status question?") and are now the understanding model's job. This only asks a state
# question — is a scheduling flow open — which drives the proactive search_slots correction
# below, and there is nothing for a model to interpret in it.
_SCHEDULING_TOOLS = ("book_appointment", "reschedule_appointment", "search_slots")


def _scheduling_in_flight(session: dict) -> bool:
    pending = (session.get("pending_action") or {}).get("tool")
    wanted = session.get("wanted_tool")
    # The goal held by the REGISTRATION child task counts too, and leaving it out was a real
    # hole rather than an omission of detail.
    #
    # THE BUG THIS FIXES: registration is a child task of the booking — when an unregistered
    # caller asks to visit, the booking goal is parked in session["registration"]["goal"] and
    # `wanted_tool` is NOT written until a LATER turn (the block that copies it across is
    # guarded on create_customer appearing in `executed`, but it runs before this turn's tools
    # execute, so it is dead on the very turn registration completes). For the whole of that
    # turn, then, every question of the form "is a booking in flight?" answered False — while
    # a booking was demonstrably in flight, since the caller had asked for one and we were
    # collecting their name to make it happen.
    #
    # The visible symptom was the chain breaking after every single step: create_customer ran,
    # the model narrated "let me find a nearby centre", the deterministic driver was consulted,
    # this returned False, and the turn ended. Then locate_centre ran, "let me check available
    # slots", ended. Then search_slots. Three turns and two wasted caller utterances for one
    # unbroken sequence of lookups — each one gated on a session key that gets written a turn
    # too late.
    registration_goal = (session.get("registration") or {}).get("goal")
    return (any(t in _SCHEDULING_TOOLS for t in (pending, wanted, registration_goal) if t)
            or bool(session.get("offered_slots")))


def _turn_is_about_scheduling(user_input: str, proposal, session: dict) -> bool:
    """Is THIS turn about slots, times or the act of booking — as opposed to a knowledge
    question that merely happens to arrive while a booking is in flight?

    The gate on the top-level "advance the flow" force below. It must be a MEANING judgement,
    not a keyword one, because "appointment time", "what are the available options" and "can
    we do it earlier" share no vocabulary — so the primary signal is the fast understanding
    classifier's own ``is_scheduling`` flag (understanding.py), which exists for exactly this
    question and already rides out on ``proposal.routing``. The rest are fallbacks for the
    legs where that flag is absent: a retry/chain plan passes ``skip_retrieval=True`` and
    therefore carries no routing, and a trivial-shaped utterance skips the classifier
    entirely. In those cases the planner's own intent label, an explicit booking phrase, or a
    day/time the caller just named still say what the turn is about.

    Deliberately NOT satisfied by "a booking is in flight" alone — that is the caller's
    STANDING goal, not what they asked THIS turn, and forcing a slot search over a mid-booking
    question about the price is exactly the subject-hijack this must not do.
    """
    if (getattr(proposal, "routing", None) or {}).get("is_scheduling"):
        return True
    if sp.is_scheduling_intent((getattr(proposal, "understood", None) or {}).get("intent")):
        return True
    if _turn_hint(user_input, proposal) in ("book_appointment", "reschedule_appointment",
                                            "cancel_appointment"):
        return True
    return (_wanted_day(user_input) is not None
            or _wanted_part_of_day(user_input) is not None
            or _wanted_hour(user_input) is not None)


def _next_scheduling_lookup(session: dict, executed: list,
                            *, in_flight: bool | None = None) -> tuple[str, dict] | None:
    """The next read-only step that advances an in-flight booking/reschedule, or None.

    Deterministic and driven off session state, this is what keeps a chain moving when the
    model — re-planning mid-turn after a registration or a lookup — narrates "You're
    registered, let me check available slots" instead of naming the next tool. The
    step-forcing rules in the main turn body (Rules 1/2/5) run only ONCE per turn and never
    inside the chain loop, so without this a hedge on the re-plan ended the turn and the
    caller had to say "ok" over and over to push the agent through its own steps —
    create_customer, then locate_centre, then search_slots, one customer utterance apiece.

    Returns only CHAINABLE lookups; a write (book/reschedule) is NEVER forced here — it still
    goes through the confirmation gate on its own turn. None means either nothing scheduling
    is in flight, or the flow has already reached offered slots and the agent should now
    speak. Every branch is gated on the step not having already run this turn, so it advances
    at most one stage per call and cannot loop.

    ``in_flight`` overrides the session-state test. It exists for the turn that STARTS a
    booking: a caller who says "I'd like to see the flat" has plainly engaged the flow, but
    nothing has been written to the session yet (``wanted_tool`` is set later in the turn, by
    the hint block), so the state test alone answers False on the one turn the caller most
    obviously asked for slots. The top-level force passes the explicit signal it already has.
    """
    if not (_scheduling_in_flight(session) if in_flight is None else in_flight):
        return None
    ran = {o.tool_name for o in executed if o.status in TOOL_SUCCEEDED}

    # A reschedule cannot search or move anything until we know WHICH visit — load it first.
    _is_reschedule = (
        (session.get("pending_action") or {}).get("tool") == "reschedule_appointment"
        or session.get("wanted_tool") == "reschedule_appointment")
    if (_is_reschedule and not session.get("appointment_id")
            and not session.get("appt_lookup_forced")
            and "my_appointments" not in ran):
        return ("my_appointments", {})

    # Every scheduling flow needs a centre before it can search for slots. With more than one
    # active centre _remember_ids leaves centre_id unset, so this returns None on the next
    # pass and the agent asks which branch — never guessing one.
    if not session.get("centre_id") and "locate_centre" not in ran:
        return ("locate_centre", {})

    # Centre established, nothing offered yet — search, carrying any day/window already named
    # so the search is targeted rather than a whole-day spread the model must pick from.
    #
    # For a FRESH booking where the caller named no branch and locate_centre returned several,
    # advance on the primary (the chosen behaviour) rather than stalling: search it this same
    # turn and let the reply name it. A RESCHEDULE never does this — its centre comes from the
    # visit being moved (resolved above), so a missing centre_id there still means "ask which
    # branch", never "guess the primary".
    _search_centre = session.get("centre_id")
    if (not _search_centre and not _is_reschedule
            and "locate_centre" in ran and session.get("primary_centre_id")):
        _search_centre = session["primary_centre_id"]
    if (_search_centre and not session.get("offered_slots")
            and "search_slots" not in ran):
        params: dict = {"centre_id": _search_centre}
        _day = session.get("wanted_day")
        if _day:
            params["date_from"] = params["date_to"] = _day
        _part = session.get("time_pref")
        if _part:
            params["part_of_day"] = _part
        return ("search_slots", params)

    return None


def _remember_ids(session: dict, tool_name: str, outcome: ToolOutcome, params: dict) -> None:
    """Carry ids the model reliably loses forward on the session.

    A successful identity check has to survive the turn, otherwise every rule requiring
    session.identity_verified would re-ask on the next sentence. The centre and the
    appointment under discussion are remembered for the same reason: the model sends names,
    descriptions and text lifted from its own prompt where an id belongs, and the
    conversation is the only dependable place to keep the real one.

    THE BUG THIS FIXES: this used to accept only a literal "success", but the executor's
    idempotency check (executor.py) returns "skipped_duplicate" — with the SAME real
    output, including the appointment_id — whenever a tool is called again with the same
    key as a prior successful run. Every other place in this file that asks "did the tool
    actually work" treats the two as equal (the chain loop's continue condition, the
    BookingTask projection) — this was the one place that did not, so a booking that
    succeeded via the idempotency path never got its appointment_id remembered. The next
    turn's session then had no record of it, and a plain "Thanks" after a genuinely
    completed booking could be answered as though nothing was booked — for no reason other
    than this one status string.
    """
    if outcome.status not in TOOL_SUCCEEDED:
        return
    out = outcome.output or {}

    # A centre id that stopped resolving is worse than none at all.
    #
    # Working memory outlives the database: seed_dsr and reset_visits delete and recreate
    # centres, and a session opened before that keeps handing the old id to every booking
    # for the next hour. The error is "Centre not found or inactive", which reads like a
    # configuration problem and is really a stale cache.
    #
    # Any tool that reports the centre missing clears it, so the next turn looks one up
    # rather than repeating a dead id.
    if (outcome.status == "blocked"
            and "centre" in str(outcome.error or outcome.failed_check or "").lower()
            and "not found" in str(outcome.error or "").lower()):
        session.pop("centre_id", None)
        session.pop("offered_centre_id", None)

    if tool_name in ("locate_centre", "search_slots", "book_appointment", "my_appointments"):
        used = out.get("centre_id") or (params or {}).get("centre_id")
        centres = out.get("centres") or []
        if not used and len(centres) == 1:
            used = centres[0].get("centre_id")
        # Several centres and the caller named no branch: remember the FIRST as the primary.
        # This does NOT become the authoritative centre_id — it is a same-turn fallback the
        # forward-booking search step reaches for so a fresh booking advances on the primary
        # in this turn (the chosen behaviour) instead of stalling for an extra "ok". A real
        # centre_id resolved from the caller's words, a successful search, or an existing
        # appointment always wins over it.
        elif not used and tool_name == "locate_centre" and len(centres) > 1:
            _primary = centres[0].get("centre_id")
            if _primary and _looks_like_uuid(_primary):
                session["primary_centre_id"] = str(_primary)
        # With SEVERAL centres, take the one the search actually used.
        #
        # locate_centre returns a list, and this only recorded a centre when the list had
        # exactly one entry. This tenant has two, so nothing was recorded — and a booking
        # three turns later failed with "Centre not found or inactive" because no centre
        # had been supplied at all. The message names the wrong problem: the centre was
        # fine, there simply was not one.
        #
        # search_slots is called with a specific centre_id, and its results are the times
        # the customer was then offered. Whichever centre those came from is the one the
        # conversation is about.
        # Only from a search that SUCCEEDED.
        #
        # Taking centre_id from the params of a failed search records whatever the model
        # invented — and the invented value then wins every later fill, so one bad guess
        # poisons the rest of the call. A search that returned slots proves the centre is
        # real, because the scheduler resolved it.
        if not used and tool_name == "search_slots" and outcome.status in TOOL_SUCCEEDED:
            used = (params or {}).get("centre_id")
        if not used and centres:
            for slot in (out.get("slots") or []):
                if slot.get("centre_id"):
                    used = slot["centre_id"]
                    break
        # Reading a booking back establishes its centre just as firmly as looking one up
        # does — a reschedule almost always starts that way.
        if not used:
            # "live" first — it holds bookings that still stand, including one earlier
            # today. "upcoming" excludes those, and "past" is empty unless the caller asked
            # for history, so checking only those two missed the commonest case: a customer
            # ringing about the visit she has, wanting to move it.
            for appt in ((out.get("live") or []) + (out.get("upcoming") or [])
                         + (out.get("past") or []) + (out.get("appointments") or [])):
                if appt.get("centre_id"):
                    used = appt["centre_id"]
                    break
        if used and _looks_like_uuid(used):
            session["centre_id"] = str(used)

    if tool_name in ("my_appointments", "book_appointment", "reschedule_appointment"):
        appt = out.get("appointment_id")
        # The time the CURRENT appointment sits at, so a reschedule can require a genuinely
        # DIFFERENT slot (RC2). Without it a staged reschedule carrying the existing time —
        # "shall I reschedule it?" read back before any new time was chosen — passed every
        # check and "moved" the visit to the same hour it was already at.
        _cur_start = out.get("starts_at")
        if not appt:
            # Both lists. An appointment the tool classes as past is still the one the
            # customer is talking about — reading it out and then not remembering it is why
            # the next tool said it was not theirs.
            for a in (out.get("live") or []) + (out.get("upcoming") or []) + (out.get("past") or []):
                if a.get("appointment_id"):
                    appt = a["appointment_id"]
                    _cur_start = _cur_start or a.get("starts_at")
                    break
        if appt and _looks_like_uuid(appt):
            session["appointment_id"] = str(appt)
        if _cur_start:
            session["appointment_current_start"] = _cur_start
        # The visit is booked/moved — the day and window they were requesting are spent.
        # Clearing them stops Rule 1 from re-searching a now-satisfied window on later
        # turns (e.g. a follow-up "is it booked?").
        if tool_name in ("book_appointment", "reschedule_appointment"):
            session.pop("time_pref", None)
            session.pop("wanted_day", None)
            # A completed booking means identity was already confirmed during the booking
            # flow (the caller gave their name for create_customer / BOOKING_REQUIREMENTS).
            # Marking visit_name_confirmed prevents the visit-status gate from re-asking
            # "may I have the name the visit is booked under?" on the very next turn — the
            # caller just booked and we already know who they are.
            if session.get("appointment_id"):
                session["visit_name_confirmed"] = True

    if tool_name == "verify_identity" and out.get("verified"):
        session["identity_verified"] = True

    # The slots just read out to the customer. Kept because the engine needs to be able to
    # book one WITHOUT asking the model for it again: "book it for tomorrow at twelve" is
    # unanswerable if the only place the twelve o'clock slot exists is a sentence the model
    # already spoke and then forgot.
    if tool_name == "search_slots" and out.get("slots"):
        session["offered_slots"] = [
            {"starts_at": s.get("starts_at"), "when": s.get("when")}
            for s in out["slots"] if s.get("starts_at")
        ][:12]
        session["offered_centre_id"] = session.get("centre_id")


def _tool_note(outcome: ToolOutcome) -> str:
    """What the model is told about a tool it just ran. Written as a 'tool' message and
    replayed in the prompt, so it learns what happened instead of repeating the mistake."""
    if outcome.status in ("success", "skipped_duplicate"):
        return f"{outcome.tool_name} succeeded: {json.dumps(_safe(outcome.output))[:600]}"
    return (f"{outcome.tool_name} did NOT run ({outcome.status}"
            f"{', check: ' + outcome.failed_check if outcome.failed_check else ''}). "
            f"Reason: {outcome.error}. Fix the cause before trying it again.")


# Asking for a person. Matched on the customer's own words, because this decision is made
# before any model-generated intent label can be trusted.
HUMAN_REQUEST_PATTERNS = (
    r"\b(?:talk|speak|connect|transfer|put me)\b[^.?!]{0,40}\b(?:human|person|someone|"
    r"agent|representative|executive|manager|colleague|advisor)\b",
    r"\breal (?:human|person)\b",
    r"\b(?:human|live) (?:agent|being|person|executive|representative)\b",
    r"\bcustomer (?:care|support) (?:agent|executive|representative|team)\b",
)

# Never make someone who is upset explain themselves twice. Anger, a complaint, or a
# dispute goes straight through — the triage gate below is for the neutral "can I speak to
# someone" that a single question can often resolve, and nothing else.
NO_TRIAGE_SENTIMENTS = {"angry", "frustrated", "upset", "negative", "distressed"}
NO_TRIAGE_INTENTS = {"complaint", "legal_complaint", "payment_dispute"}


# The one rule-forced escalation the triage gate may take a turn over.
#
# escalation_policy has `explicit_human_request`: turn.human_handoff == true -> escalate
# immediately (the rule used to also OR in turn.intent == request_human; that branch was
# removed after it forced an escalate off a free-text topic label alone while the model's
# own human_handoff boolean correctly said no — see rules/defaults.py). That rule and this
# gate are answering the same question, and the rule was winning — which is why the
# transfer happened on the first ask no matter what the code did.
#
# It is exempted, and nothing else is: legal_or_payment_dispute, vip_negative_sentiment,
# angry_lead_escalates, abusive_language, pricing_dispute and the rest all still override
# the gate outright, because each of those describes a situation the agent should not be
# handling at all. "Can I speak to someone" does not. It describes a customer who wants
# something done and has decided the agent cannot do it — which is worth one question, and
# exactly one.
#
# If you would rather it transfer immediately, this is a policy decision and belongs in the
# rules, not here: Rule Engine -> escalation_policy -> remove the exemption by renaming the
# rule, or delete this constant's contents. The gate reads the rule id, so the rules stay
# in charge.
TRIAGEABLE_ESCALATION_RULES = {"explicit_human_request"}


# A name, given in answer to "can I take your name?".
#
# Deliberately loose — people answer with "Rahul", "Rahul Yadav", "it's Rahul", and
# misspell it. Anything short, wordy and free of digits is far more likely to be a name
# than a new request, and the cost of being wrong is a record with an odd name on it,
# against the cost of dropping the caller's request entirely.
_NAME_NOISE = re.compile(r"\b(?:my name is|i am|i'?m|this is|it'?s|name is|call me)\b", re.I)


# Placeholder text a model emits when it has no real value: angle brackets, braces, or
# the wording of the field description itself. Treating any of it as data is how a
# customer gets registered as "<customer's phone number>".
_PLACEHOLDER = re.compile(
    r"^\s*[<\[{(]|[>\]})]\s*$|^\s*(?:string|null|none|n/?a|tbd|unknown|xxx+|"
    r"your |their |the )|(?:phone number|full name|customer'?s|placeholder|example)\s*$",
    re.I)


def _real_value(v) -> str | None:
    """The value if it is genuinely one, otherwise None."""
    if not isinstance(v, str):
        return None
    t = v.strip()
    if not t or _PLACEHOLDER.search(t):
        return None
    return t


def _looks_like_a_name(text: str) -> str | None:
    cleaned = _NAME_NOISE.sub(" ", (text or "").strip())
    # THE BUG THIS FIXES: the letter-only filter just below strips digit CHARACTERS, not
    # whole tokens -- "10th october at 4 pm" lost its digits and became "th october at pm",
    # and "th" (the ordinal suffix left behind) sailed through every check below as part of
    # a plausible 4-word name ("Th October At Pm"), reaching a caller as "I'm not finding a
    # visit under Th" once _first() shortened it. Drop any WHOLE token containing a digit
    # first, so an ordinal or any mixed alphanumeric fragment never leaves a stray
    # letter-only remnant for the filter below to wrongly accept.
    cleaned = " ".join(w for w in cleaned.split() if not any(c.isdigit() for c in w))
    cleaned = re.sub(r"[^A-Za-z\s.'-]", " ", cleaned).strip()
    words = cleaned.split()
    if not (1 <= len(words) <= 4):
        return None
    if any(len(w) > 20 for w in words):
        return None
    # A question is a question, not a name.
    if "?" in (text or "") or re.search(r"\b(?:what|when|where|why|how|can|could|do|does|"
                                        r"is|are|will|would|yes|no|ok|okay)\b",
                                        cleaned, re.I):
        return None
    # A filler word is not a name either — "yeah" slipping past the exclusion above (it
    # only caught "yes/no/ok/okay") is exactly how a real call got a customer registered
    # as "Yeah": a single casual acknowledgment, spoken to something that was never a name
    # prompt, happens to be 1-4 letters-only words and sails through every check above.
    if re.fullmatch(
            r"(?:yeah+|yep|yup|nah+|nope|sure|alright|right|fine|hmm+|mm+-?hmm+|um+|uh+|"
            r"okay+|k|cool|great|thanks|thank you|please|correct)",
            cleaned, re.I):
        return None
    # Nor is a scheduling/domain verb or time word — the same shape of mistake, a different
    # vocabulary. On a real call a dropped "no i want to" fragment left just "book" to reach
    # here (asked for the name a VISIT is booked under, in the status-check flow this
    # function also feeds), and "Book" sailed through every check above into
    # mismatch_reply() exactly like a real, wrong name would. None of these words can ever
    # actually be what a caller is named.
    if re.fullmatch(
            r"(?:book|cancel|reschedule|postpone|confirm|visit|appointment|booking|slot|"
            r"schedule|arrange|today|tomorrow|tonight|morning|afternoon|evening|night|"
            r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            cleaned, re.I):
        return None
    return " ".join(w.capitalize() for w in words)


async def _caller_phone(db, session: dict, conversation) -> str | None:
    """The number the caller is reaching us on, resolved at the point of use from the
    strongest source available — never from one write at session start.

    Asking somebody to recite the number they are calling from is the single clearest way
    an agent announces it is a form rather than a person, so this is worth being thorough
    about. Two sources, strongest first:

      1. ``session["caller_phone"]`` — what the client sent when it opened the session
         (api/sessions.py). Set only for an unrecognised caller, since a known customer
         already has a number on their record.
      2. ``Call.from_number`` — the CARRIER's caller ID for this conversation. This is the
         authoritative source on a real telephony call and, until now, was written to the
         database and read by nothing but a list endpoint. Every inbound caller would have
         been asked to read out a number the carrier had already handed us.

    Resolving here rather than at session start also means the number is found when the Call
    row is linked AFTER the session opens — the ordering a real webhook actually produces.
    The answer is cached back onto the session, so the lookup happens at most once a call.
    """
    cached = _real_value(session.get("caller_phone"))
    if cached:
        return cached
    # Durable Postgres record, written at session start (api/sessions.py). This is the fix for
    # the number living only in Redis: the conversation row is already loaded, so reading it
    # costs nothing, and it survives anything that can wipe the session cache. Cached back onto
    # the session so the rest of the call takes the fast path.
    persisted = _real_value(getattr(conversation, "caller_phone", None))
    if persisted:
        session["caller_phone"] = persisted
        return persisted
    row = (await db.execute(
        select(Call.from_number)
        .where(Call.conversation_id == conversation.id)
        .order_by(Call.created_at.desc()).limit(1))).scalars().first()
    number = _real_value(row)
    if not number:
        return None
    # Normalised to the same shape create_customer stores, so a caller-ID "09876543210"
    # and a spoken "9876543210" resolve to one customer rather than two records.
    number = normalize_phone(number)
    session["caller_phone"] = number
    return number


def _asks_for_human(text: str) -> bool:
    low = (text or "").lower()
    return any(re.search(pat, low) for pat in HUMAN_REQUEST_PATTERNS)


# Requests that MUST become a specific tool call, matched on the customer's own words.
#
# Prompt wording alone did not carry these. Told "stop calling me, take me off your list",
# the model said it had done so and called nothing. Told "call me back tomorrow at 6am", it
# promised a callback that was never booked. Told a budget and who decides, it said "Got it"
# and wrote nothing down. Each reply is warm, plausible, and leaves no trace — and the
# customer has no way to discover it, because nothing anywhere contradicts them.
#
# This is a BACKSTOP, not routing. It fires only when the turn produced no tool at all, and
# all it does is send the plan back naming the tool the customer's words called for. The
# model still chooses the parameters, and every gate — confirmation, rules, preconditions —
# applies exactly as before.
TOOL_HINTS = (
    # Checked FIRST, ahead of the "visit" booking pattern below, so "I wanted to visit the
    # property, I can not quite find it" hints send_document (location_map) rather than
    # book_appointment — a real call had the "visit" keyword alone drive straight into
    # offering a NEW booking slot, never addressing that the caller was trying to reach an
    # ALREADY-intended visit and was lost on the way. The tool's own definition already
    # names this exact scenario ("if the customer can't find the site, needs directions...
    # call this tool with document_type=location_map"); the gap was that nothing nudged the
    # model toward it when "visit" was also present and took precedence instead.
    (r"\b(?:can'?t|cannot|can not|couldn'?t|could not) (?:quite |really |seem to )?find\b"
     r"|\b(?:i'?m |i am )?lost\b|\bwhere (?:is|are) (?:the|your|it)\b"
     r"|\b(?:how do i|how to) (?:get|reach|find) (?:there|it|the (?:site|property|place))\b"
     r"|\b(?:need|want) directions\b",
     "send_document"),
    (r"\b(?:stop|don'?t|do not) (?:calling|contacting|call|contact) me\b"
     r"|\btake me off\b|\bremove me from (?:your|the) (?:list|database)\b"
     r"|\bunsubscribe\b|\bnever (?:call|contact) me\b",
     "record_do_not_contact"),
    (r"\b(?:call|ring|phone) me (?:back|again|later|tomorrow|on|at)\b"
     r"|\bgive me a call\b|\bcall back\b",
     "schedule_callback"),
    (r"\bcancel (?:my|the|that) (?:site )?(?:visit|appointment|booking|slot)\b",
     "cancel_appointment"),
    (r"\b(?:move|reschedul\w*|shift|postpone|push|change)\b[^.?!]{0,30}"
     r"\b(?:visit|appointment|booking|slot)\b"
     r"|\b(?:visit|appointment|booking|slot)\b[^.?!]{0,30}\b(?:to|for) (?:another|a different)\b",
     "reschedule_appointment"),
    # Four digits on their own, in reply to an identity question. The model has more than
    # once asked for them, been given them, and asked again — leaving the customer reciting
    # the same number while nothing happened.
    (r"^\s*\d{4}\s*$", "verify_identity"),
    # "please raise it", "log a complaint", "can you log that". Without this, "there's water
    # seepage in the flat I was shown, please raise it" retrieved the price of a corner flat
    # and quoted a location premium — twice — while the complaint went nowhere.
    (r"\b(?:log|raise|register|file|open)\b[^.?!]{0,30}"
     r"\b(?:complaint|ticket|issue|problem|it|that|this)\b"
     r"|\b(?:complain|complaint) about\b",
     "create_ticket"),
    # Open-ended visit inquiry — "when can I visit the project?", "what time can I come?",
    # "can I just drop by?". These ask ABOUT visiting (timing, walk-in hours) rather than
    # committing to a booked slot. Route to locate_centre so BOOKING_SEQUENCE's walk-in
    # branch fires: the agent shares office hours and invites the caller to drop in or book.
    # MUST precede book_appointment: the visit-verb + location-noun clause below also
    # matches these phrases, and _hinted_tool returns the first hit.
    (r"\bwhen (?:can|could|may|should|do|would|is it (?:possible|okay|ok|alright|fine) to)\b"
     r"[^.?!]{0,50}\b(?:visit|come|drop\s+by|stop\s+by|walk\s+in)\b"
     r"|\bwhat (?:time|day\w*|hour\w*|are (?:your|the)|is (?:your|the))\b"
     r"[^.?!]{0,40}\b(?:visit\w*|open|available|timing|hours)\b"
     r"|\bhow (?:do|can|could|would) (?:i|we|one)\b[^.?!]{0,30}\b(?:visit|come|get there)\b"
     r"|\bcan (?:i|we) just\b[^.?!]{0,40}\b(?:drop|come|walk|stop|visit)\b"
     r"|\bjust (?:drop|come)\s+(?:by|in)\b"
     r"|\b(?:office|visiting|project)\s+hours\b"
     r"|\bwalk[-\s]?in\b",
     "locate_centre"),
    # "book me in", "book that", and also the bare "book for tomorrow at twelve" — which
    # names no noun at all and was the exact sentence that started the loop.
    #
    # Also the plain-English way people ask to visit, which has no "book" in it at all:
    # "I want to see the flat", "can I visit the property", "I'd like to view the
    # apartment", "schedule a site visit", "come and tour the project". Without these the
    # phantom-booking guard (see _booking_signalled) read a genuine visit request as an
    # unsignalled book_appointment: the goal was rewritten to create_customer, so after the
    # caller gave their name and number the booking they actually asked for had vanished
    # and the agent said "you're all set — how can I help you today?" instead of offering
    # slots. The nouns are physical things you visit (flat/property/site/…), never
    # "price"/"brochure"/"details", so "can I see the price" is not mistaken for a visit.
    (r"\bbook\b[^.?!]{0,40}\b(?:visit|appointment|slot|it|that|me in)\b"
     r"|\bbook\b[^.?!]{0,40}\b(?:\d{1,2}\s*(?:am|pm)|tomorrow|today|monday|tuesday|"
     r"wednesday|thursday|friday|saturday|sunday)\b"
     r"|\b(?:go ahead|confirm) (?:and )?book\b"
     r"|\b(?:see|visit|view|tour|check out|come see|come and see|look at|drop by)\b"
     r"[^.?!]{0,20}\b(?:flat|property|project|site|apartment|apartments|unit|home|villa|"
     r"place|model (?:flat|home|unit)|sample (?:flat|home|unit)|showflat|show flat)\b"
     r"|\bsite visit\b|\bschedule (?:a |an |my |the )?(?:visit|viewing|tour|site visit)\b"
     r"|\bbook (?:a |an |my |the )?(?:viewing|tour|visit)\b"
     r"|\barrange (?:a |an |my |the )?(?:visit|viewing|tour)\b"
     # "can I come tomorrow at 3pm?" names no noun (visit/flat/property) at all, just a day
     # or time after "come" — a real call had this read as a status CHECK ("do I have a
     # visit?") instead of the booking REQUEST it plainly is, and the honest "no visit on
     # file" fallback fired on a caller who had just asked to book one.
     r"|\bcan i come\b[^.?!]{0,30}\b(?:\d{1,2}\s*(?:am|pm)|tomorrow|today|tonight|monday|"
     r"tuesday|wednesday|thursday|friday|saturday|sunday)\b",
     "book_appointment"),
    (r"\bmy budget is\b|\bbudget is (?:around|about|roughly)\b"
     r"|\b(?:my )?(?:wife|husband|father|mother|partner|family) (?:makes|will make|takes) "
     r"the (?:final )?(?:call|decision)\b|\blooking to (?:buy|purchase) (?:in|within)\b",
     "record_customer_fact"),
)

# Honouring a stop-contacting request is the one action worth taking WITHOUT the model's
# agreement. If it will not call the tool after being told to, the engine calls it: the cost
# of a wrongly-recorded preference is a staff member undoing it, and the cost of ignoring a
# real one is calling somebody who told us twice to stop.
FORCE_IF_REFUSED = {"record_do_not_contact", "verify_identity"}


# "yes", "ok do it", "go ahead" — agreement with no new detail in it. Also covers
# deferring the choice back to the agent: "pick any", "whichever works", "you choose".
# Asked which of the offered slots she wanted, a customer who said "pick any" was told
# about the range of unit sizes on offer instead of being booked into the first one —
# the knowledge-retrieval gate had a matching exemption (see rag/__init__.py NO_SUBJECT)
# but this picker, which is what actually resolves a slot, did not.
AFFIRM_ONLY = re.compile(
    r"^\s*(?:ok(?:ay)?|yes|yeah|yep|sure|please|go ahead|do it|book it|that works|"
    r"fine|alright|great|perfect|"
    r"(?:(?:just\s+)?(?:pick|choose|take|book)\s+)?any(?:\s*(?:one|of them))?|"
    r"whichever(?:\s+(?:one|works|is fine|is good))?|"
    r"whatever(?:'?s)?\s*(?:works|is fine|is good)?|"
    r"either(?:\s*(?:one|works|is fine))?|"
    r"(?:you|your)\s+(?:choose|pick|call|decide)|surprise me|doesn'?t matter|"
    r"no preference|i don'?t mind|up to you|"
    r"(?:ok(?:ay)?|yes|yeah|sure|alright)[\s,]+(?:please\s+)?(?:go ahead|do it|book it)"
    # Trailing filler carries no new information: "ok do it then" and "yes please" are the
    # same agreement as "yes". Treating them as unrecognised sent the customer back to a
    # search for a slot they had just accepted.
    r")(?:\s+(?:then|now|please|for me|that one|it))*[\s.!]*$", re.I)


# The caller walking AWAY from the time they were pushing for — by intent label
# ("decline", "not interested") or by the plain phrases people use to drop a request
# ("ok let it be", "leave it", "never mind", "another time").
#
# This is the signal that a floated day/time is DEAD, distinct from a bare "no" answer to a
# yes/no confirmation (which is handled where pending_action is released). It exists because
# session["wanted_day"]/["time_pref"] — the day the caller is currently trying to book — was
# only ever cleared on a SUCCESSFUL booking, never when the caller abandoned the attempt. So
# "I'll come today at 6pm" (no slots) → "ok let it be" left wanted_day pinned to today, and
# the very next "lets book it" was resolved against today — a day with nothing free — instead
# of the tomorrow-10AM slot still standing on the table. The caller reads that as the agent
# forgetting the offer; really it remembered a preference the caller had walked away from.
_ABANDON_INTENT_RE = re.compile(
    r"decline|reject|not[_ ]?interest|never[_ ]?mind|abandon|walk[_ ]?away|give[_ ]?up", re.I)
_ABANDON_PHRASE_RE = re.compile(
    r"\b(?:let it be|leave it|leave that|never ?mind|forget (?:it|that|about it)|"
    r"drop it|not (?:now|today)|maybe later|some other time|another time)\b", re.I)


def _abandons_current_slot(user_input: str, intent: str | None) -> bool:
    """True when the caller is dropping the time they were pushing for — not booking it,
    not naming a new one, just walking away from it. The trigger to release the transient
    day/time preference so a later change of heart resolves against the STANDING offer."""
    return bool(_ABANDON_INTENT_RE.search(str(intent or ""))
                or _ABANDON_PHRASE_RE.search(user_input or ""))


WORD_HOURS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}

# The customer types "16th"; the agent SPEAKS "the sixteenth". _wanted_day only ever
# had to parse the former — nothing checked what the agent itself said until now, and
# what it says is spelled out in words.
WORD_DAYS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
            "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
            "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
            "twentieth": 20, "twenty-first": 21, "twenty first": 21, "twenty-second": 22,
            "twenty second": 22, "twenty-third": 23, "twenty third": 23,
            "twenty-fourth": 24, "twenty fourth": 24, "twenty-fifth": 25,
            "twenty fifth": 25, "twenty-sixth": 26, "twenty sixth": 26,
            "twenty-seventh": 27, "twenty seventh": 27, "twenty-eighth": 28,
            "twenty eighth": 28, "twenty-ninth": 29, "twenty ninth": 29,
            "thirtieth": 30, "thirty-first": 31, "thirty first": 31}


def _wanted_hour(text: str) -> int | None:
    """The hour named, in 24h where that is unambiguous.

    Handles "12 pm", "4pm", "at 10" — and also "one o'clock" and "twelve noon", because
    the agent speaks its read-backs in words. "Would you like me to move it to Friday at
    one o'clock?" was unreadable to the first version of this, so answering "yes please"
    lost the one o'clock and the customer was asked what day she wanted. Again.
    """
    low = (text or "").lower()
    if re.search(r"\b(?:noon|midday)\b", low):
        return 12
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", low)
    if m:
        h = int(m.group(1)) % 12
        return h + 12 if m.group(3) == "pm" else h
    # A spoken hour, with or without "o'clock". Requiring it missed "do it at ten", which
    # is how most people actually pick from a list that was just read to them. Guarded by
    # "at" or an action verb so "one bedroom" and "two lakh" are not read as times.
    words = "|".join(WORD_HOURS)
    m = re.search(rf"\b(?:at|for|book|do it|make it|take)\s+(?:the\s+)?({words})\b", low)
    if m:
        return WORD_HOURS[m.group(1)]
    m = re.search(rf"\b({words})\b\s*o'? ?clock", low)
    if m:
        return WORD_HOURS[m.group(1)]
    m = re.search(r"\bat (\d{1,2})(?::\d{2})?\b", low)
    if m and 0 <= int(m.group(1)) <= 23:
        return int(m.group(1))
    return None


# Slot times are stored with an offset; "tomorrow" means tomorrow where the customer is.
IST = ZoneInfo("Asia/Kolkata")


def _slot_date(iso: str | None):
    try:
        return datetime.fromisoformat(iso).date() if iso else None
    except (ValueError, TypeError):
        return None


def _slot_time(iso: str | None) -> str | None:
    """The clock time of an ISO slot as "HH:MM", for a one-minute search window.

    When a caller names an exact time to book/move to, the forced search targets THAT time
    (time_from == time_to) rather than the whole day. A free slot then lands in the offered
    set and verifies; an unavailable one comes back through search_slots' own "requested
    window is full — here are the times that ARE free" path, so the agent offers the day's
    real alternatives instead of looping on a time it can never verify.
    """
    try:
        return f"{datetime.fromisoformat(iso):%H:%M}" if iso else None
    except (ValueError, TypeError):
        return None


def _wanted_day(*texts: str):
    """The date the customer named, if any. Only what is unambiguous from their words."""
    today = datetime.now(IST).date()
    for t in texts:
        low = (t or "").lower()
        if re.search(r"\btomorrow\b", low):
            return today + timedelta(days=1)
        if re.search(r"\btoday\b|\bthis afternoon\b|\bthis evening\b", low):
            return today
        # A weekday name — "tuesday", "next friday". Without this the requested day was
        # simply lost: "I want tuesday evening" parsed to no date, so the booking task had
        # no requested_day, Rule 1 never re-searched, and a stale slot from an earlier
        # week-wide search got booked instead. Resolve to the next occurrence on/after
        # today ("tuesday" on a Monday is tomorrow; on a Tuesday, today).
        _wd = re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", low)
        if _wd:
            target = ["monday", "tuesday", "wednesday", "thursday",
                      "friday", "saturday", "sunday"].index(_wd.group(1))
            ahead = (target - today.weekday()) % 7
            if ahead == 0 and re.search(r"\bnext\b", low):
                ahead = 7
            return today + timedelta(days=ahead)
        # "on the 14th", "14 August", "August 14"
        m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?"
                      r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", low)
        if not m:
            m2 = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+"
                           r"(\d{1,2})(?:st|nd|rd|th)?\b", low)
            if m2:
                day, mon = int(m2.group(2)), m2.group(1)
            else:
                # "the" is optional and the ordinal suffix is often a separate word —
                # "on 16th", "on the 16th", and "on 16 th" (a common STT split) all name
                # the same day. Requiring "the" and an attached suffix caught only the
                # last of these; "no i want something on 16th" and "why not on 16th"
                # both parsed to no date at all, which is exactly what let Bug 4's
                # search-before-claiming-unavailable check go unreachable for the two
                # real transcripts it was written to fix.
                #
                # The suffix's SECOND letter is also optional, for the same reason: STT
                # drops a trailing letter as often as it drops a whole word. "on 18t in
                # the evening" — a real transcript — is "eighteenth" with the h clipped,
                # and it parsed to no date at all under a pattern that demanded the full
                # two letters. That silently defeated every guard built on this
                # function: the day never resolved, so nothing was ever "uncovered",
                # and a fabricated "no evening slot on the 18th" reached the customer
                # with no search_slots call anywhere behind it.
                m3 = re.search(r"\bon\s+(?:the\s+)?(\d{1,2})\s*"
                              r"(?:s(?:t)?|n(?:d)?|r(?:d)?|t(?:h)?)\b"
                              r"(?!\s*floors?\b)", low)
                if not m3:
                    continue
                day, mon = int(m3.group(1)), None
        else:
            day, mon = int(m.group(1)), m.group(2)

        months = ["jan", "feb", "mar", "apr", "may", "jun",
                  "jul", "aug", "sep", "oct", "nov", "dec"]
        month = months.index(mon) + 1 if mon else today.month
        year = today.year
        try:
            found = datetime(year, month, day).date()
        except ValueError:
            continue
        # A date already past this month means next month — "the 3rd" said on the 28th.
        if found < today and not mon:
            month = month % 12 + 1
            year += 1 if month == 1 else 0
            try:
                found = datetime(year, month, day).date()
            except ValueError:
                continue
        return found
    return None


_MONTH_RE = (r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
            r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?")
_MONTHS_SHORT = ["jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"]


def _day_token_to_num(tok: str) -> int | None:
    tok = tok.strip()
    m = re.match(r"^(\d{1,2})", tok)
    if m:
        return int(m.group(1))
    return WORD_DAYS.get(tok) or WORD_DAYS.get(tok.replace("-", " "))


def _spoken_day(text: str) -> date | None:
    """The calendar day named in SPOKEN text — "the nineteenth of August", "August
    19th" — the mirror image of _wanted_day, which is tuned for how a CUSTOMER types a
    date ("on the 16th"), not how the agent SAYS one out loud.

    Built to catch a specific, serious failure: book_appointment can only ever succeed
    against a real, previously-offered slot_start (the slot_was_offered precondition
    guarantees that) — but nothing checked that the SENTENCE confirming it back to the
    customer actually named that same day. A model with the customer's stated
    preference ("the 19th") still fresh in the conversation and a genuinely-booked
    slot_start for a different, earlier day in the tool result can — and did — describe
    the wrong one: "your appointment is all set for the nineteenth of August at ten in
    the morning" when what was actually booked, and validated, was the 15th at ten.
    The customer is told a day nobody is expecting her.
    """
    low = (text or "").lower()
    day_words = "|".join(sorted((k.replace(" ", r"\s+") for k in WORD_DAYS),
                                key=len, reverse=True))
    day_tok = rf"(?:\d{{1,2}}(?:st|nd|rd|th)?|{day_words})"
    m = re.search(rf"\b({day_tok})\s+of\s+({_MONTH_RE})\b", low)
    if m:
        day_txt, mon_txt = m.group(1), m.group(2)
    else:
        m = re.search(rf"\b({_MONTH_RE})\s+(?:the\s+)?({day_tok})\b", low)
        if not m:
            return None
        mon_txt, day_txt = m.group(1), m.group(2)
    day = _day_token_to_num(day_txt)
    if day is None:
        return None
    month = next((i + 1 for i, mm in enumerate(_MONTHS_SHORT) if mon_txt.startswith(mm)), None)
    if month is None:
        return None
    today = datetime.now(IST).date()
    year = today.year
    try:
        found = date(year, month, day)
    except ValueError:
        return None
    # A spoken confirmation naming a date already past makes no sense as today's
    # booking — same next-occurrence rule _wanted_day already applies to customer input.
    if found < today:
        month2 = month % 12 + 1
        year2 = year + (1 if month2 == 1 else 0)
        try:
            found = date(year2, month2, day)
        except ValueError:
            return None
    return found


# The time-of-day window the customer named, matched to the same boundaries search_slots
# uses (scheduling_tools.PART_OF_DAY): evening is 4 PM onwards, afternoon 12–4. A day that
# only shows a single spread sample (one slot, from a week-wide search) is NOT "covered" for
# a specific window — offering 5 PM to someone who asked for the afternoon is the exact
# antipattern the prompt warns about.
_PART_OF_DAY_RE = re.compile(r"\b(morning|afternoon|evening)\b", re.I)
_PART_WINDOWS = {"morning": (0, 12), "afternoon": (12, 16), "evening": (16, 24)}


def _wanted_part_of_day(text: str) -> str | None:
    m = _PART_OF_DAY_RE.search(text or "")
    return m.group(1).lower() if m else None


def _day_covered_in_window(session: dict, day, part: str | None) -> bool:
    """True when an offered slot falls on ``day`` AND inside the named time window.

    With no window named this is just "is the day offered at all"; with one, the offered
    slot has to actually be in it, so "19th afternoon" is not satisfied by a stray 5 PM.
    """
    if day is None:
        return False
    lo, hi = _PART_WINDOWS.get(part or "", (0, 24))
    for s in (session.get("offered_slots") or []):
        d = _slot_date(s.get("starts_at"))
        if d != day:
            continue
        try:
            hour = datetime.fromisoformat(s["starts_at"]).hour
        except (ValueError, KeyError, TypeError):
            continue
        if lo <= hour < hi:
            return True
    return False


def _day_covered_at_hour(session: dict, day, hour: int | None) -> bool:
    """True when an offered slot on ``day`` matches the exact clock hour the caller named.

    The window-based cover check (``_day_covered_in_window``) treats a day as covered once
    ANY slot on it has been offered — right for a part-of-day ask, wrong for a specific time.
    Someone who asks for "today at 11 am" when only a 4 PM slot was ever offered has NOT had
    their time looked up, yet the day looked covered, so no fresh search fired and the agent
    stalled on "let me check if that's available". Used to force a search for the exact hour
    instead. Only meaningful alongside a day, and never fabricates one.
    """
    if day is None or hour is None:
        return False
    for s in (session.get("offered_slots") or []):
        if _slot_date(s.get("starts_at")) != day:
            continue
        try:
            if datetime.fromisoformat(s["starts_at"]).hour == hour:
                return True
        except (ValueError, KeyError, TypeError):
            continue
    return False


def _named_day_uncovered(user_input: str, session: dict) -> bool:
    """True when the customer named a real day and nothing already offered covers it.

    Backs the proactive twin of the no-availability-claim guard: force a real
    search_slots for a day nobody has looked up yet, rather than wait to see whether
    the model says something false about it first.
    """
    day = _wanted_day(user_input)
    if day is None:
        return False
    offered = session.get("offered_slots") or []
    return not any(_slot_date(s.get("starts_at")) == day for s in offered)


def _pick_offered_slot(session: dict, *texts: str) -> str | None:
    """The slot the customer just asked for, from the ones actually offered.

    Booking is the one action the engine could not stage for itself, because slot_start
    lives only in the model's head — and when the model would not name the tool, the
    customer was told "that is not booked yet, which day and time would you like?" while
    repeating the same day and time back at it. The slots ARE known: search_slots returned
    them earlier in this call. This finds the one they meant.
    """
    offered = session.get("offered_slots") or []
    if not offered:
        return None
    # The customer's words first, then the agent's own read-back. "Would you like me to
    # move it to Friday at one o'clock?" names the slot precisely — the model just failed
    # to also name the tool. Answering "yes please" to that should not lose the one o'clock.
    # The DAY has to match too.
    #
    # Matching on the hour alone booked a customer onto the wrong date: she asked for the
    # 14th at 4pm, the last search had returned the 13th, and the correction quietly moved
    # her to the 13th at 4pm and told her it was confirmed. Silently changing the day is
    # worse than failing — she would have turned up on a day nobody expected her.
    wanted_day = _wanted_day(*texts)
    # This turn's own words named no day — fall back to the day they last explicitly
    # asked for, rather than assume a single stale offered slot is still what they want.
    #
    # Without this, "no, something on the 18th" followed two turns later by just a name
    # (registering, nothing about a day) fell through to "only one slot was ever
    # offered, so that must be it" — and silently booked the 15th's slot, the exact day
    # she had just declined, because nothing here remembered she had declined it.
    if wanted_day is None and session.get("wanted_day"):
        try:
            wanted_day = date.fromisoformat(session["wanted_day"])
        except ValueError:
            wanted_day = None
    if wanted_day is not None:
        offered = [o for o in offered
                   if _slot_date(o.get("starts_at")) == wanted_day]
        if not offered:
            return None          # they named a day we have not searched — search it

    hour = next((h for h in (_wanted_hour(t) for t in texts) if h is not None), None)
    if hour is not None:
        # Try the hour as given, then its afternoon twin. "One o'clock" spoken about a
        # Friday afternoon means 13:00, and nobody says thirteen.
        for candidate in (hour, hour + 12) if hour <= 12 else (hour,):
            for slot in offered:
                try:
                    if datetime.fromisoformat(slot["starts_at"]).hour == candidate:
                        return slot["starts_at"]
                except (ValueError, KeyError, TypeError):
                    continue
        return None                      # they named an hour that was not on offer
    # No hour named. "Ok do it then" after a single offer means that offer — and after
    # several, the first one read out is the one they heard first and are agreeing to.
    # Returning nothing here sent the turn back to a fresh search for a time the customer
    # had already accepted.
    if AFFIRM_ONLY.match((texts[0] if texts else "") or ""):
        return offered[0]["starts_at"]
    return offered[0]["starts_at"] if len(offered) == 1 else None


# A question ABOUT an action is not a request to perform it.
#
# THE BUG THIS FIXES: "How long does a site visit take and what are the maintenance charges?"
# matched the booking pattern's bare "site visit" and became a book_appointment hint. Both
# models had read it correctly (intent product_question, is_scheduling false), but the hint
# overrode them: the planner was sent back twice to "call book_appointment" (it refused both
# times — ~2.6 s of dead air), the registration flow took booking as the goal, and the caller
# heard "Before I can book your visit, can I take your name?" with neither question answered.
# The same shape bites every request-family hint: "what happens if I cancel?", "is there a fee
# to reschedule?", "can I bring my parents on the visit?", "do I need to book in advance?".
#
# So a clause that asks about the action's properties — duration, cost, process, what to bring,
# what happens if — never yields one of these hints. Matched per CLAUSE, so "how long does a
# visit take? book me one for Saturday" still books. Only the request family is exempted:
# record_do_not_contact, verify_identity and create_ticket keep their current behaviour (a
# missed opt-out or complaint costs more than an extra clarifying turn).
_INQUIRY_DEFERRABLE = frozenset({"book_appointment", "reschedule_appointment",
                                 "cancel_appointment", "schedule_callback", "locate_centre",
                                 "send_document"})
_ACTION_INQUIRY_RE = re.compile(
    r"\bhow\s+(?:long|much\s+time|many\s+(?:hours|minutes))\b"
    r"|\bhow\s+(?:far\s+in\s+advance|early|soon\s+before)\b"
    r"|\bwhat\s+(?:happens|if\s+i)\b"
    r"|\bwhat\s+(?:should|do|must)\s+i\s+(?:bring|carry|expect|need|know)\b"
    r"|\bwhat\s+(?:documents?|id|things?|all)\s+(?:do|should|must)\s+i\b"
    r"|\bwhat(?:'s|\s+is|\s+are)\s+(?:the\s+)?(?:process|procedure|policy|steps?|rules?|"
    r"charges?|fees?|cost|timings?\s+of)\b[^.?!]{0,40}\b(?:for|to|of|on|when)\b"
    r"|\b(?:is|are)\s+there\s+(?:a\s+|any\s+)?(?:charges?|fees?|costs?|penalt(?:y|ies))\b"
    r"|\b(?:is|are)\s+(?:the\s+|a\s+)?(?:site\s+)?(?:visits?|viewings?|tours?|callbacks?)\s+"
    r"(?:free|chargeable|paid|mandatory|necessary|required|compulsory|guided)\b"
    r"|\bdo\s+i\s+(?:need|have)\s+to\b"
    r"|\bcan\s+i\s+(?:bring|carry|take)\b"
    r"|\b(?:who|which\s+\w+)\s+(?:will|would)\s+(?:be\s+there|show|meet|take)\b",
    re.IGNORECASE)
# Clause boundaries for the check above: sentence ends, semicolons, and a comma or "and/but"
# that starts a new clause.
_CLAUSE_RE = re.compile(r"(?<=[.?!])\s+|;\s*|,\s+(?:and\s+|but\s+|so\s+)?|\s+(?:and|but)\s+"
                        r"(?=(?:i|i'd|i'll|we|can|could|please|book|schedule|let's|what|how|"
                        r"is|are|do|does)\b)", re.IGNORECASE)


def _hinted_tool(text: str) -> str | None:
    low = (text or "").lower()
    clauses: list[str] | None = None
    for pattern, tool in TOOL_HINTS:
        if not re.search(pattern, low):
            continue
        if tool not in _INQUIRY_DEFERRABLE:
            return tool
        # A request-family hint counts only from a clause that is not asking ABOUT the action.
        if clauses is None:
            clauses = [c for c in _CLAUSE_RE.split(low) if c and c.strip()]
        if any(re.search(pattern, c) and not _ACTION_INQUIRY_RE.search(c) for c in clauses):
            return tool
    return None


# Intent labels that say the caller is asking for information, not asking us to act.
_INFORMATIONAL_INTENTS = frozenset({"product_question", "pricing_question", "policy_question",
                                    "smalltalk", "greeting"})


def _turn_hint(user_input: str, proposal=None) -> str | None:
    """The keyword hint for this turn, once the models have had their say.

    TOOL_HINTS is a BACKSTOP for a planner that mislabels a request — it was never meant to
    overrule two models that agree the turn is informational. When the understanding
    classifier actually ran (``routing.source == "model"``, not a trivial-skip or failure
    fallback), said ``is_scheduling`` is false, AND the planner's own intent is informational,
    a request-family hint is dropped: that is no longer a mislabel to correct, it is a keyword
    false positive ("site visit" inside "how long does a site visit take"). Any one of the
    three disagreeing keeps the hint, so a genuinely mislabelled visit request is still
    caught. Opt-out, identity and complaint hints are never dropped here.

    Use this, not `_hinted_tool`, wherever a hint DRIVES behaviour — a re-plan nudge, a forced
    tool, a remembered goal (session["wanted_tool"]), the booking signal."""
    hint = _hinted_tool(user_input)
    if hint not in _INQUIRY_DEFERRABLE or proposal is None:
        return hint
    routing = getattr(proposal, "routing", None) or {}
    intent = str((getattr(proposal, "understood", None) or {}).get("intent") or "").lower()
    if (routing.get("source") == "model" and routing.get("is_scheduling") is False
            and intent in _INFORMATIONAL_INTENTS):
        return None
    return hint


def _booking_signalled(session: dict, user_input: str, intent: str | None = None,
                       proposal=None) -> bool:
    """True when the customer has actually engaged the booking flow this call.

    A visit is signalled by a search having run / slots offered, a scheduling tool being the
    standing goal, a day the customer named, the planner reading the turn as a booking
    intent, or an explicit booking phrase in what they just said. It is NOT signalled by the
    model reaching for the book_appointment TOOL on its own. A caller who only greets and
    gives their name and number has asked for nothing — collecting their identity is
    legitimate, but it is not a booking. Without this distinction the registration machine
    treated every identity capture as a visit in flight: BOOKING IN PROGRESS lit up and the
    booking machine drove toward a visit nobody requested, inventing an appointment out of a
    caller who only wanted to leave their details.

    The reliable, phrasing-independent signal is `intent` — the planner's own read of what
    the caller wants ("I want to see the flat", "can we set up a walkthrough", "pop round
    on Saturday" all classify as book_appointment). That is what production should lean on;
    no regex keeps up with how many ways people ask to visit. The keyword match below stays
    only as a cheap backstop for the rare turn where the planner mislabels, and — crucially
    — the two are NOT symmetric: the intent LABEL is trusted as a booking signal, but the
    book_appointment TOOL proposal alone is not, which is the whole point of the guard. When
    the model proposes the tool off a greeting, the intent it assigns that turn is
    "greeting"/"smalltalk"/"create_customer", not "book_appointment" — so the phantom is
    still caught while a genuinely-phrased visit request is honoured.
    """
    if _scheduling_in_flight(session):
        return True
    if session.get("wanted_day"):
        return True
    if str(intent or "").lower() in ("book_appointment", "reschedule", "reschedule_appointment"):
        return True
    # The keyword backstop defers to the models when both read the turn as informational
    # (see _turn_hint) — "how long does a site visit take?" is not a booking.
    return _turn_hint(user_input or "", proposal) in ("book_appointment",
                                                      "reschedule_appointment")


def _cancellation_signalled(user_input: str) -> bool:
    """True when the caller's own words actually ask to cancel their visit.

    The mirror-image guard to _booking_signalled, but deliberately NOT symmetric with it:
    _booking_signalled trusts the planner's own intent label as the reliable signal. Here the
    intent label is exactly what went wrong, so it is not consulted at all -- only the
    caller's literal words.

    THE BUG THIS FIXES: right after a booking confirmation, the responder offers a WhatsApp
    location pin ("I can send you a pin if that helps"). A caller declining the PIN ("no
    thanks") got the planner itself to propose cancel_appointment, reasoning "customer
    declined after booking, needs cancellation confirmation" -- not a text-generation slip,
    the planner's own tool proposal was wrong. A vague "sure" on the very next turn then
    released the parked cancellation, and a real booking was cancelled over nothing more
    than a declined pin offer. Checked only against the turn PROPOSING the cancellation
    (gated by `not confirming_pending` at the call site) -- the confirmation turn itself
    ("sure", "yes") never contains cancel language either, and must not be blocked by this.
    """
    return _hinted_tool(user_input or "") == "cancel_appointment"


@dataclass
class TurnResult:
    response_text: str
    response_segments: list[str]
    action: str
    executed_tools: list[dict] = field(default_factory=list)
    state_changed_to: str | None = None
    escalation: dict | None = None
    end_call: bool = False
    trace_id: uuid.UUID | None = None
    confidence: dict = field(default_factory=dict)
    pending_confirmation: dict | None = None
    debug: dict = field(default_factory=dict)


class TurnCancelledAfterCommit(asyncio.CancelledError):
    """The turn was cancelled, but only once its persistence had begun -- and that
    persistence was then finished, not abandoned (see _persist_turn). Still a
    CancelledError, so pipecat's task cancellation completes exactly as before; the subclass
    only tells a caller that this turn's message, trace, tool effects and working memory ARE
    durably saved, so it must neither roll back nor carry the caller's words forward."""


async def _persist_turn(db: AsyncSession, conversation_id: uuid.UUID, session: dict,
                        bridge: StageTimer) -> None:
    """Save working memory (Redis) and commit the turn (Postgres) as one unit that a
    cancellation cannot split.

    A voice turn is cancelled whenever the caller barges in (bridge.py). Landing between
    these two awaits, that used to leave Redis advanced (turn_index, pending_action,
    offered_slots...) over a rolled-back DB -- or the reverse -- so the next turn planned
    against a state no store agreed on. Once persistence starts, the turn's work is already
    done; finishing it is always the consistent choice. It runs in its own task, shielded,
    and on cancellation this coroutine waits for it IN PLACE before re-raising, so the
    caller (which holds the per-call DB lock) never touches the session while the commit is
    still in flight -- the orphaned-shield race bridge.py's _db_lock comment describes.
    """
    async def _write() -> None:
        async with bridge.stage("session_save"):
            await working.save(conversation_id, session)
        async with bridge.stage("commit"):
            await db.commit()

    task = asyncio.ensure_future(_write())
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except Exception:
            # The write itself failed while we were being cancelled. Report a plain
            # cancellation so the caller rolls back as usual.
            logger.exception("turn persistence failed during cancellation (conversation=%s)",
                             conversation_id)
            raise asyncio.CancelledError() from None
        raise TurnCancelledAfterCommit() from None


async def _persona(db: AsyncSession, tenant_id: uuid.UUID) -> str | None:
    row = (await db.execute(select(PromptTemplate).where(
        PromptTemplate.tenant_id == tenant_id, PromptTemplate.purpose == "persona",
        PromptTemplate.is_active == True))).scalars().first()  # noqa: E712
    return row.template if row else None


def _with_bridge_timing(timing: dict[str, int], bridge: StageTimer,
                        started: float) -> dict[str, int]:
    """Assemble the final timing payload, adding the bridge breakdown (P1.08).

    ``total_ms`` is always present (it was before this change too). When bridge timing is
    enabled we also attach:
      - ``bridge_<stage>_ms``   — the individual sub-stages (session_load, input_persist,
                                  customer_load, trace_persist, session_save, commit);
      - ``bridge_ms``           — the DIRECT sum of those six sub-stages, i.e. the real
                                  DB/Redis orchestration overhead. Measured, not residual,
                                  so it can never absorb time we didn't wrap;
      - ``unattributed_ms``     — whatever wall-clock is left after every measured stage
                                  (planner/tool/responder/retrieval + the bridge sub-stages)
                                  is subtracted. This is the honest "not instrumented"
                                  bucket: an auxiliary model call a turn makes that isn't
                                  tagged planner/responder shows up HERE, not as fake bridge
                                  I/O. A large value means "go wrap that stage", not "chase
                                  DB latency". Because the leftover always lands here,
                                  nothing hides — it just stops being mislabelled.
    Disabled, the payload is byte-for-byte what it always was, so the flag is a true no-op.
    """
    out = {**timing, "total_ms": int((time.perf_counter() - started) * 1000)}
    if get_settings().bridge_timing_enabled:
        bridge.merge_into(out, prefix="bridge_")
        out["bridge_ms"] = bridge.sum_ms()
        out["unattributed_ms"] = bridge.residual_ms(out)
    return out


async def process_turn(db: AsyncSession, *, tenant_id: uuid.UUID, conversation: Conversation,
                       user_input: str, stt_confidence: float | None = None,
                       stt_seconds: float | None = None,
                       actor: str = "agent", as_of_date: date | None = None,
                       sentence_callback=None,
                       session_embed_cache: "dict[str, list[float]] | None" = None) -> TurnResult:
    started = time.perf_counter()
    timing: dict[str, int] = {}
    # Bridge instrumentation (P1.08). The StageTimer attributes the per-turn orchestration
    # overhead — session load, message persist, trace write, commit — that the benchmark
    # report lumps into the ~60 ms "BRG" stage and that nothing here previously broke out.
    # It only records perf_counter deltas, so it is safe to run every turn; its numbers are
    # surfaced on debug.timing at the end, gated by settings.bridge_timing_enabled.
    _bridge = StageTimer()
    async with _bridge.stage("session_load"):
        session = await working.load(conversation.id)
    # Snapshot BEFORE this turn's own logic can set it (e.g. a booking succeeding this turn) --
    # see _trim_repeated_visit_reminder's already_given_at_turn_start param for why this must
    # be the pre-turn value, not whatever session holds by the time that function runs.
    _visit_reminder_already_given_at_turn_start = bool(session.get("visit_reminder_given_this_call"))

    # Words in this voice transcript that sound like a term the call expects but were
    # transcribed as something else ("position" -> "possession", "rare" -> "RERA") -- checked
    # against the call lexicon voice_ws.py/plivo_ws.py stored at call start. Hints only: every
    # plan() below shows them to the model, which decides with the conversation in view; the
    # caller's words themselves are never rewritten. Empty for chat (no lexicon) and for a
    # transcript with nothing suspicious, so neither path changes at all.
    _asr_hints = (asr_hints_mod.find_hints(user_input, session.get("asr_lexicon"))
                  if session.get("asr_lexicon") else [])
    if _asr_hints:
        logger.info("voice asr hints (conversation=%s text=%r hints=%s)", conversation.id,
                    user_input, [(h.heard, h.term, h.score) for h in _asr_hints])

    # Outbound calls carry two facts the rest of the turn needs: that WE placed the call (so the
    # identity gate below runs and the prompt speaks in the outbound register), and WHY — the
    # purpose, which lives on the linked Call row, not the conversation. Seeded once per call and
    # cached on the session so the lookup happens at most once.
    if conversation.direction == "outbound":
        session["outbound"] = True
        if not session.get("call_purpose"):
            _p = (await db.execute(
                select(Call.purpose).where(Call.conversation_id == conversation.id)
                .order_by(Call.created_at.desc()).limit(1))).scalars().first()
            session["call_purpose"] = _p or outbound_flow.PURPOSE_MARKETING

    # Remember the day they most recently, explicitly named — independent of whatever
    # tool runs or does not run this turn.
    #
    # THE BUG THIS FIXES: _pick_offered_slot's "only one slot has ever been offered, so
    # a bare 'yes' must mean that one" fallback has no memory beyond the current turn's
    # own words. Asked to see the flat, offered Saturday 15th at 10 AM, the customer
    # said "no, something on the 18th" — a day nobody had searched, so nothing new was
    # offered — and then just gave her name to register. Her name mentions no day, the
    # fallback fired on the one stale slot still sitting in offered_slots, and the
    # booking that finally went through was the 15th's 10 AM silently reattached to the
    # 18th: right day, a time she never heard and never agreed to, because "yes" to a
    # question that named no time at all was read as consent to it anyway.
    _named_today = _wanted_day(user_input)
    if _named_today is not None:
        session["wanted_day"] = _named_today.isoformat()
    # The time-of-day window is part of what they asked for, and it has to outlive an
    # interruption exactly as the day does. "I want tuesday evening" then a registration
    # detour used to lose the "evening": the next search ran unwindowed and a 3 PM slot got
    # booked for an evening request. Persisted here, it drives Rule 1's windowed search even
    # on a later turn whose words are just a name.
    _named_part = _wanted_part_of_day(user_input)
    if _named_part is not None:
        session["time_pref"] = _named_part

    # The caller is accepting a handoff THIS agent offered last turn. When a reply offered to
    # put them through to a person (see _offers_human_handoff, armed at the end of that turn),
    # a bare "yes" now this turn means "yes, transfer me" — not a fresh question to be
    # re-classified, searched, and answered with the same offer again. Consumed once: the flag
    # is popped whatever the answer is, so a "no" or a change of subject simply drops the offer.
    # Acted on only when the reply reads as consent; the forced escalation is applied right
    # after plan() so the turn rides the same, tested escalation path a real human request does.
    accepted_handoff = bool(session.pop("offered_human_handoff", None)) \
        and is_affirmative(user_input)

    # Bridge the caller's ON-FILE visit (booked on an earlier call) into this turn, so a
    # returning caller can move or cancel it without the agent first re-running a lookup.
    #
    # Deliberately kept in a SEPARATE key from ``appointment_id``: the booking projection
    # (booking_task.project) reads ``appointment_id`` to drive the in-flight state machine,
    # and seeding a prior, settled visit there would make every passive turn render a
    # spurious "BOOKING IN PROGRESS" and — because that projection only ever reports THIS
    # call's booking as done — answer "is it booked?" with "not booked yet", the very fault
    # Feedback #8 reports. The durable status is surfaced by the YOUR VISIT ON FILE prompt
    # block instead; this key only makes the appointment_id available to the reschedule /
    # cancel tools (via known_appointment_id and _fit_params) when the customer acts on it.
    # Loaded only when the caller is identified and no booking has been made on THIS call.
    if conversation.customer_id and not session.get("appointment_id"):
        _on_file = await load_upcoming_appointments(
            db, tenant_id=tenant_id, customer_id=conversation.customer_id, as_of=as_of_date)
        if _on_file:
            session["on_file_appointment_id"] = _on_file[0]["appointment_id"]
        else:
            # They may have cancelled since; do not carry a stale bridge forward.
            session.pop("on_file_appointment_id", None)

    # The BookingTask state machine (docs/booking-task-architecture.md). The projection is a
    # pure read of the session; it both surfaces the booking to the model (Rule 6 summary)
    # and sequences the booking (the invariant checks below). This is the orchestrator-owned
    # state that replaced the emergent, scattered session-key reconstruction the agent used
    # to rely on.

    def _booking_summary() -> str | None:
        """Rule 6 — a one-line summary of the in-flight booking for the prompt."""
        _t = bt.project(session, customer_exists=bool(conversation.customer_id),
                        turn_index=int(session.get("turn_index", 0)),
                        active_entities=session.get("active_entities") or {})
        return _t.summary_line() if _t.slots or _t.status != "gathering" else None

    async with _bridge.stage("input_persist"):
        db.add(Message(tenant_id=tenant_id, conversation_id=conversation.id,
                       role="customer", content=user_input))
        await db.flush()

    # L8.01: Deterministic fast-path for pure backchannels and presence checks.
    # A caller who hums "mm-hmm" or says "are you there?" carries zero new content —
    # nothing to plan, retrieve, or answer with an LLM. Detect these with the same
    # regex classifier used post-plan (conversational_signal), but act on it BEFORE
    # calling plan(), so both the planner (~1173ms) and the responder (~831ms) are
    # skipped entirely. The reply is built deterministically from the prior agent line.
    #
    # Safety constraints: (a) only unambiguous signals (the consent-guarded agreement
    # words still fall through when anything is pending or a question was asked),
    # (b) requires last_agent_reply in session (i.e., not the opening turn),
    # (c) never fires when a handoff was accepted this turn.
    _fast_path_text: str | None = None
    _fast_path_signal_tag: str | None = None

    # ---- chunked phone-number capture mode (phone_capture.py), hoisted fast-path ----
    #
    # Every turn while this is active is fully deterministic -- chunk/correction/confirm
    # logic, no judgment calls -- see the module docstring. THE BUG THIS FIXES: it used to
    # be applied as a LATE override deep inside the registration flow, well after plan()
    # (and sometimes a tool) had already run and been discarded. Two separately-reported
    # bugs came from that lag:
    #   1. action="respond" with no tool trips stream_respond_no_tool_enabled, so the
    #      responder's OWN improvised sentence reached TTS via the live sentence_callback
    #      before the late override ever ran -- the caller heard the model's guess, then
    #      the real reply read right after it.
    #   2. The planner call itself (~1-2s) ran on every chunk turn for nothing -- its output
    #      was always thrown away -- which widens the exact window a caller reading digits
    #      in quick bursts can talk over. A real call had the FIRST chunk's turn cancelled
    #      by the caller's next utterance before phone_capture ever got to record it,
    #      silently losing that chunk (barge-in genuinely abandons a cancelled turn's work;
    #      see bridge.py's cancellation-safety comment -- the fix there is that this no
    #      longer corrupts the call, not that the turn's work survives). Shortening the
    #      window is what actually prevents the loss.
    # Hoisted here, before plan() is ever called, the same fast-path shape as the
    # backchannel/check-in detection just below -- and checked first: a stray "mm-hmm"
    # while mid-capture must be read by phone_capture.py as "nothing usable, nudge" (it
    # already handles that), not as a generic backchannel.
    #
    # conversation.customer_id is None is the same "no customer yet" gate the registration
    # flow uses below (customer = None if not conversation.customer_id else ...) -- checked
    # here directly since that query hasn't run yet this early.
    _phone_capture_turn_active = (
        conversation.customer_id is None and bool(session.get("phone_capture_active")))
    # True only while still mid-capture (not done, not handed off) -- gates the
    # registration flow's "ask for the phone number" trigger further below so it can never
    # re-fire on a turn phone_capture.py already owns and reset session["phone_capture_state"]
    # back to {}, silently discarding whatever chunks were just captured even though the
    # caller hears the correct per-chunk reply (since _fast_path_text wins regardless of
    # what the registration flow separately decides to speak).
    _phone_capture_pending = False
    if _phone_capture_turn_active:
        _pc_state = phone_capture.PhoneCaptureState(
            **(session.get("phone_capture_state") or {}))
        _pc_caller_id = await _caller_phone(db, session, conversation)
        _pc_result = phone_capture.handle_turn(
            _pc_state, user_input, caller_id_phone=_pc_caller_id)
        if _pc_result.done:
            session.setdefault("registration", {})["phone"] = _pc_result.phone
            session["phone_capture_active"] = False
            session.pop("phone_capture_state", None)
            # No fast-path reply -- let plan() run this turn as normal, which (now that
            # registration.phone is set) proceeds straight to create_customer if the name
            # is already known too.
        else:
            session["phone_capture_state"] = {
                "chunks": list(_pc_result.state.chunks),
                "awaiting_confirm": _pc_result.state.awaiting_confirm,
                "failed_confirmations": _pc_result.state.failed_confirmations,
            }
            if _pc_result.handed_off:
                session["phone_capture_active"] = False
                session.pop("phone_capture_state", None)
            _fast_path_text = _pc_result.reply
            _fast_path_signal_tag = "phone_capture"
            _phone_capture_pending = True

    # ---- consent: the caller's reply to a parked action (app/engine/planner/consent.py) ----
    #
    # A confirmation-required write (book / reschedule / cancel / ticket) parked last turn and
    # read back ("Shall I book your visit for …?") is decided by what the caller says NOW. The
    # deterministic read runs here, before plan(), so a plain "what time?" is answered from the
    # staged data without a planner call; the model's read (on the understanding call plan()
    # already makes) is fused in after plan(). Everything below that releases a parked action
    # keys off `_consent`, never off a model intent label — see is_affirmative for the call
    # that label used to book.
    #
    # `asked_turn` binds consent to the question just asked: a "yes" only releases a parked
    # action if the previous reply actually asked about it. An action left parked across a side
    # question or a "let me think" is not released by a "yes" to something else two turns on —
    # it gets a fresh read-back first. A pending action from before this field existed (no
    # asked_turn) keeps the old behaviour.
    _consent_pending = dict(session.get("pending_action") or {})
    _consent_staged = bool(_consent_pending.get("tool")) and not _consent_pending.get("needs")
    _consent_turn_index = int(session.get("turn_index", 0))
    _consent_asked_turn = _consent_pending.get("asked_turn")
    _pending_fresh = (_consent_asked_turn is None
                      or int(_consent_asked_turn) == _consent_turn_index - 1)
    _consent_lex = consent.read_lexical(
        user_input, pending_tool=_consent_pending.get("tool"),
        staged_start=((_consent_pending.get("params") or {}).get("slot_start")
                      if _consent_staged else None),
        today=_call_today(as_of_date))
    _consent_hold: str | None = None        # repeat | unclear | check_in | backchannel
    _consent_pin: str | None = None         # the deterministic line a hold turn speaks
    _consent_reask_line: str | None = None  # appended after answering a side question
    _consent_declined = False
    _consent_asked_this_turn = False

    # "what time?" / "say that again" / "phir se bolo" while something is parked: read the
    # staged details back and ask again — deterministic, no planner, no tool, nothing released.
    if (_fast_path_text is None and _consent_staged and not accepted_handoff
            and not _phone_capture_turn_active and _consent_lex.decision == consent.REPEAT):
        _consent_hold = consent.REPEAT
        _consent_pin = _consent_line(session, _consent_pending, kind="repeat",
                                     user_input=user_input)
        _fast_path_text = _consent_pin
        _fast_path_signal_tag = "consent_repeat"

    _fp_prev = session.get("last_agent_reply")
    if (_fast_path_text is None and _fp_prev is not None and not accepted_handoff
            and not _phone_capture_turn_active
            and int(session.get("turn_index", 0)) > 0
            # A reply that agrees to the parked action is consent, never a nod — "go ahead"
            # to "shall I book it?" must reach the confirmation gate, not the continuer path.
            and not (_consent_staged and _consent_lex.decision == consent.ACCEPT)):
        _fp_asked_q = conversational_signal.ends_with_question(_fp_prev)
        _fp_pending = bool(
            session.get("pending_action") or session.get("offered_slots")
            or session.get("awaiting_project_choice") or session.get("awaiting_visit_name")
            or int(session.get("reengage_count", 0) or 0))
        _fp_det = conversational_signal.classify(
            user_input, agent_asked_question=_fp_asked_q,
            is_opening_turn=False, pending=_fp_pending)
        if _fp_det == conversational_signal.SIGNAL_BACKCHANNEL:
            _fast_path_text = conversational_signal.backchannel_reply(
                _fp_prev, agent_asked_question=_fp_asked_q,
                variant=int(session.get("turn_index", 1)) % 3)
            _fast_path_signal_tag = "backchannel"
        elif _fp_det == conversational_signal.SIGNAL_CHECK_IN:
            _fast_path_text = conversational_signal.check_in_reply(
                _fp_prev, variant=int(session.get("turn_index", 1)) % 4)
            _fast_path_signal_tag = "check_in"
        # "Hello?" or "mm-hmm" while an action is parked: the reply re-delivers the question,
        # so the action must stay parked and the question count as asked. It used to fall
        # through as a plain "respond" turn, which cleared the parked action — the caller's
        # next "yes" then had nothing to release and was asked all over again.
        if _fast_path_signal_tag in ("check_in", "backchannel") and _consent_staged:
            _consent_hold = _fast_path_signal_tag
            _consent_pin = _fast_path_text

    # ---- steps 1-7 ----
    _t = time.perf_counter()
    if _fast_path_text is not None:
        proposal = Proposal(action="respond",
                            error=f"fast_path:{_fast_path_signal_tag}")
    else:
        proposal = await plan(
            db, tenant_id=tenant_id, conversation=conversation, user_input=user_input,
            stt_confidence=stt_confidence, asr_hints=_asr_hints,
            identity_verified=bool(session.get("identity_verified")),
            pending_action=session.get("pending_action"),
            active_entities=session.get("active_entities") or {},
            offered_slots=session.get("offered_slots") or [],
            booking_summary=_booking_summary(),
            # The previous turn asked "which project?" and this turn is the answer — so the
            # deterministic resolver may accept a positional/place/brand reply ("the second one",
            # "the Sarjapur one", "DSR"), and the fast classifier is asked to resolve a free-form
            # one. Set the moment we force the clarify (below), cleared once a project resolves.
            awaiting_project_choice=bool(session.get("awaiting_project_choice")),
            # Accepting a handoff transfers regardless of what a document says, so the retrieval
            # this turn would only be discarded — skip it and save the caller the latency. Same
            # reasoning for chunked phone-capture mode (phone_capture.py): every turn while it's
            # active is handled deterministically below and the proposal here gets overridden
            # regardless, so retrieval's result is always thrown away. Skipping it also matters
            # for a reason beyond latency — it shortens how long this turn's own turn is in
            # flight, which is exactly the window a caller reading digits in quick bursts can
            # start talking over (see the voice-pipeline cancellation note on the DB rollback in
            # app/engine/voice/bridge.py's _run_turn).
            skip_retrieval=accepted_handoff or bool(session.get("phone_capture_active")),
            # Retrieval is otherwise gated SEMANTICALLY inside plan() by the fast understanding
            # call (P1): a scheduling turn or a booking-status question routes itself away from the
            # knowledge base without a regex here deciding it first. The classifier decides from
            # meaning rather than words — the only pre-computed skip is the handoff-accept above,
            # where the turn escalates and no document could change that.
            as_of_date=as_of_date,
            session_embed_cache=session_embed_cache,
            # While an action is parked, the understanding call also reads the reply against
            # the exact question asked (consent.py Layer B) — no extra model call.
            awaiting_confirmation=(
                session.get("last_agent_reply")
                if _consent_staged and get_settings().consent_semantic_enabled else None))
    # Record the opening plan leg immediately (P4-0 telemetry). A later corrective retry or
    # chain re-plan ACCUMULATES onto these, so the turn's total planner wall-time and call
    # count stay trustworthy. This used to be overwritten further down (planner_ms reset to
    # the last leg only, planner_calls hard-reset to 1) — which erased a corrective retry's
    # extra call from the very count the replay harness reads as the 'x' column.
    timing["planner_ms"] = int((time.perf_counter() - _t) * 1000)
    timing["planner_calls"] = 0 if _fast_path_text is not None else 1

    # Fuse the deterministic read with the model's (consent.decide). Computed every turn: the
    # confirmation gate also uses it for a "yes" to a slot just read out by a search.
    _consent = consent.decide(
        user_input, lexical=_consent_lex,
        semantic=((proposal.routing or {}).get("consent") if _consent_staged else None),
        stt_confidence=stt_confidence, stt_floor=get_settings().conf_stt_readback_below)
    _consent_kind = _consent.decision
    if (_consent_staged and _consent_hold is None and not accepted_handoff
            and not _phone_capture_turn_active
            and proposal.action not in ("escalate", "end_call")
            and not proposal.forced_escalation):
        # The model heard a repeat request or a line check the words alone missed.
        if (_consent.decision == consent.REPEAT
                or (proposal.routing or {}).get("conversational_signal") == "check_in"):
            _consent_hold = consent.REPEAT
        elif _consent.decision == consent.UNCLEAR:
            # Neither layer could read it. If the planner went for a lookup of its own, the
            # caller asked something we can answer — treat it as a side question (answer, keep
            # the action parked, re-ask). Otherwise hold: never let a garble become a yes.
            if (proposal.action == "invoke_tool" and proposal.tool in READ_ONLY_TOOLS
                    and proposal.tool != _consent_pending.get("tool")):
                _consent_kind = consent.SIDE_QUESTION
            else:
                _consent_hold = consent.UNCLEAR
        if _consent_hold == consent.REPEAT:
            _consent_pin = _consent_line(session, _consent_pending, kind="repeat",
                                         user_input=user_input)
        elif _consent_hold == consent.UNCLEAR:
            _unclear_n = int(_consent_pending.get("unclear_count") or 0)
            _consent_pin = _consent_line(session, _consent_pending, kind="unclear",
                                         user_input=user_input, attempt=_unclear_n)
            if session.get("pending_action"):
                session["pending_action"]["unclear_count"] = _unclear_n + 1
    # The full read rides out on debug["consent"]. Only an actual intervention is noted on
    # proposal.error (which also gates human-handoff triage), never the read itself.
    if _consent_hold is not None:
        _n = (f"consent hold ({_consent_hold}): kept the parked "
              f"{_consent_pending.get('tool')} and asked again — {_consent.decision} via "
              f"{_consent.source}{' (' + _consent.note + ')' if _consent.note else ''}.")
        proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n

    # Accepting an offered handoff escalates, deterministically. Applied the moment plan()
    # returns so the turn rides the SAME path a rule-forced escalation does — the escalation
    # section builds the briefing, opens the Escalation, and the FORCED_HANDOVER line below
    # replaces whatever the model drafted. Keyed like the explicit_human_request rule so the
    # Call Logs read identically to a caller who asked for a person outright. Not gated on
    # compliance: handing a caller to a person is always a safe action.
    if accepted_handoff:
        proposal.action = "escalate"
        proposal.tool = None
        proposal.forced_escalation = {
            "type": "immediate",
            "reason": "Customer accepted an offered handoff to a person",
            "rule": "explicit_human_request"}
        _note = ("Caller accepted the handoff this agent offered last turn; escalated to a "
                 "person instead of re-answering.")
        proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note

    # ---- rule-required information gate: "collect X before doing Y" ----
    #
    # A `require` custom instruction ("always collect the budget before offering a site visit")
    # fired in plan() and the required slot is still missing — so this turn is turned into an
    # ASK for that information instead of proceeding to the gated action (no tool runs, so the
    # visit is never offered/booked without it). The ask text is enforced after the responder,
    # exactly like FORCED_HANDOVER, so the caller reliably hears it. Two guards keep it safe:
    #   * never override an escalation (handing to a person outranks collecting a field), and
    #   * never TRAP the caller — after two asks the requirement steps aside and the call
    #     proceeds, the same "a code guard must not strand a caller" principle used by triage.
    if (proposal.forced_requirement and proposal.action != "escalate"
            and not accepted_handoff and not proposal.forced_escalation
            and not session.get("escalated")):
        _cond = proposal.forced_requirement.get("condition") or "requirement"
        _asked_key = f"require_asked::{_cond}"
        _asked = int(session.get(_asked_key) or 0)
        if _asked < 2:
            session[_asked_key] = _asked + 1
            proposal.action = "clarify"
            proposal.tool = None
            proposal.params = {}
            _note = (f"Rule {proposal.forced_requirement.get('rule')} requires "
                     f"'{_cond}' before proceeding; asked the caller for it "
                     f"(attempt {_asked + 1}).")
            proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note
        else:
            # Asked twice, still not provided — do not strand the caller. Clear the gate so the
            # turn proceeds normally, and let the counter reset next time the slot is captured.
            proposal.forced_requirement = None
            _note = (f"Requirement '{_cond}' still unmet after {_asked} asks; proceeding so the "
                     f"caller is not trapped.")
            proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note

    # One corrective retry when the model asks to confirm an action it has not named.
    #
    # This is the failure that has cost more calls than any other: it replies "shall I
    # book it?" with tool=null, the customer says yes, and there is nothing staged to
    # release — so it says yes again, decides on its own that the booking is done, and
    # tells the customer so. Instructing it not to has been tried repeatedly and does not
    # hold. Asking again, with the fault named, does.
    #
    # The cost is one extra call ONLY on a turn the model has already got wrong. A correct
    # proposal never pays it.
    #
    # The same correction now covers a second, more expensive shape: a reply that PROMISES
    # to go and do something while naming no tool. Both faults have one cure — ask again
    # with the fault named — so they share one retry and one extra call.
    fault = None
    _force_search_day = None
    if (proposal.action in ("confirm", "invoke_tool") and not proposal.tool
            and not session.get("pending_action")):
        fault = ("your last reply asked to confirm an action but named no tool, so nothing "
                 "was staged and nothing can happen")
    elif not proposal.tool and _turn_hint(user_input, proposal):
        hinted = _turn_hint(user_input, proposal)
        fault = (f"the customer asked for something that only the {hinted} tool can "
                 f"actually do, and you named no tool — so whatever you said next left "
                 f"their record unchanged. Call {hinted}")
    # Proactive scheduling check. If the customer named a day nobody has searched, and
    # scheduling is already what this call is about, the day gets checked before the model
    # says anything about it — not just after it says something wrong. A reply that answers
    # something else entirely while ignoring the day (asked "why not the 16th", proposing
    # record_customer_fact) is just as capable of costing a real visit.
    # Tracked separately from `fault` itself (via `_force_search_day`, set alongside it) so
    # the fallback below knows WHICH correction this is — the other faults recover by trying
    # again and, failing that, either force a specific tool whose params are trivially
    # recoverable (verify_identity, record_do_not_contact) or just give up and log it. A
    # named day deserves the same forcing: _wanted_day already parsed it once to detect the
    # fault, so the params to force search_slots with are sitting right here, not a guess.
    elif (proposal.tool != "search_slots" and _scheduling_in_flight(session)
            and _named_day_uncovered(user_input, session)):
        fault = ("the customer named a day and you have not called search_slots for "
                 "it — call search_slots for that exact day before saying anything "
                 "about what is or is not available")
        _force_search_day = _wanted_day(user_input)

    # A consent hold turn ("what time?", a garble, "hello?" while something is parked) only
    # re-asks — it must not be "corrected" into running a tool the caller did not ask for.
    if _consent_hold is not None:
        fault, _force_search_day = None, None

    if fault:
        _t = time.perf_counter()
        retry = await plan(
            db, tenant_id=tenant_id, conversation=conversation,
            user_input=(f"{user_input}\n\n[SYSTEM: {fault}. Reply again and DO IT ON THIS "
                        f"TURN: action=invoke_tool, with the tool you intend to run and its "
                        f"params, filled from earlier tool results in this conversation. "
                        f"Do not say you will do something — do it.]"),
            stt_confidence=stt_confidence, asr_hints=_asr_hints,
            identity_verified=bool(session.get("identity_verified")),
            pending_action=session.get("pending_action"),
            active_entities=session.get("active_entities") or {},
            offered_slots=session.get("offered_slots") or [],
            booking_summary=_booking_summary(),
            as_of_date=as_of_date, skip_retrieval=True)
        timing["planner_ms"] = timing.get("planner_ms", 0) + int((time.perf_counter() - _t) * 1000)
        timing["planner_calls"] = timing.get("planner_calls", 1) + 1
        if retry.tool:
            proposal = retry

        # A day the customer explicitly named is exactly as recoverable as the digits
        # verify_identity forces itself with below — _wanted_day already parsed it once
        # to detect the fault, so there is nothing left to guess. Checked regardless of
        # what the retry did: a model that swaps in some OTHER tool instead of
        # search_slots (proposing book_appointment again, say) has refused exactly as
        # much as one that named nothing at all, and the customer is at exactly the
        # same risk of being booked onto a day nobody looked at.
        if _force_search_day is not None and proposal.tool != "search_slots":
            proposal.action = "invoke_tool"
            proposal.tool = "search_slots"
            proposal.params = {"date_from": _force_search_day.isoformat(),
                               "date_to": _force_search_day.isoformat()}
            _n = (f"Planner would not call search_slots for {_force_search_day} after "
                  f"being asked twice; engine invoked it directly.")
            proposal.error = ((proposal.error + " | ") if proposal.error else "") + _n
        elif not retry.tool:
            hinted = _turn_hint(user_input, proposal)
            if hinted in FORCE_IF_REFUSED:
                proposal.action = "invoke_tool"
                proposal.tool = hinted
                proposal.params = retry.params or {}
                # They said the digits; use them. The tool decides whether they are right.
                if hinted == "verify_identity" and not proposal.params.get("last_four"):
                    digits = re.fullmatch(r"\s*(\d{4})\s*", user_input or "")
                    if digits:
                        proposal.params = {"last_four": digits.group(1)}
                proposal.error = ((proposal.error + " | ") if proposal.error else "") + \
                    f"Planner would not call {hinted}; engine invoked it directly."
            else:
                proposal.error = ((proposal.error + " | ") if proposal.error else "") + \
                    "Planner named no tool on two attempts."
    # planner_ms / planner_calls were recorded on the opening leg and accumulated by the
    # corrective retry above — do NOT overwrite them here (that was the telemetry bug).
    timing["retrieval_ms"] = (proposal.retrieval or {}).get("latency_ms") or 0
    # Per-stage breakdown of the planning legs (P4-0 profiler): context assembly, the
    # understand+retrieve overlap (with each leg isolated), the rule gate and the strong
    # propose call — so a slow turn attributes to a stage rather than one opaque planner_ms.
    # Reflects the proposal actually acted on (a corrective retry's stages replace the first).
    # tool_ms and responder_ms are added later, where those stages run.
    timing["stages"] = dict(proposal.stage_ms or {})

    # ---- deterministic project disambiguation (multi-project-per-tenant) ----
    #
    # When the tenant runs several projects and this call has NOT resolved to one, a product /
    # price / configuration / booking turn cannot be answered without silently guessing a
    # project. The prompt already asks the model to ask which — but the model is not reliable
    # here: given the candidate list ("DSR Highland Greenz (…)", "Lemon Tree (Whitefield)") it
    # grabbed one and answered "3 BHK available at Lemon Tree in Whitefield" instead of asking,
    # lifting the place straight out of the list. So the ENGINE forces the clarification, the
    # same way it forces a handover line: no tool runs, and the spoken text is PINNED to a
    # fixed question below (see _force_project_clarify). Deterministic, not a hope.
    #
    # Only fires on a turn that genuinely needs a project — a greeting, small talk, a
    # stop-contacting request or a human handoff is untouched, and the moment the caller names
    # a project (resolution.mention) this turn is no longer ambiguous and never reaches here.
    _proj_meta = (proposal.retrieval or {}).get("project") or {}
    _force_project_clarify = None
    # Set further below (registration flow) only on a turn that asks the caller for their
    # name — pins that exact question verbatim after the responder call, the same
    # deterministic-override treatment as _force_project_clarify just above, so the
    # responder's licence to rephrase (Rule 4) can never substitute a different, wrong
    # purpose ("send you the floor plan") for the real one ("book your visit").
    _force_name_ask = None
    # Set further below only while chunked phone-number capture mode (phone_capture.py) is
    # active for this call -- pins that mode's deterministic reply verbatim after the
    # responder call, same treatment as _force_name_ask, so the grounded responder never
    # rephrases a digit-by-digit read-back into something conversational-sounding but wrong.
    _force_phone_capture_reply = None
    if (_proj_meta.get("ambiguous") and _proj_meta.get("candidates")
            and _turn_needs_project(proposal, user_input)):
        _force_project_clarify = _project_clarify_question(_proj_meta["candidates"])
        # The caller may be asking WHAT the call is about, not choosing a project ("what is
        # this about?", "who is this?"). The fast classifier tends to mislabel that as a
        # product question, and without this the engine would pin the SAME project menu back at
        # them verbatim — deaf to what they actually asked, and looping every turn they don't
        # happen to name a project. On an outbound call, answer it: disclose why we called,
        # THEN steer to the still-needed choice. Guarded on `outbound`, so inbound multi-project
        # disambiguation is unchanged.
        if session.get("outbound") and outbound_flow.is_purpose_question(user_input):
            _force_project_clarify = (
                outbound_flow.purpose_disclosure(session.get("call_purpose"))
                + " " + _force_project_clarify)
        # Stand down any tool the model proposed: nothing may run for a project we have not
        # identified — not a knowledge lookup, and certainly not a booking.
        proposal.action = "clarify"
        proposal.tool = None
        proposal.params = {}
        proposal.error = ((proposal.error + " | ") if proposal.error else "") + \
            "Engine forced project disambiguation (multi-project, unresolved)."
        # We are asking which project — so NEXT turn's answer ("the second one", "DSR", "the
        # Sarjapur one", or a free-form reply) must be read as a project choice. This is the
        # signal both disambiguation tiers gate on; without it the loop the caller hit repeats.
        session["awaiting_project_choice"] = True
    elif _proj_meta.get("id") and not _proj_meta.get("ambiguous"):
        # A project is settled for this turn (named, pinned, dialled, or just resolved by one
        # of the disambiguation tiers) — the question is answered, so stop treating replies as
        # project choices.
        session.pop("awaiting_project_choice", None)

    # ---- release a scheduling preference the caller has walked away from ----
    #
    # session["wanted_day"]/["time_pref"] is the day/window the caller is CURRENTLY trying to
    # book. It was only ever cleared on a successful booking, so a day the caller floated and
    # then abandoned stayed pinned and silently filtered every later slot resolution.
    #
    # The reported failure: "I'll come today at 6pm" (nothing free) → "ok let it be" →
    # "lets book it, I mean". wanted_day was stuck on today, so the affirmation was resolved
    # against today — empty — and the caller heard "no available slots", as though the agent
    # had forgotten the tomorrow-10AM slot it had just offered. It had not forgotten the
    # offer (offered_slots still held it); it was over-remembering a dead preference.
    #
    # So when the caller ABANDONS the current attempt (decline intent, or "let it be" /
    # "never mind" / "another time") WITHOUT naming a replacement day, and nothing is parked
    # awaiting a yes, drop the transient day/window. The standing OFFER (offered_slots) and
    # the standing GOAL (wanted_tool) are left untouched — so a change of heart one turn
    # later ("actually, book it") resolves against the slot still on the table, exactly as a
    # person would. A replacement day named on THIS turn is respected: _wanted_day was
    # already re-pinned above from user_input, and we only clear when the caller named none.
    if (_abandons_current_slot(user_input, (proposal.understood or {}).get("intent"))
            and _scheduling_in_flight(session)
            and _wanted_day(user_input) is None
            and not session.get("pending_action")):
        _dropped = [k for k in ("wanted_day", "time_pref") if session.pop(k, None) is not None]
        if _dropped:
            _n = (f"Caller abandoned the current slot ({user_input!r}); released stale "
                  f"{_dropped} so a later 'book it' resolves against the standing offer.")
            proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n

    customer = None
    if conversation.customer_id:
        async with _bridge.stage("customer_load"):
            customer = (await db.execute(select(Customer).where(
                Customer.id == conversation.customer_id))).scalar_one_or_none()

    # A caller already on file for THIS conversation was identified when the call connected:
    # inbound, their number was matched to the record (caller ID); outbound, we dialled that
    # number. On a phone channel that IS the identity check, so a reschedule or cancellation
    # must not then stop and ask them to "verify" who they are — which is exactly what kept
    # happening ("I want to reschedule" → "first, verify your identity") for a customer whose
    # number we already had. verify_identity still earns its place for the OTHER case, where
    # no record is linked yet and there is a claim to check against one; there, customer is
    # None and this does not fire.
    if customer is not None and not session.get("identity_verified"):
        session["identity_verified"] = True

    outcome: ToolOutcome | None = None
    # Every tool that ran this turn, in order. A turn can now run several (see the chaining
    # loop below), and the trace has to show all of them or the audit record quietly omits
    # the lookups that led to the action.
    executed: list[ToolOutcome] = []
    noted: set[int] = set()          # outcomes already written as a 'tool' message
    confirmation_source: str | None = None
    action = proposal.action
    pending = session.get("pending_action")
    state_changed_to = None

    # ---- nothing to verify against ----
    #
    # verify_identity compares the digits given against the digits on a customer record.
    # With no record there is nothing to compare, so the tool can only ever fail — and the
    # model reads that failure as "wrong digits" and asks again. A real call spent eight
    # turns on it: four digits, a name, the digits again, a booking reference.
    #
    # Refusing the tool outright and saying why is the only useful move. The customer has
    # not done anything wrong, and no number they give will work.
    # ── registration: one place, one state ──────────────────────────────────
    #
    # This replaces five session keys — wanted_tool, awaiting_name_for,
    # awaiting_phone_for, pending_name, given_phone — that were written and read in four
    # different blocks. Every ordering between them was a chance to lose the customer's
    # actual request, and five rounds of bugs came from exactly that: she agreed to a
    # booking, gave a name, gave a number, was registered, then had to ask what happened
    # to the booking.
    #
    # One dict. What they want, what we still need, what we have. Nothing else touches it.
    reg = session.get("registration") or {}

    if customer is None:
        # 1. Something needs a customer and there isn't one.
        #
        # record_customer_fact and update_customer_profile belong here too — they carry
        # the same "customer_identified" precondition as the three below, and a model
        # discussing product interest before anyone is registered reaches for
        # record_customer_fact constantly ("1.5 BHK, budget 1 crore" is exactly a fact
        # worth noting). Left untracked, that attempt was blocked and — see the
        # customer_identified branch of the blocked-tool handling above — used to get
        # parked as a generic retry-when-it-clears action instead. It never clears by
        # retrying; only registration clears it, and the customer's actual request (to
        # visit) had no record anywhere once retrying record_customer_fact ate every
        # subsequent "yes".
        if proposal.tool in ("book_appointment", "schedule_callback", "create_ticket",
                             "record_customer_fact", "update_customer_profile"):
            reg = {"goal": proposal.tool, "params": dict(proposal.params or {}),
                   "name": reg.get("name"), "phone": reg.get("phone"),
                   "phone_partial": reg.get("phone_partial")}
            # A booking the model reached for with NO sign the customer asked for one — they
            # only gave their name and number after the greeting; nothing was searched, no
            # day was named, no booking phrase was said — is a phantom. Register the caller,
            # but do not carry a "book" goal: otherwise the identity capture becomes a visit
            # the customer never requested (BOOKING IN PROGRESS lights up and the booking
            # machine drives toward a visit nobody asked for). The goal becomes
            # create_customer — a real, non-booking goal, so the name/phone capture
            # below still runs (it is gated on a goal being present) — and the tool is
            # rewritten to create_customer so a book_appointment never fires against an empty
            # offered set. Nothing downstream treats create_customer as a visit.
            if (proposal.tool == "book_appointment"
                    and not _booking_signalled(session, user_input,
                                               (proposal.understood or {}).get("intent"),
                                               proposal)):
                reg["goal"] = "create_customer"
                proposal.tool = "create_customer"
                _n = ("phantom book_appointment with no booking signal (no search/slots/day/"
                      "ask) → registration only, not treated as a visit request.")
                proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n
        # Rule 2 — registration is a child task of the booking, so it must engage whenever a
        # booking is IN FLIGHT, not only on the turn a booking tool is proposed. A visit
        # search has run (offered_slots present) or a booking is the standing goal ⇒ the
        # customer is mid-booking; details they volunteer now ("I'm Rahul, my number is …")
        # must be captured and registered, in whatever order they arrive. Without this a
        # caller who gave their name before picking a time was met with "how can I help?" —
        # the booking silently forgotten.
        elif not reg.get("goal") and (
                bool(session.get("offered_slots"))
                or session.get("wanted_tool") in ("book_appointment", "schedule_callback",
                                                  "create_ticket")):
            reg = {"goal": session.get("wanted_tool") or "book_appointment",
                   "params": dict(reg.get("params") or {}),
                   "name": reg.get("name"), "phone": reg.get("phone"),
                   "phone_partial": reg.get("phone_partial")}
        # A visit is SIGNALLED this turn — the planner read it as a booking, or a search/day/
        # booking phrase is present — but the model proposed a non-booking tool (create_customer
        # to register first, or locate_centre). Carry book_appointment as the goal so
        # registration resumes into the visit once the customer is on file. Without this the
        # caller says "I want to see the flat", gives their name, and hears "you're all set —
        # how can I help you today?" with the visit they asked for forgotten. Guarded by
        # _booking_signalled (not a raw intent read), so a bare name/number after a greeting is
        # still registration-only, never a phantom booking.
        elif (not reg.get("goal")
              and _booking_signalled(session, user_input,
                                     (proposal.understood or {}).get("intent"), proposal)):
            reg = {"goal": "book_appointment",
                   "params": dict(reg.get("params") or {}),
                   "name": reg.get("name"), "phone": reg.get("phone"),
                   "phone_partial": reg.get("phone_partial")}

        # 2. Anything the caller just said that fills a gap.
        if not reg.get("phone") and not _phone_capture_turn_active:
            # Skipped entirely while chunked phone-capture mode owned this turn (hoisted,
            # before plan() was even called) -- exactly one of the two mechanisms owns a
            # given turn's digits.
            #
            # The whole utterance is a number ("9876543210"), OR a number sits inside a
            # sentence ("I'm Rahul and my number is 9876543210"). The first form was the
            # only one caught before, so a name+number given together registered neither.
            # A mid-sentence match requires 10+ digits so a stray "18" or "5 pm" is never
            # mistaken for a phone; a standalone match keeps the looser 8+ rule.
            #
            # THE BUG THIS FIXES: a 10-digit number spoken one digit at a time with single
            # spaces ("8 1 4 1 8 1 4 1 2 8") is 19 characters long, but these windows used to
            # cap at 17/18 -- long enough for a number read in pairs/triples ("81 41 81 41
            # 28") but one character short of a fully digit-by-digit reading, which silently
            # truncated the LAST digit off every real match. The caller then had to repeat
            # the whole number, which the model (with no deterministic capture to fall back
            # on) described as needing "the last four digits" -- language that belongs to
            # the separate identity-verification flow, not a fresh registration. Widened
            # generously so any realistic spacing/punctuation is covered; the digit-count
            # check below still rejects anything that isn't actually phone-shaped.
            m = re.fullmatch(r"\s*(\+?[\d][\d\s\-()]{6,30})\s*", user_input or "")
            _min = 8
            if not m:
                m = re.search(r"(\+?\d[\d\s\-()]{8,30}\d)", user_input or "")
                _min = 10
            if (m and len(re.sub(r"\D", "", m.group(1))) >= _min
                    and "/" not in user_input and not re.search(r"\d-\d{2}-\d", user_input)):
                reg["phone"] = m.group(1).strip()
            elif session.get("last_registration_prompt_field") == "phone":
                # A caller reading a 10-digit number rarely says it in one breath — "8 1 4
                # 1", pause, "2 8 0", pause, "6 double 1" is completely normal phone
                # etiquette, and each fragment on its own is too short to pass the regex
                # above (which only recognises a SINGLE utterance that already looks like
                # a whole number). Every fragment used to be silently discarded, so the
                # caller had to repeat the entire number again to get anywhere. Accumulate
                # them instead, but only while we are the ones who just asked for the
                # number — never mid-sentence elsewhere, where a stray "6" is not a phone
                # digit at all.
                #
                # Keyed on the SPECIFIC field asked (not the generic "we asked something"
                # flag) — a free-form clarify about an unrelated thing (e.g. "could you
                # share your mobile number so we can email the floor plan") used to set the
                # generic flag too, so digits spoken for an entirely different purpose got
                # folded into the registration phone.
                fresh = extract_spoken_digits(user_input)
                if fresh:
                    combined = (reg.get("phone_partial") or "") + fresh
                    reg["phone_partial"] = combined
                    if len(combined) >= 10:
                        reg["phone"] = combined
                        reg["phone_partial"] = None
        # The number they are calling from, when they have not just said one.
        #
        # THE BUG THIS FIXES: this was `reg.setdefault("phone", session.get("caller_phone"))`
        # — and `setdefault` fills a MISSING key, never a key that is present and None. All
        # three branches above rebuild reg as `{..., "phone": reg.get("phone")}`, which writes
        # an explicit `"phone": None` whenever no number is known yet. So the key was always
        # present, the setdefault was always a no-op, and the caller's number could never
        # reach registration however reliably the session held it.
        #
        # That is why this "stopped working" with no error and nothing in the logs: the
        # caller-ID wiring was correct and stayed correct; a later refactor added those reg
        # rebuilds, and an explicit None silently disabled a feature that reads as present in
        # the code. The prompt at the ask site even still says "I have the number you're
        # calling from" — it just could never be reached.
        #
        # A truthiness check has no such failure mode, and the resolver behind it now also
        # falls back to the carrier's caller ID.
        #
        # Skipped while chunked phone-capture mode is still actively mid-collection: filling
        # reg["phone"] here silently would let the create_customer branch below fire before
        # phone_capture's own explicit grouped-confirm ever happened.
        if not reg.get("phone") and not session.get("phone_capture_active"):
            reg["phone"] = await _caller_phone(db, session, conversation)
        # Only read a name when one was actually asked for on the previous turn.
        #
        # Otherwise any short reply becomes a name: "best nuumber menas" was registered as
        # "Best Nuumber Menas". A confused question is not an answer, and a record created
        # from one is permanent.
        # Read a name when we just asked for one, OR when the customer explicitly
        # introduces themselves ("I'm Rahul", "my name is Rahul", "this is Rahul") — an
        # explicit marker is unambiguous consent to record it, so it need not have been
        # prompted. A bare short reply still requires that we asked, so a confused
        # "best number means?" is never registered as a name.
        #
        # THE BUG THIS FIXES: this used to key on the generic "we asked SOMETHING in the
        # registration flow" flag, set after ANY clarify while unregistered — including one
        # that asked for the mobile number for an unrelated reason ("could you share your
        # mobile number so we can email the floor plan?"). The caller's next utterance, a
        # bare "yeah", was then read as the answer to a name prompt that was never actually
        # asked, and a real call registered a customer named "Yeah". Keyed on the SPECIFIC
        # field we last prompted for, so only an actual name-ask opens this door.
        _introduces = re.search(r"\b(?:i'?m|i am|my name'?s?|my name is|this is|name is)\b",
                                user_input or "", re.I)
        if (not reg.get("name") and reg.get("goal")
                and (session.get("last_registration_prompt_field") == "name" or _introduces)):
            given = _looks_like_a_name(user_input)
            if given:
                reg["name"] = given
        # The model may have supplied either while proposing the tool itself — but only
        # a REAL value counts.
        #
        # It proposed create_customer with phone="<customer's phone number>" — the
        # placeholder from the field description, passed through as though it were data.
        # A falsy check does not catch that, so the placeholder overwrote a perfectly good
        # number the session already had, the tool failed on an unusable value, and the
        # agent apologised for a system error three turns running. The session knew the
        # number the whole time.
        if proposal.tool == "create_customer":
            supplied_phone = _real_value((proposal.params or {}).get("phone"))
            # A model-supplied "phone" must actually look like one. With no caller-ID to hand
            # and asked to "confirm the number", the model grabs a recent utterance instead —
            # a project name ("DSR Highland"), the caller's name — and passes it as the phone.
            # _real_value clears obvious placeholders but not a plausible string, so require
            # enough digits here: a value that is not phone-shaped is treated as "no number
            # given", and the agent asks for one properly rather than registering garbage or
            # failing create_customer in front of the caller. A real caller-ID number (from
            # _caller_phone above, already in reg) is unaffected.
            if supplied_phone and sum(c.isdigit() for c in supplied_phone) < 7:
                supplied_phone = None
            # Trust a model-supplied NAME only when the customer actually gave one this call —
            # a name prompt was shown on an earlier turn, or they introduced themselves in
            # this one. Instructed to register before booking, the model otherwise invents a
            # name on the opening "I want to see the flat" turn (which contains no name at
            # all); _real_value lets any non-placeholder string through, create_customer then
            # fires against that invented name, and the caller is never asked who they are.
            # A phone from caller-ID is fine to trust — that is genuinely on file — so only
            # the name carries this guard. Same field-specific key as the capture above: a
            # prompt that asked for the PHONE (for any reason) must not count as license to
            # trust a model-invented name.
            if session.get("last_registration_prompt_field") == "name" or _introduces:
                supplied_name = _real_value((proposal.params or {}).get("full_name"))
                reg["name"] = supplied_name or reg.get("name")
            reg["phone"] = supplied_phone or reg.get("phone")
            # Truthiness, not setdefault — same trap as "phone" above: the reg rebuilds can
            # leave an explicit None here, and setdefault would decline to fill it.
            if not reg.get("goal"):
                reg["goal"] = (session.get("registration") or {}).get("goal")

        # 3. Act on what is still missing — unless they asked something else first.
        #
        # A question deserves an answer before a form gets refilled. She asked "does my
        # slot get booked?" and was handed the number question again, verbatim, as though
        # she had said nothing. Repeating a prompt over someone's question is how a caller
        # decides the agent is not listening.
        _asked_something = bool(re.search(r"\?|^\s*(?:what|when|where|why|how|who|is|are|"
                                          r"do|does|did|can|could|will|would|has|have)\b",
                                          (user_input or "").strip(), re.I))
        _already_asked = session.get("last_registration_prompt")
        # Which field the clarify below asks for, if any — written to
        # `last_registration_prompt_field` so NEXT turn's name/phone capture can key on the
        # SPECIFIC question just asked, rather than "we asked something, anything" (see the
        # bug note at the name-capture site above).
        _prompted_field = None

        # Skipped entirely while phone-capture mode is still mid-collection this turn (not
        # done, not handed off): otherwise this block re-runs right after the hoisted
        # fast-path above, sees reg["phone"] still empty, and re-drafts the SAME initial
        # "could I have your mobile number" ask -- which would reset
        # session["phone_capture_state"] back to {}, silently discarding the chunks just
        # captured even though _fast_path_text still makes the caller hear the correct reply.
        if (reg.get("goal") or proposal.tool == "create_customer") and not _phone_capture_pending and not (
                _asked_something and _already_asked and not reg.get("phone")):
            if reg.get("name") and reg.get("phone"):
                proposal.action = "invoke_tool"
                proposal.tool = "create_customer"
                proposal.params = {"full_name": reg["name"], "phone": reg["phone"]}
            elif not reg.get("name"):
                proposal.action, proposal.tool = "clarify", None
                # Named the actual goal ("book your visit"), not "set that up" — a vague
                # "that" left the responder free to supply its OWN guess at the purpose from
                # whatever topic was more recently salient in the conversation. Grounding the
                # draft in `reg["goal"]` (deterministic session state, not a re-guess) is step
                # one; it is not sufficient on its own — see _force_name_ask below.
                _purpose = _STAGED_VERB.get(reg.get("goal"), "get you set up")
                proposal.response_draft = (
                    f"Before I can {_purpose}, can I take your name? I have the number "
                    f"you're calling from." if reg.get("phone")
                    else f"Before I can {_purpose}, can I take your name?")
                _prompted_field = "name"
                # THE BUG THIS FIXES: Rule 4 lets the responder keep the draft's TONE but
                # rephrase it — which is exactly what it did, discarding "book your visit" and
                # substituting "send you the 3 BHK floor plan" because that topic was more
                # recently salient in the conversation it was reading. A live retest of this
                # exact scenario reproduced it even with the goal-grounded draft above: "Sure,
                # could I have your name to send you the 3 BHK floor plan?" — on the turn meant
                # to register the caller for a VISIT. Getting the caller's name for the right
                # reason is the single highest-stakes line in this flow (a wrong one here is
                # how a real call registered nobody and the visit silently vanished), so it is
                # PINNED verbatim after the responder call below (`_force_name_ask`), the same
                # deterministic-override treatment already given to the project-disambiguation
                # question — not left to the model's judgement a second time.
                _force_name_ask = proposal.response_draft
            else:
                proposal.action, proposal.tool = "clarify", None
                # Plain, and says why.
                #
                # "What's the best number for you?" was read as a question ABOUT numbers —
                # "best number means?" — and the call went sideways for four turns. Asking
                # for a mobile number and saying what it is for leaves nothing to
                # interpret.
                proposal.response_draft = (
                    f"Thanks {reg['name']}. Could I have your mobile number, so we can send "
                    f"you the visit confirmation?")
                _prompted_field = "phone"
                # From the NEXT turn on, chunked phone-capture mode (see the top of this
                # `if customer is None:` block) owns every turn until a confirmed number or a
                # hand-off -- pin this first ask verbatim too, same as _force_name_ask, so the
                # responder's licence to rephrase never turns it into something that invites
                # a free-form answer instead of digits.
                session["phone_capture_active"] = True
                session["phone_capture_state"] = {}
                _force_phone_capture_reply = proposal.response_draft
        # Saved on EVERY turn while unregistered, not only when we act.
        #
        # It used to be written inside the branch above, so a turn that went off-topic
        # captured her phone number into a dict that was then discarded. She answered the
        # question, the answer was thrown away, and the next turn asked again. The state
        # has to outlive the digression — that is the entire point of having it.
        if reg:
            session["registration"] = reg
        # Remember that we have already asked, so a question in reply gets answered rather
        # than met with the same prompt a second time.
        session["last_registration_prompt"] = (
            proposal.response_draft if proposal.action == "clarify" and not proposal.tool
            else None)
        # The specific field that prompt asked for — None for every clarify that is NOT one
        # of the two deterministic templates above (including a free-form clarify about
        # something else entirely, e.g. offering to email a floor plan). Only this flag,
        # never the generic one above, may license reading the caller's next utterance as a
        # name or as registration phone digits.
        session["last_registration_prompt_field"] = _prompted_field

    else:
        # Registered. Resume what they actually rang about, once.
        if reg.get("goal") and reg["goal"] not in {o.tool_name for o in executed
                                                   if o.status in TOOL_SUCCEEDED}:
            session["wanted_tool"] = reg["goal"]
        session.pop("registration", None)

    # Resume on the SAME turn create_customer succeeds, too.
    #
    # `customer` is read at the top of the turn, so on the turn registration actually runs
    # it is still None and the branch above never executes — the goal survives in the
    # session but nothing picks it up until the customer says something that happens to
    # trigger it. A caller gave her name and number, was registered, and was then asked
    # "would you like any information?" — she had asked to book a visit two turns earlier.
    #
    # create_customer is plausibly idempotency-keyed on phone number — a customer who
    # gives their name and number twice (a misheard confirmation, say) would get
    # "skipped_duplicate" the second time, with the same real customer record in the
    # output. Checking only "success" here meant that turn silently failed to resume
    # anything, leaving her registered but with no memory of what she had rung about.
    if reg.get("goal") and any(o.tool_name == "create_customer" and o.status in TOOL_SUCCEEDED
                               for o in executed):
        session["wanted_tool"] = reg["goal"]
        session.pop("registration", None)
        # Just registered means just identified.
        #
        # She gave her name in answer to our question, we wrote it to a new record — and
        # the next reply asked her to confirm she was the person she had said she was, a
        # sentence earlier. There is nothing to verify against: the record IS what she just
        # told us. Marking identity settled here also stops the reschedule rule demanding
        # digits from someone whose number we took thirty seconds ago.
        session["identity_verified"] = True
        # THE BUG THIS FIXES: visit_identity's own gate (the "always-ask" name check before
        # answering a status/reschedule question about an existing visit) is a SEPARATE flag
        # from identity_verified above and was not covered by the same reasoning -- a caller
        # who registered THIS call, then asked to reschedule and named an unavailable time,
        # got asked "may I have the name the visit is booked under?" a second time, for a
        # name she had given thirty seconds earlier. _looks_like_a_name() then misread her
        # actual reply ("10th october at 4 pm" -- a time, not a name) and extracted "th" from
        # "10th" as a plausible name token, producing "I'm not finding a visit under Th on
        # this number." Giving your name and phone to register IS the identity check; nothing
        # about it is weaker than the outbound-call case just below that already skips this
        # same gate for exactly this reason.
        session["visit_name_confirmed"] = True

    if customer is None and proposal.tool == "verify_identity":
        note = "verify_identity proposed with no customer record; nothing to verify against."
        proposal.error = f"{proposal.error} | {note}" if proposal.error else note
        proposal.tool = None
        proposal.action = "respond"
        proposal.response_draft = (
            "I can't look that up — this number isn't in our records, so there's nothing "
            "for me to check it against. If you have booked with us before, let me put you "
            "through to a colleague who can search properly. Otherwise, tell me what you "
            "need and I'll help from here.")

    # An identity request must ASK FOR THE DIGITS, in that reply.
    #
    # The rule now says so in its reason text, and the model still opened with "I need to
    # verify your identity — could you confirm your details?" The customer says "yeah
    # sure", and only then does it name the question. Two turns, every reschedule, for a
    # question that has exactly one form.
    #
    # Rule text is guidance; this is the check on it. If the tool was blocked for identity
    # and the reply does not ask for the digits, the reply is replaced with the one
    # sentence that gets an answer.
    if (outcome is not None and outcome.status == "blocked"
            and "identity" in str(outcome.failed_check or "").lower()
            and not session.get("identity_verified")):
        _asks_digits = re.search(r"last four|four digits|4 digits",
                                 (proposal.response_draft or "").lower())
        if not _asks_digits:
            proposal.response_draft = (
                "Before I change that — could you give me the last four digits of the "
                "mobile number registered with us?")
            _note = "identity request did not ask for the digits; replaced with the question."
            proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note

    # Formal identity language — catch it even when no tool ran (outcome is None).
    #
    # The blocked-tool path above only fires when outcome carries the check failure. When Rule
    # 4 held the booking without executing (action="confirm", outcome=None), or the tool was
    # parked before running, the model's own draft can contain "I'll need to verify your
    # identity first" or "please provide the information required for verification" — which
    # goes out unchecked. A reschedule request should never sound like a compliance procedure.
    if (not session.get("identity_verified")
            and outcome is None
            and re.search(r"verify\s+(your\s+)?identity|identity\s+verif|"
                          r"for\s+security\s+purposes|security\s+(check|verification)|"
                          r"provide\s+the\s+information\s+required",
                          (proposal.response_draft or "").lower())):
        proposal.response_draft = (
            "Before I make that change — could you give me the last four digits of the "
            "mobile number registered with us?")
        _note = "formal identity language in draft (no tool ran); replaced with natural ask."
        proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note

    # ---- they just gave the digits ----
    #
    # Four digits on their own, from someone not yet verified, mean one thing. Run the
    # check, whatever the model proposed instead.
    #
    # THE LOOP THIS BREAKS: a cancellation is staged, the customer says yes, the rule gate
    # blocks it pending identity, and the agent asks for the last four digits. She gives
    # them — and the model proposes the cancellation again, because that is still what the
    # call is about. Blocked again. Asked again. The digits hint could not help: it only
    # fires when the model names NO tool, and here it names one every time. So she recites
    # the same four numbers until she gives up.
    #
    # verify_identity is chainable, so once it succeeds the chain picks the parked
    # cancellation straight back up and finishes the job in the same turn.
    if (not session.get("identity_verified") and proposal.tool != "verify_identity"
            and re.fullmatch(r"\s*\d{4}\s*", user_input or "")):
        proposal.action = "invoke_tool"
        proposal.tool = "verify_identity"
        proposal.params = {"last_four": user_input.strip()}
        # Four digits are an ANSWER, never a question.
        #
        # The model kept its own draft and replied "I don't have any information about
        # '9870' — could you clarify what you mean?" She had just been asked for those
        # exact digits. Whatever the model wrote for this turn was written about something
        # else; clearing it lets the responder speak from the verification result.
        proposal.response_draft = ""
        note = "Customer supplied identity digits; ran verify_identity ahead of the plan."
        proposal.error = f"{proposal.error} | {note}" if proposal.error else note

    # Already handed over — stop offering to hand over.
    #
    # The call was transferred, and the next three turns each answered "ok" with "I'll
    # connect you with someone who can assist further." A customer who has been told
    # somebody is coming should hear that they are coming, not the same sentence on a
    # loop. Once the handover is recorded, further human requests get an acknowledgement
    # instead of a repeat.
    if session.get("escalated") and _asks_for_human(user_input):
        proposal.action = "respond"
        proposal.tool = None
        proposal.response_draft = (
            "You're already in the queue for a colleague — they'll pick this up shortly. "
            "Is there anything I can help with while you wait?")
        _n = "already escalated; acknowledged instead of repeating the handover line."
        proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n

    # A prerequisite just cleared — resume what was waiting on it, now.
    #
    # The chain loop already re-plans after verify_identity, but the resume depends on the
    # model proposing the parked tool again, and it often proposes something else. The
    # customer verified, and the thing she verified FOR should simply happen: her consent
    # and the details are both still on the parked action.
    #
    # This is the "pop the child, resume the parent" step. The stack version of it would be
    # cleaner; this does the same job for the two prerequisites that actually interrupt.
    _pending = session.get("pending_action") or {}
    _prereq_cleared = any(
        o.tool_name in ("verify_identity", "create_customer") and o.status in TOOL_SUCCEEDED
        for o in executed)
    if (_prereq_cleared and _pending.get("tool") and not _pending.get("needs")
            and _pending["tool"] not in {o.tool_name for o in executed}):
        proposal.action = "invoke_tool"
        proposal.tool = _pending["tool"]
        proposal.params = dict(_pending.get("params") or {})
        _n = (f"prerequisite cleared; resumed {_pending['tool']} with the details the "
              f"customer already agreed to.")
        proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n

    # `action` was captured from proposal.action once, before any of the redirects above
    # ran — the registration flow (ask for name/number before booking), the four-digit
    # identity shortcut, the already-escalated acknowledgement, and the prerequisite
    # resume just above all rewrite proposal.action directly and never touch this local
    # variable, which is what every check from here on actually branches on.
    #
    # THE BUG THIS FIXES: a customer with no record yet said "shall I book it?", got
    # redirected to "can I take your name?" (proposal.action -> "clarify", proposal.tool
    # -> None — both correct), gave their name, and registered. But `action` was still
    # "confirm" from before the redirect, so the confirmation-gate logic below kept
    # treating the turn as an unresolved confirm with no tool — logged a misleading
    # "planner named no tool" fault, and left whatever was parked in pending_action from
    # BEFORE the redirect (a stale slot the customer never actually confirmed) to be
    # silently resumed the moment registration cleared, instead of being re-asked.
    action = proposal.action

    # ---- BookingTask state machine: deterministic sequencing (Rules 1, 3, 5) ----
    #
    # docs/booking-task-architecture.md. These force the MISSING step so a booking no longer
    # depends on the order the turns happened to arrive in — the emergent-state root cause
    # behind the whole family of booking bugs.
    #
    # Never over a handover or a goodbye: a booking can still be "in flight" on the turn the
    # customer asks for a person, and forcing a slot search then would talk over the
    # transfer. Rule-forced escalations are likewise left alone.
    # Skipped on a consent hold turn: forcing the booking's next step (a slot search, a centre
    # lookup) over "what time?" would answer a question the caller did not ask.
    if (action not in ("escalate", "end_call") and not proposal.forced_escalation
            and _consent_hold is None):
        _task = bt.project(session, customer_exists=customer is not None,
                           executed=executed, turn_index=int(session.get("turn_index", 0)),
                           active_entities=session.get("active_entities") or {},
                           current_state=customer.current_state if customer else None)

        # Rule 3 — book vs reschedule BY FACT, never by the model's intent label. A model
        # that relabels "change the time" as a reschedule when NOTHING is booked deadlocks
        # on a required slot_start (Tab 1). No active appointment ⇒ the tool is
        # book_appointment, whatever the label claimed.
        if proposal.tool == "reschedule_appointment" and _task.kind == "book":
            proposal.tool = "book_appointment"
            proposal.action = "invoke_tool"
            proposal.params = {k: v for k, v in (proposal.params or {}).items()
                               if k != "appointment_id"}
            _n = ("Rule 3: no active appointment exists, so this is a booking not a "
                  "reschedule; forced book_appointment.")
            proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n

        # Rule 1 & 5 — a named day must be SEARCHED before anything is offered or booked
        # (Rule 1), and a chosen time that was never offered is re-searched rather than
        # dropped into a required-field schema error (Rule 5). Either way the fix is the
        # same: run a day-scoped search_slots first. It is chainable, so the chain re-plans
        # with real slots in the same turn.
        _req_day = _wanted_day(user_input)
        if _req_day is None and session.get("wanted_day"):
            try:
                _req_day = date.fromisoformat(session["wanted_day"])
            except ValueError:
                _req_day = None
        _wants_write = proposal.tool in ("book_appointment", "reschedule_appointment")
        _offered_starts = bt.offered_starts(session)
        _proposed_start = (proposal.params or {}).get("slot_start")
        # Trust the model's resolved slot_start for WHICH DAY to search when our own regex
        # could not read one from the raw words.
        #
        # THE BUG THIS FIXES: a caller mid-reschedule answered "19th 2pm" — a natural reply
        # to "what time would you like?". The model resolved it correctly to 2026-08-19T14:00,
        # but _wanted_day requires the literal "on" before a bare ordinal ("on the 19th") and
        # _wanted_part_of_day only reads the WORDS morning/afternoon/evening, never "2pm" — so
        # both returned None. With no requested day the engine never searched the 19th; the
        # reschedule ran against an unverified slot, the slot was dropped (never offered), the
        # task fell back to the CURRENT appointment time, Rule 4 fired "chosen time equals the
        # current one", and the call looped "what day and time would you like?" forever.
        #
        # The model already did the language work — its slot_start IS the customer's day, far
        # more reliably than any regex over how they happened to phrase it. Using it only to
        # pick which day to SEARCH is safe: nothing is booked off it, the day-scoped search
        # offers real slots, and the pick is still verified against those before any write.
        # Guarded to a proposed time not already offered, so an ordinary booking where the
        # customer accepts an offered slot does not trigger a redundant re-search.
        if (_req_day is None and _wants_write and _proposed_start
                and _proposed_start not in _offered_starts):
            _req_day = _slot_date(_proposed_start)
        # The time-of-day window they asked for — this turn's words first, then the one
        # persisted from an earlier turn (so a window named before a registration detour is
        # not lost). A bare "yes"/"ok" carries no NEW window; the persisted one still holds,
        # which is correct — they never withdrew "evening".
        _part = _wanted_part_of_day(user_input) or session.get("time_pref")

        def _start_in_part(iso: str | None) -> bool:
            if not _part:
                return True
            if not iso:
                return False
            try:
                hour = datetime.fromisoformat(iso).hour
            except (ValueError, TypeError):
                return False
            lo, hi = _PART_WINDOWS[_part]
            return lo <= hour < hi

        # A chosen time is "verified" only if it was actually offered AND it sits in the
        # window the customer named — otherwise a stray 5 PM spread slot counts as a valid
        # pick for an "afternoon" ask, and they get booked into the evening.
        _chosen = (_proposed_start if _proposed_start in _offered_starts
                   else _task.slots.get("verified_slot_start"))
        _slot_verified = bool(_chosen) and _start_in_part(_chosen)
        # "Covered" respects the same window: a day present only as a week-wide spread
        # sample is not covered for a specific afternoon/evening ask.
        _covered = _day_covered_in_window(session, _req_day, _part)
        # An explicitly named clock hour ("today at 11 am") is a specific request: the day
        # counts as covered only if THAT hour was actually offered. Without this a caller who
        # asked for 11 am when only a 4 PM slot had been offered was told "let me check if
        # that's available" and the turn stalled — the day looked covered because some slot
        # existed on it, so no fresh search fired. Only applied when a day is in play, so a
        # bare "4 pm" accepting an offered slot is unaffected.
        _wanted_h = _wanted_hour(user_input)
        if _req_day is not None and _wanted_h is not None:
            _covered = _covered and _day_covered_at_hour(session, _req_day, _wanted_h)
        _searched = "search_slots" in {o.tool_name for o in executed}
        _booking_flow = (_scheduling_in_flight(session) or _wants_write
                         or _req_day is not None)
        # Never override the registration question — asking for the name is the right
        # in-flight step, and the search does not need a customer, so it can run next turn.
        #
        # `customer is None` is part of the test, not decoration: last_registration_prompt is
        # only ever CLEARED inside the unregistered branch (it is written on every turn while
        # customer is None and never touched afterwards), so a value left over from the
        # name/number exchange can outlive the registration itself. Without this, the first
        # clarify-shaped turn after a caller is on file still looked like "we are asking them
        # to register" and suppressed the forced lookups below — on a caller there is nothing
        # left to ask. You cannot be asking someone to register who is already registered.
        _asking_to_register = (customer is None and proposal.action == "clarify"
                               and session.get("last_registration_prompt"))
        # THE BUG THIS EXCLUDES: _req_day only detects "the utterance names a day", not "the
        # day is about a NEW VISIT SLOT" — "call me back tomorrow at 6am" names a day too, and
        # without this guard _booking_flow/_needs_search fired on it exactly as it would for
        # "I want to visit tomorrow", silently overriding an already-correct schedule_callback
        # proposal with a forced locate_centre/search_slots the model never asked for and a
        # callback (no centre, no slot) never needed. Four different prompt-wording attempts at
        # this exact failure (eval scenario "a callback is clamped into calling hours") all
        # failed identically because the model's own correct choice was never the problem —
        # this deterministic force ran AFTER it and discarded it regardless of prompt wording.
        _needs_search = (
            proposal.tool != "schedule_callback"
            and _booking_flow and not _searched and not _asking_to_register and (
                (_req_day is not None and not _covered)      # Rule 1: day never searched
                or (_wants_write and not _slot_verified)))   # Rule 5: pick never offered

        # Reschedule prerequisite (the reschedule analog of registration) — a visit cannot
        # be moved until we know WHICH visit. A lead already in site_visit_scheduled has one,
        # but the session may not hold its id yet; my_appointments (read-only, chainable)
        # loads the id and the centre before anything is searched or moved. Forced once:
        # _remember_ids writes appointment_id on success, so it does not re-fire. Takes
        # precedence over the search force, hence the elif below.
        _needs_appt = (_task.kind == "reschedule"
                       and not session.get("appointment_id")
                       # Fired at most once per call: if my_appointments finds nothing
                       # (a lead whose state says scheduled but whose visit was cancelled),
                       # appointment_id stays unset — without this flag the force would
                       # re-fire every turn and loop. One lookup is enough to know.
                       and not session.get("appt_lookup_forced")
                       and "my_appointments" not in {o.tool_name for o in executed}
                       and (_wants_write or _needs_search)
                       and not _asking_to_register)
        if _needs_appt:
            proposal.action = "invoke_tool"
            proposal.tool = "my_appointments"
            proposal.params = {}
            session["appt_lookup_forced"] = True
            _n = ("Rule 2/3: this is a reschedule but the appointment id is not known yet; "
                  "forced my_appointments to load it before searching or moving the visit.")
            proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n
        elif _needs_search and session.get("centre_id"):
            # A centre is established — force the windowed day search directly.
            _day = _req_day or _slot_date(_proposed_start)
            proposal.action = "invoke_tool"
            proposal.tool = "search_slots"
            proposal.params = {"centre_id": session["centre_id"]}
            if _day is not None:
                proposal.params["date_from"] = _day.isoformat()
                proposal.params["date_to"] = _day.isoformat()
            # Carry the named window through, so the search returns afternoon slots for an
            # afternoon ask rather than the whole day for the model to mis-pick from.
            if _part:
                proposal.params["part_of_day"] = _part
            else:
                # No part-of-day word, but the caller may have named an EXACT time ("2 PM").
                # The whole-day search returns only a spread SAMPLE, which routinely omits
                # that exact slot — so the pick could never be verified and the reschedule
                # looped on the current time. Target the exact time instead: a free slot is
                # then offered and verifies; an unavailable one falls through search_slots'
                # own "requested window is full — here is what's free" path, so the agent
                # offers the day's real alternatives rather than a time it can never book.
                _exact = _slot_time(_proposed_start) if (
                    _wants_write and _proposed_start
                    and _proposed_start not in _offered_starts) else None
                # The caller may have named the hour in plain words ("today at 11 am") without
                # the model resolving it into a slot_start — target that hour directly so the
                # search looks up what they actually asked for, not the whole day.
                if not _exact and _wanted_h is not None:
                    _exact = f"{_wanted_h:02d}:00"
                if _exact:
                    proposal.params["time_from"] = _exact
                    proposal.params["time_to"] = _exact
            _n = ("Rule 1/5: the customer's day/time-of-day was not yet searched, or the "
                  "chosen time was never offered; forced a windowed search_slots before "
                  "offering or booking.")
            proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n
        elif (_needs_search and not session.get("centre_id")
              and not session.get("centre_lookup_forced")
              and "locate_centre" not in {o.tool_name for o in executed}):
            # A search needs a centre, and with more than one active centre it cannot be
            # auto-filled — forcing a centre-less search_slots just fails schema validation
            # ("'centre_id' is a required property"). Establish a centre first: locate_centre
            # is chainable, so the chain re-plans and searches against a real centre in the
            # same turn (a single-centre tenant resolves automatically; a multi-centre one
            # lets the model pick from what locate_centre returns). Forced at most once per
            # call so it cannot loop when the customer keeps naming a day but no centre.
            proposal.action = "invoke_tool"
            proposal.tool = "locate_centre"
            proposal.params = {}
            session["centre_lookup_forced"] = True
            _n = ("Rule 1: a slot search needs a centre and none is established yet; forced "
                  "locate_centre before searching.")
            proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n
        elif (proposal.tool in ("search_slots", "book_appointment")
              and not session.get("centre_id")
              and not _looks_like_uuid((proposal.params or {}).get("centre_id"))
              and not session.get("centre_lookup_forced")
              and "locate_centre" not in {o.tool_name for o in executed}):
            # Rule 1b — the model PROPOSED the search/booking itself, with no real centre.
            #
            # Rule 1 above only fires when the CALLER named a day or the model proposed a
            # WRITE. A bare "ok, book me in" turn where the model jumps straight to
            # search_slots — carrying no centre, or a name/placeholder ("null") where a
            # centre_id belongs — has neither, so it slipped past every force and ran the
            # search against a centre that does not resolve. On an OUTBOUND call, where no
            # locate_centre has run yet, that is guaranteed: the tool was refused ("No active
            # centre matches 'null'") and the agent asked the caller for a centre id.
            #
            # A centre is "real" only if one is already established on the call OR the model
            # supplied an actual UUID; a name or a "null" is neither. When it is not, do
            # exactly what Rule 1 does — force locate_centre. It is chainable, so the chain
            # re-plans and searches against the resolved centre (or the primary, for a fresh
            # booking across several centres) in the SAME turn. Forced at most once per call
            # via centre_lookup_forced, so a caller who keeps saying "ok" cannot loop it.
            proposal.action = "invoke_tool"
            proposal.tool = "locate_centre"
            proposal.params = {}
            session["centre_lookup_forced"] = True
            _n = ("Rule 1b: the model proposed a slot search/booking with no resolvable "
                  "centre; forced locate_centre before searching.")
            proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n

        # ---- Rule 1c: a scheduling turn the planner ANSWERED WITH A PROMISE ----
        #
        # THE BUG THIS FIXES. An agent on a phone call cannot say a second thing without the
        # caller speaking first, so "let me check the available appointment times for you"
        # with no tool behind it does not defer the work by a moment — it costs the caller a
        # whole utterance, and they hear a progress report where they asked a question:
        #
        #     Caller: what are the available options
        #     Agent:  Could you specify what options you're interested in?
        #     Caller: appointment time
        #     Agent:  Let me check the available appointment times for DSR Highland Greenz
        #             or DSR Techno Cube in Bengaluru for you.      <- nothing ran. Turn over.
        #
        # ROOT CAUSE. `_next_scheduling_lookup` — the deterministic "what is the next step of
        # this booking" driver — already handles exactly this, but it was reachable ONLY from
        # inside the chain loop, and the chain loop runs only after some tool has already
        # executed this turn. A turn where the planner narrates instead of naming ANY tool
        # never entered the loop, so nothing advanced the flow and the driver never ran.
        #
        # Rules 1/5 above could not cover it either: both require either a day the caller
        # named (`_req_day`) or a write the model proposed (`_wants_write`), and "what times
        # are available" has neither. That is the whole gap — a caller asking the most
        # ordinary scheduling question in the language hit the one path with no forcing on it.
        #
        # So the driver is hoisted here, to the one place that sees a turn BEFORE any tool
        # runs. Same function, same session-state logic, now reachable from both call sites —
        # one definition of "the next step", instead of one definition and one hole.
        #
        # Read-only lookups ONLY (`_next_scheduling_lookup` never returns a write), so this
        # commits the caller to nothing: locate_centre and search_slots change no record, and
        # the booking they lead to still goes through the confirmation gate and the caller's
        # yes. Gated on the planner naming NO tool (a proposed tool is left to run), on the
        # turn genuinely being ABOUT scheduling (never hijacking a mid-booking price
        # question), and on not talking over the registration question.
        elif (proposal.tool is None and not executed
                and not _asking_to_register
                # A visit cannot be advanced for a project we have not identified — hold the
                # scheduling force while project disambiguation is pending.
                and not _force_project_clarify
                and _turn_is_about_scheduling(user_input, proposal, session)):
            # In flight by session state, OR the caller explicitly asked to visit on THIS
            # turn — on the turn a booking starts, nothing is in the session yet (wanted_tool
            # is written further down), so the state test alone would refuse the one turn the
            # caller most plainly asked for slots.
            _forced = _next_scheduling_lookup(
                session, executed,
                in_flight=(_scheduling_in_flight(session)
                           or _turn_hint(user_input, proposal) in ("book_appointment",
                                                           "reschedule_appointment")))
            if _forced is not None:
                proposal.action = "invoke_tool"
                proposal.tool, proposal.params = _forced[0], dict(_forced[1])
                _n = (f"Rule 1c: scheduling turn answered with a promise and no tool; forced "
                      f"{proposal.tool} so the caller hears real availability on THIS turn "
                      f"instead of a progress report.")
                proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n

        # The rules above rewrite proposal.action directly; re-sync the local `action` the
        # confirmation gate branches on (same pattern as the redirects above).
        action = proposal.action

    # ---- confirmation gate ----
    #
    # Consent is `_consent` (consent.py): the caller's own words, fused with the model's read
    # of them, ACCEPT only on positive evidence in the transcript. The planner's intent label
    # is recorded for the trace and never consulted here — that label is what booked a visit
    # off "i'm sorry what time / ja por time again".
    intent = str((proposal.understood or {}).get("intent") or "").lower()
    affirmative = _consent.accepted
    # A "yes" releases a PARKED action only when the reply before it asked about that action.
    _pending_tool = (pending or {}).get("tool")
    _affirm_pending = affirmative and _pending_fresh
    if pending and affirmative and not _pending_fresh:
        _n = (f"yes not applied to the parked {_pending_tool}: it was last asked about on turn "
              f"{_consent_asked_turn}, not the previous one; it will be read back first.")
        proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n

    # The customer has already agreed and the model has named the action — so there is
    # nothing left to confirm. Left alone, the model parks the tool instead of running it
    # and then reports "Done, you're booked" while nothing has executed: the single worst
    # failure this engine can produce, because it lies to the customer.
    #
    # Checked before parking, and deliberately independent of what the previous turn was
    # labelled — a read-back that got downgraded still ends with the customer saying yes.
    # …but only when the reply they are saying yes TO was itself a question or a read-back
    # (a confirm/clarify, or a lookup result such as offered slots).
    #
    # THE BUG THIS FIXES: a turn whose planner call failed told the caller "I'm having trouble
    # on my side" — no slot was ever spoken. Next: "let me think" → "Sure, take your time".
    # Then "yes" — to nothing in particular — and the planner, labelling its own proposal
    # "confirm", booked Friday 12:00 PM: a time the caller never heard. A model choosing
    # action=confirm is not evidence the caller was asked anything; the previous turn is.
    _prev_actions_ca = session.get("recent_actions") or []
    _asked_last_turn = bool(_prev_actions_ca) and _prev_actions_ca[-1] in (
        "action:confirm", "action:clarify", "action:tool_executed")
    customer_already_agreed = False
    if (action == "confirm" and proposal.tool and affirmative and _asked_last_turn
            and not (proposal.tool == _pending_tool and not _pending_fresh)):
        action = "invoke_tool"
        # Must be remembered, not just switched. The invoke_tool block below re-checks
        # requires_confirmation and, finding no evidence of consent, parks the tool and
        # sets the action straight back to confirm — silently undoing this line. Consent
        # given on THIS turn is the evidence, so carry it forward explicitly.
        customer_already_agreed = True

    # The model asked for confirmation and named the action. Whether consent is actually
    # needed is the CONTRACT's decision, not the model's — requires_confirmation already
    # says so per tool. Parking here as well meant a read-only lookup got staged and never
    # ran: "let me find that for you" with nothing executed, twice, while the customer
    # waited. Converting to invoke_tool lets the requires_confirmation check below make one
    # decision in one place — a booking still gets parked for a read-back, a lookup runs.
    if action == "confirm" and proposal.tool:
        action = "invoke_tool"

    # The prompt tells the model that "confirm" MUST carry a tool and params. It does not
    # always obey: it uses confirm as a conversational move — "shall I book that for
    # you?" — with no tool named. Nothing gets parked, so when the customer then says
    # yes there is no action to release, the model asks again, and the booking never
    # happens. Observed twice in a row on a five-turn happy path.
    #
    # An unenforced instruction is not a gate. If confirm arrives with no tool and
    # nothing already parked, the plan is malformed: downgrade to clarify so the agent
    # asks a question instead of promising an action it has not staged, and record the
    # fault on the trace so it is visible rather than silently swallowed.
    if action == "confirm" and not proposal.tool and not pending:
        action = "clarify"
        proposal.action = "clarify"
        note = ("Planner returned action=confirm without a tool and nothing was parked; "
                "downgraded to clarify. The model must name the tool it confirms.")
        # proposal.error is what the trace exposes as planner_error — append rather than
        # overwrite, so a genuine planner failure is not hidden by this note.
        proposal.error = f"{proposal.error} | {note}" if proposal.error else note

    # Safety net: something is parked and the customer just said yes, but the model
    # asked again instead of proceeding. Honour the customer, not the loop.
    # The tool-name match used to be required here. It let a turn slip through whenever the
    # model named some other tool while still only talking: the customer said "yes please"
    # to a slot that had been read back, the net declined to fire, and nothing ran — the
    # customer's consent wasted. Their yes refers to what was read back to them — that is the
    # whole point of reading it back — so on a turn where the model is not invoking anything,
    # the parked action wins.
    # Also when the model proposes a HARMLESS LOOKUP instead of the parked action.
    #
    # The list used to be confirm/clarify/respond only, so a turn where the model proposed
    # search_slots fell straight through — and the customer's yes was ignored. A real call
    # went "yes", "yes", "yes please" and got the same confirmation question back three
    # times, because each turn re-ran the search rather than releasing the booking she had
    # already agreed to twice.
    #
    # A read-only lookup is not a decision to do something else; it is the model hedging.
    # Her yes refers to what was read back to her, and that outranks a hedge.
    _hedging = (action == "invoke_tool" and proposal.tool in CHAINABLE_TOOLS
                and proposal.tool != (pending or {}).get("tool"))
    # The same principle, for a DIVERGENT WRITE rather than a hedge-lookup. When a scheduling
    # write is parked and read back ("shall I book?"), a bare affirmation may not execute a
    # DIFFERENT scheduling write the model drifted to on the confirmation turn. This is the
    # exact failure a caller hit: "book the earliest slot" → "shall I book?" → "ok", and the
    # model proposed cancel_appointment; the yes released nothing matching, so the cancel ran
    # (and blocked), while the booking the caller had just agreed to never happened. A real
    # change of mind ("no, cancel it instead") is not affirmative and never reaches here, so
    # honouring the parked write on a yes is safe. Non-write divergences stay with _hedging.
    _diverging_write = (action == "invoke_tool"
                        and (pending or {}).get("tool") in _SCHEDULING_WRITES
                        and proposal.tool in _SCHEDULING_WRITES
                        and proposal.tool != (pending or {}).get("tool"))
    if pending and _affirm_pending and (action in ("confirm", "clarify", "respond")
                                        or _hedging or _diverging_write):
        if _diverging_write:
            _note = (f"bare confirmation released the parked {pending.get('tool')}, not the "
                     f"{proposal.tool} the model proposed — a yes confirms what was read back")
            proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note
        action = "invoke_tool"
        proposal.tool = pending.get("tool")
        proposal.params = pending.get("params") or {}

    # A read-back on the previous turn followed by a clear yes on this one IS the
    # confirmation. Without this, the agent asks "shall I book?", the customer says
    # yes, and the gate asks again — a polite infinite loop.
    prev_actions = session.get("recent_actions") or []
    # "action:clarify" is included because the malformed-confirm guard above downgrades a
    # read-back to clarify. Without it, the turn that most needs the read-back signal is
    # the one that has just lost it.
    # "action:tool_executed" counts too. When search_slots runs and the agent says "I found
    # a slot at 10:00", that IS a read-back — the customer has been told a specific thing
    # and their next "yes, book it" is consent to exactly it. Without this the engine takes
    # a correct invoke_tool, finds no evidence of consent, re-parks it as a confirm, and
    # the booking never happens however many times the customer agrees.
    read_back_last_turn = bool(prev_actions) and prev_actions[-1] in (
        "action:confirm", "action:clarify", "action:tool_executed")
    verbally_confirmed = read_back_last_turn and affirmative

    # A fresh cancel_appointment proposal (not yet pending) must be backed by the caller's
    # own words actually asking to cancel -- see _cancellation_signalled. Checked and
    # downgraded BEFORE the invoke_tool block below, so a rejected proposal never enters the
    # execute-or-park machinery at all; it is a plain "respond" turn instead, with nothing
    # parked for a following vague "sure"/"yes" to release.
    if (action == "invoke_tool" and proposal.tool == "cancel_appointment"
            and not (pending and pending.get("tool") == "cancel_appointment")
            and not _cancellation_signalled(user_input)):
        _note = ("Model proposed cancel_appointment with no genuine cancel language in the "
                 "caller's own words this turn -- likely a decline of something unrelated "
                 "(a pin/document/callback offer) misread as wanting to cancel the visit; "
                 "downgraded to a plain acknowledgment instead.")
        proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note
        proposal.action, proposal.tool, proposal.params = "respond", None, {}
        action = "respond"

    # ---- the caller declined the parked action ----
    #
    # "No", "I changed my mind", "rehne do": the parked action is dropped and must not come
    # straight back. Two ways it used to: the planner re-proposed the SAME write with the same
    # details (re-parked and read back — "shall I book it?" right after "no"), and the
    # remembered booking goal re-staged it at the end of the turn. A decline that names new
    # details is a CHANGE, not a decline, and never reaches here.
    if _consent_staged and _consent.decision == consent.DECLINE and pending:
        _consent_declined = True
        session.pop("wanted_tool", None)
        _staged_params = pending.get("params") or {}
        if (action in ("invoke_tool", "confirm") and proposal.tool == _pending_tool
                and all(_staged_params.get(k) == v
                        for k, v in (proposal.params or {}).items())):
            proposal.action, proposal.tool, proposal.params = "respond", None, {}
            action = "respond"
            _n = f"caller declined the parked {_pending_tool}; dropped it, not re-staged."
            proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n

    # ---- a consent hold: keep the parked action waiting, run nothing ----
    #
    # "What time?", a reply neither layer could read, or "hello?" while an action is parked.
    # Whatever the planner proposed — including the parked write itself, which is how the
    # original bug booked — the action stays parked and is asked about again with the
    # deterministic line in `_consent_pin`. action="confirm" is what keeps it parked below.
    if _consent_hold is not None and pending:
        proposal.action, proposal.tool = "confirm", pending.get("tool")
        proposal.params = dict(pending.get("params") or {})
        action = "confirm"
        _consent_asked_this_turn = True

    if action == "invoke_tool" and proposal.tool:
        tool_def = await registry.get_tool(db, tenant_id, proposal.tool)
        confirming_pending = bool(pending and pending.get("tool") == proposal.tool)
        # Matching the parked tool's NAME is not consent. The customer's actual words are.
        #
        # THE BUG THIS FIXES: "shall I book Saturday 5 PM?" parks book_appointment; the
        # caller answers "no, I want tomorrow". The re-plan proposes book_appointment again,
        # its name matches the parked one, and the write fired — booking a visit the caller
        # had just declined, with no yes ever given. Same leak on reschedule. A parked
        # confirmation-required write may execute ONLY when the caller actually agreed this
        # turn (`verbally_confirmed` / `customer_already_agreed`, both of which require an
        # affirmative), so gate the pending-match branch on `affirmative` too. When the caller
        # did not affirm, `already_confirmed` stays False and the write is re-parked and read
        # back — never fired silently. `confirming_pending` is still used below to merge the
        # staged params into a bare "yes", which is its only other job.
        # A stale parked action (asked about two or more replies ago) is not released by a
        # read-back-style yes either — it is re-parked and read back fresh.
        already_confirmed = ((confirming_pending and _affirm_pending)
                             or (verbally_confirmed
                                 and not (confirming_pending and not _pending_fresh))
                             or customer_already_agreed)
        if tool_def is not None and tool_def.requires_confirmation and not already_confirmed:
            session["pending_action"] = {"tool": proposal.tool, "params": proposal.params}
            action = "confirm"
        else:
            # Perceived-latency filler: the tool below (and any it chains into) can take a
            # real moment, and the customer hears nothing until the grounded responder call
            # finishes afterward. Speak a short, non-committal acknowledgement right now over
            # the same sentence_callback the responder streams through, so the pause doesn't
            # read as silence. Fires once per turn, only when about to genuinely execute (not
            # on a parked/confirm-first branch), and only in a live streaming context --
            # sentence_callback is None on text-only/test paths, where this is a no-op.
            if sentence_callback is not None:
                # Built from the filler-audio catalog (P1.10) so the words spoken here and the
                # words pre-generated at startup cannot drift apart -- a drift would silently
                # turn every cache hit into a miss. See app/engine/voice/filler_audio.py.
                #
                # Language-matched to THIS turn: the same responder.detect_language(user_input)
                # the responder uses to choose the reply language chooses the filler language,
                # so an English turn gets an English filler, a Hindi/Hinglish turn gets a
                # Hindi/Hinglish one, and the filler never contradicts the reply that follows.
                from app.engine.voice.filler_audio import (
                    FILLER_INTENT_GENERIC, FILLER_INTENT_READ_ONLY, filler_phrase)
                _intent = (FILLER_INTENT_READ_ONLY
                          if proposal.tool in _READ_ONLY_PIPELINE_TOOLS
                          else FILLER_INTENT_GENERIC)
                _filler = filler_phrase(_intent, responder.detect_language(user_input))
                try:
                    await sentence_callback(_filler)
                except Exception:
                    pass

            ctx = _tool_context(db, tenant_id, actor, customer, conversation, session,
                                as_of_date, proposal)
            # The parked copy is a fallback for a bare "yes", not a source of truth. The
            # customer can change the details between the read-back and their agreement —
            # "shall I move it to 10:00?" / "no, four o'clock" — and taking the staged
            # params wholesale reinstated the time she had just rejected, then reported it
            # as unavailable. Anything the model supplies on THIS turn wins; the staged
            # values only fill the gaps.
            if confirming_pending:
                staged = (pending or {}).get("params") or {}
                params = {**staged, **(proposal.params or {})}
            else:
                params = proposal.params

            # Every tool call goes through the same fitting step — see _fit_params.
            params, _fit_notes = await _fit_params(
                db, tenant_id, proposal.tool, params, session, user_input,
                proposal.response_draft or "",
                project_id=getattr(conversation, "project_id", None))
            for _n in _fit_notes:
                proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n

            # ---- Rule 4 (BookingTask machine): execute a booking only on the invariant ----
            #
            # docs/booking-task-architecture.md. A booking/reschedule may fire only when the
            # task's ONE invariant holds: a chosen slot, a customer on file, AND the
            # customer's consent. This is a cross-check layered ON TOP OF the other gates
            # (the confirmation gate above already parks an unconfirmed write; the tool's
            # preconditions check the slot and customer), NOT a replacement. A write that
            # reaches here has normally cleared those gates, so the invariant agrees and the
            # booking proceeds; its value is making the "never book without consent+slot+
            # customer" guarantee explicit, deterministic, and logged — catching any future
            # ordering that would slip a booking through.
            _r4_block = None
            if proposal.tool in ("book_appointment", "reschedule_appointment"):
                _r4 = bt.project(session, customer_exists=customer is not None,
                                 executed=executed,
                                 current_state=customer.current_state if customer else None,
                                 active_entities=session.get("active_entities") or {})
                _r4.confirmed = bool(already_confirmed)
                _r4.prereqs["customer_exists"] = customer is not None
                _r4.slots["verified_slot_start"] = ((params or {}).get("slot_start")
                                                    or _r4.slots.get("verified_slot_start"))
                # A reschedule must land on a DIFFERENT time (RC2). Feed the current slot in
                # and gate on ready_to_reschedule, which is ready_to_book plus "the new time
                # is not the one they already have".
                _is_resched = proposal.tool == "reschedule_appointment"
                if _is_resched and session.get("appointment_current_start"):
                    _r4.slots["current_slot_start"] = session["appointment_current_start"]
                _ready = _r4.ready_to_reschedule() if _is_resched else _r4.ready_to_book()
                if not _ready:
                    _miss = []
                    if not _r4.slots.get("verified_slot_start"):
                        _miss.append("a chosen time")
                    elif _is_resched and _r4.ready_to_book():
                        # Slot + customer + consent all present, yet not ready ⇒ the chosen
                        # time IS the current one. Ask for a new time, don't re-book the same.
                        _miss.append("a different time from the one already booked")
                    if customer is None:
                        _miss.append("a registered customer")
                    if not already_confirmed:
                        _miss.append("the customer's consent")
                    _r4_block = ", ".join(_miss) or "the booking invariant"

            # RECOVERY: the customer named a NEW time to move to, but it was never searched,
            # so it cannot be verified and the reschedule cannot fire. Do NOT loop "what day
            # and time would you like?" — search that day now (the reschedule analog of the
            # Rule 1/5 search force, applied at the point the invariant actually fails), then
            # the chain re-plans, offers real slots, and reads one back. Without this the call
            # got stuck: "aug 19 2pm" was dropped as unverified, the machine fell back to the
            # current appointment time, RC2 fired "same time", and it asked again forever.
            # _resched_day comes from the model's own resolved slot_start (or our day parse) —
            # the reliable signal — so any phrasing the planner understood is honoured.
            # NB: this if/elif/else runs for EVERY tool (create_customer, knowledge_lookup, …),
            # so it must reference only always-bound names. `_is_resched` and `_req_day` /
            # `_proposed_start` are bound ONLY under the reschedule branch above — touching
            # them unconditionally here raised UnboundLocalError and 500'd every create_customer
            # turn. Gate all of that behind `proposal.tool == reschedule`, computed first, so
            # the day lookups are only evaluated when those names actually exist.
            _is_resched_now = proposal.tool == "reschedule_appointment"
            _resched_day = (_req_day or _slot_date(_proposed_start)) if _is_resched_now else None
            _already_searched = "search_slots" in {o.tool_name for o in executed}
            # Only search a day we have NOT already offered slots for. Once the day is covered,
            # a still-unverified pick means the customer's exact time is not free — re-searching
            # the same day every turn would loop, so fall through to RC2 (ask for a different
            # time) instead.
            _resched_covered = (_day_covered_in_window(
                session, _resched_day,
                _wanted_part_of_day(user_input) or session.get("time_pref"))
                if _is_resched_now else True)
            if (_is_resched_now and _r4_block is not None and _resched_day is not None
                    and session.get("centre_id") and not _already_searched
                    and not _resched_covered
                    and ("a different time" in _r4_block or "a chosen time" in _r4_block)):
                # Keep the reschedule parked (minus the unverified slot) so the pick the
                # customer makes from the fresh search fills a clean gap.
                session["pending_action"] = {"tool": "reschedule_appointment",
                                              "params": {k: v for k, v in (params or {}).items()
                                                         if k != "slot_start"},
                                              "needs": "slot_start"}
                _search_params = {"centre_id": session["centre_id"],
                                  "date_from": _resched_day.isoformat(),
                                  "date_to": _resched_day.isoformat()}
                _part_now = _wanted_part_of_day(user_input) or session.get("time_pref")
                if _part_now:
                    _search_params["part_of_day"] = _part_now
                _search_params, _sfn = await _fit_params(
                    db, tenant_id, "search_slots", _search_params, session, user_input,
                    proposal.response_draft or "",
                    project_id=getattr(conversation, "project_id", None))
                proposal.tool = "search_slots"
                proposal.params = _search_params
                _t = time.perf_counter()
                outcome = await execute(ctx, "search_slots", _search_params or {},
                                        principal_kind="agent")
                timing["tool_ms"] = int((time.perf_counter() - _t) * 1000)
                _remember_ids(session, "search_slots", outcome, _search_params or {})
                executed.append(outcome)
                _n = (f"Rule 4/RC: reschedule target {_resched_day.isoformat()} was not yet "
                      f"searched; forced a day-scoped search_slots before moving the visit.")
                proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n
            elif _r4_block is not None and "a different time" in _r4_block:
                # Same-time reschedule (RC2): the customer agreed to "reschedule" before
                # naming a new time, so the parked slot is the one they already have. Staging
                # "shall I move it to <same time>?" is nonsense — ask for the new time
                # instead, and strip the stale slot so a future pick fills a clean gap rather
                # than re-confirming the current one.
                session["pending_action"] = {"tool": proposal.tool,
                                              "params": {k: v for k, v in (params or {}).items()
                                                         if k != "slot_start"},
                                              "needs": "slot_start"}
                action = "clarify"
                proposal.action, proposal.tool = "clarify", None
                proposal.response_draft = SAFE_PROMPT_BY_INTENT["reschedule_appointment"]
                outcome = None
                _n = "Rule 4: reschedule held — chosen time equals the current one; asked for a new time."
                proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n
            elif _r4_block is not None:
                # Do NOT fabricate a booking. Hold it for confirmation so the next turn can
                # fill the gap, and record why on the trace. The downstream guards turn this
                # into a proper read-back rather than a false "booked".
                session["pending_action"] = {"tool": proposal.tool, "params": params or {}}
                action = "confirm"
                _n = f"Rule 4: booking held — invariant not met (missing {_r4_block})."
                proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n
            else:
                _t = time.perf_counter()
                outcome = await execute(ctx, proposal.tool, params or {}, principal_kind="agent")
                timing["tool_ms"] = int((time.perf_counter() - _t) * 1000)
                _remember_ids(session, proposal.tool, outcome, params or {})
                executed.append(outcome)

            # ---- keep going, in the same turn ----
            #
            # THE PROBLEM THIS SOLVES: process_turn used to run exactly one tool and then
            # stop and speak. There is no mechanism anywhere for the agent to say a second
            # thing without the customer speaking first, so a workflow as ordinary as
            # "find the centre, find a slot, read it back" cost three separate customer
            # utterances. The caller ended up saying "ok" over and over to push the agent
            # through its own steps, and when they stopped saying it, the call stalled:
            #
            #     Agent: Let me check the available slots for the evening of August 12th.
            #     Lead:  yes please
            #     Agent: The next available slot is Wednesday, August 12th at 10:00 AM.
            #     Lead:  i want in the evening
            #     Agent: Let me find an available slot in the evening for August 12th.
            #     Lead:  ok
            #     Agent: Let me find an available slot in the evening for August 12th.
            #     Lead:  when
            #
            # Nobody on a phone call behaves like that. They wait, hear nothing, and ask
            # what happened. So after a lookup succeeds the engine re-plans immediately,
            # with the result in hand, and runs the next tool too — the customer hears one
            # complete answer instead of a progress report.
            #
            # Write tools are deliberately absent from CHAINABLE_TOOLS: anything that
            # changes a record ends the turn, so the confirmation gate and the customer's
            # yes still govern every commitment. A chain that arrives at a write proposes
            # it, parks it, and reads it back — which is exactly the behaviour we want:
            # "I have 4 PM free on the 12th, shall I move it?" in a single turn.
            chain_start = time.perf_counter()
            seen = {(proposal.tool, json.dumps(params or {}, sort_keys=True, default=str))}
            steps = 0
            # Whether the CALLER'S turn was about scheduling, decided once from the turn's own
            # proposal (which still carries the classifier's routing flags) and held constant
            # for the whole chain — a re-planned leg passes skip_retrieval=True and so carries
            # no routing of its own, and the answer must not drift mid-chain regardless.
            #
            # This gates the deterministic driver below. Without it, reordering the driver
            # ahead of the re-plan would hijack any chain that merely happens to run while a
            # booking is in flight: a caller mid-booking who asks about the price gets
            # knowledge_lookup, and the driver — seeing a live booking and no centre — would
            # answer "locate_centre" and march them into a slot search they never asked for.
            # The old ordering was accidentally protected from this (it only consulted the
            # driver once the model had narrated); the new ordering has to say so explicitly.
            _chain_is_scheduling = _turn_is_about_scheduling(user_input, proposal, session)
            # `outcome` is None when Rule 4 held the booking instead of executing it — there
            # is nothing to chain from, so skip the loop entirely.
            while (outcome is not None
                   and outcome.status in ("success", "skipped_duplicate")
                   and outcome.tool_name in CHAINABLE_TOOLS
                   and steps < MAX_TOOL_STEPS
                   and (time.perf_counter() - chain_start) * 1000 < CHAIN_BUDGET_MS):
                steps += 1
                # The next plan has to be able to see what the last tool returned, and the
                # prompt is built from the message history, so the result is written before
                # re-planning rather than at the end of the turn.
                db.add(Message(tenant_id=tenant_id, conversation_id=conversation.id,
                               role="tool", content=_tool_note(outcome)))
                noted.add(id(outcome))
                await db.flush()

                # No second retrieval — the documents, if any were needed, were fetched by
                # the opening plan of this same turn, and re-running it cost about 700 ms.
                #
                # The MAIN model, not the fast one: measured, gpt-4o-mini is slower than
                # gpt-4o from here. The per-call cost is almost entirely fixed overhead, so
                # a "cheaper" model buys nothing on the clock.
                #
                # This was the hidden cost in the profile: chain steps call plan() again,
                # nothing timed them, and they landed in "other" — 5.5 seconds of it on a
                # slot search, which read like a database problem and was three model calls.
                # If the customer's own words called for a tool that still has not run,
                # say so here. The hint backstop earlier only fires when the turn proposed
                # NO tool at all — but the common miss is proposing the right first step
                # and then losing the thread: asked to cancel a visit, the agent looked the
                # booking up, then asked "do you want to keep it or make any changes?" and
                # cancelled nothing. The lookup was correct. Stopping there was not.
                # ---- ask the model only when there is actually a choice to make ----
                #
                # The deterministic next step is computed FIRST, and when one exists the
                # re-plan is skipped entirely. This is both a correctness fix and the single
                # biggest latency win available here.
                #
                # CORRECTNESS: the old order re-planned, waited for the model to narrate
                # ("let me find a nearby centre"), and only THEN consulted the driver. So a
                # sequence that is not a decision at all — no centre yet, therefore
                # locate_centre; centre but no slots, therefore search_slots — was put to a
                # model on every step, and the answer was overridden every time it came back
                # wrong. Asking a question whose answer you intend to discard is not a
                # safeguard; it is a coin flip you have to keep catching.
                #
                # LATENCY: each of those discarded plan() calls costs a full round trip
                # (~1.3-2 s of mostly fixed overhead). On the create_customer ->
                # locate_centre -> search_slots chain that is two calls, ~3-4 s, spent to be
                # told something the session already knew — and it was the CHAIN_BUDGET_MS
                # spend that made the third step likely to be cut off, which is the other way
                # this same sequence lost its last step. Skipping them keeps the whole chain
                # comfortably inside the budget.
                #
                # The model still gets its say wherever judgement is genuinely required: the
                # driver returns None the moment slots are on the table, and from there the
                # re-plan below decides what to do with them.
                # Closing the wanted_tool-written-late gap so a fresh booking advances past
                # locate_centre to the slot search in ONE turn (the multi-centre stall: several
                # centres came back, none got auto-picked, and the search never got forced).
                #
                # The boost is deliberately narrow — it fires ONLY when locate_centre was the
                # tool that just ran AND the caller's words hint a book/reschedule. locate_centre
                # exists only to feed a slot search, so "we just located, heading to book" is a
                # safe read; requiring the hint too keeps a bare "where are your offices?" (a
                # locate with no booking) from being pushed onto a search. Crucially it does NOT
                # fire after my_appointments — "what's my visit booked for" is a VIEW (whose text
                # also trips the coarse "book" hint), and a view must never become a slot search.
                _just_located = bool(executed) and executed[-1].tool_name == "locate_centre"
                _forced = (_next_scheduling_lookup(
                               session, executed,
                               in_flight=(_scheduling_in_flight(session)
                                          or (_just_located
                                              and _turn_hint(user_input, proposal) in (
                                                  "book_appointment", "reschedule_appointment"))))
                           if _chain_is_scheduling else None)
                if _forced is not None:
                    proposal.action = "invoke_tool"
                    proposal.tool, proposal.params = _forced[0], dict(_forced[1])
                    _n = (f"Chain: {proposal.tool} is the deterministic next step of the "
                          f"booking; ran it directly without re-planning.")
                    proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n
                else:
                    hint = _turn_hint(user_input, proposal)
                    already = {o.tool_name for o in executed}
                    nudge = ""
                    if hint and hint not in already:
                        nudge = (f"\n\n[SYSTEM: the customer asked for something that needs "
                                 f"{hint}, and it has not run yet. If you now have what it "
                                 f"needs, call it — do not ask them to restate what they "
                                 f"already told you.]")

                    _t = time.perf_counter()
                    nxt = await plan(
                        db, tenant_id=tenant_id, conversation=conversation,
                        user_input=user_input + nudge, stt_confidence=stt_confidence,
                        asr_hints=_asr_hints,
                        identity_verified=bool(session.get("identity_verified")),
                        pending_action=session.get("pending_action"), as_of_date=as_of_date,
                        active_entities=session.get("active_entities") or {},
                        offered_slots=session.get("offered_slots") or [],
                        booking_summary=_booking_summary(),
                        skip_retrieval=True)
                    timing["planner_ms"] = timing.get("planner_ms", 0) + int((time.perf_counter() - _t) * 1000)
                    timing["planner_calls"] = timing.get("planner_calls", 1) + 1

                    # Carry the trace forward, so the decision record shows every rule and
                    # chunk that took part in the turn rather than only the final step's.
                    nxt.knowledge_used = (proposal.knowledge_used or []) + (nxt.knowledge_used or [])
                    nxt.rules_evaluated = (proposal.rules_evaluated or []) + (nxt.rules_evaluated or [])
                    proposal = nxt

                    if proposal.action != "invoke_tool" or not proposal.tool:
                        break                # it has what it needs; let it speak

                nxt_def = await registry.get_tool(db, tenant_id, proposal.tool)
                # Consent already given for exactly this tool, on an earlier turn, and the
                # attempt was blocked rather than refused. Verifying her identity is what
                # she was asked to do to unblock it — asking "shall I go ahead?" a second
                # time after she has already said yes is how a four-turn task becomes ten.
                staged_now = session.get("pending_action") or {}
                # Resuming the parked tool is only legitimate when consent for it EXISTS: it was
                # given earlier and the attempt was blocked (blocked_by — e.g. identity), or the
                # caller agreed to it on THIS turn. Matching the name alone used to be enough, so
                # a turn that merely ran a lookup ("what time?" → search_slots) could re-plan into
                # the parked booking and run it with no yes at all.
                resuming = (staged_now.get("tool") == proposal.tool
                            and (nxt_def is None or not nxt_def.requires_confirmation
                                 or bool(staged_now.get("blocked_by")) or _affirm_pending))
                if resuming:
                    proposal.params = {**(staged_now.get("params") or {}),
                                       **(proposal.params or {})}
                elif nxt_def is not None and nxt_def.requires_confirmation:
                    # Park and read back. The customer's yes is still required.
                    session["pending_action"] = {"tool": proposal.tool,
                                                 "params": proposal.params}
                    action = "confirm"
                    break
                elif proposal.tool not in CHAINABLE_TOOLS:
                    # A WRITE TOOL THE CUSTOMER NEVER ASKED FOR. Stop; do not run it.
                    #
                    # THE BUG THIS FIXES: four tools change records without requiring
                    # confirmation — schedule_callback, record_customer_fact,
                    # update_customer_profile and record_do_not_contact. Chaining let the
                    # model reach them off the back of an unrelated lookup. Asked "is there
                    # any offer running right now", the agent looked something up, re-planned
                    # with the result in hand, drifted, and booked a callback the customer
                    # had not mentioned — then reported it as done. The same path could have
                    # put someone on the do-not-contact list without them ever asking.
                    #
                    # Chaining exists to let the agent finish a LOOKUP it already started,
                    # never to let it acquire new intentions between turns. The tool is not
                    # forbidden — it can still run on the next turn, through the normal
                    # path, where the customer's own words are what proposed it.
                    proposal.error = ((proposal.error + " | ") if proposal.error else "") + \
                        f"Chain stopped before {proposal.tool}: writes need their own turn."
                    break

                key = (proposal.tool, json.dumps(proposal.params or {}, sort_keys=True,
                                                 default=str))
                if key in seen:
                    # Same tool, same params, second time. The model is stuck, and running
                    # it again would only burn the latency budget.
                    proposal.error = ((proposal.error + " | ") if proposal.error else "") + \
                        f"Chain stopped: {proposal.tool} repeated with identical params."
                    break
                seen.add(key)

                ctx = _tool_context(db, tenant_id, actor, customer, conversation, session,
                                    as_of_date, proposal)
                # The chain path executes too, so it fits its params the same way. Doing
                # this in only one of the two call sites is what made the last fix look
                # live and behave as though it were not.
                _chain_params, _fit_notes = await _fit_params(
                    db, tenant_id, proposal.tool, proposal.params, session, user_input,
                    proposal.response_draft or "",
                    project_id=getattr(conversation, "project_id", None))
                for _n in _fit_notes:
                    proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n
                _t = time.perf_counter()
                outcome = await execute(ctx, proposal.tool, _chain_params,
                                        principal_kind="agent")
                timing["tool_ms"] = timing.get("tool_ms", 0) + int((time.perf_counter() - _t) * 1000)
                _remember_ids(session, proposal.tool, outcome, _chain_params)
                executed.append(outcome)

            timing["tool_steps"] = len(executed)
            confirmation_source = ("pending_action" if confirming_pending
                                   else ("verbal_read_back" if verbally_confirmed else "not_required"))
            # `outcome` is None when Rule 4 held the booking without executing a tool — the
            # same-time reschedule (RC2) path sets outcome=None, action="clarify", and has
            # already parked pending_action and written the "ask for a new time" prompt.
            # There is no tool result to inspect here, so skip the blocked-tool handling
            # entirely rather than dereferencing None (which crashed the turn with
            # AttributeError: 'NoneType' object has no attribute 'status').
            if outcome is not None and action != "confirm":
                # A BLOCKED tool never ran. The customer has already agreed, and the
                # details they agreed to are in the parked action — so throwing it away
                # here means that once the blocker clears there is nothing left to resume.
                #
                # THE BUG THIS FIXES: she said "reschedule to the 12th at 4 PM", the agent
                # read it back, she said yes, and the rule gate blocked it pending identity
                # verification. Clearing pending_action at that moment discarded the date
                # and the time. She verified successfully, and the agent then said it did
                # not have the details to confirm — and escalated. Her consent was real and
                # so were the details; only the engine had forgotten them.
                if outcome.status == "blocked" and outcome.failed_check == "customer_identified":
                    # Do NOT park this one the way an identity or slot block gets parked.
                    #
                    # THE BUG THIS FIXES: asked about a 1.5 BHK and a budget, the model
                    # reasonably tried record_customer_fact to note it — blocked, no
                    # customer exists yet. That block got parked exactly like an identity
                    # check (both match "identified" in the reasoning), on the assumption
                    # it would resolve itself and could just be retried. It cannot: retrying
                    # record_customer_fact will never succeed until registration happens
                    # through the separate name/phone flow below, which is what actually
                    # clears this. Meanwhile the customer said "yes" — meaning "yes, I'd
                    # like to visit" — and that "yes" released the PARKED record_customer_fact
                    # instead, which failed the same way again. The visit was never asked
                    # about again; every later "yes" kept feeding the wrong parked task.
                    #
                    # The registration block below is the actual mechanism for "something
                    # needs a customer and there isn't one" — it already resumes
                    # book_appointment/schedule_callback/create_ticket once one exists (see
                    # reg["goal"] a bit further down, now also record_customer_fact and
                    # update_customer_profile). Leaving pending_action alone here — clearing
                    # it rather than parking a copy of this — stops it from competing with
                    # that mechanism for the customer's next "yes".
                    session["pending_action"] = None
                elif outcome.status == "blocked":
                    # Keep the consent, but count the failures.
                    #
                    # A blocked action is normally worth holding: the customer already said
                    # yes and the blocker (identity, say) is about to clear. When the
                    # blocker is the PARAMETERS though — a slot that is not free — holding
                    # it re-releases the same dead booking on every subsequent turn. A real
                    # call went round four times: the customer picked ten, then twelve,
                    # then eleven, and each time the parked 4pm booking was retried and
                    # refused, so the agent offered the same three times again.
                    #
                    # Two strikes, then the stale action goes and the next turn plans
                    # afresh from what the customer actually said.
                    prior = (session.get("pending_action") or {})
                    strikes = prior.get("strikes", 0) + 1 if prior.get("tool") == outcome.tool_name else 1

                    # Count strikes against the PARAMS, not the task.
                    #
                    # This used to discard the whole parked action after two blocks. The
                    # strike counter was added to stop a loop where the same dead booking
                    # was retried forever — but a prerequisite block is not a dead booking.
                    # An identity check plus one busy slot is two strikes, and the customer's
                    # agreed date and time went in the bin: "I don't have the details to
                    # confirm", then an escalation for a task the agent was one step from
                    # finishing.
                    #
                    # A block on a PREREQUISITE (identity, registration) says the task is
                    # fine and something must happen first — the consent and the details
                    # stay. Only a block on the parameters themselves, twice, means the
                    # params are wrong; and even then the slot is cleared rather than the
                    # task, so the next turn asks for a time instead of starting over.
                    _prereq_block = any(w in str(outcome.failed_check or outcome.error or "").lower()
                                        for w in ("identity", "verify", "identified",
                                                  "customer must be", "consent"))
                    if strikes >= 2 and not _prereq_block:
                        _kept = {k: v for k, v in (params or {}).items()
                                 if k not in ("slot_start",)}
                        session["pending_action"] = {"tool": outcome.tool_name,
                                                     "params": _kept,
                                                     "blocked_by": outcome.failed_check,
                                                     "strikes": 0,
                                                     "needs": "slot_start"}
                        note = (f"{outcome.tool_name} blocked {strikes}x on the same params; "
                                f"cleared slot_start and kept the rest so they are asked for "
                                f"a time rather than starting again.")
                        proposal.error = f"{proposal.error} | {note}" if proposal.error else note
                    else:
                        session["pending_action"] = {"tool": outcome.tool_name,
                                                     "params": params or {},
                                                     "blocked_by": outcome.failed_check,
                                                     "strikes": strikes}
                else:
                    session["pending_action"] = None
                state_changed_to = outcome.state_changed_to
                action = "tool_executed" if outcome.status in ("success", "skipped_duplicate") else "spoke"
                if outcome.status == "failed" and (outcome.failure_action or {}).get("action") == "escalate":
                    action = "escalate"
                # A booking/reschedule that just succeeded against this bound customer_id IS
                # a stronger identity check than the visit-status gate's own name/phone
                # question — it proves the record we hold is really theirs, the same
                # conclusion a name match would reach. Without this, a caller who had JUST
                # been booked was asked "may I have the name the visit is booked under?" on
                # their very next question about that same visit, reading as though the
                # agent forgot who it was just talking to. Scoped to a SUCCESS on a write the
                # caller themselves drove this call — an outbound-confirmed identity check
                # (the other place this flag is set) still applies unaffected.
                if (action == "tool_executed" and customer is not None
                        and outcome.tool_name in _SCHEDULING_WRITES):
                    session["visit_name_confirmed"] = True
                    # A fresh booking/reschedule confirmation IS the visit reminder for this
                    # call -- whatever words the model uses to state it ("all set", "booked",
                    # "confirmed"...), the caller has now heard the day and time. Setting this
                    # deterministically here, rather than relying on _VISIT_REMINDER_RE to
                    # recognise this turn's own confirmation sentence, means a LATER repeat is
                    # always caught even when the model's phrasing here is one _VISIT_REMINDER_RE
                    # doesn't happen to match. Safe to set before _trim_repeated_visit_reminder
                    # runs on this same turn's text: that trim never drops the FIRST sentence of
                    # a reply even on a repeat, and the confirmation is always the lead sentence
                    # of a booking-success reply.
                    session["visit_reminder_given_this_call"] = True
    elif action not in ("confirm",):
        session["pending_action"] = None

    # ---- the caller asked something else, or asked for time: the parked action waits ----
    #
    # "How much is parking?" or "let me think" in reply to "shall I book?" is neither a yes
    # nor a no. Both paths above clear the parked action whenever the turn ends in a reply or
    # a successful lookup, so the booking the caller was one word from agreeing to used to
    # vanish the moment they asked a question. Put it back. After a side question the reply
    # also returns to it ("…and shall I go ahead and book your visit for …?"), a bounded
    # number of times; after a "let me think" it waits quietly. Either way it can only be
    # released by a fresh yes to a fresh read-back (asked_turn), never silently.
    _ran_write = any(o.tool_name not in READ_ONLY_TOOLS for o in executed)
    if (_consent_staged and _consent_hold is None and not _consent_declined
            and _consent_kind in (consent.SIDE_QUESTION, consent.HOLD)
            and not session.get("pending_action") and not _ran_write
            and action not in ("escalate", "end_call")
            and not proposal.forced_escalation):
        # asked_turn is carried as-is: it still names the turn that last ASKED, so a later
        # "yes" without a fresh read-back stays stale. Only a re-ask below refreshes it.
        _kept = dict(_consent_pending)
        if _consent_kind == consent.SIDE_QUESTION:
            _reasks = int(_consent_pending.get("reask_count") or 0)
            if _reasks < max(0, int(get_settings().consent_max_reasks)):
                _kept["reask_count"] = _reasks + 1
                _consent_reask_line = _consent_line(session, _consent_pending, kind="reask",
                                                    user_input=user_input)
        session["pending_action"] = _kept
        _n = (f"parked {_kept.get('tool')} kept across a {_consent_kind} reply"
              f"{'; re-asked after the answer' if _consent_reask_line else ''}.")
        proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n

    # ---- Rule 2: registration is a child task — resume the booking at confirming ----
    #
    # docs/booking-task-architecture.md failure mode D. When registration interrupts a
    # booking, the customer's chosen slot is kept in registration.params. The moment
    # create_customer succeeds, the booking must resume automatically with that slot — a
    # read-back ("shall I confirm 4 PM?") — instead of leaving it to the model, which
    # reliably drifts to "you're registered, anything else?" and forces the customer to ask
    # "is it booked?" to re-drive it. Only fires when a slot was actually chosen before
    # registration; if the name came before any slot, the normal flow asks for a day/time.
    if (not session.get("pending_action") and action != "confirm"
            and any(o.tool_name == "create_customer" and o.status in TOOL_SUCCEEDED
                    for o in executed)):
        _reg = session.get("registration") or {}
        if _reg.get("goal") == "book_appointment":
            _offered = {s.get("starts_at"): s for s in (session.get("offered_slots") or [])
                        if s.get("starts_at")}
            _chosen = (_reg.get("params") or {}).get("slot_start")
            if _chosen not in _offered:
                _chosen = _pick_offered_slot(session, user_input, proposal.response_draft or "")
            _centre = session.get("centre_id") or (_reg.get("params") or {}).get("centre_id")
            if _chosen in _offered and _centre:
                session["pending_action"] = {
                    "tool": "book_appointment",
                    "params": {"slot_start": _chosen, "centre_id": _centre},
                    "staged_by": "registration_resume"}
                action = "confirm"
                session.pop("registration", None)
                session.pop("wanted_tool", None)
                session["identity_verified"] = True   # the record IS what they just gave us
                _name = (_reg.get("name") or "").split()
                _first = _name[0] if _name else "there"
                _when = _offered[_chosen].get("when") or "the time you chose"
                proposal.response_draft = (
                    f"Thanks {_first} — shall I confirm your visit for {_when} at the "
                    f"site office?")
                _n = ("Rule 2: registration complete; resumed the booking at confirming "
                      "with the slot the customer already chose.")
                proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n

    # ---- park what the customer asked for, if the model would not ----
    #
    # Last resort, and only for tools that REQUIRE CONFIRMATION — so this stages a question,
    # never an action. Nothing is written; the customer still has to say yes.
    #
    # The failure it answers: asked to cancel a visit, the agent looked the booking up and
    # then said "do you want to go ahead with it or make any changes?" — a question that
    # cannot be answered, about a request the customer had already made plainly. The
    # lookup ran, the chain nudged, and the model still would not name the tool. Parking it
    # here turns that into "your visit is on the fourteenth at noon — shall I cancel it?",
    # which is the sentence they were owed two turns earlier.
    # What they came for, remembered across turns.
    #
    # The hint used to read only the CURRENT message. So "move my site visit to Friday"
    # registered on turn one, and the moment the customer switched to "yes please" the
    # engine forgot what the call was about — nothing was staged, and three turns later it
    # told her the visit had not been moved and asked what day she wanted. She had said.
    fresh_hint = _turn_hint(user_input, proposal)
    # You cannot book a time nobody has looked up. If they name a slot before any search
    # has run, the thing they need next is the search — not a booking staged against an
    # empty list, and not "the slot you wanted isn't available", which is what they were
    # told about a slot that had never been checked.
    if (fresh_hint in ("book_appointment", "reschedule_appointment")
            and not session.get("offered_slots")
            and "search_slots" not in {o.tool_name for o in executed}):
        fresh_hint = "search_slots"
    # verify_identity is never what they RANG ABOUT. Letting it overwrite the remembered
    # goal meant that giving the digits erased the reschedule the whole call was for.
    # Not on a turn that just declined the parked action: "no, don't book it" contains a
    # booking word, and re-remembering it as the goal re-stages the very booking refused.
    if fresh_hint and fresh_hint != "verify_identity" and not _consent_declined:
        session["wanted_tool"] = fresh_hint
    hinted = fresh_hint or session.get("wanted_tool")
    if hinted and hinted in {o.tool_name for o in executed if o.status in TOOL_SUCCEEDED}:
        session.pop("wanted_tool", None)


    if (hinted and action not in ("escalate", "end_call", "confirm")
            and hinted not in {o.tool_name for o in executed}
            and not session.get("pending_action") and not _consent_declined):
        hint_def = await registry.get_tool(db, tenant_id, hinted)
        if hint_def is not None and hint_def.requires_confirmation:  # noqa: SIM102
            # Seed the parked params from what the call already established. Staging an
            # empty object meant the customer said yes and then hit a schema error asking
            # for an appointment id — about the booking the agent had just read out.
            #
            # Filtered against the tool's own schema, because most of these set
            # additionalProperties: false. Helpfully adding a centre_id to a cancellation
            # would be rejected outright, turning a fix into a different failure.
            allowed = set(((hint_def.input_schema or {}).get("properties") or {}).keys())
            staged_params = {k: v for k, v in (proposal.params or {}).items() if k in allowed}
            for key in ("appointment_id", "centre_id"):
                if key in allowed and key not in staged_params and session.get(key):
                    staged_params[key] = session[key]
            if "slot_start" in allowed:
                # Same check as _fit_params: a supplied time is only trustworthy if it
                # is one that was actually searched and offered.
                _offered_starts = {s.get("starts_at")
                                   for s in (session.get("offered_slots") or [])
                                   if s.get("starts_at")}
                if (staged_params.get("slot_start")
                        and staged_params["slot_start"] not in _offered_starts):
                    staged_params.pop("slot_start", None)
                if not staged_params.get("slot_start"):
                    picked = _pick_offered_slot(session, user_input,
                                                proposal.response_draft or "")
                    if picked:
                        staged_params["slot_start"] = picked
            # A booking with no time is not a booking. If we cannot say which slot they
            # meant, stage nothing — parking something that will fail on release is worse
            # than leaving the model to ask.
            required = set(((hint_def.input_schema or {}).get("required") or []))
            missing = {k for k in required | {"slot_start"} & allowed
                       if k in allowed and not staged_params.get(k)}
            if not missing:
                session["pending_action"] = {"tool": hinted, "params": staged_params,
                                             "staged_by": "hint"}
                action = "confirm"
                note = (f"Planner would not propose {hinted}; engine staged it for "
                        f"confirmation.")
                proposal.error = f"{proposal.error} | {note}" if proposal.error else note

    # ---- ask what it is about, once, before transferring ----
    #
    # "I want to talk to a human" was handed straight over. That is a real answer to a real
    # request, but it means the agent never found out whether the thing they wanted was
    # something it could have done in one turn — and an agent that transfers on the first
    # ask is not doing the job it exists to do.
    #
    # So: ask what the call is about ONCE, then transfer. The gate is deliberately narrow
    # and fails open — it only ever delays a transfer when ALL of these hold:
    #   * they used plain "put me through to a person" words, and this is the first time
    #   * the model chose to escalate itself; no rule forced it
    #   * they are not angry, and this is not a complaint, dispute or legal matter
    #   * nothing has already gone wrong this turn (an LLM failure escalates immediately —
    #     that is the engine admitting it is broken, and second-guessing it would strand
    #     the customer)
    # Anything else, including a second request in any wording, goes through untouched.
    asked_for_human = _asks_for_human(user_input)
    if asked_for_human:
        session["human_requests"] = int(session.get("human_requests", 0)) + 1

    # Every condition, recorded. This gate has now failed to fire twice for reasons I could
    # only guess at from the transcript, and guessing costs a round trip each time. The
    # verdict goes into debug.triage so the next failure names itself.
    understood = proposal.understood or {}
    triage_checks = {
        "escalating": action == "escalate",
        "asked_for_human": asked_for_human,
        "first_ask": int(session.get("human_requests", 0)) == 1,
        "not_rule_forced": (not proposal.forced_escalation
                            or (proposal.forced_escalation or {}).get("rule")
                            in TRIAGEABLE_ESCALATION_RULES),
        "not_after_tool": not proposal.escalate_after_tool,
        "no_planner_error": not proposal.error,
        "not_already_escalated": not session.get("escalated"),
        "sentiment_ok": str(understood.get("sentiment") or "").lower() not in NO_TRIAGE_SENTIMENTS,
        "intent_ok": str(understood.get("intent") or "").lower() not in NO_TRIAGE_INTENTS,
    }
    triage_debug = {"fired": all(triage_checks.values()),
                    "blocked_by": [k for k, v in triage_checks.items() if not v],
                    "sentiment": understood.get("sentiment"),
                    "intent": understood.get("intent"),
                    "planner_error": proposal.error,
                    "forced_by": (proposal.forced_escalation or {}).get("rule")}
    if all(triage_checks.values()):
        _t = time.perf_counter()
        triage = await plan(
            db, tenant_id=tenant_id, conversation=conversation,
            user_input=(f"{user_input}\n\n[SYSTEM: they have asked for a person for the "
                        f"first time and are not upset. Before transferring, ask warmly "
                        f"what it is about — you may be able to sort it out yourself in "
                        f"one step. Make clear you will put them through if they would "
                        f"still rather speak to someone. action=clarify.]"),
            stt_confidence=stt_confidence, asr_hints=_asr_hints,
            identity_verified=bool(session.get("identity_verified")),
            pending_action=session.get("pending_action"), as_of_date=as_of_date,
            active_entities=session.get("active_entities") or {},
            offered_slots=session.get("offered_slots") or [],
            skip_retrieval=True)
        timing["planner_ms"] = timing.get("planner_ms", 0) + int((time.perf_counter() - _t) * 1000)
        timing["planner_calls"] = timing.get("planner_calls", 1) + 1
        # Judge the reply it wrote, not the label it attached to it. Asked to find out
        # what the call was about, the model reliably writes the right question and then
        # tags the turn "escalate" anyway — because the customer did ask for a person. A
        # draft that puts a question to them IS the triage step; taking the label at face
        # value threw that away and transferred on the first ask every time.
        drafted = (triage.response_draft or "").strip()
        # A question mark is not enough. The model wrote "I can help with many queries.
        # May I know what it's about?" — technically a question, but it opens by talking
        # the customer out of the transfer, and she had to ask twice. Someone who asked for
        # a person should hear that they will get one; the question is to route them
        # faster, not to keep them.
        low_draft = drafted.lower()
        offers_transfer = re.search(
            r"\b(?:transfer|put you through|connect you|colleague|sales executive|"
            r"someone who can)\b", low_draft)
        # A REFUSAL to transfer also contains the word "transfer".
        #
        # The model wrote "I'm afraid I can't transfer this call, but I can help you with
        # the additional charges" — and the substring check waved it through. The call was
        # handed to a person on that very turn, so the customer was told transfers were
        # impossible while being transferred, and carried on talking to an agent that had
        # already let go of her.
        #
        # Never claim an inability the system does not have. Transfers work; the whole
        # point of triage is to route faster, not to talk anyone out of one.
        refuses = re.search(
            r"\b(?:can'?t|cannot|can not|unable to|not able to|"
            r"do not|don'?t)\b[^.?!]{0,30}"
            r"\b(?:transfer|put you through|connect|hand)\b", low_draft)
        if refuses or not (drafted and "?" in drafted and offers_transfer):
            drafted = TRIAGE_QUESTION
            triage.response_draft = TRIAGE_QUESTION
        if drafted:
            triage.knowledge_used = (proposal.knowledge_used or []) + (triage.knowledge_used or [])
            triage.rules_evaluated = (proposal.rules_evaluated or []) + (triage.rules_evaluated or [])
            proposal = triage
            action = "clarify"
            session["triaged_human_request"] = True
            # Clear the rule's verdict as well as the action. The escalation block below
            # fires on `forced` REGARDLESS of action, so leaving it set meant the call
            # asked "what is it regarding?" and transferred her in the same breath — she
            # was put through to a colleague while being invited to explain herself to the
            # agent. Deciding to triage means deciding not to escalate, this turn.
            proposal.forced_escalation = None
        # If it did not produce a question, the transfer stands. A caller asking for a
        # person must never be trapped by a failure in the code trying to keep them.

    # ---- outbound lead guard: keep leading, never reactively hand a pitch call to a human ----
    #
    # WE placed this call. Once identity is confirmed, an outbound call exists to move the
    # customer forward — and for a pitch purpose the one move that always works WITHOUT any
    # inventory lookup is offering a site visit. Handed a thin knowledge base, the reactive
    # engine instead ran a product lookup, found nothing, apologised and tagged the turn
    # `escalate` ("handed to a person") — the inbound reflex, the opposite of leading. This
    # replaces that SOFT, model-chosen escalation with a lead-forward reply (ground the caller in
    # their registered interest, offer the visit); the text override lands after the responder.
    #
    # Deliberately narrow — fires only when ALL hold, so a genuine handover is untouched:
    #   * outbound, identity CONFIRMED (never inbound; never before the person is confirmed)
    #   * the MODEL chose to escalate — no rule forced it, it is not a post-tool compliance
    #     handover, and the caller neither asked for a person nor accepted an offered handoff
    #   * the caller is not upset and this is not a complaint/dispute (reuse the triage guards)
    #   * we have not already escalated on this call
    # An explicit "put me through", an angry caller or a forced rule escalates exactly as before.
    # Gated on `outbound`, so inbound behaviour is byte-for-byte unchanged.
    _outbound_lead_text = None
    if (action == "escalate" and session.get("outbound")
            and session.get("outbound_stage") == outbound_flow.STAGE_CONFIRMED
            and not asked_for_human and not accepted_handoff
            and not proposal.forced_escalation and not proposal.escalate_after_tool
            and not session.get("escalated")
            and str((proposal.understood or {}).get("sentiment") or "").lower()
                not in NO_TRIAGE_SENTIMENTS
            and str((proposal.understood or {}).get("intent") or "").lower()
                not in NO_TRIAGE_INTENTS):
        _outbound_lead_text = outbound_flow.lead_forward_reply(
            session.get("call_purpose"),
            getattr(customer, "full_name", None),
            (getattr(proposal, "enquiry_project", None)
             or getattr(proposal, "project_name", None)),
            getattr(proposal, "enquiry_interest", None))
        action = "respond"
        proposal.action = "respond"
        proposal.tool = None
        proposal.forced_escalation = None
        _n = ("Outbound lead guard: model tried to hand this confirmed outbound call to a human "
              "for lack of an answer; kept leading and offered a site visit instead.")
        proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n

    # P6-F5: the explicit_human_request rule (escalation_policy) fires off the MODEL's own
    # human_handoff_requested flag or a "human"-labelled intent -- not the caller's literal
    # words -- and a caller simply asking for a channel we don't have (email) can trip that
    # model-side signal even though nothing they said asked for a person.
    #
    # THE BUG THIS FIXES: "can you send the details on email?" forced action="escalate" via
    # explicit_human_request, while the SAME planner call's own reasoning read "Offer WhatsApp
    # as alternative" -- the model's structured fields (human_handoff_requested / intent) and
    # its own drafted reasoning disagreed, and the rule-forced override won, bypassing the
    # prompt rule (prompts.py, "ASKED FOR A CHANNEL YOU DON'T HAVE...") entirely because that
    # rule only shapes the response TEXT, never the rule-forced ACTION.
    #
    # Must run BEFORE the escalation block below, not after: that block is what actually
    # creates the Escalation row and sets session["escalated"] = True, and undoing the
    # action/text afterwards (as a first attempt at this fix did) still left a phantom
    # escalation record behind, suppressing every later turn's own legitimate escalation
    # payload for the rest of the call. Same reasoning as the _outbound_lead_text guard just
    # above, which resets forced_escalation for its own false-escalate case before reaching
    # here -- this is that pattern applied to a channel-request false escalate instead of an
    # outbound-silence one.
    #
    # Checked against the caller's own words (email_claim.is_email_channel_request), never
    # the model's self-reported flags, so a genuine "email me, or let me talk to someone"
    # still escalates untouched.
    if (action == "escalate"
            and (proposal.forced_escalation or {}).get("rule") == "explicit_human_request"
            and email_claim.is_email_channel_request(user_input)):
        action = "respond"
        proposal.action = "respond"
        proposal.tool = None
        proposal.forced_escalation = None
        _n = ("Rule explicit_human_request fired on a channel request (email), not a "
              "genuine human-handoff ask; reverted to a respond turn with the honest "
              "WhatsApp offer rather than transfer for a missing email channel.")
        proposal.error = f"{proposal.error} | {_n}" if proposal.error else _n

    # ---- escalation ----
    escalation_payload = None
    forced = proposal.forced_escalation
    if action == "escalate" or forced or proposal.escalate_after_tool:
        reason = (forced or {}).get("reason") or proposal.error or "Agent could not proceed"
        esc_type = (forced or {}).get("type", "immediate")
        briefing = _briefing(customer, conversation, proposal, outcome)
        # escalations.reason is String(100), and the text that lands here is not written to
        # fit it: a rule's block reason arrives wrapped as "Blocked by rule <id>: <reason>",
        # which for the price-negotiation rule alone is 138 characters. Postgres raises
        # StringDataRightTruncation, the whole turn 500s, and the customer asking to speak
        # to a human is the one person who must never hit an error. Truncated for the
        # column; the full text is kept in the briefing, which is what the colleague picking
        # this up actually reads.
        if len(reason) > 100:
            briefing = f"Reason (full): {reason}\n\n{briefing}"
            reason = reason[:97].rstrip() + "..."
        trigger_rule = ((forced or {}).get("rule") or None)
        if trigger_rule and len(trigger_rule) > 100:
            trigger_rule = trigger_rule[:100]
        # Targeted escalation (spec §4.4/§5): a routing instruction may name a transfer
        # target. Resolve it against the tenant's active registry so the right team/queue
        # picks it up; an unknown or inactive target degrades to a generic escalation
        # (target=None) with the reason recorded — never a hard failure.
        _target_key = (forced or {}).get("target")
        resolved_target = None
        if _target_key:
            try:
                from app.db.models import TransferTarget
                _tt = (await db.execute(select(TransferTarget).where(
                    TransferTarget.tenant_id == tenant_id,
                    TransferTarget.key == str(_target_key),
                    TransferTarget.is_active.is_(True)))).scalar_one_or_none()
                resolved_target = _tt.key if _tt else None
                # Carry the human name forward so a rule-forced handover can SAY where the
                # caller is going ("putting you through to the Financial Desk") instead of a
                # generic line — but only when the target actually resolved to an active row.
                if _tt is not None and isinstance(forced, dict):
                    forced["target_display"] = _tt.display_name
            except Exception:
                logger.exception("transfer-target resolution failed — generic escalation")
        if customer is not None and not session.get("escalated"):
            db.add(Escalation(tenant_id=tenant_id, customer_id=customer.id,
                              conversation_id=conversation.id, reason=reason,
                              trigger_rule=trigger_rule, target=resolved_target,
                              confidence=proposal.confidence.score if proposal.confidence else None,
                              briefing=briefing, status="pending"))
            session["escalated"] = True
            publish(tenant_id, "escalation.created",
                    customer_id=str(customer.id), conversation_id=str(conversation.id),
                    reason=reason, escalation_type=esc_type, target=resolved_target)
            escalation_payload = {"type": esc_type, "reason": reason, "briefing": briefing,
                                  "target": resolved_target}
        # Deliberately NOT set when an escalation is already open on this conversation.
        # The record was created once — correctly — but returning the payload every turn
        # made every consumer announce the handover again, so one complaint looked like
        # three. The action stays "escalate", so behaviour is unchanged; only the
        # announcement stops repeating.
        if proposal.escalate_after_tool and outcome is not None:
            action = "tool_executed"       # the compliance action ran; escalation rides along

    # ---- step 11: grounded response ----
    # The ACTION STATE of this turn (RC1). The responder is handed the ground truth —
    # done / failed / blocked / staged — so it writes the correct tense and can never claim a
    # completion for an action that was merely parked or blocked. A staged or blocked write
    # runs no tool, so this is what stops such a turn from being described as if it happened.
    # ---- outbound pitch ladder (hybrid: deterministic stage, LLM-generated grounded content) --
    #
    # After identity is confirmed on a PITCH call, the caller's short positive replies advance a
    # gradual pitch — verify → value → visit offer → booking — NEVER repeating the opener or
    # jumping straight to a visit. The STAGE MACHINE is deterministic (reliable, loop-free); the
    # CONTENT is written by the grounded responder that ALREADY runs this turn (dynamic), so the
    # pitch LEADS with the real configurations, sizes and price rather than canned praise. We
    # force the retrieval that puts those facts in the responder's prompt — a vector search, NOT
    # an LLM call — and hand it a stage directive via `pitch=` below. A plain decline steps aside
    # gracefully; anything with real content (a price/parking question) is left to the normal
    # engine to ANSWER (classify_pitch_reply returns 'other' and this block does nothing). Gated
    # on outbound + CONFIRMED + a live ladder stage, so inbound and every non-pitch call are
    # byte-for-byte unchanged. The canned rung line rides along as the responder's tone hint AND
    # its failure fallback, so a responder outage still leads the caller forward.
    _pitch_directive = None      # stage directive handed to the responder (grounded generation)
    _pitch_draft = None          # canned rung line — responder tone hint + failure fallback
    _pitch_knowledge: list[dict] = []   # material force-retrieved for this rung
    _pitch_handled = False
    _pitch_stage = session.get("pitch_stage")
    if (session.get("outbound")
            and session.get("outbound_stage") == outbound_flow.STAGE_CONFIRMED
            and _pitch_stage in outbound_flow.PITCH_LADDER_ACTIVE
            and action not in ("end_call", "escalate")
            and not accepted_handoff and not _force_project_clarify):
        _pitch_reply = outbound_flow.classify_pitch_reply(user_input)
        _pitch_name = getattr(customer, "full_name", None)
        _pitch_project = session.get("pitch_project")
        _pitch_unit = session.get("pitch_unit")
        if _pitch_reply == "advance":
            _pitch_voc = await _tenant_vocabulary(db, tenant_id)
            _next_stage, _pitch_draft = outbound_flow.advance_pitch(
                _pitch_stage, _pitch_name, _pitch_project, _pitch_voc)
            session["pitch_stage"] = _next_stage
            _pitch_directive = {"stage": _next_stage, "project": _pitch_project,
                                "unit": _pitch_unit}
            _pitch_handled = True
            # Force the RAG lookup that grounds this rung (a vector search, NOT an LLM call). An
            # empty query (the visit/time ask needs no facts) skips it. Scoped to the pinned
            # project so one project's cost sheet never answers for another; any hiccup degrades
            # to no extra material (the responder then honestly offers a visit) rather than a 500.
            _pq = outbound_flow.pitch_query(_next_stage, _pitch_project, _pitch_unit)
            if _pq:
                try:
                    from app.engine.rag.retrieval import retrieve as _retrieve_pitch
                    _pr = await _retrieve_pitch(
                        db, tenant_id=tenant_id, query=_pq,
                        synonyms=getattr(proposal, "retrieval_synonyms", None),
                        project_id=getattr(conversation, "project_id", None))
                    _pitch_knowledge = _pr.as_context() or []
                except Exception:
                    _pitch_knowledge = []
            _note = (f"Outbound pitch ladder: {_pitch_stage} → {_next_stage}; led forward and "
                     f"grounded the pitch from {len(_pitch_knowledge)} retrieved passage(s).")
            proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note
        elif _pitch_reply == "decline":
            _next_stage, _pitch_draft = outbound_flow.decline_pitch(_pitch_name)
            session["pitch_stage"] = _next_stage
            _pitch_directive = {"stage": outbound_flow.PITCH_WIND_DOWN,
                                "project": _pitch_project, "unit": _pitch_unit}
            _pitch_handled = True
            _note = ("Outbound pitch ladder: caller declined; accepted gracefully and offered to "
                     "send details instead of pushing.")
            proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note
        # 'other' → real content; leave the turn to the normal engine (answered, not laddered).

    # ---- conversational signal: fold a presence-check / nod INTO the responder call ----
    #
    # A short caller turn that is ABOUT the conversation ("am I audible?", "mm-hmm") must not be
    # answered literally, and a canned verbatim repeat reads like customer service — it should
    # RESUME the agent's own last line in the sales persona ("yes, you're coming through — as I
    # was saying, …"). Detection stays DETERMINISTIC (fast; and the CONSENT GUARD lives here — an
    # ambiguous "yes/ok" is a nod only when nothing is waiting on a real "yes"); the GENERATION is
    # handed to the grounded responder that ALREADY runs this turn, via `continuation=` below, so
    # the natural in-persona reply costs NO extra model call. The deterministic re-delivery
    # survives only as the fallback if that responder call fails (see the block after it).
    # Skipped when the pitch ladder already owns this turn — a bare "yeah" there means ADVANCE
    # the pitch, not "nod at the last line".
    _signal_kind = None
    _prev_agent = None
    _signal_asked_q = False
    _signal_by = "deterministic"
    # THE BUG THIS FIXES: a real call had the caller say a genuine wind-down ("ठीक है
    # धन्यवाद", Devanagari "ok thanks") that the understanding LLM's own `conversational_signal`
    # self-report separately read as a "backchannel" nod -- its own misclassification, since a
    # caller winding the call down and a caller nodding along to keep it going are mutually
    # exclusive readings of the same utterance, never both at once. Left unguarded, this fed a
    # "continue the agent's last line" framing into the responder, which produced an unrelated
    # presence-check reply ("I'm here, you can hear me now...") instead of letting the normal
    # wind-down path (closure.reads_like_no_further_need, checked further down) answer it.
    # Excluded here, at the one place both _det_signal and the LLM's self-report are read, so
    # neither path can ever win against the caller's own closure/wind-down words.
    # Also not on a consent hold turn: its line is already decided (`_consent_pin`), and this
    # block's own fallback would relabel the turn "respond" and drop the parked action.
    if (action not in ("end_call", "escalate") and not _force_project_clarify
            and not _pitch_handled
            and not closure.reads_like_no_further_need(user_input)
            and not closure.caller_signalled_closure(user_input)
            and _consent_hold is None):
        _prev_agent = session.get("last_agent_reply")
        if _prev_agent is None:
            # Turn one: the opener was spoken (and persisted) outside process_turn, so read it
            # back from the transcript — a first-turn "can you hear me?" still resumes the opener.
            _prev_agent = (await db.execute(
                select(Message.content).where(
                    Message.conversation_id == conversation.id, Message.role == "agent")
                .order_by(Message.created_at.desc()).limit(1))).scalars().first()
        _signal_asked_q = conversational_signal.ends_with_question(_prev_agent)
        # Anything that could be waiting on a literal "yes". If any is live, an ambiguous
        # agreement word stays with the normal affirmation path; the unambiguous continuers
        # ("mm-hmm", "go on") are read as a nod regardless.
        _pending = bool(
            session.get("pending_action") or session.get("offered_slots")
            or session.get("awaiting_project_choice") or session.get("awaiting_visit_name")
            or int(session.get("reengage_count", 0) or 0) or accepted_handoff)
        _det_signal = conversational_signal.classify(
            user_input, agent_asked_question=_signal_asked_q,
            is_opening_turn=(int(session.get("turn_index", 0)) == 0), pending=_pending)
        _signal = _det_signal
        # Hybrid second pass: when the fast deterministic reader finds nothing, fall to the LLM's
        # semantic read from the understanding call (multilingual / long-tail — "aap sun paa rahe
        # ho?"). The same consent guard is re-applied so a mislabelled pending "yes" is never a nod.
        if _signal == conversational_signal.SIGNAL_NONE:
            _llm_signal = str((proposal.routing or {}).get("conversational_signal") or "none")
            if _llm_signal == "check_in":
                _signal = conversational_signal.SIGNAL_CHECK_IN
            elif _llm_signal == "backchannel" and not _signal_asked_q and not _pending:
                _signal = conversational_signal.SIGNAL_BACKCHANNEL
        if _signal == conversational_signal.SIGNAL_CHECK_IN:
            _signal_kind = "check_in"
        elif _signal == conversational_signal.SIGNAL_BACKCHANNEL:
            _signal_kind = "backchannel"
        _signal_by = "deterministic" if _det_signal != conversational_signal.SIGNAL_NONE else "llm"

    astate = _action_state(action, outcome, session)
    staged_desc = None
    blocked_reason = None
    if astate == "staged":
        _p = session.get("pending_action") or {}
        _when = _staged_when(session, _p.get("params"))
        # A reschedule staged with the CURRENT time (the customer agreed to "reschedule"
        # before naming a new time) must not read back that identical time — "shall I move it
        # to 5 PM?" when 5 PM is what they already have. Drop the time so the responder asks
        # for the new one; Rule 4 blocks the same-time execution regardless.
        if (_p.get("tool") == "reschedule_appointment"
                and bt._same_instant((_p.get("params") or {}).get("slot_start"),
                                     session.get("appointment_current_start"))):
            _when = None
        staged_desc = {"verb": _STAGED_VERB.get(_p.get("tool"), "do that"), "when": _when}
    elif astate == "blocked" and outcome is not None:
        blocked_reason = (outcome.failure_action or {}).get("template") or outcome.error
    elif astate == "none" and proposal.rule_blocked:
        # THE BUG THIS FIXES: dnc_state_stops_everything (and any other rule-engine BLOCK
        # with scope "all") fires before any tool is ever attempted, so outcome stays None
        # and _action_state's own "blocked" branch (which needs outcome) never engages --
        # the responder reached the rest of this function with no signal at all that
        # anything was blocked, and improvised: a fabricated scope claim ("we don't have
        # rentals") on one real call, re-pitching a lead who had just asked to be left
        # alone on another. "policy_blocked" is deliberately its own action_state, not
        # reused "blocked" -- that branch's own wording ("ask the customer for exactly what
        # is needed to clear it") assumes a missing precondition the customer can supply,
        # which is wrong here: nothing about a DNC/compliance block is the caller's to fix.
        astate = "policy_blocked"
        blocked_reason = proposal.rule_blocked.get("reason")
    # P0 — the grounded responder is the ONLY path to the customer.
    #
    # The PRD requires the customer-facing reply to be regenerated AFTER execution, from
    # verified results only. `proposal.response_draft` is written during PLANNING — before
    # any tool ran — so emitting it raw is how "I've booked your visit" reaches a caller on a
    # turn where nothing executed. So the draft is demoted to an internal tone hint and is
    # never emitted verbatim: every reply goes through the grounded sentence-writer, which is
    # handed the deterministic ACTION STATE (done/staged/blocked/failed) and cannot claim a
    # completion for an action that was merely parked or blocked. This is the structural fix
    # that let the whole post-hoc claim/promise regex net be removed — with no ungrounded
    # draft ever reaching the caller, there was nothing left for it to police.
    # For read-only tools, pass sentence_callback so TTS can begin on the first
    # sentence while the responder LLM is still generating the rest. Write tools are
    # excluded — the full grounding check must finish before any completion claim plays.
    #
    # P2.02 (REVERTED — see THE BUG THIS FIXES below): a plain "respond" turn with no
    # tool at all used to get this same treatment when stream_respond_no_tool_enabled is
    # on, on the claim that it was "safe by the same reasoning as the tool-turn case
    # above": grounding_check/scope_claim/booking_claim/etc. below still validate the
    # FULL text regardless of what streamed early, and bridge.py replays everything if
    # the validated text ends up differing from what was pre-pushed.
    #
    # THE BUG THIS FIXES: that reasoning is wrong about what "replay" actually fixes. A
    # real call hit it directly -- decision_traces + raw TTS logs for the same turn:
    #   12:28:25.108  Generating TTS [Would you like me to move your visit to a morning
    #                 slot, perhaps 10 AM tomorrow?]            <- streamed live, SPOKEN
    #   12:28:25.139  voice turn done response='Of course -- may I have the name the
    #                 visit is booked under, so I can pull it up for you?'  <- the REAL,
    #                 validated text, 31ms later, completely different
    # bridge.py's dedup only SKIPS re-speaking a final sentence that matches one already
    # streamed -- replaying the correct text when nothing matches does not un-speak the
    # wrong sentence the caller already heard. The caller heard both: the responder's own
    # unvalidated guess, then the real answer read right after it. This is exactly the
    # same failure mode the phone-capture fast-path above was built to avoid, just hit
    # here through the ordinary respond-with-no-tool path instead -- grounding_check,
    # scope_claim and booking_claim (is_ungrounded_visit_claim /
    # is_ungrounded_cancellation_claim) all run AFTER this point and can rewrite the text
    # wholesale, and a plain "respond" turn is exactly the shape that reaches them.
    #
    # The read-only-tool-success case just below is left untouched -- that text is
    # grounded in a VERIFIED tool result rather than free-form generation, a materially
    # different risk profile, and nothing in this investigation found it misbehaving.
    _no_tool_respond_ok = False
    _pipeline_cb = (
        sentence_callback
        if (sentence_callback is not None
            and (_no_tool_respond_ok
                 or (outcome is not None
                     and outcome.tool_name in _READ_ONLY_PIPELINE_TOOLS
                     and outcome.status in TOOL_SUCCEEDED)))
        else None
    )
    # P3.04 — RESPONDER TEMPLATE CACHE (the ~150ms RSP saving on write-tool confirmations).
    #
    # When this turn's outcome is a write/confirm tool that GENUINELY completed (action state
    # "done"), its customer-facing reply is deterministic — every fact is in outcome.output.
    # templated_reply.render() builds that confirmation directly with NO model call; the
    # responder LLM is skipped entirely on those turns. It returns None (→ LLM path runs,
    # unchanged) the instant anything is off: non-success, staged/blocked, no template,
    # a missing field, or a reframing signal (presence-check, pitch, project-clarify, outbound
    # lead) that genuinely needs the model. Gated on RESPONDER_TEMPLATE_CACHE_ENABLED.
    # CRUCIAL: every post-responder guard still runs on the templated text, unchanged.
    _templated = (
        templated_reply.render(
            outcome,
            action_state=astate,
            language=responder.detect_language(user_input),
            reframing=bool(_signal_kind or _pitch_directive
                           or _force_project_clarify or _outbound_lead_text),
        )
        if get_settings().responder_template_cache_enabled
        else None
    )

    # Everything a measured figure in the reply may legitimately come from (figure_claim.py):
    # the passages and tool results the responder is given, the curated facts, the caller's
    # visits on file, the caller's own words and our previous line. Built ONCE and handed to
    # the responder's stream and to the final pass below, so both judge every figure alike.
    _figure_grounding = figure_claim.grounding_text(
        knowledge=(proposal.knowledge_used or []) + (_pitch_knowledge or []),
        tool_outputs=[o.output for o in executed if getattr(o, "output", None)]
                     + ([outcome.output] if outcome is not None and outcome.output else []),
        project_facts=proposal.project_facts,
        extra=[user_input or "", _prev_agent or "",
               json.dumps(proposal.existing_appointments or [], default=str)])

    _t = time.perf_counter()
    if _fast_path_text is not None:
        text = _fast_path_text
        resp_meta: dict = {"verifier_ok": True, "tokens": {}, "grounded": False,
                           "model": "fast_path"}
    elif _force_phone_capture_reply is not None:
        # Only set below at the registration flow's "could I have your mobile number" ask
        # (action="clarify" -- an active capture turn goes through _fast_path_text above
        # instead, hoisted before plan() entirely). Same early-exit shape as
        # _fast_path_text/_templated, and for the same reason: the responder would get
        # overridden by this exact text at the end of the function regardless (see
        # _force_name_ask's identical pattern just above), so running it first is pure
        # waste -- skip it and keep the saving.
        text = _force_phone_capture_reply
        resp_meta = {"verifier_ok": True, "tokens": {}, "grounded": False,
                     "model": "phone_capture"}
        timing["responder_ms"] = 0
    elif _consent_pin is not None and action == "confirm":
        # A consent hold turn speaks a deterministic line built from the staged data
        # (consent.readback), so the responder call is skipped — pinned again below anyway.
        text = _consent_pin
        resp_meta = {"verifier_ok": True, "tokens": {}, "grounded": False,
                     "model": "consent_hold"}
        timing["responder_ms"] = 0
    elif _templated is not None:
        text, resp_meta = _templated.text, _templated.meta
        timing["responder_ms"] = 0
    else:
        # tool_name comes from `outcome`, never from `proposal.tool` — they are NOT
        # always the same tool.
        #
        # THE ROOT CAUSE THIS FIXES: a chain that proposes a write tool requiring
        # confirmation does not execute it — book_appointment gets PARKED, `outcome`
        # stays whatever the last tool that actually ran was (search_slots, still
        # "success"), but `proposal.tool` has already moved on to "book_appointment".
        # Passing proposal.tool here told the responder "book_appointment: status
        # success, result: <search_slots' slot list>" — a real success record for a
        # tool that never ran, paired with someone else's output. It read the sentence
        # any reasonable model would write from that: a completed booking. Feeding the
        # responder a result for a tool that did not run was the disease behind the whole
        # "claimed a completed book" family; the ACTION STATE above is the cure.
        text, resp_meta = await responder.generate(
            # On a pitch-ladder turn the canned rung line is the tone hint (and the fallback if
            # this call fails); otherwise the planner's own draft. The pitch directive below tells
            # the model to write the GROUNDED version of it from the force-retrieved material.
            draft=(_pitch_draft if _pitch_draft is not None else proposal.response_draft),
            tool_name=outcome.tool_name if outcome else None,
            tool_status=outcome.status if outcome else None,
            tool_output=_hide_from_responder(outcome.output) if outcome else None,
            tool_error=outcome.error if outcome else None,
            failure_action=outcome.failure_action if outcome else None,
            # Merge any pitch-rung material force-retrieved above into the knowledge the responder
            # grounds on, so the pitch cites real configurations/sizes/price.
            knowledge=((proposal.knowledge_used or []) + _pitch_knowledge
                       if _pitch_knowledge else proposal.knowledge_used),
            persona=await _persona(db, tenant_id),
            asked=user_input,
            # The planner's resolved reading of this turn (intent + entities), so the reply is
            # framed around what the customer means — not just the literal words they used.
            understood=proposal.understood,
            # The caller's visit(s) on file, so the final sentence-writer never offers a NEW visit
            # to someone already booked and can point them to the one they have on a related turn.
            existing_appointments=proposal.existing_appointments,
            # Tenant-declared equivalences, so the reader reliably bridges domain vocabulary it
            # might not know ("DG"/"power backup" ≈ "generator") when it grounds a "yes" by meaning.
            synonyms=proposal.retrieval_synonyms,
            # Curated always-answerable facts (RCC structure, location, RERA) as VERIFIED material,
            # so an answer that lives only in the curated set is no longer stripped as ungrounded.
            project_facts=proposal.project_facts,
            # The resolved project this call is about, so the reply never applies another project's
            # policy/scope/price/configurations to it (responder Rule 15) — the reasoning-layer
            # backstop to retrieval's project scoping.
            project_name=getattr(proposal, "project_name", None),
            action_state=astate, staged=staged_desc, blocked_reason=blocked_reason,
            # Presence-check / nod: tell the responder to CONTINUE the agent's last line in persona
            # instead of answering the literal words. Folded into this existing call — no extra
            # round trip. None on every ordinary turn, so behaviour there is byte-for-byte unchanged.
            continuation=({"kind": _signal_kind, "last_line": _prev_agent}
                          if _signal_kind else None),
            # Outbound pitch rung: LEAD with the force-retrieved facts and end on the forward
            # question. None on every non-pitch turn, so behaviour there is byte-for-byte unchanged.
            pitch=_pitch_directive,
            # Advisory custom instructions, already scope-filtered and rendered by the planner
            # (app/custom_instructions). The subordinate TENANT PREFERENCES block that shapes
            # tone/priority in the sentence the caller actually hears. None when none apply.
            tenant_preferences=getattr(proposal, "tenant_preferences", None),
            figure_grounding=_figure_grounding,
            sentence_callback=_pipeline_cb)
    if _fast_path_text is None and _templated is None:
        timing["responder_ms"] = int((time.perf_counter() - _t) * 1000)

    timing["responder_source"] = (resp_meta or {}).get("source", "llm")
    if timing["responder_source"] == "template":
        timing["responder_template"] = (resp_meta or {}).get("template_id")

    if not text or not text.strip():
        text = "Sorry, could you repeat that?"

    # Pin the forced project-disambiguation question (multi-project-per-tenant). The grounded
    # responder has just written SOMETHING for this clarify turn, but the caller must hear the
    # exact 'which project?' question — with each project's place — not a paraphrase that might
    # slip back into naming one. Same late-override discipline as FORCED_HANDOVER below.
    if _force_project_clarify:
        text = _force_project_clarify

    # Pin the name-ask the same way, and for the same reason: a live retest of the real bug
    # this fixes showed the responder discard "book your visit" and substitute "send you the
    # 3 BHK floor plan" as the reason for taking the caller's name, pulling the wrong topic
    # from conversation history even with a goal-grounded draft. Getting this question's
    # purpose right is the highest-stakes line in the registration flow — it is the exact
    # question whose answer becomes the customer record — so it is never left to a rephrase.
    if _force_name_ask:
        text = _force_name_ask

    # Pin every chunked-phone-capture reply the same way: a digit-by-digit read-back
    # ("9-8-4-5-0") or a grouped confirm ("Is 98450 12367 correct?") must reach the caller
    # exactly as phone_capture.py produced it. The responder has no idea this mode is
    # active and, left to it, "sounds natural" easily means turning a bare digit read-back
    # into a full sentence or silently dropping it as uninformative.
    if _force_phone_capture_reply:
        text = _force_phone_capture_reply

    # Pin the outbound lead-forward reply over whatever the responder wrote for the suppressed
    # escalation (see the outbound lead guard above). The responder demotes response_draft to a
    # tone hint and regenerates from the — now empty — action state, so the deterministic
    # lead line is applied here as a late override, the same discipline as the identity gate.
    if _outbound_lead_text:
        text = _outbound_lead_text

    # Pin the consent hold line last among these: on a hold turn nothing ran and the parked
    # action is the only open question, so the caller hears exactly that question — never a
    # rephrase that drops the time they asked to hear again. Only while the hold still owns
    # the turn (action stays "confirm"); an escalation or goodbye decided later wins.
    if _consent_pin is not None and action == "confirm":
        text = _consent_pin

    # ---- outbound, identity already confirmed: a mid-call "what is this call about?" is
    #      answered with WHY WE CALLED — for a SERVICE call, the client's existing commitment
    #      (their booked visit) — never with a project brochure. The fast classifier reads the
    #      question as a product enquiry and the planner reaches for the project sheet (the exact
    #      reported bug: a service-call "what is it about?" got answered with the project's
    #      location, acreage and configurations). This pins the purpose disclosure over whatever
    #      the responder wrote. Skipped when the multi-project disambiguation above already
    #      handled the same question; gated on outbound + CONFIRMED, so inbound and pre-identity
    #      behaviour are byte-for-byte unchanged. The tenant lookup runs only when the question
    #      actually fires (rare), so it adds no per-turn cost to the common path.
    if (not _force_project_clarify
            and session.get("outbound")
            and session.get("outbound_stage") == outbound_flow.STAGE_CONFIRMED
            and outbound_flow.is_purpose_question(user_input)):
        _trow = (await db.execute(
            select(Tenant).where(Tenant.id == tenant_id))).scalar_one_or_none()
        _appt = (getattr(proposal, "existing_appointments", None) or [None])[0]
        _voc = await _tenant_vocabulary(db, tenant_id)
        text = outbound_flow.purpose_disclosure(
            session.get("call_purpose"), getattr(_trow, "name", None),
            commitment=outbound_flow.commitment_phrase(_appt, _voc),
            interest=getattr(proposal, "enquiry_interest", None),
            project=(getattr(proposal, "enquiry_project", None)
                     or getattr(proposal, "project_name", None)))
        action = "respond"
        proposal.error = ((proposal.error + " | ") if proposal.error else "") + \
            "Outbound purpose question after confirmation; disclosed the call's purpose."

    # The one fact that matters most, checked against what the tool actually did — not
    # against what the model wrote.
    #
    # THE BUG THIS FIXES: slot_was_offered (contracts.py) already guarantees
    # book_appointment can only ever succeed against a real, previously-offered time —
    # an invented slot_start is refused before the database is ever touched. Nothing
    # guaranteed the SENTENCE confirming it back to the customer named that same day.
    # A model with "something on the 19th" fresh from the customer's own words, and a
    # genuinely-booked slot_start for an earlier day sitting right there in the tool
    # result, described the customer's wish instead of the database row: "your
    # appointment is all set for the nineteenth of August at ten in the morning" — a
    # real, valid, successfully booked appointment, just for the 15th, not the 19th.
    # She would have arrived on a day nobody was expecting her.
    #
    # Grounding the reply on the ACTION STATE is powerless against this: the action really
    # did complete (state is "done"), so nothing about the sentence looks like a false
    # claim. It correctly describes A booking — just not the one that happened. Only a
    # deterministic check of the reply against the tool's own output catches it, which is
    # why this stays even though the responder is now the only path to the customer.
    if (outcome is not None
            and outcome.tool_name in ("book_appointment", "reschedule_appointment")
            and outcome.status in TOOL_SUCCEEDED):
        _real_start = (outcome.output or {}).get("starts_at")
        _real_date = _slot_date(_real_start)
        if _real_start and _real_date is not None:
            _dt = datetime.fromisoformat(_real_start)
            # The DATE the reply names, and the HOUR the reply names, both checked against
            # the slot that was actually booked. The date guard alone let a wrong TIME
            # through: the customer asked for 3 PM, 3 PM was taken, the engine correctly
            # booked the real free slot (2 PM), but the model still SAID "3 PM" — same day,
            # so the date check passed, and the customer was told a time that was not theirs.
            _said_date = _spoken_day(text)
            _said_hour = _wanted_hour(text)
            _date_wrong = _said_date is not None and _said_date != _real_date
            _time_wrong = _said_hour is not None and _said_hour != _dt.hour
            if _date_wrong or _time_wrong:
                _hour12 = _dt.hour % 12 or 12
                _ampm = "AM" if _dt.hour < 12 else "PM"
                text = (f"Your visit is confirmed for {_dt:%A, %d %B} at "
                        f"{_hour12}:{_dt.minute:02d} {_ampm}, at the site office.")
                _wrong = []
                if _date_wrong:
                    _wrong.append(f"date (said {_said_date}, booked {_real_date})")
                if _time_wrong:
                    _wrong.append(f"time (said {_said_hour}:00, booked {_dt.hour}:00)")
                _note = (f"Responder stated the wrong {' and '.join(_wrong)}; "
                         f"replaced with the real booked slot.")
                proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note

    # ---- an affirmed feature must be one the material actually states (Rule 13) ----
    #
    # THE BUG THIS FIXES: asked "is there a park-facing flat?", the agent — once the 3 BHK was
    # in focus and its floor-plan chunks ("east and west facing", near amenity text about a
    # trampoline park) were retrieved — answered "yes, there are park-facing 3 BHKs". No unit is
    # park-facing; the model stretched adjacent words into a feature the data never states.
    #
    # The reply is grounded by MEANING, not by string matching: the responder reads the query and
    # the chunks semantically (Rule 13 — bridging synonyms and paraphrase, "power backup" IS the
    # "generator") and self-assesses whether they support its answer; grounding_check acts on that
    # verdict. This is deliberately NOT a deterministic synonym check — an earlier one wrongly
    # suppressed a true "yes" about the generator because the model cited the caller's words, not
    # the document's. An affirm the model itself marks ungrounded (or a verifier outage with no
    # verbatim proof) is replaced with an honest "I can't confirm that".
    _grounding = fg.grounding_text(proposal.knowledge_used,
                                   outcome.output if outcome is not None else None,
                                   proposal.project_facts)
    _verdict = grounding_check.check(resp_meta, _grounding)
    if not _verdict.ok:
        _note = f"Rule 13 (semantic grounding): {_verdict.reason}; replaced with honest fallback."
        proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note
        text = UNSUPPORTED_FEATURE_FALLBACK.get(responder.detect_language(user_input),
                                                UNSUPPORTED_FEATURE_FALLBACK["english"])

    # ---- a rent/lease/sale-scope boundary must be in the material, not invented from silence ----
    #
    # THE BUG THIS FIXES: asked "what's the approx rent?", the agent said "we only sell and do
    # not offer rentals" — a policy invented from the absence of any rent passage. The seeded
    # facts say NOTHING about sale-vs-rent, so nothing grounds it.
    #
    # DESIGN — detection and grounding are split by who is reliable at each:
    #   * DETECTION (is this sentence a scope claim?) is a UNION of two signals — the responder's
    #     own semantic flag (Rule 14, scope_claim=="closed"), which catches paraphrases, AND a
    #     regex over the sentence, which catches the common phrasings even on an LLM outage.
    #   * GROUNDING (is that scope actually in the source?) is decided ONLY in code, against the
    #     real material — the model is never trusted here, because it rationalises "we only sell"
    #     as obviously true and marks a fabrication grounded. A scope claim stands ONLY when the
    #     material contains rent/lease/sale vocabulary; absent it, the claim has no source.
    # This makes the outcome independent of how the CALLER phrased the question and of the model's
    # (unreliable) grounding opinion: any turn that yields a fabricated scope denial converges on
    # the same honest, callback-offering line.
    _claims_scope = (scope_claim.asserts_closed_scope(text)
                     or (resp_meta or {}).get("scope_claim") == "closed")
    if text and _claims_scope and not scope_claim.scope_declared(_grounding):
        _note = ("Reply declared a rent/lease/sale-scope boundary the material never states "
                 "(no rent/lease/sale vocabulary in facts, knowledge or tool output); replaced "
                 "rather than let a fabricated policy stand.")
        proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note
        # THE BUG THIS FIXES: this fallback's own "I can have our sales team call you back"
        # is itself a re-contact offer -- fine on an ordinary turn, but on a turn a policy
        # rule (e.g. DNC) just blocked, offering a callback hands the exact thing this
        # caller opted out of straight back to them in the very same breath.
        _scope_fallback = (UNGROUNDED_SCOPE_FALLBACK_POLICY_BLOCKED if proposal.rule_blocked
                           else UNGROUNDED_SCOPE_FALLBACK)
        text = _scope_fallback.get(responder.detect_language(user_input),
                                   _scope_fallback["english"])

    # ---- a policy-blocked turn (e.g. DNC) may not pitch further or offer re-contact ----
    #
    # THE BUG THIS FIXES: a real call had the DNC rule fire, then the model answered a
    # genuinely grounded factual question correctly ("we only sell, no rentals") but tacked
    # on "Would you like to hear about the apartments we have for purchase?" anyway -- the
    # prompt instruction (responder.py's "policy_blocked" ACTION STATE block, added
    # alongside this guard) tells the model not to, but a prompt instruction is a request,
    # not a guarantee, same reasoning as every other guard in this module. Caught here
    # regardless of whether the claim above also fired, since the forward-pitch can ride on
    # an otherwise-honest, correctly-grounded sentence.
    if text and proposal.rule_blocked and _FORWARD_PITCH_RE.search(text):
        text = _strip_forward_pitch(text, language=responder.detect_language(user_input))
        _note = ("Reply pitched further or offered re-contact on a turn a policy rule "
                 "blocked (e.g. DNC); the forward-pitch clause was dropped.")
        proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note

    # ---- the agent may not claim the caller has a visit that does not exist ----
    #
    # THE BUG THIS FIXES: an UNIDENTIFIED caller ("I'd like to see the 3BHK") was told "you
    # already have a visit booked", and then "your visit is confirmed for Saturday at 2 PM" —
    # a booking with an invented day and time that the caller never made. With no customer_id,
    # existing_appointments is empty and nothing was looked up, so there is no such visit; the
    # model conjured one from the booking intent. Unlike a wrong-slot claim there is no tool
    # behind it at all, so ACTION STATE cannot catch it — only a check of the SENTENCE against
    # whether an appointment genuinely exists.
    #
    # Grounded ONLY by ground truth: a row in existing_appointments (a real on-file visit for
    # an identified caller), or an appointment a lookup/booking actually established this turn.
    # A my_appointments that returned NOTHING does not count. When neither holds and the reply
    # asserts an on-file visit anyway, it is replaced with an honest line that invents nothing.
    _appt_grounded = bool(proposal.existing_appointments)
    if not _appt_grounded:
        for _o in executed:
            if (_o.status in TOOL_SUCCEEDED
                    and _o.tool_name in ("my_appointments", "book_appointment",
                                         "reschedule_appointment")):
                _out = _o.output or {}
                if (_out.get("appointment_id") or _out.get("starts_at")
                        or _out.get("live") or _out.get("upcoming")
                        or _out.get("past") or _out.get("appointments")):
                    _appt_grounded = True
                    break
    if booking_claim.is_ungrounded_visit_claim(text, has_appointment_on_file=_appt_grounded):
        _intent = str((proposal.understood or {}).get("intent") or "").lower()
        _wants_book = (_intent in ("book_appointment", "book", "schedule_visit",
                                   "reschedule", "reschedule_appointment")
                       or _turn_hint(user_input, proposal) in ("book_appointment",
                                                       "reschedule_appointment"))
        text = booking_claim.honest_fallback(wants_to_book=_wants_book,
                                             language=responder.detect_language(user_input))
        _note = ("Reply claimed the caller has a visit on file, but none exists "
                 "(existing_appointments empty and no lookup/booking established one); "
                 "replaced with an honest line rather than invent a booking.")
        proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note

    # ---- the agent may not claim a visit was CANCELLED unless it actually was this turn ----
    #
    # THE BUG THIS FIXES: a real call had the planner correctly decide action="respond", no
    # tool called, reasoning "customer declined location pin offer" -- the responder (a
    # separate LLM call that writes the final spoken text from the conversation's context, not
    # just the planner's draft) independently said "Your visit... has been cancelled",
    # resolving an unanswered "keep it or cancel it?" question from several turns earlier on
    # its own. The real appointment was untouched in the database. Mirror-image of the
    # ungrounded-booking-claim guard above, in the opposite direction: grounded ONLY by a
    # cancel_appointment that actually succeeded THIS turn, never guessed from conversation
    # history.
    _cancel_grounded = any(
        _o.status in TOOL_SUCCEEDED and _o.tool_name == "cancel_appointment"
        for _o in executed)
    if booking_claim.is_ungrounded_cancellation_claim(text, cancellation_grounded=_cancel_grounded):
        text = booking_claim.cancellation_honest_fallback(
            language=responder.detect_language(user_input))
        _note = ("Reply claimed the visit was cancelled, but no cancel_appointment succeeded "
                 "this turn; replaced with an honest line stating it is still booked rather "
                 "than let a fabricated cancellation stand.")
        proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note

    # ---- the agent may not promise, perform, or ask to confirm an email send ----
    #
    # THE BUG THIS FIXES: asked "can you send the details on email?", the reply was "Sure, I
    # can arrange for the details to be emailed from sales@dsrinfra.com. Could you please
    # confirm the email address you'd like us to use?" -- there is no email tool anywhere
    # (send_document only sends over WhatsApp), so both the invented sender address and the
    # promise itself are fabricated. Unlike the booking/cancellation claims above there is no
    # grounding state to check: email is never available, so this runs unconditionally.
    if email_claim.is_fabricated_email_claim(text):
        text = email_claim.honest_fallback(language=responder.detect_language(user_input))
        _note = ("Reply promised or claimed an email send, but no email capability exists "
                 "anywhere in the tool layer; replaced with an honest WhatsApp offer rather "
                 "than let a fabricated email claim stand.")
        proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note

    # ---- the agent may not tell the caller that our own documents disagree ----
    #
    # THE BUG THIS FIXES: a real call said "there's a slight size-range discrepancy between
    # the brochure and cost sheet", then, asked "what is the discrepancy?", read both figures
    # out. Pushed back with the other figure, it still said "the brochure might be outdated"
    # after the prompt rule existed. Cuts the leaking clause (or sentence); the honest fixed
    # line only when nothing clean is left. Same function the responder's stream applies per
    # sentence, so the streamed and validated text stay identical. See source_conflict_claim.py.
    _scrubbed = source_conflict_claim.scrub(text, asked=user_input)
    if _scrubbed != text:
        text = _scrubbed or source_conflict_claim.honest_fallback(
            language=responder.detect_language(user_input))
        _note = ("Reply exposed a conflict between internal documents to the caller; the "
                 "leaking part was removed" + ("" if _scrubbed else
                                               " and the reply replaced with an honest line") + ".")
        proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note

    # ---- every measured figure must come from the material ----
    #
    # THE BUG THIS FIXES: with no distance in front of it, the reply said "approximately 5
    # kilometers from Wipro" (and "seven kilometres" on another call); the KB says 2.5 km.
    # Any distance/duration/area/percentage not supported by the same grounding text the
    # responder's stream was checked against is cut and replaced by an honest "I don't have
    # the exact …" line — never by a figure of our own. See figure_claim.py.
    _unsupported = figure_claim.unsupported(text, _figure_grounding)
    if _unsupported:
        text = figure_claim.scrub(text, _figure_grounding,
                                  language=responder.detect_language(user_input))
        _note = ("Reply stated figure(s) the material does not support ("
                 + ", ".join(q.text for q in _unsupported) + "); removed.")
        proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note

    # ---- an unrecognizable utterance near a real booking may not be read as a decision ----
    #
    # THE BUG THIS FIXES: two real calls had the model read a meaningless, garbled word --
    # "noni", "typique" -- as an instruction about the caller's existing visit. "noni" was
    # classified "customer says no" at the model's own self-reported confidence 0.99 (so a
    # low-confidence gate would never have caught it -- the model was confident, and wrong),
    # and the reply pivoted out of nowhere into "would you like to keep it, cancel, or
    # reschedule?" See app/engine/planner/unclear_input.py for the detection rule.
    #
    # Gated on the INPUT alone, not on pattern-matching the reply text: a retest of this exact
    # fix showed the model phrase the identical mistake a different way next time ("I've noted
    # you won't be coming to the visit tomorrow. Would you like to reschedule it?" -- only one
    # of the three menu words, so a reply-text pattern match missed it). The utterance being
    # unrecognizable is the one stable signal; nothing reliable can be said about how the
    # model will phrase its guess, so the guard must not depend on that phrasing at all.
    if _appt_grounded and unclear_input.looks_unrecognizable(user_input):
        text = unclear_input.ask_to_repeat(responder.detect_language(user_input))
        _note = (f"Reply drew a conclusion about the booking off an unrecognizable "
                 f"utterance ({user_input!r}); replaced with a request to repeat rather "
                 f"than guess.")
        proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note

    # ---- a spoken "Weekday, D Month" must name the day that date actually falls on ----
    #
    # THE BUG THIS FIXES: a real call had the agent ask "Would 5 PM on Monday, 10 October
    # work for your evening visit?" when 10 October that year is a Saturday -- the SAME call
    # called it "Saturday, 10 October" both one turn before and one turn after. No tool ran
    # on the turn that got it wrong (see date_claim.py for the full trace), so the date/time-
    # vs-tool-output check above never had a tool result to catch it against. A weekday is a
    # pure function of its date, so this needs no ground truth and runs on every reply.
    text, _weekday_notes = date_claim.correct_weekday_date_claims(text)
    if _weekday_notes:
        _note = "Weekday/date mismatch corrected: " + "; ".join(_weekday_notes)
        proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note

    # ---- a scheduling turn may not end on a promise ----
    #
    # The backstop to Rule 1c above. That force makes the agent DO the lookup on turns where
    # one can be done; this catches the residue where none can — a multi-site tenant whose
    # caller has not chosen a site yet, so locate_centre has already run and search_slots has
    # nothing to search against. The responder is told never to promise a future action
    # (responder.py rule 2), but with no tool result and no knowledge to ground on it has
    # nothing else to write, and "let me check the times for you" is what comes out.
    #
    # On a phone call that sentence is not a small stylistic problem: the agent cannot speak
    # again until the caller does, so a promise costs them an utterance and returns nothing.
    # Replaced with a question they can actually answer, which is what turns a dead turn back
    # into a live one. Deliberately narrow — it fires only when NO tool ran this turn, the
    # turn was about scheduling, and the sentence itself promises a scheduling lookup.
    #
    # NB this reads the AGENT'S OWN sentence, not the caller's. It is not a revival of the
    # deleted routing regexes (which tried to infer a caller's MEANING from their words); it
    # detects an engine fault in text the engine just wrote, where the phrasings are few and
    # ours to recognise.
    if (not executed and action not in ("escalate", "end_call", "confirm")
            and _turn_is_about_scheduling(user_input, proposal, session)
            and sp.reads_like_lookup_promise(text)):
        text = sp.stalled_reply(centre_known=bool(session.get("centre_id")))
        _note = ("Reply promised a scheduling lookup on a turn where no tool ran; replaced "
                 "with a question the caller can answer rather than a progress report.")
        proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note

    # ---- end the call deliberately: honour a farewell, but cap the "anything else?" loop ----
    #
    # end_call is the model's decision alone, and it kept ending the call when the caller merely
    # declined an OFFER — "no" to "shall I book a visit?" is not "goodbye". Hanging up cannot be
    # taken back, so a bare "no" never ends the call. But the earlier fix (downgrade EVERY
    # unproven end_call to "anything else?") had the opposite failure: a caller saying "ok thanks"
    # then "no" then "no" was asked "anything else?" forever, because none of those are the hard
    # farewell the guard demanded. Neither extreme is right; the human pattern is a BOUNDED
    # wind-down — ask once, then sign off and let the caller interrupt if they still have a need.
    #
    # The whole policy lives in closure.decide_end_call — one pure, unit-tested function — so the
    # transcript, the returned result and the tests can never drift from it. It resolves the
    # proposed hang-up into: whether to end, the reply to speak, the new wind-down counter, and a
    # trace note. Behaviour, highest precedence first:
    #   * hard farewell ("bye", "that's all", "you can hang up") -> end immediately (turn one).
    #   * OUTBOUND, right person reached, no live question, unambiguous disinterest
    #     ("not interested", "not looking to buy") -> close gracefully turn one (operator policy:
    #     accept the decline, never push) — this is what stops the "anything else?" loop pestering
    #     a caller who already declined.
    #   * soft wind-down (short ack / bare "no", no question) -> ask "anything else?" ONCE, then
    #     sign off and end on the next one. The counter is unified across every path that can ask,
    #     so the loop is bounded to a SINGLE re-engage (fixes the leak where a suppressed sign-off
    #     asked without counting and a later "no" asked again — the double-ask this replaces).
    #   * no closure signal at all -> keep the call open (a mis-planned end_call on a real turn
    #     must never hang up).
    # The outbound branches are inert unless session["outbound"] is set and identity is CONFIRMED,
    # so inbound behaviour is unchanged. Placed BEFORE the agent message and the "action:" trace
    # are persisted below, so the stored transcript and the returned result agree on both.
    #
    # P1-F13: Make the first AND second soft wind-down detection deterministic.
    #
    # Second wind-down (reengage_count >= 1): "anything else?" was already asked; the model may
    # stay at respond/confirm instead of end_call. Force end_call so decide_end_call can close
    # the bounded loop. Original fix — fires regardless of action type.
    #
    # First wind-down (reengage_count == 0, action == respond): the model missed classifying the
    # caller's initial soft wind-down entirely, so reengage_count was never set. Force end_call
    # here → decide_end_call asks "anything else?" and sets reengage_count = 1. Guarded more
    # tightly than the second-wind-down case:
    #   • action == "respond" only — never override a booking confirm or invoke_tool mid-flight.
    #   • not _scheduling_in_flight — "ok" while slots are offered stays as a booking reply.
    #   • not session.get("pending_action") — no pending confirmation action can be skipped.
    # has_live_question guards both cases: a real question in the same utterance is never suppressed.
    #
    # P6-F2: an UNAMBIGUOUS hard farewell ("bye", "bye bye", "that's all", "you can hang up" --
    # caller_signalled_closure, NOT the softer reads_like_no_further_need below) overrides even
    # an escalate decision. The soft-wind-down promotion just below deliberately excludes
    # escalate, on the reasoning that a bare "no"/"ok" is genuinely ambiguous against a real
    # human-handoff need and the escalation should win. A hard farewell is not that kind of
    # signal -- it is the SAME unambiguous, caller's-own-words evidence call_closure.py's own
    # module docstring describes as the one thing allowed to hang up a call, and it is exactly
    # as strong whether the model's own action that turn was "respond" or "escalate". THE BUG
    # THIS FIXES: a real call had the model read "no i'm not i'm not bye bye" as needing a
    # human and chose escalate; because escalate was excluded here, the "bye bye" in the very
    # same utterance never got checked at all, and the caller was transferred instead of told
    # goodbye. Decided separately from, and before, the soft-wind-down block so a hard farewell
    # is never silently absorbed into the softer check's own gating (reengage_count, scheduling
    # in flight, pending_action) -- none of those conditions are relevant to a caller who has
    # already, unambiguously, said goodbye.
    if (action == "escalate"
            and closure.caller_signalled_closure(user_input)
            and not closure.has_live_question(user_input)):
        action = "end_call"

    _rc = int(session.get("reengage_count", 0) or 0)
    if (action not in ("end_call", "escalate")
            and closure.reads_like_no_further_need(user_input)
            and not closure.has_live_question(user_input)
            and (_rc >= 1
                 or (action == "respond"
                     and not _scheduling_in_flight(session)
                     and not session.get("pending_action")))):
        action = "end_call"
    if action == "end_call":
        _decision = closure.decide_end_call(
            user_input, text,
            outbound=bool(session.get("outbound")),
            identity_confirmed=(
                session.get("outbound_stage") == outbound_flow.STAGE_CONFIRMED),
            reengage_count=int(session.get("reengage_count", 0) or 0),
        )
        if _decision.text is not None:
            text = _decision.text
        if _decision.reengage_count is None:
            session.pop("reengage_count", None)
        else:
            session["reengage_count"] = _decision.reengage_count
        if not _decision.end:
            action = "respond"
        proposal.error = (f"{proposal.error} | {_decision.note}"
                          if proposal.error else _decision.note)
    else:
        # Any turn the agent does NOT try to end is real engagement — reset the wind-down count
        # so a later "no" starts the cap fresh rather than ending on its first pass.
        session.pop("reengage_count", None)

    # ---- outbound: confirm we reached the right person before pursuing the purpose ----
    #
    # WE placed this call, so — per the voice-pipeline spec §3.5 — the agent must confirm it is
    # speaking to the intended person before any pitch or business. The opener already asked
    # ("Am I speaking with X?"); this reads the caller's answer and branches deterministically.
    # A post-plan override (like the visit-status gate below): whatever the planner drafted for
    # an unconfirmed-identity turn is replaced, so the caller never hears a pitch aimed at the
    # wrong person. Runs only until identity is settled — once confirmed / a callback is offered
    # / the call is closing, the normal engine drives the rest.
    _outbound_handled = False
    if session.get("outbound") and session.get("outbound_stage") not in (
            outbound_flow.STAGE_CONFIRMED, outbound_flow.STAGE_CALLBACK,
            outbound_flow.STAGE_CLOSED):
        _tenant_row = (await db.execute(
            select(Tenant).where(Tenant.id == tenant_id))).scalar_one_or_none()
        _biz = getattr(_tenant_row, "name", None)
        _who = getattr(customer, "full_name", None)
        _purpose = session.get("call_purpose") or outbound_flow.PURPOSE_MARKETING
        _branch = outbound_flow.classify_identity_reply(user_input)

        if _branch == outbound_flow.BRANCH_OPT_OUT:
            # "Stop calling me." NOT intercepted: the normal DNC path records the opt-out
            # structurally (record_do_not_contact) exactly as an inbound one, and no rule,
            # campaign or purpose can override it. Letting the planner's reply stand keeps that
            # one path — and one audit trail — for every opt-out however it arrives.
            pass
        else:
            _outbound_handled = True
            if _branch == outbound_flow.BRANCH_CONFIRMED:
                session["outbound_stage"] = outbound_flow.STAGE_CONFIRMED
                session.pop("outbound_identity_tries", None)
                # Lead with — and VERIFY — what the lead already enquired about, when we
                # resolved it (resolve_project's outbound-only enquiry-interest signal): the
                # project AND the unit ("2 BHK") they asked about. The opener recalls it as a
                # question before pitching, so the caller shares the context. Both None → the
                # opener degrades to an honest discovery question rather than a false recall.
                # For a SERVICE call, lead with the client's existing commitment (their booked
                # visit) — a service call confirms/reminds it (PRD), never pitches. None → the
                # service opener falls back to the generic "your enquiry with us" wording.
                _appt = (getattr(proposal, "existing_appointments", None) or [None])[0]
                _enq_proj_c = (getattr(proposal, "enquiry_project", None)
                               or getattr(proposal, "project_name", None))
                _enq_unit_c = getattr(proposal, "enquiry_interest", None)
                _opener_voc = await _tenant_vocabulary(db, tenant_id)
                text = outbound_flow.purpose_opener(
                    _purpose, _biz, _who, _enq_proj_c, _enq_unit_c,
                    commitment=outbound_flow.commitment_phrase(_appt, _opener_voc))
                action = "respond"
                # Arm the gradual pitch ladder when this is a PITCH purpose AND we actually have
                # a registered interest to lead from — the opener is rung one (it asked "…is that
                # right?"), and each later positive reply advances one rung (value → visit offer →
                # booking) instead of repeating the opener or jumping straight to a visit. A
                # service/feedback call, or a discovery opener with nothing on file, does not
                # ladder — the model leads those — so the flag is cleared for them.
                if outbound_flow.is_lead_purpose(_purpose) and (_enq_proj_c or _enq_unit_c):
                    session["pitch_stage"] = outbound_flow.PITCH_VERIFY
                    session["pitch_project"] = (_enq_proj_c or "").strip() or None
                    session["pitch_unit"] = (_enq_unit_c or "").strip() or None
                else:
                    session.pop("pitch_stage", None)
                    session.pop("pitch_project", None)
                    session.pop("pitch_unit", None)
                _note = "Outbound identity gate: confirmed the right person; opened the purpose."
            elif _branch == outbound_flow.BRANCH_WRONG_PERSON:
                session["outbound_stage"] = outbound_flow.STAGE_CLOSED
                text = outbound_flow.wrong_person_reply(_biz)
                action = "end_call"
                # Flag the record for a human to verify — operator policy is flag-for-review,
                # NOT auto-suppress, so a one-off wrong answer never permanently blocks a lead.
                if customer is not None:
                    _tags = list(getattr(customer, "tags", []) or [])
                    if "wrong_number" not in _tags:
                        customer.tags = _tags + ["wrong_number"]
                _note = "Outbound identity gate: wrong person/number; apologised, flagged for review, ended."
            elif _branch in (outbound_flow.BRANCH_UNAVAILABLE, outbound_flow.BRANCH_NOT_NOW):
                session["outbound_stage"] = outbound_flow.STAGE_CALLBACK
                text = outbound_flow.offer_callback_reply(_who)
                action = "respond"
                _note = ("Outbound identity gate: not reachable now (%s); offered a callback."
                         % _branch)
            elif _branch == outbound_flow.BRANCH_GATEKEEPER:
                text = outbound_flow.gatekeeper_reply(_biz, _purpose, _who)
                action = "respond"
                _note = "Outbound identity gate: screening question; disclosed purpose and re-asked."
            else:  # unclear
                _tries = int(session.get("outbound_identity_tries", 0)) + 1
                session["outbound_identity_tries"] = _tries
                if _tries <= 2:
                    text = outbound_flow.reask_identity(_biz, _who)
                    action = "respond"
                    _note = "Outbound identity gate: unclear answer; re-asked to confirm identity."
                else:
                    # Twice unclear — stop re-asking and let the planner's own reply carry the
                    # call rather than looping the same question a third time.
                    _outbound_handled = False
            if _outbound_handled:
                proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note

    # ---- a returning caller's visit status is confirmed by name before it is read back ----
    #
    # An inbound caller is matched to a record by caller ID at connect time (hence the greeting
    # already knows their name), and booking status is then read ONLY against that record. When
    # the caller ID matched the wrong record — or the visit sits under a slightly different one
    # — the agent denied a visit the caller plainly has, then offered a new one. Operator policy
    # for a returning caller: ALWAYS ask the name first, then answer from THEIR OWN record (no
    # cross-record search). The name is a soft identity check — a match answers the status
    # honestly (which may still be "no visit"); a mismatch means we hold the wrong record and
    # must not read its booking back as theirs. Runs last so this deterministic text is final;
    # gated on an identified caller, so unknown callers keep the booking_claim honest fallback.
    # Skipped when the outbound identity gate above already produced this turn's reply.
    if customer is not None and not _outbound_handled:
        _matched = False
        if session.get("awaiting_visit_name"):
            # Turn B — we asked for the name last turn. Read it and answer from the record.
            # Consume the Turn-A question kind on every path so it never leaks to a later turn.
            _asked_assignee = bool(session.pop("visit_q_assignee", False))
            given = _looks_like_a_name(user_input)
            # The name STT hears is not always the name that was said. On a real call a
            # customer named "Kapil" was transcribed as "Capelle", then "Capel" — two
            # honest attempts, neither matching, because the WORD itself was wrong, not
            # the matching logic. The last 4 digits of the registered mobile are a far
            # more reliable thing for STT to get right than an uncommon name, and
            # mismatch_reply() already invites this as the alternative — so a caller who
            # answers with digits (instead of, or after, a name) is accepted the same way
            # a name match is, rather than being sent back to retry the same name a third
            # time.
            _digits = extract_spoken_digits(user_input)
            _on_record = re.sub(r"\D", "", getattr(customer, "phone_normalized", "") or "")
            phone_confirmed = (bool(_digits) and bool(_on_record)
                               and _digits[-4:] == _on_record[-4:])
            if phone_confirmed or given:
                session["awaiting_visit_name"] = False
                _matched = phone_confirmed or visit_identity.names_match(
                    given, getattr(customer, "full_name", None))
                if _matched:
                    session["visit_name_confirmed"] = True
                    # Same reasoning as the scheduling-write success above: confirmed_reply()
                    # states the visit's day/time deterministically here, so a later repeat of
                    # that same information should be trimmed regardless of whether this exact
                    # wording happens to match _VISIT_REMINDER_RE.
                    session["visit_reminder_given_this_call"] = True
                    _appts = proposal.existing_appointments or []
                    text = visit_identity.confirmed_reply(
                        given or getattr(customer, "full_name", None) or "",
                        has_visit=bool(_appts),
                        when=_appts[0].get("when") if _appts else None,
                        # The caller's original question (Turn A) may have been "who is
                        # assigned?" — an assignee is not stored on the record, so answer it
                        # honestly now rather than inviting them to ask again.
                        asked_assignee=_asked_assignee)
                else:
                    text = visit_identity.mismatch_reply(given)
                action = "respond"
                _note = (f"Visit-status gate: caller "
                         + ("confirmed via the last 4 digits of their registered number"
                            if phone_confirmed else f"named {given!r}")
                         + ("; matched the record, answered from their own appointments."
                            if _matched else
                            "; did not match the record on this number; did not read it back."))
                proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note
            else:
                # Neither a name nor phone digits in the reply — stop waiting rather than
                # loop; let the turn stand.
                session["awaiting_visit_name"] = False
        # Not on a consent hold turn: "when is it?" / "what time?" right after "shall I book
        # …?" is about the parked slot just read out, not about a visit on file — and this
        # gate would drop the parked booking to ask for a name instead.
        elif (_consent_hold is None and not session.get("visit_name_confirmed")
              and visit_identity.should_ask_name(
                routing=proposal.routing,
                intent=(proposal.understood or {}).get("intent"),
                # The planner ran my_appointments ⇒ it read this turn as being about the
                # caller's own bookings, even when the classifier mislabelled the intent.
                ran_my_appointments=any(
                    o.tool_name == "my_appointments" and o.status in TOOL_SUCCEEDED
                    for o in executed),
                # "Write turn" here means the CALLER actually asked to book/reschedule/cancel
                # — read from their own words or an executed write — NOT merely that the
                # scheduling machinery is in flight. A pure status question ("who is assigned
                # to my visit?") was being mis-read as a reschedule, which spun up a booking
                # task and lit _booking_signalled, and that speculative state then suppressed
                # this gate. Keying the exclusion on the caller's command instead means a
                # genuine "reschedule my visit" is still never gated, while a status question
                # swept into a phantom reschedule is.
                is_write_turn=(
                    _turn_hint(user_input, proposal) in _SCHEDULING_WRITES
                    or any(o.tool_name in _SCHEDULING_WRITES
                           and o.status in TOOL_SUCCEEDED for o in executed)),
                # The caller's own words name an existing visit — fires the gate even when the
                # planner ran no lookup or mis-routed toward a new booking (my_appointments
                # never ran), so a returning caller's visit is never denied without confirming
                # who they are first.
                asked_about_visit=visit_identity.mentions_existing_visit(user_input))):
            # A returning caller is asking about an existing visit. WHO they are is normally
            # established by asking the name (always-ask policy) because inbound caller-ID can
            # match the wrong record. But on an OUTBOUND call WE dialled THIS record's number
            # and the caller already confirmed at the identity gate that they are {name} — a
            # strictly stronger check than the caller-ID this gate exists to hedge. So skip the
            # redundant name question and answer straight from their own record. Guard is
            # keyed on `outbound` + a settled `STAGE_CONFIRMED` (never CALLBACK/CLOSED, where
            # identity was NOT confirmed), so inbound never enters here and its always-ask path
            # is byte-for-byte unchanged.
            if session.get("outbound") and (
                    session.get("outbound_stage") == outbound_flow.STAGE_CONFIRMED):
                # Identity is settled — the name check is already satisfied for the rest of the
                # call, and this turn is answered deterministically from the dialled record's
                # appointments (the same reliable text Turn B produces on a name match).
                session["visit_name_confirmed"] = True
                session["visit_reminder_given_this_call"] = True
                _appts = proposal.existing_appointments or []
                text = visit_identity.confirmed_reply(
                    getattr(customer, "full_name", None) or "",
                    has_visit=bool(_appts),
                    when=_appts[0].get("when") if _appts else None,
                    asked_assignee=visit_identity.asks_about_assignee(user_input))
                action = "respond"
                _note = ("Visit-status gate: identity already confirmed on this outbound call; "
                         "answered from the dialled record without re-asking the name.")
                proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note
            else:
                # Turn A — a returning caller is asking about an existing visit. Defer the
                # status answer and confirm the name first (always-ask policy).
                session["awaiting_visit_name"] = True
                # Remember whether they asked WHO is assigned, so Turn B can answer it honestly
                # (the assignee is not stored on the appointment record).
                session["visit_q_assignee"] = visit_identity.asks_about_assignee(user_input)
                # Neutralise any speculative booking/reschedule the planner started from this
                # status question, so a phantom flow cannot churn ("let me look that up…") on
                # the following turns. Only reached when the caller issued no real change
                # command, so this state is safe to drop; a genuine reschedule never lands here.
                for _k in ("booking_task", "wanted_tool", "wanted_day", "appt_lookup_forced"):
                    session.pop(_k, None)
                if (session.get("pending_action") or {}).get("tool") in _SCHEDULING_WRITES:
                    session.pop("pending_action", None)
                text = visit_identity.ask_for_name()
                action = "respond"
                _note = ("Visit-status gate: returning caller asked about an existing visit; "
                         "asked for the name to confirm identity before reading the status back.")
                proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note

    # ---- after answering a side question, return to the parked action ----
    #
    # Appended (not pinned) so the answer to their question stands. Skipped when the reply
    # already ends on a question of its own — two questions in one breath leave the caller
    # unsure which to answer — and when a later gate dropped the parked action after all. A
    # re-ask that was not spoken leaves the action stale, so a later "yes" still gets a fresh
    # read-back rather than releasing it.
    if (_consent_reask_line and action not in ("escalate", "end_call")
            and (session.get("pending_action") or {}).get("tool") == _consent_pending.get("tool")
            and not (text or "").rstrip().endswith("?")):
        text = f"{(text or '').rstrip()} {_consent_reask_line}".strip()
        _consent_asked_this_turn = True

    # ---- conversational signal: deterministic fallback if the responder call failed ----
    #
    # Detection and the persona GENERATION happened before the responder (which was handed
    # `continuation=` and has already written the natural, in-character reply into `text`). This
    # only steps in when a signal WAS detected but that model call failed (verifier_ok False) —
    # the reliable floor, so a "Hello?" / "mm-hmm" is never met with a greeting or silence when
    # the LLM is unavailable: confirm we're here and re-deliver the last line deterministically.
    # Skipped when a later gate (outbound identity, pitch ladder) owns the turn's reply.
    if (_signal_kind and not _outbound_handled and not _pitch_handled
            and action not in ("end_call", "escalate")
            and not _force_project_clarify):
        if not resp_meta.get("verifier_ok"):
            if _signal_kind == "check_in":
                text = conversational_signal.check_in_reply(_prev_agent)
            else:
                text = conversational_signal.backchannel_reply(
                    _prev_agent, agent_asked_question=_signal_asked_q)
            _note = (f"Conversational signal ({_signal_by}): {_signal_kind}; responder call failed, "
                     "fell back to deterministic re-delivery of the last line.")
        else:
            _note = (f"Conversational signal ({_signal_by}): {_signal_kind}; responder continued "
                     "the last line in persona (folded into its call, no extra model call).")
        action = "respond"
        proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note

    # The model must learn what happened, otherwise it repeats the same mistake every turn.
    # This is written as a 'tool' message and replayed in the prompt. The chaining loop
    # already wrote notes for the intermediate steps before re-planning on them, so only an
    # outcome that has not been recorded yet is written here.
    if outcome is not None and outcome.status in ("success", "skipped_duplicate"):
        session.pop("last_safe_reply", None)

    if outcome is not None and id(outcome) not in noted:
        db.add(Message(tenant_id=tenant_id, conversation_id=conversation.id,
                       role="tool", content=_tool_note(outcome)))

    db.add(Message(tenant_id=tenant_id, conversation_id=conversation.id,
                   role="agent", content=text))
    db.add(Message(tenant_id=tenant_id, conversation_id=conversation.id,
                   role="system", content=f"action:{action}"))

    # ---- step 12: decision trace ----
    conf = proposal.confidence
    trace = DecisionTrace(
        tenant_id=tenant_id, conversation_id=conversation.id,
        customer_id=customer.id if customer else None,
        turn_index=int(session.get("turn_index", 0)),
        input_text=user_input, retrieved_memory=proposal.memory_used,
        retrieved_chunks=proposal.knowledge_used,
        rules_evaluated=proposal.rules_evaluated,
        planner_reasoning=proposal.reasoning,
        proposed_tool=proposal.tool, tool_params=_safe(proposal.params),
        validation_result=({**_safe(outcome.validation or {}),
                            "status": outcome.status,
                            "failed_check": outcome.failed_check,
                            "error": outcome.error,
                            "failure_action": _safe(outcome.failure_action),
                            # The whole chain, so the Call Logs drawer shows that the turn
                            # ran locate_centre then search_slots rather than appearing to
                            # have answered from nowhere.
                            "chain": [{"tool": o.tool_name, "status": o.status}
                                      for o in executed] if len(executed) > 1 else None}
                           if outcome else None),
        confidence=conf.score if conf else None,
        confidence_band=conf.band if conf else None,
        action_taken=action,
        latency_ms=int((time.perf_counter() - started) * 1000),
        model_used=proposal.model,
        tokens={"understanding": proposal.understanding_tokens,
               "planner": proposal.tokens, "responder": resp_meta.get("tokens", {})},
        # tts_characters is set below, once the response text has gone through its final
        # rounding/trim passes -- those run after this point, so the character count of what
        # actually gets spoken isn't known yet here. stt_seconds is a voice-channel-only
        # signal: None on a chat turn (bridge.py is voice-only and is the only caller that
        # ever passes it), which is also what gates tts_characters below.
        stt_seconds=stt_seconds,
        prompt_versions={"base": "1", "responder_grounded": resp_meta.get("grounded", False)})
    async with _bridge.stage("trace_persist"):
        db.add(trace)
        await db.flush()

    publish(tenant_id, "conversation.turn", conversation_id=str(conversation.id),
            customer_id=str(customer.id) if customer else None,
            customer_said=user_input[:200], agent_said=text[:200], action=action,
            confidence=conf.score if conf else None,
            tool=outcome.tool_name if outcome else None)
    if state_changed_to:
        publish(tenant_id, "customer.state_changed",
                customer_id=str(customer.id) if customer else None,
                to_state=state_changed_to)

    # Record which turn last ASKED about the parked action — a read-back (action "confirm"), a
    # consent hold line, or a re-ask after a side question. Next turn's "yes" releases it only
    # if it answers this turn (see `_pending_fresh`).
    _parked = session.get("pending_action")
    if isinstance(_parked, dict) and _parked.get("tool") and (
            action == "confirm" or _consent_asked_this_turn):
        _parked["asked_turn"] = int(session.get("turn_index", 0))
    session["turn_index"] = int(session.get("turn_index", 0)) + 1
    session["recent_actions"] = (session.get("recent_actions", []) + [f"action:{action}"])[-5:]
    # The exact line the caller just heard, so the NEXT turn can repeat it verbatim if they say
    # "Hello?" / "sorry, what?" (the line dropped) or nod for us to continue — see the
    # conversational-signal override above. Kept as the final agent text of every turn.
    session["last_agent_reply"] = text
    # What the call is about, updated from what the planner understood this turn.
    #
    # These entities were extracted and discarded on every single turn. Persisting them is
    # what lets the NEXT turn's retrieval know that "both cost breakdowns" means the
    # 2.5 BHK ones. Taken from the final proposal, so a corrective re-plan stores the
    # entities it actually acted on.
    session["active_entities"] = merge_entities(
        session.get("active_entities") or {},
        (proposal.understood or {}).get("entities"),
        user_input)

    # Promote a stated unit choice from the transient focus bag into durable memory. The
    # entities above are a retrieval-ranking hint that drifts across a booking; a chosen
    # configuration ("3 BHK") is a DECISION, and answering "the flat I selected" from a hint
    # that has since drifted is how a 3 BHK caller was quoted the 2.5 BHK sheet. Pinned on the
    # session (stable referent for this call, and the value book_appointment binds to the
    # visit) and, for an identified caller, written through to the CRM so it survives the call.
    # Freshest statement wins. A unit named THIS turn overrides the standing choice — that is
    # what makes "actually, make it the 2 BHK" a change of mind and not a second, coexisting
    # interest. Only if nothing was named this turn does the pinned choice hold; the bag scan
    # is the last resort (first mention of a call, before anything is pinned). Falling back to
    # the pin rather than re-scanning the bag is deliberate: the bag can still hold the OLD
    # unit under a stale slot key, and re-deriving from it would flip a settled choice back.
    _unit_voc = await _tenant_vocabulary(db, tenant_id)
    _incoming = preference.desired_unit((proposal.understood or {}).get("entities"), _unit_voc)
    _unit = (_incoming or session.get("desired_unit")
             or preference.desired_unit(session.get("active_entities") or {}, _unit_voc))
    if _unit:
        session["desired_unit"] = _unit
        # conversation.customer_id, not the local `customer` — a create_customer that ran
        # THIS turn attaches the caller to the conversation but does not refresh the local,
        # so keying off it would drop the preference on the very turn they register. Persist
        # only when the unit is newly stated or has changed, so an unchanged choice does not
        # cost a write on every subsequent turn of the call.
        _cid = customer.id if customer is not None else conversation.customer_id
        if _cid is not None and session.get("desired_unit_persisted") != _unit:
            await preference.remember_product_interest(
                db, tenant_id=tenant_id, customer_id=_cid,
                conversation_id=conversation.id, unit=_unit,
                # Scope the interest to the project this call resolved to.
                project_id=getattr(conversation, "project_id", None))
            session["desired_unit_persisted"] = _unit

    # BookingTask projection, persisted every turn. Built from the FINAL session state and
    # this turn's tool outcomes, so next turn reads a task that reflects what actually
    # happened, and the projection is carried on the decision trace for observability.
    _final_task = bt.project(
        session, customer_exists=customer is not None, executed=executed,
        turn_index=int(session.get("turn_index", 0)) - 1,
        active_entities=session.get("active_entities") or {},
        current_state=customer.current_state if customer else None)
    bt.store(session, _final_task)

    # Arm the handoff-accept for next turn. If THIS reply offered to put the caller through
    # to a person (and we are not already transferring them), a bare "yes" next turn is that
    # acceptance — see the consume step at the top of process_turn. Re-derived from the final
    # reply every turn, so the flag lives exactly one turn: a reply that does not offer a
    # handoff clears it, and there is no path by which a stale offer lingers into an unrelated
    # turn. Skipped on an escalation turn (FORCED_HANDOVER itself reads like an offer) so the
    # accept never re-arms itself into a loop.
    if action != "escalate" and not session.get("escalated") and _offers_human_handoff(text):
        session["offered_human_handoff"] = True
    else:
        session.pop("offered_human_handoff", None)

    # A rule sent this call to a person. Say so, if the reply does not already. When the rule
    # named a transfer target, the reply must name THAT target — a generic "I'll connect you to
    # someone" is replaced with "...to our Financial Desk" so the caller hears where they are
    # going. A reply that already names the target is left as the model wrote it.
    _target_display = (proposal.forced_escalation or {}).get("target_display")
    _names_target = bool(_target_display) and _target_display.lower() in (text or "").lower()
    if (action == "escalate" and proposal.forced_escalation and not _names_target
            and (_target_display or not _reads_like_handover(text))):
        note = (f"Rule {(proposal.forced_escalation or {}).get('rule')} forced a handover; "
                f"reply {'did not name the target' if _target_display else 'did not mention it'}"
                f" and was replaced.")
        proposal.error = f"{proposal.error} | {note}" if proposal.error else note
        text = _forced_handover_line(_target_display)

    # A rule requires a piece of information before the agent may proceed ("collect budget
    # before offering a visit"). The turn was already turned into a clarify above; here we make
    # sure the spoken line actually ASKS for it — the responder, rewriting from scratch, may not
    # know to. Enforced deterministically like the handover line, so the caller reliably hears
    # the ask rather than being quietly moved on.
    if proposal.forced_requirement and action not in ("escalate", "end_call"):
        note = (f"Rule {proposal.forced_requirement.get('rule')} requires "
                f"'{proposal.forced_requirement.get('condition')}' first; asked for it.")
        proposal.error = f"{proposal.error} | {note}" if proposal.error else note
        text = _required_info_line(proposal.forced_requirement)

    # A tool was BLOCKED by a slot `require` at the tool gate ("collect budget before offering a
    # visit"): any preconditions such as registration ran first, and the gated action is now held
    # until the information is given. Speak the rule's OWN question (deterministic — no extra
    # model call) and count the ask, so the requirement stands aside after a couple of tries
    # (the anti-trap in engine.unmet_requirements reads this counter). Fully generic and
    # delete-safe: the tool, slot and question are all data on the compiled rule, so removing the
    # instruction removes the behaviour with no code left behind.
    _req_fa = (outcome.failure_action or {}) if outcome is not None else {}
    if (outcome is not None and outcome.status == "blocked" and _req_fa.get("slot")
            and action not in ("escalate", "end_call")):
        _rcond = _req_fa.get("condition") or _req_fa.get("slot")
        _asks = dict(session.get("__require_asks__") or {})
        _asks[_rcond] = int(_asks.get(_rcond, 0)) + 1
        session["__require_asks__"] = _asks
        _rnote = (f"Rule {_req_fa.get('rule')} requires '{_rcond}' before "
                  f"{outcome.tool_name}; asked for it (attempt {_asks[_rcond]}).")
        proposal.error = f"{proposal.error} | {_rnote}" if proposal.error else _rnote
        text = _required_info_line(_req_fa)

    text = round_spoken_prices(text)
    text = fix_lakh_crore_mismatch(text)
    # Abbreviations and symbols to spoken words ("sq ft" -> "square feet", "₹" -> "rupees").
    # Same three rewrites, same order, as responder._speakable on the streamed sentences.
    text = to_spoken_form(text)
    text = _strip_booking_noise(text)
    text = _collapse_duplicate_facts(text)
    text = _trim_repeated_anything_else(text, session)
    text = _trim_repeated_visit_offer(text, session)
    text = _trim_repeated_pin_offer(text, session)
    text = _trim_unsolicited_reschedule_offer(text, user_input)
    text = _trim_unprompted_callback_offer(text, user_input)
    text = _trim_repeated_visit_reminder(
        text, session, user_input,
        already_given_at_turn_start=_visit_reminder_already_given_at_turn_start)

    # P6-F3: the end_call block above only advances reengage_count through decide_end_call,
    # and its own `else` branch unconditionally CLEARS the count on any turn that doesn't
    # reach "end_call" -- including a turn whose action is "tool_executed"/"respond" but
    # whose FINAL drafted text still asks an "anything else?"-style question of its own
    # accord (a tool-confirmation sentence the model appended one to, not something that
    # went through decide_end_call at all).
    #
    # THE BUG THIS FIXES: a real call had the agent say "I've marked your number as
    # do-not-call... Is there anything else I can do for you right now?" in the SAME turn as
    # record_do_not_contact (action="tool_executed", reengage_count untouched, then cleared
    # to None by the else branch above since this turn never became "end_call"). The
    # caller's very next "no thanks" was read as the FIRST decline (count still 0/None) and
    # asked again ("...or shall I let you go?") instead of ending -- the caller had, in
    # substance, already been asked and declined once.
    #
    # Checked against THIS turn's own FINAL text (after every trim above), not the model's
    # action label, so it catches the ask in whatever wording carried it and whatever action
    # the turn was. Only sets the count when nothing already did (never overwrites a value
    # decide_end_call itself just set to a higher precedent, and never fires on a turn that
    # is itself ending the call).
    if (action not in ("end_call", "escalate")
            and not session.get("reengage_count")
            and closure.reads_like_reengage_offer(text)):
        session["reengage_count"] = 1

    # Character count of what ElevenLabs actually gets asked to speak -- set here, not at
    # trace construction above, because `text` only reaches its final form after the price-
    # rounding/anything-else/visit-reminder trims just above. None on a chat turn (no TTS
    # ever runs), matching stt_seconds already being None there.
    trace.tts_characters = len(text) if stt_seconds is not None else None

    # Saved last, not right after the handoff flag above: _trim_repeated_anything_else and
    # _trim_repeated_visit_reminder both set a session flag ("said this once already") that
    # the NEXT turn's trim depends on. Saving before they run persisted the session a step too
    # early -- the flag was set on this function's local dict but the save that would have
    # written it to Redis had already happened, so every following turn loaded a session
    # without it and reads as "first time" forever. The repeat is never actually trimmed.
    # Redis + Postgres together, never split by a barge-in cancellation -- see _persist_turn.
    await _persist_turn(db, conversation.id, session, _bridge)

    return TurnResult(
        response_text=text, response_segments=responder.split_sentences(text),
        action=action,
        # Every tool the turn ran, in order — a chained turn runs more than one, and
        # reporting only the last would hide the lookup that produced the answer.
        executed_tools=[{"tool_name": o.tool_name, "status": o.status,
                         "output": _safe(o.output), "error": o.error,
                         "failed_check": o.failed_check,
                         "failure_action": _safe(o.failure_action)} for o in executed],
        state_changed_to=state_changed_to, escalation=escalation_payload,
        end_call=(action == "end_call"), trace_id=trace.id,
        confidence={"score": conf.score if conf else 0, "band": conf.band if conf else "low",
                    "self_reported": conf.self_confidence if conf else 0,
                    "adjustments": conf.adjustments if conf else []},
        pending_confirmation=session.get("pending_action"),
        debug={"retrieval": proposal.retrieval,
               # Likely-misheard words flagged to the planner this turn (voice only) -- what
               # the transcript said, the term it probably was, and how strongly it matched.
               "asr_hints": [{"heard": h.heard, "term": h.term, "score": h.score}
                             for h in _asr_hints],
               # The fast understanding pre-pass (P1): what it classified the turn as and
               # whether that routed retrieval on or off. Empty on retry legs.
               "routing": proposal.routing,
               "knowledge_used": proposal.knowledge_used,
               "as_of_date": str(as_of_date) if as_of_date else None,
               "timing": _with_bridge_timing(timing, _bridge, started),
               "confirmation": confirmation_source,
               "understood_intent": intent, "read_as_affirmative": affirmative,
               # The consent read for this turn (consent.py): both layers, the fused decision,
               # the words it rests on, and whether a parked action was held / re-asked.
               "consent": {**_consent.as_trace(), "kind": _consent_kind,
                           "parked": _consent_pending.get("tool") if _consent_staged else None,
                           "fresh": _pending_fresh if _consent_staged else None,
                           "hold": _consent_hold,
                           "reasked": bool(_consent_reask_line and _consent_asked_this_turn),
                           "declined": _consent_declined},
               "candidates": proposal.candidates,
               "rules_fired": [r for r in proposal.rules_evaluated if r.get("matched")],
               "planner_error": proposal.error,
               "triage": triage_debug,
               "response_grounded": resp_meta.get("grounded", False),
               # The BookingTask projection for this turn — the orchestrator's own record
               # of where the booking has got to, logged for observability.
               "booking_task": _final_task.to_dict(),
               "missing_info": proposal.missing_info,
               # What this turn cost, priced from the raw usage already on `trace` (never
               # persisted as a $ figure -- see app/observability/pricing.py for why). Shared
               # verbatim by chat (this dict) and voice (bridge.py forwards result.debug as
               # the RTVI turn_debug message), so both channels' Console chips get it for
               # free once the frontend reads it.
               "usage": {
                   "tokens": trace.tokens,
                   "tts_characters": trace.tts_characters,
                   "stt_seconds": trace.stt_seconds,
                   "cost_usd": pricing.turn_cost(
                       tokens_by_stage=trace.tokens, model_used=trace.model_used,
                       tts_characters=trace.tts_characters,
                       stt_seconds=(float(trace.stt_seconds)
                                   if trace.stt_seconds is not None else None)).as_dict(),
               }})


def _briefing(customer: Customer | None, conversation: Conversation,
              proposal: Proposal, outcome: ToolOutcome | None) -> str:
    lines = []
    if customer is not None:
        lines.append(f"Customer: {customer.full_name or 'unknown'} ({customer.phone})")
        lines.append(f"Journey stage: {customer.current_state}")
    lines.append(f"Channel: {conversation.channel} / {conversation.direction}")
    u = proposal.understood or {}
    lines.append(f"Latest intent: {u.get('intent')} (sentiment: {u.get('sentiment')})")
    if proposal.reasoning:
        lines.append(f"Agent's reading: {proposal.reasoning}")
    if outcome is not None:
        lines.append(f"Last action: {outcome.tool_name} -> {outcome.status}"
                     + (f" ({outcome.error})" if outcome.error else ""))
    if proposal.missing_info:
        lines.append("Outstanding: " + ", ".join(proposal.missing_info))
    return "\n".join(lines)


def _safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _safe(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_safe(x) for x in v]
    if isinstance(v, uuid.UUID):
        return str(v)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v