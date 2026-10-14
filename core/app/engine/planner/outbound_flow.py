"""Outbound call opening — confirm we reached the right person before pursuing the purpose.

WHY THIS EXISTS. An outbound call is different from an inbound one in two ways the engine must
handle before any of its normal machinery runs:

  1. WE called THEM. The voice-pipeline spec (§3.5) requires the agent confirm it is speaking to
     the intended person first ("Am I speaking with Mr Sharma?") — a cold pitch to whoever
     answers is both confusing and non-compliant. Until identity is confirmed, the agent must
     not pursue the reason for the call.
  2. The call has a PURPOSE (marketing | service | feedback_survey | …) that shapes what a good
     conversation looks like. The same engine answers questions and books visits identically;
     the purpose only changes the opening and the tone, which is a prompt concern.

This module is PURE and self-tested (``python -m app.engine.planner.outbound_flow``). It owns
two decisions and nothing else: how to CLASSIFY the caller's answer to the identity question,
and WHAT to say for each branch. The wiring in turn_runner supplies the call's stage and
purpose and applies the result; the prompt layer (prompts.py) renders the purpose guidance.

THE BRANCHES (operator-confirmed policy):
  * confirmed      → open the purpose and hand back to the normal engine.
  * wrong_person   → apologise, do NOT pitch, flag the record for review (not auto-suppressed),
                     end the call.
  * unavailable    → the intended person is not the one who answered; offer a callback.
  * not_now        → right person, bad moment; offer a callback.
  * gatekeeper     → "who is this / what's this about"; disclose briefly and re-ask identity.
  * opt_out        → "stop calling me"; NOT handled here — left to the normal DNC path so the
                     opt-out is recorded structurally (record_do_not_contact), exactly as inbound.
  * unclear        → re-ask the identity question once.
"""
import re

from app.packs.base import Vocabulary

# Vocal Backend Tasksheet P6 Generic Sprint 3: every function below that used to hardcode
# "site visit" now takes an optional `vocabulary` — resolved per-tenant by the caller via
# `PackRegistry.vocabulary_for_tenant()` (this module stays PURE; it does no DB access
# itself). When omitted, this fallback reproduces the original literal wording exactly, so
# an unmigrated call site (or this file's own `_selftest()` below) sees zero behavior change.
_FALLBACK_VOCABULARY = Vocabulary(commitment_noun="site visit")

# ------------------------------------------------------------------ call stages (session)
STAGE_IDENTITY = "identity"      # opener asked; awaiting / classifying the identity answer
STAGE_CONFIRMED = "confirmed"    # right person reached; normal engine drives the rest
STAGE_CALLBACK = "callback"      # a callback was offered; normal engine schedules it next turn
STAGE_CLOSED = "closed"          # call is ending (wrong person / done)

# ------------------------------------------------------------------ purposes
PURPOSE_MARKETING = "marketing"
PURPOSE_SERVICE = "service"
PURPOSE_FEEDBACK = "feedback_survey"
PURPOSE_RETENTION = "retention"
PURPOSE_FOLLOW_UP = "follow_up"
PURPOSE_UPSELL = "upsell"

# ------------------------------------------------------------------ branch classification
BRANCH_CONFIRMED = "confirmed"
BRANCH_WRONG_PERSON = "wrong_person"
BRANCH_UNAVAILABLE = "unavailable"
BRANCH_NOT_NOW = "not_now"
BRANCH_GATEKEEPER = "gatekeeper"
BRANCH_OPT_OUT = "opt_out"
BRANCH_UNCLEAR = "unclear"

# "stop calling me" — highest priority, because it must be honoured no matter what else the
# sentence contains. Left for the normal DNC tool path to ACT on; classified here only so the
# gate steps aside for it rather than treating "don't call" as a plain negative.
_OPT_OUT_RE = re.compile(
    r"\b(?:stop|quit|cease)\b[^.?!]{0,20}\b(?:call|calling|contact|contacting|phon|ring|disturb)"
    r"|\bdo ?n['o]?t\b[^.?!]{0,20}\b(?:call|contact|phone|ring)\b"
    r"|\b(?:remove me|take me off|unsubscribe|opt me out|opt-out|leave me alone|"
    r"no more calls?|don'?t (?:ever )?call (?:me )?again)\b", re.I)

# The intended person is not the one who answered / is not there.
_UNAVAILABLE_RE = re.compile(
    r"\b(?:not (?:here|in|available|at home|around)|isn'?t (?:here|in|available|around)|"
    r"un-?available|not (?:his|her|the) (?:phone|number)|"
    r"(?:he|she|they)('?s| is| are)? (?:out|away|not here|busy|at work|unavailable)|"
    r"take a message|wrong person|speak(?:ing)? (?:to|with) (?:his|her|their)|"
    r"this is (?:his|her|their))\b", re.I)

# Right person, wrong moment.
_NOT_NOW_RE = re.compile(
    r"\b(?:busy|driving|in a meeting|not a good time|bad time|"
    r"can'?t (?:talk|speak)|call (?:me )?(?:back )?later|call back|"
    r"another time|some other time|later please|not right now|not now|at work)\b", re.I)

# Suspicious / screening: who is this, what is this (call) about. "this" and "it" both refer to
# the call ("what is it about?", "what's this call about?"), so both are accepted.
_GATEKEEPER_RE = re.compile(
    r"\bwho(?:'?s| is| are)?\b[^.?!]{0,15}\b(?:this|calling|you|speaking)\b"
    r"|\bwhat(?:'?s| is)?\b[^.?!]{0,20}\b(?:(?:this|it)(?: call)? (?:about|regarding)|"
    r"the reason|calling about|(?:this|it) (?:is )?(?:in )?regard|you (?:want|calling))\b"
    r"|\bwhy (?:are|r|did) (?:you|u) (?:calling|phoning|ringing|call(?:ed)?)\b"
    r"|\bhow did you get (?:my|this) (?:number|contact)\b", re.I)

# Explicitly the wrong number / no such person.
_WRONG_PERSON_RE = re.compile(
    r"\bwrong (?:number|person)\b"
    r"|\b(?:no ?(?:one|body)|there'?s no ?(?:one|body)?)\b[^.?!]{0,20}"
    r"\b(?:named|called|by (?:that|this) name|here|lives here)\b"
    r"|\byou'?ve got the wrong\b"
    r"|\bnever heard of\b", re.I)

# Plain agreement that it is the right person.
_CONFIRMED_RE = re.compile(
    r"\b(?:yes|yeah|yep|yup|speaking|that'?s me|this is (?:he|she|me)|"
    r"you are|correct|that'?s right|that'?s correct|go ahead|sure|"
    r"who'?s asking|haan|ha?a?n?ji?)\b", re.I)

# Plain refusal / "no" with nothing more specific — after the identity question, this reads as
# "you have the wrong person".
_BARE_NEGATIVE_RE = re.compile(r"\b(?:no|nope|nah|nahi|not me|isn'?t me|i'?m not)\b", re.I)


def classify_identity_reply(text: str) -> str:
    """Which branch the caller's answer to 'am I speaking with X?' falls into.

    Priority is deliberate: an opt-out is honoured over anything else; an explicit "who is
    this?" is treated as screening (disclose and re-ask) rather than as a refusal; a clear
    "yes/speaking" confirms; and only a bare, unqualified negative is read as the wrong person.
    """
    t = (text or "").strip()
    if not t:
        return BRANCH_UNCLEAR
    if _OPT_OUT_RE.search(t):
        return BRANCH_OPT_OUT
    if _WRONG_PERSON_RE.search(t):
        return BRANCH_WRONG_PERSON
    if _UNAVAILABLE_RE.search(t):
        return BRANCH_UNAVAILABLE
    if _NOT_NOW_RE.search(t):
        return BRANCH_NOT_NOW
    # A confirmation wins over a screening question ("yes, who's this?") — they have identified
    # themselves, and the purpose opener discloses who we are anyway.
    if _CONFIRMED_RE.search(t) and not _BARE_NEGATIVE_RE.search(t):
        return BRANCH_CONFIRMED
    if _GATEKEEPER_RE.search(t):
        return BRANCH_GATEKEEPER
    if _BARE_NEGATIVE_RE.search(t):
        return BRANCH_WRONG_PERSON
    return BRANCH_UNCLEAR


# ------------------------------------------------------------------ the lines the caller hears

def _first(name: str | None) -> str:
    toks = re.findall(r"[A-Za-z]+", (name or "").strip())
    return toks[0].capitalize() if toks else ""


def _biz(business: str | None) -> str:
    return (business or "").strip() or "our sales team"


def identity_opener(business: str | None, name: str | None) -> str:
    """The agent's first line on an outbound call: confirm we reached the right person. Shared
    by the voice socket and the chat session so both channels open the same way."""
    first = _first(name)
    if first:
        return (f"Hello, am I speaking with {first}? This is {_biz(business)} calling — "
                f"is now a good time for a quick word?")
    return (f"Hello, this is {_biz(business)} calling. Am I speaking with the right person, "
            f"and is now a good time for a quick word?")


def _greet(name: str | None) -> str:
    first = _first(name)
    return f", {first}" if first else ""


def _interest_phrase(interest: str | None) -> str:
    """A stated enquiry unit ("2 BHK", "3bhk apartment") rendered as a natural spoken fragment
    with an article — "a 2 BHK", "an apartment" — or "" when nothing usable is on file.

    Kept deliberately light: the CRM text is used almost verbatim (it is already what the lead
    said), tidying only whitespace and the article, so an operator's phrasing is never mangled
    and a value we don't understand still reads correctly."""
    t = re.sub(r"\s+", " ", (interest or "").strip())
    if not t:
        return ""
    article = "an" if re.match(r"^[aeiou8]", t, re.I) else "a"
    return f"{article} {t}"


def _enquiry_recall(interest: str | None, project: str | None) -> str:
    """The 'what you enquired about' fragment the outbound opener VERIFIES before pitching —
    concrete enough that the caller recognises it, so the call never assumes a context the
    caller never heard in-call. Uses whichever of {unit, project} we actually have:

      * unit + project → "a 2 BHK at DSR Highland Greenz"
      * unit only      → "a 2 BHK"
      * project only   → "our DSR Highland Greenz project"
      * neither        → "" (the caller has no enquiry on file; the opener stays a discovery
                          question rather than a false recall)."""
    unit = _interest_phrase(interest)
    proj = (project or "").strip()
    if unit and proj:
        return f"{unit} at {proj}"
    if unit:
        return unit
    if proj:
        return f"our {proj} project"
    return ""


def commitment_phrase(appointment: dict | None, vocabulary: Vocabulary | None = None) -> str:
    """A spoken reference to a commitment the client ALREADY has with us — "your site visit on
    Sunday 17 August at 4:00 PM" for the real-estate pack's default vocabulary — from their
    upcoming appointment, or "" when there is none.

    This is what a SERVICE call is about (PRD: confirm/reschedule/remind an existing commitment,
    never a pitch). Pure; the appointment dict is an ``existing_appointments`` entry ({"when",
    "unit", …}); a missing/blank ``when`` degrades to the generic "your {commitment_noun}"."""
    if not appointment:
        return ""
    noun = (vocabulary or _FALLBACK_VOCABULARY).commitment_noun
    when = (appointment.get("when") or "").strip()
    return f"your {noun} on {when}" if when else f"your {noun}"


def _service_subject(commitment: str | None, interest: str | None, project: str | None) -> str:
    """What a SERVICE call is ABOUT, most concrete first — so the agent always leads with
    something real, the way a person would, instead of a vague "your enquiry with us":

      * a booked visit          -> "your site visit on Wednesday at 1 PM"   (commitment)
      * else a registered enquiry -> "your enquiry about a 3 BHK at DSR Highland Greenz"
                                   /  "your enquiry about DSR Highland Greenz"
      * else                    -> "" (no concrete subject; the caller uses the generic wording).

    Pure. ``commitment`` comes from ``commitment_phrase``; ``interest``/``project`` are the
    durably-loaded registered interest (unit + project) already used by the marketing openers."""
    commit = (commitment or "").strip()
    if commit:
        return commit
    unit = _interest_phrase(interest)                # "a 3 BHK" / "an apartment" / ""
    proj = (project or "").strip()
    if unit and proj:
        return f"your enquiry about {unit} at {proj}"
    if proj:
        return f"your enquiry about {proj}"
    if unit:
        return f"your enquiry about {unit}"
    return ""


def purpose_opener(purpose: str | None, business: str | None, name: str | None,
                   project: str | None = None, interest: str | None = None,
                   commitment: str | None = None) -> str:
    """Identity is confirmed — open the reason for the call in the right register. Deliberately
    an INVITATION, not a monologue: it discloses who we are and why we called and hands the
    floor back, so the normal (grounded) engine answers whatever the caller says next.

    ``project`` and ``interest`` are what the lead already enquired about (the project, e.g.
    "DSR Highland Greenz", and the unit, e.g. "2 BHK"), when we resolved them from their live
    product interest. A call WE placed about a stated interest must RECALL and VERIFY that
    interest before it pitches — "you'd enquired about a 2 BHK at DSR Highland Greenz, is that
    right?" — so the caller shares the context we are building on instead of hearing a bare
    "what you were after" that presumes a conversation they never had. When neither is known
    (no enquiry on file) the marketing/upsell/follow-up openers degrade to an honest discovery
    question rather than asserting a remembered interest that does not exist.

    ``commitment`` is the client's existing commitment for a SERVICE call (their booked visit,
    from ``commitment_phrase``). A service call is about THAT — confirm/remind it — so the opener
    leads with it ("about your site visit on Wednesday at 1 PM") instead of a vague "your enquiry
    with us"; empty when nothing is booked, and the opener falls back to the generic wording."""
    g = _greet(name)
    b = _biz(business)
    recall = _enquiry_recall(interest, project)
    commit = (commitment or "").strip()
    p = (purpose or PURPOSE_MARKETING).strip().lower()
    if p == PURPOSE_SERVICE:
        if commit:                                   # a booked visit → confirm/remind it
            return (f"Thank you{g}. I'm calling from {b} about {commit} — I wanted to confirm "
                    f"you're still all set, and check if there's anything you need before then.")
        subject = _service_subject(None, interest, project)
        if subject:                                  # no visit, but a registered enquiry on file
            return (f"Thank you{g}. I'm calling from {b} about {subject} — I wanted to make sure "
                    f"everything's on track and see if there's anything I can help you with.")
        return (f"Thank you{g}. I'm calling from {b} about your enquiry with us — I wanted to "
                f"make sure everything is in order. How can I help?")
    if p == PURPOSE_FEEDBACK:
        return (f"Thanks{g}. I'm calling from {b} for some quick feedback on your recent site "
                f"visit — it will only take a minute. Is that alright?")
    if p in (PURPOSE_RETENTION, PURPOSE_FOLLOW_UP):
        if recall:
            return (f"Thanks{g}. I'm calling from {b} to follow up on your enquiry about "
                    f"{recall} — is that still what you're looking for?")
        return (f"Thanks{g}. I'm calling from {b} to follow up on your interest with us. Could "
                f"I ask what kind of home you're looking for, so I can point you the right way?")
    if p == PURPOSE_UPSELL:
        if recall:
            return (f"Thanks{g}. I'm calling from {b} — you'd enquired about {recall}, is that "
                    f"right?")
        return (f"Thanks{g}. I'm calling from {b} — we've got a few things I think are worth "
                f"your while. May I take a moment to walk you through them?")
    # marketing (and any unknown purpose): warm, brief, ONE clear interest question. The opener
    # is deliberately just the verification — the value and the visit offer come next, gradually,
    # up the pitch ladder (see advance_pitch), never dumped here or asked before interest is
    # confirmed. Ending on the question is also what lets the caller's "yeah" read as an ANSWER
    # (not a content-free nod that the conversational-signal path would echo the opener back at).
    if recall:
        return (f"Great, thank you{g}. I'm calling from {b} — you'd enquired with us about "
                f"{recall}, is that right?")
    return (f"Great, thank you{g}. I'm calling from {b} — we've recently had some property "
            f"options I thought might interest you. Could I ask what you're looking for, so I "
            f"share the right ones?")


def wrong_person_reply(business: str | None) -> str:
    """Wrong number: apologise, promise to correct our records (the record is flagged for
    review, not auto-suppressed), and end. No pitch to someone we did not mean to call."""
    return ("I'm so sorry to trouble you — it sounds like I've reached the wrong number. "
            "I'll update our records so you aren't disturbed again. Apologies for the "
            "interruption, and have a good day.")


def offer_callback_reply(name: str | None) -> str:
    """Unavailable / bad moment: offer a callback and ask for a better time. The normal engine
    (schedule_callback) records whatever time the caller then gives."""
    g = _greet(name)
    return (f"No problem at all{g} — I don't want to hold you up. When would be a better time "
            f"for us to call you back?")


# Purposes where the agent LEADS toward a conversion (a site visit) rather than servicing an
# existing commitment. Marketing/upsell/follow-up/retention are a pitch; service/feedback are not.
LEAD_PURPOSES = frozenset({PURPOSE_MARKETING, PURPOSE_UPSELL, PURPOSE_RETENTION, PURPOSE_FOLLOW_UP})


def is_lead_purpose(purpose: str | None) -> bool:
    return (purpose or PURPOSE_MARKETING).strip().lower() in LEAD_PURPOSES


def lead_forward_reply(purpose: str | None, name: str | None,
                       project: str | None = None, interest: str | None = None) -> str:
    """The agent has nothing fresh to recite from the knowledge base but — because WE placed this
    call — must keep LEADING rather than apologise and hand the caller to a human. Grounds them in
    what they registered interest in and makes the ONE concrete forward move the purpose is for.

    For a lead purpose (marketing/upsell/follow-up) that move is a SITE VISIT — the real
    conversion, and the one thing the agent can always offer without any inventory lookup. For
    any other outbound purpose it degrades to an honest "is there anything I can help with?"
    rather than a false visit pitch. Pure and unit-tested; used by turn_runner's outbound lead
    guard to replace a reactive escalation on a confirmed outbound call."""
    g = _greet(name)
    if is_lead_purpose(purpose):
        recall = _enquiry_recall(interest, project)
        about = f" about {recall}" if recall else ""
        return (f"I'd genuinely love to help you take this forward{g}. The best way to get a "
                f"real feel for it is to see it in person — would you like me to arrange a "
                f"quick site visit{about}? I'll find a time that suits you.")
    return (f"Of course{g} — I'm here to help. Is there anything about it I can sort out for "
            f"you right now?")


# ---- outbound pitch ladder (lead purposes, identity confirmed) --------------------------
#
# WE placed the call; once the caller CONFIRMS interest the agent LEADS a gradual pitch instead
# of asking for a visit in the very opener — or, worse, repeating the opener when the caller
# just nods. Each positive nudge ("yes", "go on", "sure") advances ONE rung; a plain decline
# steps aside gracefully; anything with real content (a price/parking question) is left to the
# normal engine to ANSWER, never advanced. The current rung lives on the session (``pitch_stage``).
#
#   verify → value → offer → scheduling      (each rung ENDS on the question inviting the next)
#   any rung → wind_down                      (on a plain decline)
#
# Deliberately just a few rungs so it reads like a person leading a call, not a script. Real
# specifics (price, sizes, amenities) are the model's job from PROJECT FACTS when the caller
# asks; these rungs carry only honest, non-fabricated framing and the forward question.
PITCH_VERIFY = "verify"          # opener asked "…is that right?"; interest not yet confirmed
PITCH_VALUE = "value"            # interest confirmed; sharing why it's worth their time
PITCH_OFFER = "offer"            # value given; offering the site visit
PITCH_SCHEDULING = "scheduling"  # visit accepted; the normal booking engine takes over
PITCH_WIND_DOWN = "wind_down"    # declined; closing gracefully

# The rungs where a bare "yes / go on" should advance the pitch. Once scheduling or wind-down is
# reached the ladder is done and the normal engine drives the rest.
PITCH_LADDER_ACTIVE = frozenset({PITCH_VERIFY, PITCH_VALUE, PITCH_OFFER})

# A plain decline — "no", "not interested", "maybe later". Kept separate from the identity
# opt-out (that is a "stop calling me", handled by the DNC path); this is just "not now for the
# pitch", which we accept gracefully and stop pushing.
_PITCH_DECLINE_RE = re.compile(
    r"\b(?:not (?:interested|keen|looking|really)|no thanks?|no thank you|"
    r"don'?t (?:want|need|bother)|already (?:bought|booked|sorted|purchased)|"
    r"changed my mind|leave it|drop it|maybe later|some other time|not (?:right )?now)\b"
    r"|^\s*(?:no|nope|nah|na)[\s.,!?-]*$", re.I)

# A bare positive reply and NOTHING else — the caller is nudging us on, not asking anything. The
# whole utterance must be these words (anchored ^…$), so "yeah, what about parking?" is NOT an
# advance — it carries a question, and falls through to 'other' for the engine to answer.
_PITCH_ADVANCE_RE = re.compile(
    r"^\s*(?:(?:yes|yeah|yep|yup|ya|ok(?:ay)?|okey|sure|alright|all right|please|"
    r"go on|go ahead|carry on|continue|do(?: it| that| please)?|please do|sounds good|"
    r"that works|why not|of course|definitely|absolutely|certainly|great|cool|perfect|nice|"
    r"love to|i'?d love to|i'?d like that|tell me( more)?|let'?s|sure thing|haan|haa?n?ji?|"
    r"ji|theek hai|theek|achha|acha)[\s,.!?-]*)+$", re.I)

# Common multi-word confirmations to the opener's "…is that right?" — a whole-utterance yes with
# nothing else asked. Kept tight and anchored so a negation ("that's not right") or a question
# ("is that right?") never slips through as an advance.
_PITCH_CONFIRM_RE = re.compile(
    r"^\s*(?:(?:yes|yeah|yep|yup|ya)[\s,]*)?"
    r"(?:that'?s|that is|it'?s|it is)\s+(?:right|correct|true|me)[\s.,!?-]*$"
    r"|^\s*(?:that'?s right|that'?s correct|correct|exactly|precisely|spot on|"
    r"you'?re right|right you are|that'?s me|speaking|you got it|you'?ve got it)[\s.,!?-]*$",
    re.I)


def classify_pitch_reply(text: str | None) -> str:
    """How the caller answered the current pitch rung:

      'advance' — a bare yes / go-on that moves the pitch forward one rung.
      'decline' — a plain no / not-interested; step aside gracefully.
      'other'   — real content (a question, a detail, a time). The NORMAL engine handles it;
                  the ladder never swallows it. When in doubt this is 'other'.

    Pure and unit-tested. Decline is checked before advance so "no thanks" is a decline, never a
    stray 'no'+'thanks' advance."""
    t = (text or "").strip()
    if not t:
        return "other"
    if _PITCH_DECLINE_RE.search(t):
        return "decline"
    if _PITCH_ADVANCE_RE.match(t) or _PITCH_CONFIRM_RE.match(t):
        return "advance"
    return "other"


def advance_pitch(stage: str, name: str | None, project: str | None,
                  vocabulary: Vocabulary | None = None) -> tuple[str, str]:
    """Advance the pitch one rung on a positive reply → (next_stage, line).

    Pure. The line grounds in the project name when known and ALWAYS ends on the question that
    invites the next step, so the caller has exactly one clear thing to respond to and the reply
    never reads as a monologue. ``stage`` must be one of PITCH_LADDER_ACTIVE."""
    g = _greet(name)
    proj = (project or "").strip()
    noun = (vocabulary or _FALLBACK_VOCABULARY).commitment_noun
    if stage == PITCH_VERIFY:
        subj = proj or "This project"
        return PITCH_VALUE, (
            f"That's great to hear{g}. {subj} has genuinely been a favourite — people love the "
            f"location, the layouts and the amenities. Would you like to hear a bit more about it?")
    if stage == PITCH_VALUE:
        where = f" to {proj}" if proj else ""
        return PITCH_OFFER, (
            "It's really the kind of home that's best experienced in person — the space and the "
            "finish don't quite come across over a call. Would you like me to arrange a quick "
            f"{noun}{where} so you can see it for yourself?")
    # PITCH_OFFER accepted → hand to the normal booking engine, which finds the slot.
    return PITCH_SCHEDULING, (
        f"Wonderful{g} — I'd be glad to set that up. What day and time would suit you best?")


def decline_pitch(name: str | None) -> tuple[str, str]:
    """A plain decline anywhere on the ladder: accept it gracefully, offer to send details, and
    stop pitching (operator policy — never push). Returns (PITCH_WIND_DOWN, line)."""
    g = _greet(name)
    return PITCH_WIND_DOWN, (
        f"No problem at all{g} — I won't take up more of your time. Would it help if I sent the "
        f"details across so you can look through them whenever suits you?")


def pitch_query(stage: str, project: str | None, unit: str | None) -> str:
    """The RAG query that fetches the CONCRETE material a pitch rung should lead with — the
    real configurations/sizes for the 'value' rung, the price/amenities for the 'offer' rung.
    turn_runner runs this search (a vector lookup, NOT an LLM call) and hands the passages to the
    responder so the pitch cites facts instead of generic praise. Empty string when the rung
    needs no new facts (the visit/time ask), so no search runs. ``stage`` is the DESTINATION
    rung being delivered this turn (value | offer | scheduling)."""
    proj = (project or "").strip()
    unit = (unit or "").strip()
    if stage == PITCH_VALUE:
        return " ".join(x for x in (unit, "configurations sizes floor plans square feet", proj) if x)
    if stage == PITCH_OFFER:
        return " ".join(x for x in (unit, "price cost per square foot amenities", proj) if x)
    return ""


# The "why we called" phrase per purpose, shared by the gatekeeper reply (which also re-asks
# identity) and the post-confirmation purpose disclosure (which does not) so the two never drift.
_PURPOSE_REASON = {
    PURPOSE_SERVICE: "about your enquiry with us",
    PURPOSE_FEEDBACK: "for quick feedback on your recent site visit",
    PURPOSE_RETENTION: "to follow up on your interest",
    PURPOSE_FOLLOW_UP: "to follow up on your interest",
    PURPOSE_UPSELL: "about a few new options that may interest you",
}


def _purpose_reason(purpose: str | None) -> str:
    return _PURPOSE_REASON.get((purpose or PURPOSE_MARKETING).strip().lower(),
                               "about our latest property options")


def is_purpose_question(text: str | None) -> bool:
    """True when the caller is asking what the call is about / who is calling — a meta question
    about the CALL, not an answer to what we just asked.

    The identity gate handles this BEFORE the right person is confirmed; this predicate lets the
    same "what is this about?" be recognised AFTER confirmation too, where the fast classifier
    tends to mislabel it a product question and the engine would otherwise loop a canned prompt
    (e.g. the project menu) straight back at the caller. Pure and unit-tested."""
    return bool(_GATEKEEPER_RE.search(text or ""))


def purpose_disclosure(purpose: str | None, business: str | None = None,
                       commitment: str | None = None, interest: str | None = None,
                       project: str | None = None) -> str:
    """A brief statement of WHY we called — the reason half of the gatekeeper reply, WITHOUT the
    identity re-ask. Used to answer a mid-call "what is this about?" once identity is already
    confirmed, so the disclosure never re-opens the settled identity question.

    For a SERVICE / feedback / follow-up call it discloses the most CONCRETE subject we hold
    (``_service_subject``): the client's booked visit (``commitment``) if any, else their
    registered enquiry (``interest`` + ``project``) — "about your enquiry about a 3 BHK at DSR
    Highland Greenz" — so a service call never answers "what is this about?" with a vague "your
    enquiry with us", let alone a project brochure. Marketing/upsell ignore all three (there is
    no commitment to a cold pitch); with nothing concrete on file the generic reason stands."""
    b = (business or "").strip()
    lead = f"This is {b}, calling" if b else "We're calling"
    p = (purpose or PURPOSE_MARKETING).strip().lower()
    if p in (PURPOSE_SERVICE, PURPOSE_FEEDBACK, PURPOSE_RETENTION, PURPOSE_FOLLOW_UP):
        subject = _service_subject(commitment, interest, project)
        if subject:
            return (f"{lead} about {subject} — I wanted to make sure everything's on track and "
                    f"see if there's anything I can help you with.")
    return f"{lead} {_purpose_reason(purpose)}."


def gatekeeper_reply(business: str | None, purpose: str | None, name: str | None) -> str:
    """Screening question: disclose who we are and why, briefly, then re-ask identity. A
    transactional purpose is stated plainly (TRAI transparency), never disguised."""
    b = _biz(business)
    first = _first(name)
    who = f"Am I speaking with {first}?" if first else "Am I speaking with the right person?"
    return f"Of course — this is {b}, calling {_purpose_reason(purpose)}. {who}"


def reask_identity(business: str | None, name: str | None) -> str:
    """Unclear answer: ask the identity question once more, plainly."""
    first = _first(name)
    if first:
        return f"Sorry, I didn't quite catch that — am I speaking with {first}?"
    return "Sorry, I didn't quite catch that — am I speaking with the right person?"


# ------------------------------------------------------------------ prompt guidance (by purpose)

# One block, rendered into the system prompt only on an outbound call, that tells the model the
# call's frame and objective. It complements — never replaces — the deterministic identity gate:
# the gate guarantees the precondition (right person, good time) before business begins; this
# shapes tone and goal once it does.
_PURPOSE_GUIDANCE = {
    PURPOSE_MARKETING: (
        "This is an OUTBOUND MARKETING call — we rang the customer. Be warm, brief and "
        "respectful of their time. Goal: spark interest in what fits their stated needs and, "
        "when they are interested, offer a {commitment_noun}. LEAD the call — on every turn "
        "present something concrete about the project from the PROJECT FACTS you were given "
        "(location, configurations, price, what stands out) and move toward booking a "
        "{commitment_noun}; do NOT wait to be asked. If you do not have a specific detail or "
        "fresh inventory to quote, do NOT apologise and stop, and NEVER hand the call to a human "
        "just because you lack an answer — instead offer the {commitment_noun} (that needs no "
        "lookup) or offer to send details for later, and keep leading. Only involve a person if "
        "the customer explicitly asks for one, is upset, or raises a complaint. If they are "
        "genuinely not interested, accept it gracefully, offer to send details, and close — "
        "never push."),
    PURPOSE_SERVICE: (
        "This is an OUTBOUND SERVICE call — we rang the customer about a commitment they ALREADY "
        "have with us: a booked {commitment_noun} or an open enquiry (when one is on file it is "
        "shown in the visit/booking block above). LEAD with THAT commitment — confirm they are "
        "still on for it, remind them of the day and time, and ask whether they need anything or "
        "want to reschedule. If they ask what the call is about, tell them plainly it is about "
        "THEIR {commitment_noun}/commitment — NEVER answer with a description of the project, "
        "its location or its configurations, and never turn a service call into a sales pitch. "
        "Be helpful and efficient."),
    PURPOSE_FEEDBACK: (
        "This is an OUTBOUND FEEDBACK call after a completed {commitment_noun}. Goal: ask "
        "briefly how it went and note their feedback. Keep it short and do not sell."),
    PURPOSE_RETENTION: (
        "This is an OUTBOUND FOLLOW-UP call to re-engage a customer who went quiet. Goal: "
        "reconnect around what has actually changed (new inventory, an offer) and gauge "
        "interest. Be low-pressure."),
}
_PURPOSE_GUIDANCE[PURPOSE_FOLLOW_UP] = _PURPOSE_GUIDANCE[PURPOSE_RETENTION]
_PURPOSE_GUIDANCE[PURPOSE_UPSELL] = _PURPOSE_GUIDANCE[PURPOSE_MARKETING]


def purpose_guidance(purpose: str | None, vocabulary: Vocabulary | None = None) -> str:
    """Vocal Backend Tasksheet P6 Generic Sprint 3: the templates above hold a
    `{commitment_noun}` placeholder wherever they used to hardcode "site visit" — safe to
    `.format()` here (unlike planner/prompts.py's BASE_SYSTEM) since none of this guidance
    text contains a literal `{`/`}` of its own."""
    noun = (vocabulary or _FALLBACK_VOCABULARY).commitment_noun
    template = _PURPOSE_GUIDANCE.get((purpose or PURPOSE_MARKETING).strip().lower(),
                                     _PURPOSE_GUIDANCE[PURPOSE_MARKETING])
    return template.format(commitment_noun=noun)


def _selftest() -> None:
    c = classify_identity_reply
    assert c("Yes, speaking") == BRANCH_CONFIRMED
    assert c("yeah that's me") == BRANCH_CONFIRMED
    assert c("this is he") == BRANCH_CONFIRMED
    assert c("Speaking, who's this?") == BRANCH_CONFIRMED       # confirm wins over screening
    assert c("No, wrong number") == BRANCH_WRONG_PERSON
    assert c("no") == BRANCH_WRONG_PERSON
    assert c("there's nobody by that name here") == BRANCH_WRONG_PERSON
    assert c("he's not here right now") == BRANCH_UNAVAILABLE
    assert c("she is not available") == BRANCH_UNAVAILABLE
    assert c("can I take a message") == BRANCH_UNAVAILABLE
    assert c("I'm driving, call me later") == BRANCH_NOT_NOW
    assert c("this is not a good time") == BRANCH_NOT_NOW
    assert c("who is this?") == BRANCH_GATEKEEPER
    assert c("what is this regarding") == BRANCH_GATEKEEPER
    assert c("how did you get my number") == BRANCH_GATEKEEPER
    assert c("please stop calling me") == BRANCH_OPT_OUT
    assert c("take me off your list") == BRANCH_OPT_OUT
    assert c("do not call me again") == BRANCH_OPT_OUT
    assert c("") == BRANCH_UNCLEAR
    assert c("hmm the weather") == BRANCH_UNCLEAR

    assert "am I speaking with Rohan" in identity_opener("DSR", "Rohan Mehta")
    assert "right person" in identity_opener("DSR", None)
    assert "Rohan" in purpose_opener("marketing", "DSR", "Rohan Mehta")
    assert "feedback" in purpose_opener("feedback_survey", "DSR", "Priya").lower()
    assert "in order" in purpose_opener("service", "DSR", "Priya")

    # SERVICE call leads with the client's booked commitment when one is on file (PRD: a service
    # call confirms/reminds an existing visit, never pitches). commitment_phrase renders it.
    assert commitment_phrase({"when": "Sunday 17 August at 4:00 PM"}) == (
        "your site visit on Sunday 17 August at 4:00 PM")
    assert commitment_phrase({"when": ""}) == "your site visit"   # booked, time unreadable
    assert commitment_phrase(None) == "" and commitment_phrase({}) == ""   # nothing booked
    _svc = purpose_opener("service", "DSR", "Rohan", commitment="your site visit on Wednesday")
    assert "about your site visit on Wednesday" in _svc and "all set" in _svc
    assert "How can I help" not in _svc                       # led with the commitment, not vague
    # _service_subject: most concrete first — visit, else registered enquiry, else nothing.
    assert _service_subject("your site visit on Wed", "3 BHK", "DSR HG") == "your site visit on Wed"
    assert _service_subject(None, "3 BHK", "DSR HG") == "your enquiry about a 3 BHK at DSR HG"
    assert _service_subject(None, None, "DSR HG") == "your enquiry about DSR HG"
    assert _service_subject(None, "3 BHK", None) == "your enquiry about a 3 BHK"
    assert _service_subject(None, None, None) == "" and _service_subject("  ", " ", " ") == ""
    # NO booked visit but a registered enquiry → the service opener leads with the ENQUIRY, not
    # a vague "your enquiry with us" (the exact reported service-call gap).
    _svce = purpose_opener("service", "DSR", "Rohan", project="DSR Highland Greenz",
                           interest="3 BHK")
    assert "about your enquiry about a 3 BHK at DSR Highland Greenz" in _svce
    assert "How can I help" not in _svce
    # nothing concrete on file at all → the generic service opener still stands
    assert "in order" in purpose_opener("service", "DSR", "Rohan")

    # _interest_phrase / _enquiry_recall: the concrete "what you enquired about" fragment.
    assert _interest_phrase("2 BHK") == "a 2 BHK"
    assert _interest_phrase("apartment") == "an apartment"
    assert _interest_phrase("  3bhk   villa ") == "a 3bhk villa"   # tidied, near-verbatim
    assert _interest_phrase("8 BHK penthouse").startswith("an ")   # spoken "eight" → "an"
    assert _interest_phrase("") == "" and _interest_phrase(None) == ""
    assert _enquiry_recall("2 BHK", "DSR Highland Greenz") == "a 2 BHK at DSR Highland Greenz"
    assert _enquiry_recall("2 BHK", None) == "a 2 BHK"
    assert _enquiry_recall(None, "DSR Highland Greenz") == "our DSR Highland Greenz project"
    assert _enquiry_recall(None, None) == "" and _enquiry_recall("  ", "  ") == ""

    # VERIFY-FIRST: when the enquiry unit + project are known, the marketing/upsell/follow-up
    # openers RECALL the concrete interest and ask to confirm it BEFORE pitching new options —
    # they never assert a bare "what you were after" the caller never heard.
    _mk = purpose_opener("marketing", "DSR", "Priya", "DSR Highland Greenz", "2 BHK")
    assert "a 2 BHK at DSR Highland Greenz" in _mk and "is that right?" in _mk
    assert "what you were after" not in _mk
    _up = purpose_opener("upsell", "DSR", "Priya", "Lemon Tree", "3 BHK")
    assert "a 3 BHK at Lemon Tree" in _up and "is that right?" in _up
    _fu = purpose_opener("follow_up", "DSR", "Priya", "Lemon Tree", "3 BHK")
    assert "a 3 BHK at Lemon Tree" in _fu and "still what you're looking for" in _fu
    # Unit unknown but project known → recall the project; still a verification question.
    _mkp = purpose_opener("marketing", "DSR", "Priya", "DSR Highland Greenz")
    assert "our DSR Highland Greenz project" in _mkp and "is that right?" in _mkp
    # Nothing on file → an HONEST discovery question, never a false recall or a dangling "at".
    _mkn = purpose_opener("marketing", "DSR", "Priya")
    assert "is that right?" not in _mkn and "what you're looking for" in _mkn
    assert " at " not in _mkn
    assert " at " not in purpose_opener("marketing", "DSR", "Priya", "   ", "   ")
    assert "wrong" in wrong_person_reply("DSR").lower()
    assert "better time" in offer_callback_reply("Rohan")
    assert "DSR" in gatekeeper_reply("DSR", "marketing", "Rohan")
    assert "speaking with Rohan" in reask_identity("DSR", "Rohan Mehta")

    # is_purpose_question: a meta "what is this about?" is recognised (so it is disclosed, not
    # looped) — but a project choice or a plain "go ahead" is NOT mistaken for one.
    assert is_purpose_question("what is this about?")
    assert is_purpose_question("what's this regarding")
    assert is_purpose_question("what is it about?")           # "it" refers to the call too
    assert is_purpose_question("what's this call about")
    assert is_purpose_question("why did you call me")
    assert is_purpose_question("who is this")
    assert is_purpose_question("why are you calling me")
    assert not is_purpose_question("the second one")
    assert not is_purpose_question("yes tell me")
    assert not is_purpose_question("DSR Highland Greenz")
    assert not is_purpose_question("is it a 2 bhk")           # a product question, not a meta one
    # purpose_disclosure: purpose-aware, and the identity re-ask is gone.
    assert "enquiry" in purpose_disclosure("service")
    assert purpose_disclosure("marketing").startswith("We're calling")
    assert "DSR" in purpose_disclosure("marketing", "DSR")
    assert "speaking with" not in purpose_disclosure("service", "DSR").lower()
    # a service disclosure with a commitment names the VISIT, not a generic reason.
    _svcd = purpose_disclosure("service", "DSR", commitment="your site visit on Wednesday")
    assert "about your site visit on Wednesday" in _svcd and "help" in _svcd
    # no visit but a registered enquiry → the disclosure names the ENQUIRY, never "enquiry with us".
    _svcd2 = purpose_disclosure("service", "DSR", interest="3 BHK", project="DSR Highland Greenz")
    assert "about your enquiry about a 3 BHK at DSR Highland Greenz" in _svcd2
    assert "enquiry with us" not in _svcd2
    # a commitment/enquiry is irrelevant to a marketing pitch — that disclosure ignores them.
    assert "site visit" not in purpose_disclosure(
        "marketing", "DSR", commitment="your site visit on Wednesday", interest="3 BHK")
    # lead_forward_reply: the pitch-purposes drive to a SITE VISIT (no inventory lookup needed),
    # grounding in the registered interest when known; other purposes degrade to a help offer,
    # and NONE of them apologise-and-transfer to a human.
    assert is_lead_purpose("marketing") and is_lead_purpose("upsell")
    assert is_lead_purpose("follow_up") and not is_lead_purpose("service")
    assert not is_lead_purpose("feedback_survey")
    _lf = lead_forward_reply("marketing", "Priya", "DSR Highland Greenz", "2 BHK")
    assert "site visit" in _lf and "about a 2 BHK at DSR Highland Greenz" in _lf
    assert ", Priya" in _lf
    _lfp = lead_forward_reply("upsell", None, "DSR Highland Greenz")
    assert "site visit about our DSR Highland Greenz project" in _lfp
    _lfn = lead_forward_reply("marketing", "Priya")            # nothing on file → no false recall
    assert "site visit" in _lfn and " about " not in _lfn
    _lfs = lead_forward_reply("service", "Priya")              # not a pitch purpose → help offer
    assert "site visit" not in _lfs and "help" in _lfs
    for _r in (_lf, _lfp, _lfn, _lfs):                         # never a human handoff
        assert "colleague" not in _r and "someone" not in _r and "connect you" not in _r

    # ---- outbound pitch ladder: gradual lead, one rung per positive nudge -------------------
    # classify_pitch_reply: bare positives advance, plain declines step aside, real content is
    # left to the engine (never swallowed by the ladder).
    for adv in ("yes", "yeah", "yeah.", "ok", "okay", "sure", "go on", "please do", "tell me more",
                "haan", "theek hai", "sounds good", "why not", "of course", "absolutely",
                # common multi-word confirmations to "…is that right?"
                "that's right", "yes that's right", "that is correct", "correct", "exactly",
                "yeah that's me", "speaking", "it's right"):
        assert classify_pitch_reply(adv) == "advance", adv
    for dec in ("no", "nope", "not interested", "no thanks", "no thank you", "maybe later",
                "already booked", "not right now", "changed my mind"):
        assert classify_pitch_reply(dec) == "decline", dec
    for oth in ("what's the price?", "yeah what about parking", "is there a 2 bhk",
                "tomorrow at 5 pm", "how far is the school", "", "   ",
                "that's not right", "is that right?", "no that's wrong"):
        assert classify_pitch_reply(oth) == "other", oth

    # advance_pitch: verify → value → offer → scheduling, each rung ENDS on a question and the
    # visit is NEVER asked in the very first rung (the reported "don't ask for a visit up front").
    _s1, _v1 = advance_pitch(PITCH_VERIFY, "Priya", "DSR Highland Greenz")
    assert _s1 == PITCH_VALUE and _v1.rstrip().endswith("?")
    assert "DSR Highland Greenz" in _v1 and "favourite" in _v1
    assert "site visit" not in _v1                     # value rung gauges interest, no visit yet
    _s2, _v2 = advance_pitch(PITCH_VALUE, "Priya", "DSR Highland Greenz")
    assert _s2 == PITCH_OFFER and _v2.rstrip().endswith("?")
    assert "site visit to DSR Highland Greenz" in _v2  # NOW the visit is offered
    _s3, _v3 = advance_pitch(PITCH_OFFER, "Priya", "DSR Highland Greenz")
    assert _s3 == PITCH_SCHEDULING and _v3.rstrip().endswith("?")
    assert "day and time" in _v3                       # hands off to booking with a time question
    # No project on file → the rungs still read cleanly (no dangling "to").
    _, _vp = advance_pitch(PITCH_VALUE, "Priya", None)
    assert "site visit so you" in _vp and " to  " not in _vp
    _, _vv = advance_pitch(PITCH_VERIFY, None, None)
    assert _vv.startswith("That's great to hear.") and "This project" in _vv
    # decline_pitch: graceful, offers details, never pushes.
    _sd, _vd = decline_pitch("Priya")
    assert _sd == PITCH_WIND_DOWN and "details" in _vd and _vd.rstrip().endswith("?")
    # pitch_query: the value/offer rungs pull real material; the visit/time ask pulls nothing.
    assert "square feet" in pitch_query(PITCH_VALUE, "DSR Highland Greenz", "3 BHK")
    assert "3 BHK" in pitch_query(PITCH_VALUE, "DSR Highland Greenz", "3 BHK")
    assert "price" in pitch_query(PITCH_OFFER, "DSR Highland Greenz", "3 BHK")
    assert pitch_query(PITCH_SCHEDULING, "DSR Highland Greenz", "3 BHK") == ""
    assert pitch_query(PITCH_VALUE, None, None) and "  " not in pitch_query(PITCH_VALUE, None, None)
    # The active-ladder set is exactly the three pitching rungs.
    assert PITCH_LADDER_ACTIVE == {PITCH_VERIFY, PITCH_VALUE, PITCH_OFFER}
    assert PITCH_SCHEDULING not in PITCH_LADDER_ACTIVE and PITCH_WIND_DOWN not in PITCH_LADDER_ACTIVE

    assert "MARKETING" in purpose_guidance("marketing")
    assert "SERVICE" in purpose_guidance("service")
    assert "FEEDBACK" in purpose_guidance("feedback_survey")
    assert purpose_guidance("anything_else")  # falls back, non-empty
    print("outbound_flow selftest: all passed")


if __name__ == "__main__":
    _selftest()
