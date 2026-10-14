"""Guard: the agent may not tell a caller they have a visit unless a real one exists.

The only text the caller hears is the responder's, which regenerates the reply from the
draft, so a confident sentence proves nothing about what actually happened this turn. Two
shapes of this same fabrication reach the caller:

  * "you already have a visit booked" / "your visit is confirmed for Saturday at 2 PM" --
    asserts a PRE-EXISTING appointment, a fact that comes from the database
    (``existing_appointments``), not from anything that ran this turn. An unidentified
    caller ("I'd like to see the 3BHK") has an empty ``existing_appointments``, so the model
    can invent a visit from the booking intent alone, complete with a day and time the
    caller never chose.
  * "I've booked a visit for you" / "your 3 BHK visit on October 7th is booked" -- asserts
    the AGENT ITSELF just completed a booking THIS turn. The planner can be entirely honest
    here (``action_taken`` genuinely still "clarify", no tool proposed) while the responder
    fabricates a completion sentence anyway -- a real production call confirmed a non-
    existent booking three times over, with an invented confirmation, while the decision
    trace shows the planner never had the caller's name and never called book_appointment.

Neither has a tool call the deterministic ACTION STATE can check against for an unidentified
caller, so only a check of the SENTENCE itself, against whether an appointment genuinely
exists or was just established, can catch either.

This is the deterministic net for both: if the reply asserts an on-file or just-completed
visit and nothing grounds it (no row in ``existing_appointments``, no book_appointment /
reschedule_appointment that actually succeeded this turn), the claim is replaced. Pure and
self-tested (``python -m app.engine.planner.booking_claim``); the wiring in turn_runner
supplies the grounding facts.
"""
import re

from app.engine.planner.text_normalize import normalize_quotes

# Assertions that the caller ALREADY has / currently has a visit. Present-tense statements of
# fact, not offers or questions — "would you like to book?" and "shall I book it?" are handled
# by the offer/question exclusion below, never here.
_CLAIM_PATTERNS = (
    r"\byou(?:'| a)?re (?:already )?booked\b",
    r"\byou (?:already )?have (?:a|an|your|one|another) (?:visit|appointment|booking|slot|site visit)\b",
    r"\byou already have\b",
    r"\byour (?:visit|appointment|booking|site visit|slot) (?:is|was|has been|remains) "
    r"(?:confirmed|booked|scheduled|set|arranged|on|for|at)\b",
    r"\b(?:there(?:'s| is)|i (?:can )?see) (?:a|an|your) (?:visit|appointment|booking|site visit)\b"
    r"[^.?!]{0,40}\b(?:on file|on record|booked|scheduled|confirmed|for you|under your)\b",
    r"\byou(?:'| ha)?ve got (?:a|an|your) (?:visit|appointment|booking)\b",
    # Same "your <visit-noun> ... is booked" claim as the pattern above, but with a
    # descriptive clause (configuration, date, time) between the noun and the verb --
    # "your 3 BHK visit on October 7th in the evening is booked" never matched that pattern,
    # which requires the verb immediately after the noun.
    r"\byour\b[^.?!]{0,60}?\b(?:visit|appointment|booking|site visit|slot)\b[^.?!]{0,60}?\b"
    r"(?:is|was|has been|remains|is now)\s+(?:confirmed|booked|scheduled|set up|arranged)\b",
    # First-person "I've/I have booked/confirmed/arranged/set up ... visit" -- the agent
    # claiming IT just completed a booking this turn, as opposed to asserting a pre-existing
    # one. Same fabrication risk (nothing grounds it without a book_appointment/
    # reschedule_appointment that actually succeeded this turn), just phrased as the agent's
    # own completed action rather than a fact about the caller's history -- the grounding
    # check below (_appt_grounded in turn_runner.py) already covers exactly this case via
    # `executed`, it just never ran because nothing detected the claim. Deliberately anchored
    # to a completed-action auxiliary ("'ve"/"have") so a genuine offer ("I'll set up a
    # visit", "I can book that for you") is never caught -- future tense promises nothing yet.
    r"\bi(?:'ve| have) (?:already )?(?:booked|confirmed|arranged|set up|scheduled) "
    r"(?:a|an|your|the|one) (?:visit|appointment|booking|site visit)\b",
)

# The claim is void when the sentence around it is a NEGATION ("I don't see a booking",
# "you do not have a visit") or an OFFER/QUESTION ("would you like to book", "shall I set one
# up"). These share the same nouns as a genuine claim, so a bare noun match is not enough.
_NEGATION = re.compile(
    r"\b(?:don'?t|do not|doesn'?t|does not|no|not|never|without|couldn'?t find|can'?t find|"
    r"cannot find|unable to find)\b", re.I)
_OFFER = re.compile(
    r"\b(?:would you like|shall i|do you want|can i (?:help|set|book|arrange|get)|"
    r"i can (?:help|set|book|arrange)|let me know if you|if you(?:'d| would) like|"
    r"like me to|want me to)\b", re.I)


# A claim introduced by one of these is a CONDITION, not a statement of fact: "once you confirm,
# your visit for Saturday is booked" / "if your visit is confirmed…". Checked against the words
# directly before the matched claim, so a condition elsewhere in the sentence doesn't hide a
# real claim.
_CONDITION_WORDS = r"(?:if|once|when|after|until|unless|before|as soon as)"
_CONDITIONAL_LEAD = re.compile(
    # the sentence opens with a condition ("Once you say yes, your visit … is booked")
    rf"^\W*{_CONDITION_WORDS}\b(?!\s+again\b)"
    # or the claim's "your" comes straight after one ("…and if your visit is confirmed")
    rf"|\b{_CONDITION_WORDS}\s*$")


def _sentences(text: str) -> list[str]:
    text = normalize_quotes(text)
    return [s for s in re.split(r"(?<=[.?!])\s+|\n+", text or "") if s.strip()]


def asserts_existing_visit(text: str) -> bool:
    """True when SOME sentence states, as fact, that the caller has a visit — and that
    sentence is neither a negation of one, an offer to make one, nor a condition."""
    for sent in _sentences(text):
        low = sent.lower()
        if _NEGATION.search(low) or _OFFER.search(low):
            continue
        for p in _CLAIM_PATTERNS:
            m = re.search(p, low)
            if m and not _CONDITIONAL_LEAD.search(low[:m.start()]):
                return True
    return False


def is_ungrounded_visit_claim(text: str, *, has_appointment_on_file: bool) -> bool:
    """The reply claims an on-file visit that nothing supports.

    ``has_appointment_on_file`` is the ONLY thing that makes such a claim legitimate: a real
    row in ``existing_appointments``, or an appointment a lookup/booking established THIS turn.
    Computed by the caller (turn_runner) from ground truth, never guessed here.
    """
    if has_appointment_on_file:
        return False
    return asserts_existing_visit(text)


# Honest replacements — they assert NO booking either way (an unidentified caller's history is
# genuinely unknown to us) and move to the constructive next step instead of inventing a visit.
#
# Keyed by language and written by hand, not translated by the LLM at call time: this text
# replaces something the model got WRONG (a fabricated visit), so the replacement itself must
# never go anywhere near a model that could get it wrong too. A caller speaking Hindi got this
# in English regardless of what they'd said all call — the string was never anything but the
# one English sentence, so "match the customer's language" (prompts.py) had nothing to act on
# here at all; this bypasses the responder LLM by design (see turn_runner.py's call site).
FALLBACK_WANTS_VISIT = {
    "english": ("I don't have a visit on record for you on this call. I'd be happy to set one "
               "up — can I take your name to get started?"),
    "hindi": ("इस कॉल पर मुझे आपकी कोई विजिट रिकॉर्ड में नहीं दिख रही। मुझे एक बुक करने में खुशी होगी — "
             "क्या मैं आपका नाम ले सकता हूं?"),
    "hinglish": ("Is call par mujhe aapki koi visit record mein nahi dikh rahi. Main ek book "
                "karne mein khushi se madad karunga — shuru karne ke liye aapka naam bata "
                "sakte hain?"),
}
FALLBACK_STATUS_QUERY = {
    "english": ("I'm not seeing a visit on record for you on this call. If you booked on an "
               "earlier call, tell me the name or phone number it's under and I'll check for "
               "you."),
    "hindi": ("इस कॉल पर मुझे आपकी कोई विजिट रिकॉर्ड में नहीं दिख रही। अगर आपने पहले किसी कॉल पर बुक की थी, "
             "तो वह नाम या मोबाइल नंबर बताइए जिस पर बुक हुई थी, मैं जांच कर बताता हूं।"),
    "hinglish": ("Is call par mujhe aapki koi visit record mein nahi dikh rahi. Agar aapne "
                "pehle kisi call par book ki thi, to woh naam ya mobile number bataiye jis par "
                "book hui thi, main check karke bataata hoon."),
}


def honest_fallback(*, wants_to_book: bool, language: str = "english") -> str:
    """The replacement line. A caller trying to book is moved toward booking; a caller asking
    about a booking is offered a lookup once they identify themselves. `language` should come
    from responder.detect_language() on the caller's own words this turn — never guessed here."""
    table = FALLBACK_WANTS_VISIT if wants_to_book else FALLBACK_STATUS_QUERY
    return table.get(language, table["english"])


# ---------------------------------------------------------------------------------------
# The mirror-image guard: the agent may not tell a caller their visit was CANCELLED unless
# cancel_appointment actually succeeded this turn.
#
# THE BUG THIS FIXES: a real call had the planner correctly decide action="respond", no tool
# called, reasoning "customer declined location pin offer" -- entirely unrelated to the
# booking. The responder (a separate LLM call that writes the final spoken text from the
# conversation's context, not just the planner's draft) independently said "Your visit... has
# been cancelled" -- latching onto an unanswered "keep it or cancel it?" question the agent
# itself had asked several turns earlier and resolving it on its own, disconnected from what
# the planner decided and what actually ran. The real appointment was untouched in the
# database; the caller was told otherwise. Same failure shape booking_claim already guards
# against for a fabricated BOOKING ("I've booked a visit for you" with nothing booked) --
# this is the same thing in the cancellation direction, which had no guard at all.
_CANCEL_CLAIM_PATTERNS = (
    # "your visit ... has been cancelled" -- a descriptive clause (date, time) is allowed
    # between the noun and the verb, same qualifier gap booking_claim's own patterns above
    # already had to close: "your 3 BHK visit on October 7th in the evening is booked" needed
    # the same fix, and the exact real call this guard exists for has the identical shape
    # ("your visit scheduled for tomorrow, Saturday 10 October at 11:00 AM, has been
    # cancelled") -- a verb-immediately-after-noun pattern alone would miss it.
    r"\byour\b[^.?!]{0,60}?\b(?:visit|appointment|booking|site visit|slot)\b[^.?!]{0,60}?\b"
    r"(?:has been|is|was|has)\s+(?:cancelled|canceled|removed|called off)\b",
    # "I've cancelled your 3 BHK visit" -- a configuration/qualifier between "your" and
    # "visit" is just as common here as in the booking-claim forms above; a live retest of
    # this exact fix caught it missing this gap on the first try.
    r"\b(?:i'?ve|i have) (?:already )?(?:cancelled|canceled|removed|called off) "
    r"(?:your|the|that)\b[^.?!]{0,30}?\b(?:visit|appointment|booking|site visit)\b",
    r"\byour\b[^.?!]{0,60}?\b(?:visit|appointment|booking|site visit|slot)\b[^.?!]{0,60}?\b"
    r"(?:is|has been)\s+(?:no longer|not)\s+(?:scheduled|booked|on (?:file|the books|the calendar))\b",
    r"\b(?:we'?ve|we have) (?:cancelled|canceled|removed) (?:your|the) "
    r"(?:visit|appointment|booking|site visit)\b",
)

# Own offer/negation exclusions, deliberately NOT the shared _NEGATION/_OFFER above: those
# include a bare "\bno\b", which collides with "no longer scheduled" -- part of the claim
# language itself here, not a negation of it. "Shall I cancel it?", "would you like to
# cancel", "I haven't/can't/didn't cancel" must never match; a bare "no longer" must.
_CANCEL_OFFER = re.compile(
    r"\b(?:would you like|shall i|do you want|should i|want me to|like me to) "
    r"[^.?!]{0,30}\bcancel\b", re.I)
_CANCEL_NEGATION = re.compile(
    r"\b(?:haven'?t|hasn'?t|can'?t|cannot|couldn'?t|didn'?t|won'?t|never) "
    r"[^.?!]{0,20}\bcancel", re.I)


def asserts_visit_cancelled(text: str) -> bool:
    """True when SOME sentence states, as fact, that the visit was just cancelled — and that
    sentence is neither an offer/question about cancelling nor a negation of one."""
    for sent in _sentences(text):
        low = sent.lower()
        if _CANCEL_NEGATION.search(low) or _OFFER.search(low) or _CANCEL_OFFER.search(low):
            continue
        for p in _CANCEL_CLAIM_PATTERNS:
            m = re.search(p, low)
            if m and not _CONDITIONAL_LEAD.search(low[:m.start()]):
                return True
    return False


def is_ungrounded_cancellation_claim(text: str, *, cancellation_grounded: bool) -> bool:
    """The reply claims a visit was cancelled that nothing supports.

    ``cancellation_grounded`` is the ONLY thing that makes such a claim legitimate: a
    cancel_appointment call that actually succeeded THIS turn. Computed by the caller
    (turn_runner) from the real tool outcomes, never guessed here.
    """
    if cancellation_grounded:
        return False
    return asserts_visit_cancelled(text)


# Honest replacement — states plainly that the visit is still on (the only thing we actually
# know, since nothing cancelled it) rather than silently dropping the sentence, which would
# leave the caller with no idea their "cancellation" was never real.
#
# THE BUG THIS FIXES: the original fallback ended with "Let me know if you'd actually like
# me to cancel or move it." -- a real call hit this guard twice in one conversation (both
# times correctly: the responder had fabricated a cancellation claim from a bare "no"
# declining an unrelated offer, decision_traces confirmed the planner's own reasoning was
# right both times -- "ask next step", "declined reminder offer") and spoke the IDENTICAL
# canned invitation both times. The guard was doing its job -- stopping a false claim -- but
# its own replacement text re-introduced the exact unprompted "would you like to cancel"
# pattern a separate prompt rule (prompts.py, "declining X is not Y") exists to stop. State
# the fact and stop there; the caller can always ask to cancel or move it themselves; this
# fallback firing at all is never a sign the caller signalled wanting to.
CANCELLATION_FALLBACK = {
    "english": "Your visit is still on the books as scheduled — nothing's been cancelled.",
    "hindi": "आपकी विजिट अभी भी शेड्यूल के अनुसार बुक है — कुछ भी कैंसिल नहीं हुआ है।",
    "hinglish": "Aapki visit abhi bhi schedule ke according booked hai — kuch cancel nahi hua hai.",
}


def cancellation_honest_fallback(*, language: str = "english") -> str:
    """The replacement line when a cancellation was claimed but never happened. `language`
    should come from responder.detect_language() on the caller's own words this turn."""
    return CANCELLATION_FALLBACK.get(language, CANCELLATION_FALLBACK["english"])


def _selftest() -> None:
    # Fabrications that MUST be caught (no grounding).
    fabricated = [
        "I see you want to see the 3 BHK in person, but you already have a visit booked.",
        "Your visit is confirmed for Saturday at 2 PM.",
        "You're already booked for a site visit.",
        "You have an appointment scheduled for tomorrow.",
        "There is a booking on file for you.",
        # From a real production call (2026-10-06): the planner never had the caller's name
        # and never proposed book_appointment (decision trace: action_taken stayed "clarify"
        # every turn), yet the responder told the caller this three times, with an invented
        # confirmation number -- no row in appointments, no customer_id on the conversation.
        "Your 3 BHK visit on October 7th in the evening is booked, and the confirmation "
        "will be sent to 78745784.",
        "Great, I've booked a visit for a 3 BHK on 7 October 2026 in the evening.",
        # A qualifier between the noun and the verb (seen live on a GTM demo, after only a
        # slot search).
        "Your visit for the 2 BHK at Sumadhura Edition is confirmed for tomorrow, Tuesday "
        "06 October at 11:00 AM. Looking forward to seeing you then!",
        "Your site visit on Tuesday at 11 AM has been booked.",
        "Your appointment with our sales team is now confirmed.",
    ]
    for t in fabricated:
        assert is_ungrounded_visit_claim(t, has_appointment_on_file=False), t
        # ...but NEVER when a real appointment backs it.
        assert not is_ungrounded_visit_claim(t, has_appointment_on_file=True), t

    # Offers, questions and negations are NOT claims — must never be replaced.
    safe = [
        "Would you like me to book a visit for you?",
        "Shall I book your visit for Saturday at 2 PM?",
        "I can set up a site visit — what day works?",
        "I don't have a visit on record for you on this call.",
        "You do not have any appointment booked with us yet.",
        "I couldn't find a booking under that number.",
        "Let me know if you'd like to book a viewing.",
        # Future tense ("I'll") is a promise, not a completion claim -- must never be caught
        # by the new first-person pattern just because it shares "visit" and a verb stem.
        "Great, I'll set up a visit for a 3 BHK on the evening of October 7th.",
        # Qualified forms that are still offers, questions, negations or conditions.
        "Would you like your visit for Saturday to be confirmed?",
        "Your visit for Saturday is not confirmed yet — shall I go ahead?",
        "Once you say yes, your visit for tomorrow at 11 AM is booked.",
        "If your visit for Saturday is confirmed, we'll send the location.",
        # Descriptions, not claims: no strong verb after the qualifier.
        "Your visit to the site is for about 30 minutes.",
        "Your visit for the 2 BHK can be tomorrow at 11 AM or 4 PM.",
    ]
    for t in safe:
        assert not is_ungrounded_visit_claim(t, has_appointment_on_file=False), t

    assert honest_fallback(wants_to_book=True) == FALLBACK_WANTS_VISIT["english"]
    assert honest_fallback(wants_to_book=False) == FALLBACK_STATUS_QUERY["english"]
    assert honest_fallback(wants_to_book=True, language="hindi") == FALLBACK_WANTS_VISIT["hindi"]
    assert honest_fallback(wants_to_book=False, language="hinglish") == FALLBACK_STATUS_QUERY["hinglish"]
    assert honest_fallback(wants_to_book=True, language="klingon") == FALLBACK_WANTS_VISIT["english"]

    # --- cancellation claims ---
    cancel_fabricated = [
        # the exact real call: planner said action=respond, no tool ran, reasoning was
        # "customer declined location pin offer" -- nothing to do with cancelling at all
        "Your visit scheduled for tomorrow, Saturday 10 October at 11:00 AM, has been "
        "cancelled. Let me know if you'd like to reschedule.",
        "Your appointment is cancelled.",
        "I've cancelled your visit.",
        # a second live retest of this exact fix: this phrasing (verb BEFORE "your", a
        # configuration qualifier between "your" and "visit") slipped through the first
        # version of the "I've cancelled your..." pattern above
        "I've cancelled your 3 BHK visit scheduled for tomorrow, Saturday 10 October at "
        "10 AM. Let me know if you'd like to arrange another time.",
        "We've removed your visit from the calendar.",
        "Your booking is no longer scheduled.",
    ]
    for t in cancel_fabricated:
        assert is_ungrounded_cancellation_claim(t, cancellation_grounded=False), t
        assert not is_ungrounded_cancellation_claim(t, cancellation_grounded=True), t

    cancel_safe = [
        "Shall I cancel your visit for Saturday?",
        "Would you like me to cancel it?",
        "I haven't cancelled anything yet.",
        "I can't cancel that without your confirmation first.",
        "Your visit is still booked for Saturday at 2 PM.",
    ]
    for t in cancel_safe:
        assert not is_ungrounded_cancellation_claim(t, cancellation_grounded=False), t

    assert cancellation_honest_fallback() == CANCELLATION_FALLBACK["english"]
    assert cancellation_honest_fallback(language="hindi") == CANCELLATION_FALLBACK["hindi"]
    assert cancellation_honest_fallback(language="klingon") == CANCELLATION_FALLBACK["english"]

    print("booking_claim selftest: all passed")


if __name__ == "__main__":
    _selftest()
