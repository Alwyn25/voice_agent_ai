"""Guard: the agent may not promise, perform, or ask to confirm an email send.

There is no email tool anywhere in the tool layer (``tools/definitions.py``'s ``send_document``
only sends over WhatsApp; ``app/services/messaging/`` has no email service). A prompt rule
(prompts.py, "ASKED FOR A CHANNEL YOU DON'T HAVE...") tells the model to offer WhatsApp instead
of transferring when a caller asks for email -- but a prompt instruction is a request, not a
guarantee, the same way the booking/cancellation claim guards exist because telling the model
"don't invent a booking" was not enough on its own.

THE BUG THIS FIXES: a real test asked "can you send the details on email?" and the reply was
"Sure, I can arrange for the details to be emailed from sales@dsrinfra.com. Could you please
confirm the email address you'd like us to use?" -- the model stopped escalating (the original
bug), but replaced it with a worse fabrication: a specific invented sender address that exists
nowhere in the tenant's data, a false claim that email-sending is possible at all, and a request
for the caller's email to act on a promise that can never be kept.

Unlike booking_claim's visit/cancellation guards, there is no "grounding" state to check here --
email is NEVER available, full stop, so any sentence committing to it is always fabricated. Pure
and self-tested (``python -m app.engine.planner.email_claim``); turn_runner calls this
unconditionally, with no ground-truth lookup needed.
"""
import re

from app.engine.planner.text_normalize import normalize_quotes

# Sentences that COMMIT to an email send: a promise ("I'll email", "I can arrange to email"),
# a completed-action claim ("I've emailed", "sent to your email"), or a request for the
# caller's email address to act on either -- all of these imply a capability that does not
# exist. Deliberately NOT matching a bare mention of the word "email" alone (the caller's own
# request repeats it, and the honest redirect to WhatsApp needs to say "can't email it" without
# tripping its own guard).
_EMAIL_ADDRESS_RE_TEXT = r"[\w.+-]+@[\w-]+\.[\w.-]+"
_EMAIL_COMMIT_PATTERNS = (
    # "I'll email you" / "we'll email" / "I will email" / "I can email that" / "we can
    # arrange ... emailed" -- "we" catches the real live-retest miss: the model drafts a
    # company-voice promise ("we'll email the brochure to you shortly") just as often as a
    # first-person one, and the fabrication is identical either way.
    r"\b(?:i|we)(?:'ll| will|'m going to|'re going to) (?:also )?(?:email|mail) "
    r"(?:you|that|this|it|the)\b",
    r"\b(?:i|we) can (?:also )?(?:email|mail) (?:you|that|this|it|the)\b",
    r"\b(?:(?:i|we) can )?arrange(?:d)? (?:for|to have)[^.?!]{0,40}\b(?:emailed|mailed)\b",
    # "I've emailed" / "we've emailed" / "it's been emailed" / "sent to your email"
    r"\b(?:i'?ve|i have|we'?ve|we have) (?:already )?(?:emailed|mailed)\b",
    r"\b(?:sent|emailed|mailed) to your email\b",
    r"\b(?:emailed|mailed) (?:from|to)\s+" + _EMAIL_ADDRESS_RE_TEXT + r"\b",
    # reversed direction: inviting the CALLER to email the business, as opposed to the agent
    # emailing the caller -- "you can email us at sales@dsrinfra.com" is the same fabricated
    # channel-promise in the other direction (nothing turns that inbox into a document send
    # to THIS caller), seen on a live retest of this exact fix.
    r"\byou (?:can|could|may) (?:also )?email (?:us|them|me)\b",
    # asking the caller to confirm/provide/have their email address to act on -- implies a
    # send is about to happen regardless of the verb used to ask for it
    r"\b(?:confirm|provide|share|give me|have|get|take|what(?:'s| is)) (?:your|the) email\b",
    r"\byour email address\b",
    r"\bemail address you(?:'d| would) like (?:us|me) to use\b",
)

# Never caught: a negation ("I can't email", "there's no email option") or an offer that
# explicitly declines email in favour of WhatsApp -- exactly the behaviour the fix wants.
_NEGATION = re.compile(
    r"\b(?:can'?t|cannot|can not|don'?t|do not|doesn'?t|does not|no|not|unable to|"
    r"isn'?t able to|won'?t be able to)\b[^.?!]{0,20}\b(?:email|mail)\b", re.I)


def _sentences(text: str) -> list[str]:
    text = normalize_quotes(text)
    return [s for s in re.split(r"(?<=[.?!])\s+|\n+", text or "") if s.strip()]


def is_fabricated_email_claim(text: str) -> bool:
    """True when some sentence promises, performs, or asks to confirm an email send -- a
    capability that never exists. Always checked unconditionally; there is no ground-truth
    state that could ever make such a claim true."""
    for sent in _sentences(text):
        low = sent.lower()
        if _NEGATION.search(low):
            continue
        for p in _EMAIL_COMMIT_PATTERNS:
            if re.search(p, low):
                return True
    return False


# Honest replacement -- states the real limitation and offers the real alternative, mirroring
# the prompt rule's own wording so the two never contradict each other.
EMAIL_FALLBACK = {
    "english": ("I can't send that over email, but I can send it to you on WhatsApp right "
                "now — would that work?"),
    "hindi": ("मैं वह ईमेल पर नहीं भेज सकता, लेकिन अभी आपको WhatsApp पर भेज सकता हूं — क्या यह ठीक रहेगा?"),
    "hinglish": ("Main woh email par nahi bhej sakta, lekin abhi aapko WhatsApp par bhej "
                "sakta hoon — kya yeh chalega?"),
}


def honest_fallback(*, language: str = "english") -> str:
    """The replacement line. `language` should come from responder.detect_language() on the
    caller's own words this turn -- never guessed here."""
    return EMAIL_FALLBACK.get(language, EMAIL_FALLBACK["english"])


# ---------------------------------------------------------------------------------------
# The upstream mirror of the same bug: the planner can route a bare email-channel request
# into an ESCALATE action via the `explicit_human_request` rule (escalation_policy, keyed on
# the model's own `human_handoff_requested` flag or a "human"-labelled intent -- see
# confidence.INTENT_ALIASES/rules/defaults.py) even though nothing the caller actually SAID
# asked for a person. A real test showed the planner's own reasoning was "Offer WhatsApp as
# alternative" (i.e. it correctly understood the right move) while its structured output's
# human_handoff/intent fields disagreed and forced an escalate anyway -- the two halves of
# one model call contradicting each other, the same unreliable-self-report problem every
# other guard in this package exists to catch downstream of.
#
# Checked against the CALLER's literal words, never the model's self-reported flags: a
# genuine "email me, or let me talk to someone" still escalates (human words present), only
# a bare channel request does not.
_EMAIL_WORD_RE = re.compile(r"\b(?:email|e-?mail|mail(?:ed)?)\b", re.I)
_HUMAN_REQUEST_RE = re.compile(
    r"\b(?:human|a person|manager|supervisor|representative|someone (?:else|real)|"
    r"real (?:agent|person)|live agent|talk to (?:a |an )?(?:person|human|someone|agent)|"
    r"speak (?:to|with) (?:a |an )?(?:person|human|someone|agent|manager)|escalate)\b", re.I)


def is_email_channel_request(text: str) -> bool:
    """True when the caller's own words ask for something via email/a channel we don't have,
    and say nothing that would independently justify a human handover."""
    low = normalize_quotes(text).lower()
    return bool(_EMAIL_WORD_RE.search(low)) and not bool(_HUMAN_REQUEST_RE.search(low))


def _selftest() -> None:
    fabricated = [
        "Sure, I can arrange for the details to be emailed from sales@dsrinfra.com. Could "
        "you please confirm the email address you'd like us to use?",
        "I'll email you the details right away.",
        "I've already emailed the brochure to you.",
        "Sure, the details have been emailed to your email.",
        "Sure, I can email that to you — what's your email?",
        "I can mail you the floor plan from info@dsrinfra.com.",
        "Done, it's been sent to your email.",
        # Live retest of this exact fix: reversed direction (inviting the caller to email
        # the business) plus a different ask-for-address verb ("have" instead of
        # "confirm"/"provide") -- both slipped through the first version of this guard.
        "You can email us at sales@dsrinfra.com for the details; may I have your email "
        "address so the sales team can follow up?",
        "Sure, you can also email us at info@dsrinfra.com.",
        # Live retest, second gap: company-voice "we" instead of "I" -- the actual tool that
        # ran (send_document) only sends over WhatsApp, so this claims a channel that was
        # never used for the thing that actually happened this turn.
        "Sure, we'll email the brochure to you shortly.",
        "We've already emailed the cost sheet to you.",
        # Live retest, third gap: a real production reply used a CURLY apostrophe
        # (U+2019, what prose actually looks like) where every pattern above has a
        # straight one -- this slipped past every "we'll"/"I've" check silently. See
        # text_normalize.py for the fix (normalize before matching, not the patterns).
        "Sure, we’ll email the brochure to you shortly.",
    ]
    for t in fabricated:
        assert is_fabricated_email_claim(t), t

    safe = [
        "I can't send that over email, but I can send it to you on WhatsApp right now — "
        "would that work?",
        "I'm not able to email it, but WhatsApp works just as well — shall I send it there?",
        "There's no email option, but I can share it on WhatsApp instead.",
        "Would you like me to send this on WhatsApp instead?",
        "I can send you the brochure on WhatsApp right now.",
        "Sure, let me get your WhatsApp number.",
        "We can't email it, but we can send it on WhatsApp right now — would that work?",
        "We don't have an email option, but we can share it on WhatsApp instead.",
        # Live retest, third gap -- a genuine factual answer to "what's your contact email"
        # (a real address FROM the tenant's own knowledge base, not an invented one) is not a
        # send-promise at all; a blanket "any email address anywhere" pattern this guard used
        # to carry caught this and overwrote a correct answer with the WhatsApp fallback.
        # Removed once every fabricated case above was confirmed still caught by a targeted
        # pattern without it.
        "Our official email is sales@dsrinfra.com.",
        "You can reach us at sales@dsrinfra.com or through our website.",
    ]
    for t in safe:
        assert not is_fabricated_email_claim(t), t

    assert honest_fallback() == EMAIL_FALLBACK["english"]
    assert honest_fallback(language="hindi") == EMAIL_FALLBACK["hindi"]
    assert honest_fallback(language="klingon") == EMAIL_FALLBACK["english"]

    channel_only = [
        "can you send the details on email?",
        "can you email me that?",
        "please mail me the brochure",
        "send it on email",
    ]
    for t in channel_only:
        assert is_email_channel_request(t), t

    genuine_human = [
        "can you email me, or let me talk to a person instead",
        "I want to speak with a human",
        "put me through to a manager",
        "can I talk to someone real",
    ]
    for t in genuine_human:
        assert not is_email_channel_request(t), t

    no_email_at_all = [
        "I want to speak with a human",
        "what's the price of the 2 BHK",
    ]
    for t in no_email_at_all:
        assert not is_email_channel_request(t), t

    print("email_claim selftest: all passed")


if __name__ == "__main__":
    _selftest()
