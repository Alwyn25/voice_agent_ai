"""Knowledge retrieval (RAG) — pre-model turn triage.

WHAT LIVES HERE NOW. A single, domain-NEUTRAL prefilter — `is_trivial_turn` — that
recognises turns which need neither a document nor any classification at all: greetings,
bare agreement, a dictated number, a deferral, a meta-question, a slot-pick. Those are
answered from the conversation itself, so the engine skips BOTH the fast understanding
call (P1) and retrieval for them. That is a latency optimisation (the plan's §5.5
"skip-calls" routing), not a semantic decision — every pattern below keys off the SHAPE
of the utterance, never its subject, so an engine that sells flats and one that sells
brake pads share this file unchanged.

WHAT USED TO LIVE HERE. A `needs_knowledge()` router that also tried to decide, from raw
words, whether a turn was a *scheduling* turn or a *booking-status* turn (the `SCHEDULING`
regex, the own-records / wants-a-person heuristics). Those are genuine *meaning*
judgements — "a generator is power backup", "is my visit booked is about their record" —
that a keyword pattern can only ever approximate for one operator's vocabulary. They now
come from the fast understanding model (`planner.understanding`), which decides them
semantically; the regexes were deleted with P1. This module no longer routes by subject.
"""
import re

# Turns that never need a document, whatever else is in them.
# Built from parts so combinations work: "yes please", "ok thanks", "yeah go ahead" are
# all just agreement, and listing every pairing by hand would miss most of them.
_CHIT = (r"hi|hey|hello|yes|yeah|yep|no|nope|nah|ok(?:ay)?|sure|thanks?|thank you|"
         r"please|bye|goodbye|correct|right|fine|great|perfect|got it|understood|"
         r"go ahead|do it|book it|sounds good|that works|cheers|alright|"
         r"good morning|good evening|good afternoon|see you|nothing|that's all|"
         r"that is all|leave it|never mind|"
         # Typed greetings arrive misspelled far more often than anything else, and
         # "Helllo" retrieving eight chunks is pure waste on the opening turn of a call.
         r"h+e+l+o+|h+i+|h+e+y+|y+e+a*h*|o+k+")
CHITCHAT = re.compile(rf"^\s*(?:{_CHIT})(?:[\s,]+(?:{_CHIT}))*[\s.,!?]*$", re.I)

# "book the 4", "the 2 one", "make it 3", "12 please" — choosing from times just offered.
# Capped in length so a real question containing a number is untouched.
PICKING_A_TIME = re.compile(
    r"^\s*(?:(?:ok(?:ay)?|yes|yeah|sure|please|lets?|let's)\s+)*"
    r"(?:book|take|make it|do|go with|the|at|for)?\s*(?:the\s+)?"
    r"\d{1,2}\s*(?:o'? ?clock|am|pm|:\d{2})?"
    r"(?:\s+(?:one|slot|please|then|works|is good|sounds good))?[\s.!]*$", re.I)

# Bare agreement. The subject lives in the previous turn, not in these words, so there is
# nothing here worth embedding.
AGREEMENT = re.compile(
    r"^\s*(?:yes|yeah|yep|sure|ok(?:ay)?|alright|please|go ahead|do it|"
    r"sounds good|that works|perfect|great|fine|correct|right|absolutely|definitely)"
    r"(?:[\s,]+(?:please|do|go ahead|sure|thanks?|then|now|it|that))*[\s.!]*$", re.I)

# Someone dictating a number is answering a question, not asking one.
BARE_DATA = re.compile(r"^\s*[\d\s+\-()]{3,20}$")

# Deferring the choice back to the agent — no subject to search on, same as bare
# agreement. "Pick any" was embedded and searched as a product question, matched the
# configurations document, and the customer choosing between the slots just read out
# was told about 1 BHK through 3 BHK availability instead of having one picked for them.
NO_SUBJECT = re.compile(
    r"^\s*(?:(?:just\s+)?(?:pick|choose|take|book)\s+)?"
    r"(?:any(?:\s*(?:one|of them))?|whichever(?:\s+(?:one|works|is fine|is good))?|"
    r"whatever(?:'?s)?\s*(?:works|is fine|is good)?|either(?:\s*(?:one|works|is fine))?|"
    r"(?:you|your)\s+(?:choose|pick|call|decide)|surprise me|doesn'?t matter|"
    r"no preference|i don'?t mind|up to you)[\s.!]*$", re.I)

# A bare question about the AGENT'S OWN LAST WORD — "what detail", "what information",
# "what do you mean" — names no subject a document could answer. The answer is a
# restatement of whatever the agent just asked for, not a knowledge lookup. Asked "what
# detail" right after being asked for a name and phone number to register a booking,
# this retrieved the brochure and answered "flat sizes, costs, and configurations" — a
# confident answer to a question nobody asked, over a customer trying to finish booking.
#
# Deliberately anchored to the WHOLE utterance, same as NO_SUBJECT above. "what details
# do you have on the 2 BHK" names a real subject and must still search normally; only
# the bare form, with nothing else in the sentence, is this.
BARE_META_QUESTION = re.compile(
    r"^\s*(?:what|which)\s+(?:detail|details|information|info|context|documents?)"
    r"(?:\s+(?:do you (?:need|mean|want)|you need|is (?:that|this|needed)))?[\s?.!]*$"
    r"|^\s*what do you mean[\s?.!]*$", re.I)


def is_trivial_turn(text: str) -> bool:
    """True for a turn that needs neither a document nor a fast-model classification.

    Greetings, bare agreement, a dictated number, a deferral back to the agent, a
    meta-question about the agent's own last word, and a slot-pick all carry no subject to
    search and no ambiguity to resolve — the answer is in the conversation, not the
    knowledge base. Recognising them here lets the planner skip the understanding call AND
    retrieval, which is where §5.5's "chitchat / confirmations / slot-picks need 0-1 calls"
    saving comes from.

    This is a pure latency guard, NOT a router: it never decides WHERE an answer comes
    from, only that this shape of utterance needs no lookup. Every branch is about the
    shape of the words, never their subject, so it carries no domain vocabulary.
    """
    low = (text or "").lower().strip()
    # Too short to classify or search — "", "hi", "ok". Whatever it is, no document helps.
    if len(low) < 4:
        return True
    return bool(CHITCHAT.match(low) or BARE_DATA.match(low) or AGREEMENT.match(low)
                or NO_SUBJECT.match(low) or BARE_META_QUESTION.match(low)
                or PICKING_A_TIME.match(low))
