"""Step 2 — fast understanding & routing (the two-tier design's first tier).

WHY THIS EXISTS. Two decisions have to be made BEFORE the strong planner model runs, and
therefore before any model has said anything: (1) could a document answer this turn, so is
retrieval worth firing; (2) is this a scheduling turn or a booking-status turn, whose
answer comes from the scheduler or the customer's record and never from a document. With no
understanding step, those decisions were made by regex over the customer's raw words —
`rag.needs_knowledge`, `SCHEDULING`, `_is_scheduling_aside`, `_asks_booking_status`. A
keyword pattern can only ever approximate meaning for one operator's vocabulary: it never
learned that a "generator" is "power backup" or that "is my visit sorted" is a status
question, and every gap it left became another pattern (the plan's P1 diagnosis).

WHAT IT DOES. One cheap, fast-model structured call that reads the latest customer message
in the context of the recent turns and returns, semantically:

  intent               — a short label, the model's own words
  entities             — the concrete things named this turn (slot → value)
  needs_knowledge      — would answering require the company's documents/knowledge base
  is_scheduling        — is this about slots / times / the act of booking (scheduler, not docs)
  is_status_question   — is this about the state of THEIR OWN existing booking (record, not docs)

The engine combines these into one routing decision (`do_retrieve`) and deletes the regex
router. This is the classify/extract tier the PRD specified as step 2; it was collapsed
into the single planner call to save a round trip, and that collapse is what forced the
regexes into existence.

LATENCY. This adds a model call, which `focus.py` rightly warns against putting in series
on the retrieval path. It is NOT run in series: `planner._understand_and_retrieve` fires it
CONCURRENTLY with a speculative retrieval, so it hides behind an embedding+search that is
happening anyway (the plan's §5.5). It runs on the FAST model — a tiny prompt with a
five-field answer — so on its own it is ~200-400 ms, and it is skipped entirely for trivial
turns (`rag.is_trivial_turn`).

FAILURE POSTURE. If the call fails, the safe default is `needs_knowledge=True` and the two
routing flags False — i.e. retrieve. A wasted embedding is nearly free; an answer invented
because retrieval was wrongly skipped is not.
"""
import json
import time
from dataclasses import dataclass, field

from app.engine.rag import is_trivial_turn
from app.providers.llm.base import LLMError
from app.providers.llm.router import get_llm

# The number of recent turns handed to the classifier for coreference. A follow-up like
# "give me both cost breakdowns" or "what about the north-facing one" only means something
# against what was just said; without a little history the classifier cannot tell a genuine
# product question from a bare continuation. Kept small — this is a fast, cheap call and the
# latest message carries most of the signal.
_CONTEXT_TURNS = 4

UNDERSTANDING_SYSTEM = """You are the fast first-pass classifier for an AI phone agent. \
You do NOT answer the customer and you do NOT choose actions — a second, stronger model \
does that. Your only job is to read the LATEST customer message, in the context of the \
recent turns, and label it so the engine knows where the answer lives.

Judge by MEANING, not by keywords. Bridge vocabulary when you decide needs_knowledge — a \
"generator" is "power backup", a "daycare" is a "creche", a "society" is the "project".

Return these fields about the LATEST customer message:

- intent: a short lowercase label in your own words for what they want, e.g. \
"product_question", "book_appointment", "reschedule", "cancel", "greeting", "affirm", \
"booking_status".

- entities: the concrete things they named THIS turn, as a flat object of \
{"<slot>": "<value>"} — a configuration ("2.5 BHK"), a size, a day, a time, a name, a \
facing. Use {} if they named nothing concrete. Encode it as a JSON string in \
"entities_json".

- needs_knowledge: true if answering them would require looking something up in the \
company's documents or knowledge base — product details, dimensions, amenities, \
specifications, materials, pricing, location, policies. false for greetings, agreement, \
small talk, or anything answerable without any document.

- is_scheduling: true if the message is about appointment availability, slots, times, or \
the act of booking / rescheduling / cancelling a visit — the sort of thing the scheduler \
answers, not a document. A question about the PROPERTY ("do you have a 3 BHK", "how big is \
it") is NOT scheduling. A question about a TIME or a day ("anything Friday afternoon", \
"something on the 19th", "move it to the morning") IS scheduling, even with no \
booking-word in it.

- is_status_question: true if they are asking about THEIR OWN existing appointment or \
booking — its state OR any detail of it: "is my visit booked?", "did it go through?", "am I \
confirmed?", "when is my visit?", "where is it?", "WHO is assigned to it / who's my sales \
person?", "what flat is it for?". Anything asking ABOUT a visit they already have. This is \
answered from their record, never from a document — a request for the assigned salesperson's \
CONTACT is still a status question, not a knowledge lookup.

When is_scheduling or is_status_question is true, needs_knowledge is almost always false — \
the answer is not in a document. Set the flags independently and honestly regardless.

- conversational_signal: whether the LATEST message is ABOUT the conversation or the phone \
line ITSELF rather than about property, booking or the caller's records. This is how the \
agent tells a caller who lost the audio apart from one saying hello for the first time. One of:
  "greeting"    — an opening hello on the caller's FIRST breath of the call, before any real \
exchange has happened.
  "check_in"    — a PRESENCE or AUDIBILITY probe: they think the line dropped, went silent, \
or they could not hear the agent — "hello?", "are you there?", "can you hear me?", "you cut \
out", "you're breaking up", "come again?", "sorry, what?", and the SAME meaning in Hindi / \
Hinglish ("हैलो?", "सुन रहे हैं आप?", "aap sun paa rahe ho?", "awaaz nahi aa rahi"). A bare \
"hello"/"hi" AFTER the agent has already been speaking is a check_in (they are answering \
silence), NOT a greeting.
  "backchannel" — a CONTINUER / nod that only keeps the floor open for the agent, carrying no \
new request and no answer: "mm-hmm", "uh-huh", "go on", "keep going", "right", "I see", \
"got it", and in Hindi/Hinglish "हाँ हाँ", "achha", "theek hai" used as a listening noise.
  "none"        — anything else. ANY turn that carries real content, asks a question, states \
a detail, or is a yes/no ANSWER to something the agent just asked is "none". A "yes"/"ok"/ \
"sure" that could be agreeing to a booking, a visit or a handoff is "none", never \
backchannel — the engine reads consent separately. When in doubt, "none".

- project_choice: usually null. ONLY when the message below carries a "## PROJECT OPTIONS" \
block (a numbered list of projects the caller was just asked to choose between), work out \
which one the LATEST message is pointing at and return that project's EXACT name from the \
list. The caller may point at it in any way — by position ("the second one", "the first"), \
by a place ("the one off Sarjapur Road", "the Whitefield project"), by a partial or brand \
name ("DSR", "the Legend one"), or by a description ("the cheaper one", "the one with \
villas"). Return the single exact name string if one option clearly fits; return null if no \
options block is present, if the message names none of them, or if it is genuinely \
ambiguous between two. Never invent a name that is not in the list."""

# Strict Structured Outputs shape (every object closed, every property required), matching
# the convention in planner._response_schema. entities travel as a JSON string because a
# strict schema cannot describe an object with arbitrary, tool-defined keys.
UNDERSTANDING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "entities_json", "needs_knowledge", "is_scheduling",
                 "is_status_question", "conversational_signal", "project_choice"],
    "properties": {
        "intent": {"type": ["string", "null"]},
        "conversational_signal": {
            "type": "string",
            "enum": ["greeting", "check_in", "backchannel", "none"],
            "description": "Whether the latest message is a greeting, a presence/audibility "
                           "check, a backchannel/nod, or none of these (real content).",
        },
        "entities_json": {
            "type": "string",
            "description": "JSON-encoded flat object of the concrete slots named this turn, "
                           "e.g. {\"bhk_type\": \"2.5 BHK\"}. \"{}\" if nothing was named.",
        },
        "needs_knowledge": {"type": "boolean"},
        "is_scheduling": {"type": "boolean"},
        "is_status_question": {"type": "boolean"},
        "project_choice": {
            "type": ["string", "null"],
            "description": "Only when a PROJECT OPTIONS block is present: the EXACT name of "
                           "the project the caller is pointing at, else null.",
        },
    },
}

# The consent read (consent.py, Layer B). Appended to the prompt and the schema ONLY on a turn
# where an action is parked awaiting the caller's yes — every other turn sends the unchanged
# prompt and schema above, so it costs nothing there. The engine never trusts this alone: an
# "accept" is honoured only when the quoted evidence is really in the transcript (see
# consent.decide), which is what stops a garbled reply from being read as a yes.
CONSENT_SYSTEM = """

- consent: the agent has just asked the caller to confirm an action (shown under \
"## AWAITING CONFIRMATION"). Read the LATEST customer message as a reply to THAT question, by \
meaning, in any language or phrasing, and return:
  decision — exactly one of:
    "accept"        — they clearly agree to the action as asked ("yes", "sure, let's do it", \
"go for it", "haan kar do", "हाँ बुक कर दीजिए").
    "decline"       — they refuse or change their mind ("no", "I changed my mind", "leave it", \
"not now", "rehne do", "मत कीजिए").
    "change"        — they want different details (another time/day/place: "make it 11", \
"can we do tomorrow?").
    "repeat"        — they want to hear the details again or did not catch them ("what time?", \
"say that again", "sorry?", "phir se bolo", "कितने बजे?").
    "side_question" — they ask about something else without answering ("is there parking?").
    "hold"          — they want time before deciding ("wait", "let me think", "not sure").
    "unclear"       — you cannot tell, the message is garbled or cut off, or it only partly \
fits a category. When in doubt, "unclear" — never guess "accept".
  evidence — the EXACT words copied from the latest customer message that carry the decision \
(for "accept", the words of agreement themselves). "" if there are none. Never paraphrase."""

CONSENT_SCHEMA_PROPERTY = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "evidence"],
    "properties": {
        "decision": {"type": "string",
                     "enum": ["accept", "decline", "change", "repeat", "side_question",
                              "hold", "unclear"]},
        "evidence": {"type": "string",
                     "description": "Exact words from the latest customer message that "
                                    "carry the decision; \"\" if none."},
    },
}


def _schema_for(awaiting_confirmation: str | None) -> dict:
    """The understanding schema, with the consent field added only while awaiting a yes."""
    if not awaiting_confirmation:
        return UNDERSTANDING_SCHEMA
    return {**UNDERSTANDING_SCHEMA,
            "required": [*UNDERSTANDING_SCHEMA["required"], "consent"],
            "properties": {**UNDERSTANDING_SCHEMA["properties"],
                           "consent": CONSENT_SCHEMA_PROPERTY}}


@dataclass
class Understanding:
    """The fast pre-pass result. `source` records HOW it was produced so the trace can tell
    a model classification from a skipped-trivial turn from a failure fallback."""
    intent: str | None = None
    entities: dict = field(default_factory=dict)
    needs_knowledge: bool = True
    is_scheduling: bool = False
    is_status_question: bool = False
    # The semantic read of whether this turn is ABOUT the conversation/line itself — a greeting,
    # a presence/audibility check ("can you hear me?"), a backchannel nod, or none (real
    # content). "none" on every path where the model did not run (trivial/fallback), where the
    # deterministic conversational_signal module in turn_runner is the fast, English, consent-
    # guarded first pass; this LLM field is the multilingual / long-tail second pass.
    conversational_signal: str = "none"
    # The project the caller pointed at, when a PROJECT OPTIONS block was supplied (the caller
    # is answering "which project?"). The exact candidate name, or None. Ignored on every turn
    # where no options were offered — see understand(candidates=...).
    project_choice: str | None = None
    # The consent read (consent.py Layer B): {"decision", "evidence"} as the model returned
    # them, or None on every turn where no action was awaiting a yes (or the model did not
    # run). Raw on purpose — consent.decide validates it against the transcript.
    consent: dict | None = None
    source: str = "model"                       # model | trivial | fallback
    model: str = ""
    tokens: dict = field(default_factory=dict)
    latency_ms: int = 0

    @property
    def do_retrieve(self) -> bool:
        """The single routing decision the engine acts on: fetch documents only when a
        document could answer AND the turn is not routed to the scheduler or the customer's
        own record. Combined here, in code, rather than trusted to any one model flag."""
        return (self.needs_knowledge and not self.is_scheduling
                and not self.is_status_question)

    def as_trace(self) -> dict:
        """The audit-trail view — what the engine understood and decided, before the strong
        planner call. Kept flat and JSON-friendly for the decision trace."""
        out = {"intent": self.intent, "entities": self.entities,
               "needs_knowledge": self.needs_knowledge, "is_scheduling": self.is_scheduling,
               "is_status_question": self.is_status_question, "do_retrieve": self.do_retrieve,
               "conversational_signal": self.conversational_signal,
               "project_choice": self.project_choice,
               "source": self.source, "model": self.model, "latency_ms": self.latency_ms}
        if self.consent is not None:
            out["consent"] = self.consent
        return out


def _parse_entities(value) -> dict:
    """entities_json arrives as a JSON string under Structured Outputs; the mock provider
    and tests may pass a dict directly. Accept either, and never let a malformed value reach
    the caller as though it were real extraction."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _match_candidate(choice, candidates: list[str] | None) -> str | None:
    """The offered candidate the model's ``project_choice`` refers to, or None. The classifier
    is asked for an exact name, but a lenient match (case-insensitive, and accepting the bare
    project name behind a "Name (location)" label) keeps a trivially-reworded answer from being
    thrown away, while an off-list string is still rejected — it must resolve to one we offered.
    """
    if not choice or not candidates:
        return None
    want = str(choice).strip().lower()
    if not want:
        return None
    for cand in candidates:
        low = cand.strip().lower()
        # "DSR Highland Greenz (off Sarjapur Road)" — the identity is the part before " (".
        bare = low.split(" (", 1)[0].strip()
        if want in (low, bare):
            return cand
    return None


def _recent_context(history: list[dict] | None) -> str:
    """The last few turns as plain "Role: text" lines, for coreference. The current
    customer utterance is passed separately and is NOT included here."""
    lines = []
    for turn in (history or [])[-_CONTEXT_TURNS:]:
        role = str(turn.get("role", "")).strip()
        content = str(turn.get("content", "")).strip()
        # System bookkeeping lines ("action: …") are noise to a meaning classifier.
        if not content or role == "system":
            continue
        speaker = "Customer" if role == "customer" else "Agent"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


async def understand(user_input: str, *, history: list[dict] | None = None,
                     model: str | None = None,
                     candidates: list[str] | None = None,
                     asr_note: str | None = None,
                     awaiting_confirmation: str | None = None) -> Understanding:
    """Classify and route the latest customer message. Never raises — a model failure
    degrades to the safe "retrieve anyway" default rather than losing the turn.

    ``model`` overrides the classifier model; it defaults to the fast tier at the call site
    (planner passes the literal string ``"fast"``), which every provider maps onto its own
    fast model.

    ``candidates`` — the project options the caller was just asked to choose between, in the
    order they were offered. Supplied ONLY on a turn where the project is still ambiguous, so
    the classifier can resolve a free-form answer ("the cheaper one") that the deterministic
    selectors missed. This rides on the call that already runs — it adds NO extra model call.
    When None/empty, project_choice is left null and nothing changes.

    ``asr_note`` — on a voice turn, the words in the transcript that were most likely
    mis-heard and the term each probably was (prompts.asr_hint_note). Placed right before the
    message so routing follows the likely term: "when is the position" is a product question
    about possession, not a status check. None (chat, or nothing flagged) changes nothing.

    ``awaiting_confirmation`` — the exact question the agent just asked to confirm a parked
    action ("Shall I book your visit for …?"). When given, the same call also returns the
    ``consent`` read (consent.py Layer B). None — every turn with nothing parked — sends the
    unchanged prompt and schema. A trivial turn ("yes", "ok") still skips the model: the
    deterministic consent layer reads those on its own.
    """
    # Trivial turns (greetings, agreement, slot-picks, bare data) carry no subject and no
    # ambiguity — skip the model call entirely. This is the §5.5 skip-calls saving and it
    # keeps "hi" / "yes" / "book the 4" at zero classification cost.
    #
    # BUT: when the caller is answering "which project?", even a one-word reply ("DSR", "the
    # second") must be resolved — that is the whole point of passing candidates — so a trivial
    # utterance is NOT skipped while options are on the table.
    if is_trivial_turn(user_input) and not candidates:
        return Understanding(needs_knowledge=False, source="trivial")

    started = time.perf_counter()
    context = _recent_context(history)
    options_block = ""
    if candidates:
        listed = "\n".join(f"{i}. {name}" for i, name in enumerate(candidates, start=1))
        options_block = ("## PROJECT OPTIONS\nThe caller was just asked which of these "
                         "projects they mean. Resolve project_choice to the exact name of the "
                         f"one the latest message points at, or null.\n{listed}\n\n")
    awaiting_block = (f"## AWAITING CONFIRMATION\nAgent asked: {awaiting_confirmation}\n\n"
                      if awaiting_confirmation else "")
    user = (f"## RECENT TURNS\n{context}\n\n" if context else "") + \
           options_block + awaiting_block + \
           (f"## {asr_note}\n\n" if asr_note else "") + \
           f"## LATEST CUSTOMER MESSAGE\n{user_input}"
    try:
        result = await get_llm().complete_json(
            system=(UNDERSTANDING_SYSTEM + CONSENT_SYSTEM if awaiting_confirmation
                    else UNDERSTANDING_SYSTEM),
            user=user, model=model,
            # Deterministic: this is a classification, not phrasing — variation here is only
            # a chance to route the turn differently on a re-run. A five-field answer needs
            # very little room; the consent field adds ~20-30 tokens when present.
            temperature=0.0, max_tokens=360 if awaiting_confirmation else 300,
            schema=_schema_for(awaiting_confirmation), schema_name="understanding")
        d = result.data or {}
        _consent = d.get("consent") if awaiting_confirmation else None
        return Understanding(
            intent=(str(d["intent"]) if d.get("intent") else None),
            entities=_parse_entities(d.get("entities_json", d.get("entities"))),
            # Default-True on a missing flag is deliberate: a stripped answer should retrieve,
            # not stay silent. The two routing flags default False for the same reason —
            # absent evidence must not divert the turn away from the knowledge base.
            needs_knowledge=bool(d.get("needs_knowledge", True)),
            is_scheduling=bool(d.get("is_scheduling", False)),
            is_status_question=bool(d.get("is_status_question", False)),
            # An off-enum or missing value degrades to "none" — the safe default that leaves
            # the turn to the normal engine rather than overriding a reply on a bad read.
            conversational_signal=(
                d.get("conversational_signal")
                if d.get("conversational_signal") in ("greeting", "check_in", "backchannel")
                else "none"),
            # Accept the choice only if it is actually one of the options we offered — the
            # model must select, never rename or invent. Off-list answers become None.
            project_choice=_match_candidate(d.get("project_choice"), candidates),
            consent=(_consent if isinstance(_consent, dict) else None),
            source="model", model=result.model, tokens=result.tokens,
            latency_ms=int((time.perf_counter() - started) * 1000))
    except LLMError:
        # Safe default — retrieve. Skipping retrieval when it was actually needed is how the
        # agent invents answers; a wasted embedding when it was not is nearly free.
        return Understanding(needs_knowledge=True, source="fallback",
                             latency_ms=int((time.perf_counter() - started) * 1000))
