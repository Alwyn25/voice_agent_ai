"""Step 11 — grounded response generation.

The model writes the final sentence using ONLY verified material: the result of the
tool that just ran, plus any retrieved knowledge. It is told explicitly that it may
not introduce a number, date, time or price that is not in that block.

NOTE: this call is NO LONGER skipped for clarify/acknowledgement turns. An earlier design
skipped it to save a round trip, but that let an ungrounded planner DRAFT reach the caller
(the "I've booked your visit" class of claim). The grounded responder is now the ONLY path
to the customer — see turn_runner.process_turn ("the grounded responder is the ONLY path")
— so every reply is regenerated here from verified results, deliberately trading the saved
round trip for the guarantee that no ungrounded draft is ever spoken.
"""
import json
import re
from typing import Any

from app.config import get_settings
from app.providers.llm.base import LLMError
from app.providers.llm.router import get_llm
from app.engine.planner.email_claim import honest_fallback as _email_honest_fallback
from app.engine.planner.email_claim import is_fabricated_email_claim
from app.engine.planner import figure_claim, source_conflict_claim
from app.services.money import fix_lakh_crore_mismatch, round_spoken_prices
from app.services.spoken_form import PROMPT_RULE as SPOKEN_FORM_RULE
from app.services.spoken_form import to_spoken_form

RESPONDER_SYSTEM = """You write the single spoken reply an AI phone agent gives next.

RULES:
1. Use ONLY the facts in VERIFIED RESULTS, PROJECT FACTS and RETRIEVED KNOWLEDGE. If a time,
   price, date, name or number is not in one of those, you may not say it. PROJECT FACTS are
   curated, verified and always true — treat them as solid as VERIFIED RESULTS, and when one of
   them answers the question, answer straight from it rather than saying you will find out.
2. Never promise a future action. If the result says something was done, say it was done;
   if it says it failed, be honest and offer the alternative given.
3. One or two short sentences — occasionally three if the customer needs a real choice.
   Natural spoken language, no lists, no markdown, no emoji. Write in whichever language
   the PERSONA below says to use for this turn — English, Hindi, or Hinglish — not
   necessarily English. This still applies even when RETRIEVED KNOWLEDGE itself is long or
   tells you to "give the whole list" for a DIFFERENT kind of question than this one — that
   instruction is scoped to its own question type (e.g. "what's available"), not every
   later turn the document happens to get retrieved for. Asked the cost of the 2 BHK they
   already named, answer the cost of the 2 BHK — do not also re-recite every configuration
   and size just because that list sits in the same source material.
   SCOPE-LOCK: When the caller specifies a configuration (e.g. "2 BHK", "3 BHK", "studio"),
   your answer must cover ONLY that configuration. Mentioning other unit types not asked
   about is a rule violation — it does not make the answer more helpful, it makes it longer
   and harder to hear over the phone. Do not volunteer other configurations to appear
   thorough. This is absolute: even if you see 1 BHK, 2 BHK, and 3 BHK prices in the
   material, when the caller asked about 2 BHK you say only the 2 BHK price.
4. Keep the tone of the DRAFT but correct any fact it gets wrong.
5. State recorded facts exactly as they appear in the results — do not swap who they are
   about. If the result says the decision maker is the customer's wife, say their wife,
   not the customer. When a tool merely records something, confirm that it is recorded
   rather than restating it as a claim about the customer.
6. Sound like the PERSONA below, not like a form being filled in. Answer what was asked,
   then leave the customer something to respond to — the choice in front of them, or the
   one thing you need from them. A reply that ends with nothing to answer makes them say
   "ok" and wait, and the call goes nowhere. "Anything else I can help with?" is not a way
   forward; it is a way out.
7. Do not reuse a sentence you have already used in this call. If the same situation comes
   round twice, say it a different way — repeating yourself word for word is the fastest
   way to sound like a recording rather than a person.
8. ANSWER THE QUESTION IN "THE CUSTOMER JUST ASKED". Everything else in this prompt is
   material you MAY use, not a list of things to mention. Retrieved passages are whatever
   the search returned — some will be about a different subject entirely, and saying
   something true about the wrong subject is the most common way this call goes wrong. If
   what they asked cannot be answered from the material here, say that.
   EQUIVALENT WORDING STILL COUNTS — BUT ONLY WITH EVIDENCE. The documents may name the very
   thing they asked about under a different word: a "creche" for their "daycare", "power
   backup" or "DG" for their "generator", "RCC structure" for their "bricks". When a passage
   in front of you plainly describes that thing, it IS the answer — give it, using the term
   the document uses ("yes, there's a creche"). This is NOT licence to say yes to everything:
   affirm only what a retrieved passage actually states. Do not stretch a loosely-related
   passage into a match, do not infer a feature the passages never mention, and do not invent
   a detail to be helpful. No evidence in the passages means you do not have it — say so
   plainly rather than guessing a "yes".
   BUILD THE ANSWER FROM THE PASSAGES. For a knowledge question, first settle what they
   actually want to know — the UNDERSTANDING block, when present, gives the intent and the
   subject the planner already resolved (e.g. "the flat" = the 3 BHK) — then give a direct,
   complete answer assembled from the relevant passages, in your own words. Draw together
   what is relevant across them; do not stop at a partial fact when the passages hold the
   rest, do not quote raw, and do not dump everything retrieved. But if the passages do not
   actually contain what they asked — not even under another name — do NOT invent it, do NOT
   guess, and do NOT pad the gap with something adjacent that is merely nearby in the text:
   say plainly you don't have that specific detail.
   Two concrete patterns of over-answering worth naming directly: a booking result may
   include an "ends_at" alongside the visit's start time — that exists for the calendar,
   not to volunteer to the caller, so do not add "it will end at X" unless they specifically
   ask how long the visit runs. Likewise a stored address is one long string that usually
   ends in a PIN code — read the locality and road, not the trailing 6-digit code, which is
   not how anyone gives directions aloud.
   AFTER BOOKING: once a visit is confirmed, close with the day, time, and locality only,
   then offer to send a Google Maps pin ("I can send you the location pin on WhatsApp if
   that helps"). Do not state the full address, do not read the PIN code, do not mention
   the end time. One sentence of confirmation, one offer of the pin — nothing else.
9. VERIFIED RESULTS outrank RETRIEVED KNOWLEDGE whenever they overlap. A tool that looked
   up this customer's own records is talking about THEM; a retrieved document is talking
   about the project in general. Asked when their site visit is, someone whose booking was
   just read out of the database was told the site is open ten to six-thirty — accurate
   about the brochure, useless to them, and it reads as though their booking has been
   lost. If a tool returned the answer, lead with it.
10. SELF-CHECK YOUR OWN CLAIM. Set "asserts_completed" to whichever of these your
    response states as an accomplished fact, in ANY wording — "booked", "registered",
    "all set", "confirmed", "arranged", "locked in", or any other way of saying it:
    "book" (a visit/appointment exists), "reschedule" (an appointment moved to a new
    time), "cancel" (one was cancelled), "ticket" (a complaint/ticket was logged), "dnc"
    (they are off the contact list). Set it to null if you are asking a question,
    reporting a failure, or your response makes no such claim. This is checked
    automatically against what actually happened this turn — classify your own sentence
    honestly, whether or not you expect it to pass.
11. OBEY THE ACTION STATE, IF ONE IS GIVEN. When an "## ACTION STATE" block is present it is
    the ground truth about what has and has not happened this turn — it outranks the DRAFT's
    tense and your own instinct. "staged" and "blocked" mean the action has NOT happened:
    you must phrase it as a question or a request, in the FUTURE tense, and "asserts_completed"
    MUST be null. Only "done" permits a completed claim.
12. THE CALLER'S VISIT ON FILE. When a "## YOUR VISIT ON FILE" block is present, the caller
    ALREADY HAS a visit booked. Two hard rules follow:
    (a) NEVER offer to book or schedule a new visit, and never ask "would you like to schedule
        a visit?" — to someone already booked it reads as though you have lost their booking.
        If the turn is about seeing the property, point them to the visit they already have and,
        if they want a different time, offer to MOVE it — never to add a second.
    (b) Mention that visit only when THIS turn is about the property or the visit itself (its
        configuration, price, or seeing it). On an unrelated turn, do not bring it up. When you
        do, read its day and time back exactly as written in the block.
    (c) THE CONVERSE, EQUALLY HARD: when there is NO "## YOUR VISIT ON FILE" block and no
        VERIFIED RESULTS establishing an appointment this turn, this caller has NO visit on
        record with us. Never tell them they already have a visit, and never state a day or
        time for one — you have no source for it, so any such sentence is invented. If they ask
        whether they have a booking, say you don't see one on this call and offer to set one up
        or to look it up once they identify themselves.
13. GROUND EVERY "YES" TO A FEATURE, BY MEANING. When the customer asks whether the PROJECT
    has some feature, amenity, view, unit attribute or configuration ("is there a park-facing
    flat?", "do you have a gym?", "is there a generator?", "is there a 4 BHK?"), decide your
    answer by READING THE MATERIAL FOR MEANING, not by matching words:
      - The material names things in ITS OWN words. A feature counts as present when a passage
        DESCRIBES it, under any synonym, paraphrase or domain term: "creche" IS a daycare, "DG"
        and "power backup" ARE a generator, "RCC structure" IS what "bricks" is asking about,
        the SYNONYMS block below (if present) lists more the tenant treats as equivalent. Read
        it the way a knowledgeable human agent would and answer YES when the meaning is there,
        even if the exact word is not.
      - But a related word is NOT the feature. "West facing" is a compass direction, not a park
        view; a "trampoline park" in the amenities is not a "park-facing flat"; a "2 BHK" is not
        a "4 BHK". Do not stretch an adjacent or partial match into a yes.
      - If the material does not actually establish the thing they asked about — under any
        wording — you do NOT have it. Say plainly you don't have that detail; do not guess a yes.
    Then self-assess, honestly, so a wrong guess can be caught:
      - "feature_answer": "affirm" if your reply says the thing exists/is available, "deny" if
        your reply says it is not, "none" if this turn is not a feature yes/no (you are asking
        something, giving prices or sizes, or talking about the caller's OWN booking/records).
      - "answer_grounded": true only if the VERIFIED RESULTS, PROJECT FACTS or RETRIEVED
        KNOWLEDGE genuinely support your answer when read for meaning; false if you are
        inferring, guessing, or
        stretching. On an "affirm" this MUST be your honest verdict — an affirm you mark
        false is replaced automatically with an honest "I can't confirm that", so do not
        affirm something you cannot ground.
      - "support_quote": the sentence or phrase from the material, copied verbatim, that carries
        that meaning (for "affirm"); null otherwise. Quote the passage's actual words.
14. GROUND EVERY STATEMENT ABOUT WHAT WE DO OR DON'T OFFER. A reply that tells the caller we do
    NOT provide something, or that we ONLY provide something — "we don't offer rentals", "we
    only sell, not lease", "there's no resale here", "we don't do commercial" — is a claim about
    the PROJECT'S SCOPE, and it must come from the material, never from the mere absence of a
    matching passage. Not finding "rentals" in the material is NOT evidence we don't rent. If
    nothing in front of you states the scope, do not assert it: say honestly you don't have that
    detail and offer a sales-team callback or a handoff to someone who can confirm. Self-assess:
      - "scope_claim": "closed" if your reply states such a boundary on what we offer (we do
        not do X, or we only do Y); "none" otherwise.
      - "scope_grounded": true ONLY if VERIFIED RESULTS, PROJECT FACTS or RETRIEVED KNOWLEDGE
        actually state that scope (e.g. "for sale only", a stated list of what's offered); false
        if you are inferring it from silence. A "closed" scope_claim you mark scope_grounded
        false is replaced automatically with an honest line that offers a callback — so never
        assert a scope you cannot see in the material.
15. ONE PROJECT AT A TIME. When a "## SPEAKING ABOUT" block names the project this call is about,
    every fact you state must be about THAT project. Retrieved passages and project facts can name
    a DIFFERENT project (this developer markets several); a policy, scope, price, configuration or
    amenity stated about another project is NOT evidence about this one. Never carry one project's
    "we only sell / no rentals", its prices, or its configuration list over to the project named
    in the block. If the only material that would answer names a different project, you do NOT have
    it for this one — say plainly you don't have that detail for {this project} and offer a
    callback or a visit. Do not name or compare the other project unless the caller asked about it.
16. A VISIT AND A DOCUMENT ARE NEVER THE SAME NOUN PHRASE. A call often starts on one topic — a
    floor plan, a brochure, a price query — before the customer asks to book a visit; the two
    stay separate things throughout the rest of the call, on EVERY turn, not only the one where
    a booking is confirmed. A visit/viewing/site visit is something a person does in person; a
    floor plan/brochure/document is paper (or a file) that gets sent. Never fuse them into one
    phrase — "the 3 BHK floor plan visit", "a viewing for the floor plan", "set one up for the
    3 BHK floor plan" (when "one" means a VISIT) are all the same error and are never correct,
    on an offer, a clarifying question, or a confirmation alike. Say "the visit" or "the 3 BHK
    visit"; mention the floor plan only when this turn is actually about sending or discussing
    that document. When VERIFIED RESULTS confirms a booking/reschedule/ticket specifically, name
    only what that result actually confirms — the configuration, time and place of the visit,
    nothing an earlier topic in the call supplied instead.
17. """ + source_conflict_claim.PROMPT_RULE + """ Retrieved passages are labelled with
    internal document titles in [brackets]; those labels are for you, never to be read out.
18. """ + SPOKEN_FORM_RULE + """

Respond with JSON: {"response": "<what to say>", "asserts_completed": "<book|reschedule|cancel|ticket|dnc|null>", "feature_answer": "<affirm|deny|none>", "answer_grounded": <true|false>, "support_quote": "<verbatim phrase from the material, or null>", "scope_claim": "<closed|none>", "scope_grounded": <true|false>}"""

RESPONDER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["response", "asserts_completed", "feature_answer", "answer_grounded",
                 "support_quote", "scope_claim", "scope_grounded"],
    "properties": {
        "response": {"type": "string"},
        "asserts_completed": {
            "type": ["string", "null"],
            "enum": ["book", "reschedule", "cancel", "ticket", "dnc", None],
        },
        # The reply's own SEMANTIC self-assessment for a feature/availability question (Rule 13):
        # its stance, whether the material genuinely supports it read for MEANING (synonyms and
        # paraphrase included), and the verbatim passage carrying that meaning. The meaning
        # judgement is the model's — no string rule can bridge "generator" ⇄ "power backup" — and
        # grounding_check acts on it, with support_quote as an audit trail and degraded-mode net.
        "feature_answer": {"type": "string", "enum": ["affirm", "deny", "none"]},
        "answer_grounded": {"type": "boolean"},
        "support_quote": {"type": ["string", "null"]},
        # The reply's SEMANTIC self-assessment for a SCOPE/POLICY claim (Rule 14): whether it
        # tells the caller a whole category is not offered ("we don't rent") or is the only thing
        # offered ("we only sell"), and whether the material actually states that scope. A closed
        # scope the model marks ungrounded was invented from silence — grounding_check replaces it
        # with an honest, callback-offering line. Separate from feature_answer so denying a single
        # amenity (left untouched, Rule 13) is never confused with bounding what the project does.
        "scope_claim": {"type": "string", "enum": ["closed", "none"]},
        "scope_grounded": {"type": "boolean"},
    },
}


def _speakable(text: str) -> str:
    """The deterministic rewrites every spoken sentence gets, in one fixed order: rupee
    rounding, lakh/crore correction, then abbreviations/symbols to spoken words. turn_runner.py
    applies the same three, in the same order, to the final text -- keep them in step."""
    return to_spoken_form(fix_lakh_crore_mismatch(round_spoken_prices(text)))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


# THE BUG THIS FIXES: a literal '"response": "' (with a space after the colon) never
# matched real streamed output -- OpenAI's JSON streaming is compact, no space
# ("response":"..."), so this marker silently never fired and sentence_callback was
# never called on any real turn; streaming quietly no-opped every time, falling back to
# the non-streaming path with no visible error. \s* tolerates either form.
_RESPONSE_FIELD_PATTERN = re.compile(r'"response"\s*:\s*"$')
# How much of the tail to keep scanning for the marker across chunk boundaries -- must
# stay comfortably larger than the pattern's own text so a match spanning two streamed
# tokens is never lost at the truncation point.
_MARKER_TAIL_CHARS = 40


async def _stream_first_sentence(*, system: str, user: str, model: str,
                                   temperature: float, max_tokens: int,
                                   schema: dict, schema_name: str,
                                   sentence_callback,
                                   asked: str | None = None,
                                   figure_grounding: str | None = None) -> tuple[str, int]:
    """Stream a JSON completion and call sentence_callback for EVERY sentence found
    inside the 'response' field, as each one completes -- not just the first. Returns
    (full_raw_json, latency_ms).

    P3.02: previously this stopped after the first sentence (name kept for a minimal
    diff), leaving sentences 2+ to wait for the whole response to finish generating
    AND pass grounding/scope/booking-claim validation before reaching TTS at all --
    on a multi-sentence reply, only sentence 1 got any early-start benefit. Now every
    sentence is pushed the moment its boundary is seen, so a well-behaved multi-
    sentence reply is spoken essentially as it's generated; bridge.py's post-turn
    segment loop still replays anything the validators changed, unchanged safety net.

    Raises LLMError on any failure so callers can fall back to complete_json().
    Sentence boundary detection: [.!?] followed by a space character, or the closing
    quote of the JSON field (the last sentence has no trailing space to trigger on).
    """
    import time as _time
    started = _time.perf_counter()
    provider = get_llm()
    if not hasattr(provider, "stream_text"):
        raise LLMError("streaming not supported by this provider")

    async def _emit(sentence: str) -> None:
        """Every streamed sentence -- mid-reply AND the final one flushed when the field
        closes -- gets the same guards and rewrites turn_runner.py applies to the validated
        text, so what is spoken early is what the final pass produces. bridge.py dedups the
        two by exact text and replays on any difference; the final sentence used to be sent
        raw, so a last sentence with a price or "sq ft" in it was spoken twice.

        - email: there is no email tool anywhere, so a sentence promising one must never
          reach the caller even for the window before the final pass would replace it.
        - source conflict: a clause exposing an internal-document disagreement is cut; a
          sentence that is nothing but the leak is never spoken (source_conflict_claim.py).
        - unsupported figures: a distance/duration/area/percentage the grounding material does
          not contain is cut and replaced, once per kind, by an honest "I don't have the exact
          …" line (figure_claim.py). `_figures_said` is shared across the reply's sentences so
          this yields exactly what turn_runner's single pass over the whole text does.
        - rounding + spoken form: a raw "1,54,32,011" or "sq ft" never reaches TTS
          (round_spoken_prices, to_spoken_form -- see _speakable)."""
        if is_fabricated_email_claim(sentence):
            sentence = _email_honest_fallback()
        sentence = source_conflict_claim.scrub(sentence, asked=asked)
        if sentence and figure_grounding is not None:
            sentence = figure_claim.scrub(sentence, figure_grounding,
                                          language=detect_language(asked), said=_figures_said)
        if not sentence:
            return
        # A scrub can turn one sentence into two (kept clause + the missing-figure line); push
        # them as separate segments, matching how bridge.py splits the validated text.
        for piece in split_sentences(sentence):
            try:
                await sentence_callback(_speakable(piece))
            except Exception:
                pass

    _figures_said: set = set()

    full_buf = ""
    marker_buf = ""
    in_field = False
    pending = ""
    escape_next = False

    async for token in provider.stream_text(
        system=system, user=user, model=model,
        temperature=temperature, max_tokens=max_tokens,
        schema=schema, schema_name=schema_name,
    ):
        full_buf += token
        for char in token:
            if not in_field:
                marker_buf += char
                if _RESPONSE_FIELD_PATTERN.search(marker_buf):
                    in_field = True
                    marker_buf = ""
                elif len(marker_buf) > 200:
                    marker_buf = marker_buf[-_MARKER_TAIL_CHARS:]
            else:
                if escape_next:
                    pending += char
                    escape_next = False
                elif char == "\\":
                    escape_next = True
                elif char == '"':
                    in_field = False
                    # Field closed -- flush whatever's left as the final sentence. It
                    # won't have hit the space+punctuation boundary below because the
                    # closing quote follows the last word with no trailing space.
                    tail = pending.rstrip()
                    pending = ""
                    if tail:
                        await _emit(tail)
                else:
                    pending += char
                    # Sentence boundary: punctuation followed by a space
                    if len(pending) > 3 and pending[-1] == " " and pending[-2] in ".!?":
                        sentence = pending[:-1].rstrip()
                        pending = ""
                        if sentence:
                            await _emit(sentence)

    return full_buf, int((_time.perf_counter() - started) * 1000)


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _action_state_block(action_state: str | None, staged: dict | None,
                        blocked_reason: str | None) -> str | None:
    """The ground-truth about what happened this turn, so the reply's tense cannot lie.

    This is Rule 6's counterpart for the RESPONDER: the planner already surfaces booking
    STATE into its own prompt; this surfaces the ACTION state into the sentence-writer. It
    is the structural fix for the whole "claimed a completed book/reschedule … reply
    replaced" family — the model was drafting "your appointment is booked" for an action
    that had only been STAGED (parked awaiting the customer's yes) or BLOCKED (waiting on
    identity), because nothing told it those had not happened. Told plainly, it writes the
    right tense and the after-the-fact guards have nothing left to catch.
    """
    if not action_state or action_state in ("done", "none"):
        return None
    verb = (staged or {}).get("verb") or "do that"
    when = (staged or {}).get("when")
    if action_state == "staged":
        line = (f"NOT DONE YET. The action ({verb}) is staged and waiting for the customer to "
                f"say yes. Ask, in the FUTURE tense — e.g. \"Shall I {verb}"
                + (f" for {when}" if when else "") + "?\" — and read back the exact time above "
                "if there is one. Do NOT say it is done, booked, confirmed or set. "
                "asserts_completed MUST be null.")
    elif action_state == "blocked":
        line = (f"CANNOT PROCEED YET. {blocked_reason or 'A required step is not done.'} "
                "Ask the customer for exactly what is needed to clear it. Do NOT claim the "
                "action happened. asserts_completed MUST be null.")
    elif action_state == "policy_blocked":
        # THE BUG THIS FIXES: a deterministic rule (e.g. the caller is on the do-not-contact
        # list) blocked every tool/pitch this turn, with nothing telling the model so -- it
        # fabricated a scope claim ("we don't have rentals") on one real call and kept
        # pitching a lead who had just asked to be left alone on another. Unlike "blocked"
        # above, this is never something the CALLER can clear -- there is no missing
        # precondition to ask for, so the wording must not imply one.
        _reason = blocked_reason or "a policy rule is blocking further action"
        line = (f"A COMPLIANCE RULE IS ACTIVE THIS TURN: {_reason}. "
                "You may still answer a FACTUAL question "
                "honestly from VERIFIED RESULTS, PROJECT FACTS or RETRIEVED KNOWLEDGE if the "
                "answer is genuinely there. Do NOT invent a scope claim ('we don't have X', "
                "'we only do Y') that is not actually stated in that material -- if you don't "
                "have the answer, say so plainly instead of guessing one. Do NOT pitch, offer "
                "a callback, suggest alternatives, or move the conversation forward in any "
                "way -- this caller is not to be sold to on this call. If there is nothing "
                "factual to answer, acknowledge them briefly and let the call wind down.")
    elif action_state == "failed":
        line = ("FAILED this turn. Be honest that it did not happen and offer the alternative "
                "in VERIFIED RESULTS. asserts_completed MUST be null.")
    else:
        return None
    return "## ACTION STATE (ground truth — outranks the draft)\n" + line


def _understanding_block(understood: dict | None) -> str | None:
    """The planner's resolved reading of the turn, handed to the sentence-writer so it frames
    the answer around what the customer MEANS, not just the raw words.

    The raw utterance alone under-determines a bare follow-up — "what about that one?" names
    no subject the responder can see, though the planner already resolved it to (say) the
    3 BHK. Surfacing the resolved intent and entities here lets the reply answer the intended
    question instead of the literal one. Only the two fields that aid framing are included,
    and only when non-empty, so an empty understanding adds nothing to the prompt.
    """
    if not understood:
        return None
    intent = str(understood.get("intent") or "").strip()
    entities = understood.get("entities") or {}
    if not isinstance(entities, dict):
        entities = {}
    entities = {k: v for k, v in entities.items() if v}
    if not intent and not entities:
        return None
    lines = []
    if intent:
        lines.append(f"intent: {intent}")
    if entities:
        lines.append("subject/entities: " + json.dumps(entities, separators=(",", ":")))
    return ("## UNDERSTANDING (what they mean, resolved by the planner — frame the answer "
            "around THIS)\n" + "\n".join(lines))


def _synonyms_block(synonyms: list[list[str]] | None) -> str | None:
    """The tenant's declared equivalent-terms groups, handed to the sentence-writer so it
    bridges domain vocabulary it might not know on its own when it reads the material for
    meaning (Rule 13). The model already knows common synonyms ("creche"≈"daycare"); this
    covers the tenant-specific ones ("DG"/"power backup"≈"generator", "bhp"≈"horsepower")
    reliably rather than by luck. A hint for MEANING, not a matching rule — retrieval already
    used these to fetch the right chunk; this just makes the reader connect the words."""
    groups = [g for g in (synonyms or []) if isinstance(g, list) and len(g) > 1]
    if not groups:
        return None
    lines = "\n".join("- " + " = ".join(str(t) for t in g) for g in groups[:40])
    return ("## SYNONYMS (these terms mean the same thing here — treat any as the others when "
            "reading the material)\n" + lines)


def _project_facts_block(facts: list[dict] | None) -> str | None:
    """The tenant's curated always-answerable facts (location, RCC structure, RERA, scale),
    surfaced to the sentence-writer as VERIFIED material it may answer from and ground on.

    These reach the planner's DRAFT already, but the DRAFT is only a tone hint — the grounded
    responder rewrites every reply from the material it is given here, and was given only the
    tool result and the retrieved chunks. So a fact that lives ONLY in this curated set — "RCC
    framed structure, not brick" for "are bricks used?" — was stripped as ungrounded and the
    caller heard "the brochure doesn't mention it". Handing the facts to the responder closes
    that gap: they are as authoritative as VERIFIED RESULTS (Rule 1) and can support a feature
    answer (Rule 13). None when the tenant has curated no facts."""
    lines = [f"- {f['label']}: {f['value']}" for f in (facts or [])
             if isinstance(f, dict) and f.get("label") and f.get("value")]
    if not lines:
        return None
    return ("## PROJECT FACTS (curated, verified — as authoritative as VERIFIED RESULTS; answer "
            "directly from these and you may ground a feature answer on them)\n" + "\n".join(lines))


def _visit_on_file_block(appts: list[dict] | None) -> str | None:
    """The caller's live visit(s), surfaced to the sentence-writer so it never offers a NEW
    one to someone already booked (Rule 12). Same wording as the planner's YOUR VISIT ON FILE
    block, so both surfaces read the visit back identically. None when there is nothing on
    file (unknown caller, or no live visit)."""
    lines = [
        f"- {a['when']} (reference {a.get('reference', '')}, status: {a.get('status', 'booked')}"
        + (f", for the {a['unit']}" if a.get("unit") else "") + ")"
        for a in (appts or []) if a.get("when")]
    if not lines:
        return None
    return ("## YOUR VISIT ON FILE (the caller ALREADY has this visit — never offer a new "
            "one; see rule 12)\n" + "\n".join(lines))


# A fast, deterministic language read on the CURRENT turn's own words — not the LLM's
# judgment, and not something that can be diluted by a long block of English RETRIEVED
# KNOWLEDGE sitting right above it in the same prompt. The PERSONA instruction alone told
# the model to "match the customer's language", but on a short follow-up next to a page of
# English source material it kept answering in English anyway — the instruction was true in
# principle but easy to outweigh. Stating the verdict for THIS turn as its own block removes
# the inference step entirely.
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_HINGLISH_WORDS = re.compile(
    # "me" is deliberately excluded despite being a common Hinglish shorthand for "mein" --
    # it is also an ordinary, unambiguous English word ("send me the brochure"), and this
    # match is an OR over the whole turn, so including it would misclassify plain English
    # sentences as Hinglish. Every other addition here is a spelling variant unlikely to
    # collide with genuine English ("apke" vs the already-present "aapke", "h" standing in
    # for "hai", "koi" meaning "any/some").
    r"\b(?:hai|hain|h|kya|kitna|kitni|kitne|ka|ki|ke|ko|se|mein|aur|bhi|nahi|nahin|nhi|"
    r"chahiye|karna|kare|karo|krna|krdo|batao|bata|bataye|hoga|hogi|mujhe|hume|humein|aap|"
    r"ap|aapka|aapki|aapke|apka|apki|apke|usme|usmein|isme|ismein|uska|uski|uske|iska|iski|"
    r"iske|voh|vo|woh|wo|yeh|ye|vahi|wahi|wala|wali|dono|kaisa|kaisi|kaise|thoda|bhai|yaar|"
    r"koi|abhi|abi|khud|bss|bas)\b",
    re.I)


def detect_language(text: str) -> str:
    """"english" | "hindi" | "hinglish", for THIS turn's raw words only — no memory of
    earlier turns. A blank turn (e.g. the opening line) reports "english", the default."""
    text = (text or "").strip()
    if not text:
        return "english"
    if _DEVANAGARI.search(text):
        return "hindi"
    if _HINGLISH_WORDS.search(text):
        return "hinglish"
    return "english"


_LANGUAGE_INSTRUCTION = {
    # Every number (prices, sizes, counts, dates) stays in English even inside an otherwise-
    # Devanagari sentence -- NEVER converted into Hindi number-words. This is the same reason
    # the Hinglish rule below already keeps numbers in English: the multilingual TTS voice
    # pronounces a number according to the language of the words around it, and asking it to
    # pronounce a Hindi-worded number came out mispronounced with an off accent on a real
    # call. Bare digits sitting inside a Devanagari sentence turned out to be read
    # inconsistently too -- correct English pronunciation on some turns, an attempted (and
    # wrong) Hindi reading on others, unpredictably. ElevenLabs' own guidance is that a
    # multilingual voice needs the number spelled out in words to remove the ambiguity
    # entirely, not just written in Latin digits next to Devanagari text — a digit character
    # carries no language signal by itself, a word does.
    "hindi": ("Reply in Hindi, Devanagari script — this turn's words were Hindi. EXCEPTION: "
             "every number (size, count, date — a plain quantity, not a price) is spelled "
             "out as English WORDS, not digits and never Hindi number-words: \"six nineteen "
             "to six forty-six square feet\", not \"619 to 646\" and not the Hindi words for "
             "those numbers. A price stays as the rounded crore/lakh phrase in digits+English "
             "(\"1.54 crore\", \"96.25 lakh\") — that one stays as-is, only plain quantities "
             "need to become words."),
    "hinglish": ("Reply in Hinglish: natural Hindi-English code-mixing in Roman script, the "
                "way the caller just wrote it — this turn's words were Hinglish. Do this "
                "even though the material above (retrieved knowledge, tool results) is in "
                "English — translate the facts, do not switch to English because the "
                "source material is in English."),
    "english": "Reply in English — this turn's words were English.",
}


def _language_block(language: str) -> str:
    return "## REPLY LANGUAGE FOR THIS TURN\n" + _LANGUAGE_INSTRUCTION[language]


def _continuation_block(continuation: dict | None) -> str | None:
    """When the caller's turn was a presence/audibility check or a listening nod, this tells the
    sentence-writer to CONTINUE the agent's own last line — in character — rather than answer the
    literal words. It is what makes "am I audible?" produce "yes, you're coming through — as I was
    saying, …<the pitch/point>" in the sales voice, instead of a canned "Yes, I'm here —" glued to
    a verbatim repeat. Detection is deterministic (turn_runner); this is only the GENERATION half,
    folded into the responder call that already runs — no extra model call.

    ``continuation`` is ``{"kind": "check_in"|"backchannel", "last_line": "<agent's last line>"}``.
    The last line is treated as already-verified, already-spoken material to carry forward, so the
    grounding rules never strip the facts (the time/price) the caller may have missed.
    """
    if not continuation:
        return None
    kind = continuation.get("kind")
    last = re.sub(r"\s+", " ", str(continuation.get("last_line") or "").strip())
    if kind not in ("check_in", "backchannel") or not last:
        return None
    # Said once, pointedly, because a real call glued a rotated "Yes, I'm here and coming
    # through." prefix onto a WORD-FOR-WORD copy of a long previous line — "keeping its facts
    # exactly" and "resume where you were" read, to the model, as license to paste the whole
    # sentence back rather than restate its substance briefly. A caller who just missed one
    # line does not need the whole pitch read to them twice over; a short recap beats a replay.
    _no_verbatim = (" Paraphrase it — do NOT repeat it word-for-word or near-verbatim, and do "
                    "NOT recite it in full; a short recap of the key fact(s) is enough.")
    if kind == "check_in":
        head = ("THE CALLER COULD NOT HEAR YOU. They asked whether you are there / audible — they "
                "missed your last line. In ONE short reply: briefly confirm you are on the line and "
                "coming through, THEN restate the POINT of your last line below, in character, "
                "keeping its facts (times, prices, names) accurate." + _no_verbatim +
                " Do NOT re-greet, do NOT ask what they need, do NOT restart, do NOT turn it into "
                "a generic offer of help.")
    else:
        head = ("THE CALLER GAVE A LISTENING NOD (\"mm-hmm\", \"go on\", \"ok\") — they are holding "
                "the floor open for you, not asking anything and not answering a question. Carry on "
                "from your last line below, in character, and move the conversation forward one "
                "step." + _no_verbatim + " Do NOT re-greet and do NOT restart. If your last line "
                "ended on a question they did not answer, put it to them again, lightly.")
    return ("## CONVERSATIONAL SIGNAL — HOW TO REPLY (outranks the DRAFT and the material below)\n"
            + head
            + "\n\n## YOUR LAST LINE (already verified and spoken — restate its substance briefly, "
            "in your own voice and your own words; do NOT invent anything beyond it)\n" + last)


def _pitch_block(pitch: dict | None) -> str | None:
    """Outbound pitch rung → a GENERATION directive folded into the responder call that already
    runs (no extra model call). The deterministic ladder in turn_runner decides WHICH rung we are
    on and force-retrieves the real material (configurations, sizes, price) into this same prompt;
    this tells the sentence-writer to LEAD with those concrete facts — never generic praise — and
    end on the one question that moves the call forward. It cannot loosen the grounding rules: it
    says to lead with the material, and to fall back to a visit offer (not an invented number)
    when the material lacks the specifics.

    ``pitch`` is ``{"stage","project","unit"}`` where stage ∈ value | offer | scheduling |
    wind_down."""
    if not pitch:
        return None
    stage = str(pitch.get("stage") or "")
    proj = str(pitch.get("project") or "").strip()
    unit = str(pitch.get("unit") or "").strip()
    what = ("the " + unit if unit else "it") + (f" at {proj}" if proj else "")
    if stage == "value":
        body = (
            f"The caller just confirmed they are interested in {what}. LEAD now with the CONCRETE "
            f"substance from the material above — the actual configurations/options for "
            f"{unit or 'the home'}, their sizes in square feet, and any layout differences. Give "
            f"one or two SPECIFIC, real facts drawn from the material (generic praise like "
            f"'people love it' or 'it's a favourite' is BANNED — say something a brochure would "
            f"say, with numbers), then ask if they'd like to hear a bit more or would like to see "
            f"it. If the material does not contain the specifics, say briefly and honestly what "
            f"you can and move to offering a visit — never invent a figure.")
    elif stage == "offer":
        body = (
            f"The caller wants to hear more about {what}. Add ONE more concrete detail from the "
            f"material — the price or per-square-foot rate, another configuration, or a standout "
            f"amenity — then recommend seeing it in person and ask if they'd like you to arrange a "
            f"quick site visit. Do not invent any figure the material does not contain.")
    elif stage == "scheduling":
        body = (
            "The caller wants to go ahead with a site visit. Warmly ask what day and time would "
            "suit them best so you can arrange it. Do not quote any new facts or figures this turn.")
    elif stage == "wind_down":
        body = (
            "The caller is not interested right now. Accept it graciously, offer to send the "
            "details across for whenever suits them, and do not push. One short line.")
    else:
        return None
    return ("## OUTBOUND PITCH — HOW TO REPLY (outranks the DRAFT; still obey the grounding rules "
            "— state only facts present in the material, never invent one)\n" + body
            + "\n\nTwo to three short spoken sentences, in the PERSONA's language, ending on the "
            "question that invites the next step. Do not repeat a sentence you have used before.")


async def generate(*, draft: str, tool_name: str | None, tool_status: str | None,
                   tool_output: dict | None, tool_error: str | None,
                   failure_action: dict | None, knowledge: list[dict] | None,
                   persona: str | None = None, asked: str | None = None,
                   understood: dict | None = None,
                   existing_appointments: list[dict] | None = None,
                   action_state: str | None = None, staged: dict | None = None,
                   blocked_reason: str | None = None,
                   synonyms: list[list[str]] | None = None,
                   project_facts: list[dict] | None = None,
                   project_name: str | None = None,
                   continuation: dict | None = None,
                   pitch: dict | None = None,
                   tenant_preferences: str | None = None,
                   figure_grounding: str | None = None,
                   sentence_callback=None) -> tuple[str, dict]:
    """Returns (response_text, meta). Falls back to the draft if the model fails.

    ``action_state`` (``done|failed|staged|blocked|none`` or None) makes the reply's tense
    truthful: for ``staged``/``blocked`` the model is told the action has NOT happened, so it
    asks instead of claiming. None (the flag-off path) omits the block and behaves as before.

    ``understood`` is the planner's structured reading (intent + resolved entities); when
    given it is surfaced so the reply is framed around the intended question, not the literal
    words. Optional — omitting it preserves the raw-utterance-only behaviour.

    ``continuation`` (``{"kind","last_line"}`` or None) makes THIS turn a presence-check or
    backchannel reply: the model confirms/rolls with it and CONTINUES the agent's last line in
    persona, instead of answering the literal words. Folded into this existing call so it costs
    no extra round trip; turn_runner supplies it only when its deterministic detector fired.
    """
    blocks = [_language_block(detect_language(asked))]
    _state_block = _action_state_block(action_state, staged, blocked_reason)
    if _state_block:
        blocks.append(_state_block)
    # Which project this call is about — so the sentence-writer never applies another project's
    # policy/scope/price/configurations to it (Rule 15). Sits at the very top, above the material,
    # because it governs how ALL of the material below must be read.
    if project_name:
        blocks.append(f"## SPEAKING ABOUT\nThis call is about {project_name}. Every fact you state "
                      f"must be about {project_name}; material naming a different project does not "
                      f"describe it.")
    # What the customer actually said, first and above everything else.
    #
    # Without it this call sees a tool result and a pile of retrieved chunks with no way to
    # tell which the customer wanted. Asked "is there a slot at 1pm on the 14th", it was
    # handed the slot search AND some pricing chunks, and answered "the 1 BHK cost sheet
    # isn't published yet" — a true sentence, about a question nobody had asked.
    if asked:
        blocks.append(f"## THE CUSTOMER JUST ASKED\n{asked}")
    # A presence-check / backchannel: continue the agent's own last line in persona instead of
    # answering the literal words. High in the prompt, right under the raw question it reframes.
    _continuation = _continuation_block(continuation)
    if _continuation:
        blocks.append(_continuation)
    # Outbound pitch rung directive: LEAD with the concrete configurations/sizes/price the ladder
    # force-retrieved into this prompt, end on the forward question. High in the prompt, next to
    # the question it reframes, same discipline as the continuation directive above.
    _pitch = _pitch_block(pitch)
    if _pitch:
        blocks.append(_pitch)
    # The planner's resolved intent/subject, right below the raw question — so a bare
    # follow-up is answered about what they mean, not the literal words.
    _understanding = _understanding_block(understood)
    if _understanding:
        blocks.append(_understanding)
    # The caller's own visit on file — high in the prompt, next to the question, so the reply
    # is framed knowing they are already booked (never pitched a fresh visit) on a related turn.
    _visit = _visit_on_file_block(existing_appointments)
    if _visit:
        blocks.append(_visit)
    if tool_name:
        blocks.append("## VERIFIED RESULTS\n" + json.dumps({
            "action": tool_name, "status": tool_status,
            "result": _jsonable(tool_output or {}),
            "error": tool_error,
            "what_to_do_instead": (failure_action or {}).get("template"),
        }, indent=2))
    # Curated project facts sit below the caller's own tool result (which still outranks them,
    # Rule 9) and above the retrieved chunks — they are verified and always answerable.
    _facts = _project_facts_block(project_facts)
    if _facts:
        blocks.append(_facts)
    if knowledge:
        blocks.append("## RETRIEVED KNOWLEDGE\n" + "\n\n".join(
            f"[{k.get('source', 'doc')}] {k.get('content')}" for k in knowledge))
    # Tenant synonym hints sit just under the material they help the reader interpret.
    _synonyms = _synonyms_block(synonyms)
    if _synonyms:
        blocks.append(_synonyms)
    blocks.append("## DRAFT (tone reference only)\n" + (draft or ""))

    # Advisory custom instructions (spec §4.1): a tenant-preferences block that refines
    # tone/priority, declared SUBORDINATE to the hard rules and output contract above. It
    # MUST reach the responder — the only text the caller hears — not just the planner. It
    # sits in the cacheable prefix (with PERSONA), so for the common case (no intent-scoped
    # rows) it is stable across a call's turns and free after turn one. Already scope- and
    # cap-filtered and rendered upstream (app/custom_instructions); None when nothing applies.
    system = RESPONDER_SYSTEM + (f"\n\n## PERSONA\n{persona}" if persona else "")
    if tenant_preferences:
        system += "\n\n" + tenant_preferences
    user_content = "\n\n".join(blocks)

    # For read-only-tool turns, try streaming so TTS can start on each sentence as it
    # completes, not just the first, while the LLM continues generating the rest.
    # Falls back to complete_json() on any failure — behaviour is identical to the
    # non-streaming path for the caller.
    _stream_raw: str | None = None
    _stream_latency_ms = 0
    if sentence_callback is not None:
        try:
            _stream_raw, _stream_latency_ms = await _stream_first_sentence(
                system=system, user=user_content,
                # 0.3 was low enough that the agent produced near-identical sentences on
                # every call — the customer-facing complaint was that it "feels permanently
                # seeded, the same every time". This is the one model call whose job is
                # PHRASING, and the facts it may use are already fixed by VERIFIED RESULTS
                # above, so variation here changes how something is said and not what is said.
                # The MAIN model, despite this being the cheaper job — because measurement
                # said so. A bare call from here costs 1295 ms on gpt-4o and 1423 ms on
                # gpt-4o-mini: the small model is SLOWER. Nearly all of a call is fixed
                # overhead — network round trip and time to first token — and the mini model
                # wins on price, not on latency. Moving this call to mini to "save time" made
                # every turn worse by about 130 ms.
                model=None, temperature=0.7, max_tokens=300,  # main tier
                schema=RESPONDER_SCHEMA, schema_name="responder_response",
                sentence_callback=sentence_callback, asked=asked,
                figure_grounding=figure_grounding)
        except Exception:
            _stream_raw = None  # streaming failed; fall through to complete_json

    try:
        if _stream_raw is not None:
            try:
                _data = json.loads(_stream_raw)
            except json.JSONDecodeError as _e:
                raise LLMError(f"stream JSON parse error: {_e}") from _e
            data = _data
            # Best-effort label for cost tracking (app/observability/pricing.py looks this
            # up by exact literal model name) -- _stream_first_sentence doesn't currently
            # report back which provider/model actually streamed, so this assumes OpenAI's
            # main model, matching this path's behaviour before the fallback chain existed.
            # A pre-existing gap, not something this rename introduces.
            result_model = get_settings().openai_model_main
            result_tokens: dict = {}
            result_latency = _stream_latency_ms
        else:
            result = await get_llm().complete_json(
                system=system, user=user_content,
                model=None, temperature=0.7, max_tokens=300,  # main tier
                schema=RESPONDER_SCHEMA, schema_name="responder_response")
            data = result.data
            result_model = result.model
            result_tokens = result.tokens
            result_latency = result.latency_ms

        text = str(data.get("response") or "").strip()
        if not text:
            raise LLMError("empty response")
        _claim = data.get("asserts_completed")
        _claim = _claim if _claim in ("book", "reschedule", "cancel", "ticket", "dnc") else None
        # A staged/blocked/failed action did not happen, full stop. Even if the model self-
        # reported a completion, the action state is ground truth — null the claim so the
        # downstream guard is never handed a "completion" for something merely parked.
        if action_state in ("staged", "blocked", "failed", "policy_blocked"):
            _claim = None
        # Feature yes/no SEMANTIC self-assessment (Rule 13). The model judged meaning; the
        # caller (grounding_check) acts on that judgement. `verifier_ok` records that the
        # structured self-assessment is present at all — its absence (the LLMError path below)
        # is what puts grounding_check into degraded mode.
        _feat = data.get("feature_answer")
        _feat = _feat if _feat in ("affirm", "deny", "none") else "none"
        _grounded = bool(data.get("answer_grounded"))
        _quote = data.get("support_quote")
        _quote = _quote.strip() if isinstance(_quote, str) and _quote.strip() else None
        # Scope/policy self-assessment (Rule 14). A "closed" claim the model cannot ground is a
        # scope invented from silence ("we don't rent") — grounding_check replaces it.
        _scope = data.get("scope_claim")
        _scope = _scope if _scope in ("closed", "none") else "none"
        _scope_grounded = bool(data.get("scope_grounded"))
        return text, {"model": result_model, "tokens": result_tokens,
                      "latency_ms": result_latency, "grounded": True,
                      "asserts_completed": _claim, "action_state": action_state,
                      "feature_answer": _feat, "answer_grounded": _grounded,
                      "support_quote": _quote, "scope_claim": _scope,
                      "scope_grounded": _scope_grounded, "verifier_ok": True}
    except LLMError as e:
        # The self-assessment never ran: verifier_ok=False puts grounding_check into degraded
        # mode, where a reply is trusted only if it quoted the source verbatim.
        return (draft or "Let me check that and come back to you."), \
               {"grounded": False, "error": str(e), "asserts_completed": None,
                "feature_answer": "none", "answer_grounded": False,
                "support_quote": None, "scope_claim": "none", "scope_grounded": False,
                "verifier_ok": False}