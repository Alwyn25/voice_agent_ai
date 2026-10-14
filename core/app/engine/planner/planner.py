"""The planner — steps 1 to 7 of process_turn.

  1 assemble context      (deterministic)
  2 understanding         (fast-model classify, run CONCURRENTLY with step 3 — never serial)
  3 knowledge retrieval   (hook; implemented in the RAG stage)
  4 candidate actions     (deterministic — the model only ever sees this list)
  5 rule gate             (deterministic)
  6 propose               (LLM)
  7 confidence banding    (deterministic)

Execution, grounded response and the audit trace are steps 8-12 (next stage).
"""
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Conversation, PromptTemplate
from app.engine.planner import prompts
from app.engine.planner.confidence import (ConfidenceResult, INTENT_ALIASES,
                                            normalize_intent, score_confidence)
from app.engine.planner.turn_context import TurnContext, build_turn_context, memory_reference
from app.engine.planner.focus import contextual_query, focus_terms, merge_entities, needs_context_anchor
from app.engine.planner.understanding import Understanding, understand
from app.engine.rag.retrieval import retrieve
from app.engine.rules import engine as rule_engine
from app.engine.rules.context import build_context as build_rule_context
from app.engine.tools import registry
from app.engine.tools.types import READ_ONLY_TOOLS
from app.providers.llm.base import LLMError
from app.providers.llm.router import get_llm

logger = logging.getLogger(__name__)

VALID_ACTIONS = ("respond", "invoke_tool", "clarify", "confirm", "escalate", "end_call")

# Block scopes that mean "this caller is not to be contacted/sold to at all" -- DNC and a
# marketing-specific opt-out. ONLY these set proposal.rule_blocked (-> the responder's
# no-pitch "policy_blocked" state, see turn_runner.py). A block scoped to one tool
# ("tool:book_appointment", e.g. a weekend-booking restriction or a tenant's "no visits on
# Sundays" custom instruction) is an ordinary business rule, not a compliance stop: the
# caller asking about a different day is still a hot lead, and the responder offering
# "Would Saturday work?" is the right answer, not something to be suppressed.
_NO_FURTHER_CONTACT_SCOPES = frozenset({"all", "marketing"})


def is_no_further_contact_block(verdict) -> bool:
    """True only for a block verdict that means "stop contacting/selling to this caller
    entirely" (see _NO_FURTHER_CONTACT_SCOPES) -- the only kind that may set
    proposal.rule_blocked and switch the responder into no-pitch mode."""
    return verdict.body.get("scope") in _NO_FURTHER_CONTACT_SCOPES

# Tools that must execute even when a rule forces escalation. An opt-out request is
# honoured immediately; the escalation then happens on top of it.
ALWAYS_HONOUR_TOOLS = ("record_do_not_contact",)


@dataclass
class Proposal:
    action: str
    understood: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    tool: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    response_draft: str = ""
    missing_info: list[str] = field(default_factory=list)
    confidence: ConfidenceResult | None = None
    candidates: list[str] = field(default_factory=list)
    rules_evaluated: list[dict] = field(default_factory=list)
    forced_escalation: dict | None = None
    escalate_after_tool: bool = False
    # A rule-engine BLOCK (e.g. dnc_state_stops_everything's scope:"all") that fired this
    # turn -- the deterministic reason nothing could run, carried through to the grounded
    # RESPONDER so it has an honest explanation instead of improvising one. Without this a
    # blocked turn with no tool outcome (outcome stays None, _action_state returns "none")
    # reached the responder with ZERO signal that anything was blocked at all, and it
    # fabricated a scope claim ("we don't have rentals") or kept pitching a lead who had
    # just asked to be left alone -- see turn_runner.py's wiring of this into "policy_blocked".
    rule_blocked: dict | None = None
    # A rule (typically a custom instruction) that REQUIRES a piece of information be collected
    # before proceeding — e.g. "collect budget before offering a visit". Set when the required
    # slot is still missing; the turn is turned into a clarify that asks for it (the ask text is
    # enforced after the responder, mirroring forced_escalation). None when nothing is pending.
    forced_requirement: dict | None = None
    memory_used: dict = field(default_factory=dict)
    knowledge_used: list[dict] = field(default_factory=list)
    # The caller's live visits on file (from turn_context). Carried onto the proposal so the
    # grounded RESPONDER — the only text that reaches the caller, and which regenerates the
    # reply from scratch — knows the caller already has a visit. Without this the planner's
    # draft could reference the visit, but the responder rewrote it blind and reached for
    # "would you like to schedule a visit?" to a caller already booked.
    existing_appointments: list[dict] = field(default_factory=list)
    retrieval: dict = field(default_factory=dict)
    # Tenant-declared synonym groups (tenant.settings["retrieval_synonyms"]), carried out of
    # the turn context so the post-responder feature-grounding guard (Rule 13) can bridge the
    # same equivalences retrieval does — "generator" ⇄ "power backup" / "DG" — when it verifies
    # an affirmed feature against the material. Without it the guard rejected a real "yes" whose
    # only support in the text was under a synonym the caller did not use.
    retrieval_synonyms: list[list[str]] = field(default_factory=list)
    # Tenant-curated always-answerable facts (tenant.settings["project_facts"]) — location,
    # RCC structure, RERA, scale. Carried out of the turn context so the grounded RESPONDER,
    # the only text that reaches the caller, can answer directly from them and GROUND a reply
    # on them. Without this they reached only the planner DRAFT, and the responder — which sees
    # only VERIFIED RESULTS and RETRIEVED KNOWLEDGE — stripped the answer: "are bricks used?"
    # was told "the brochure doesn't mention it" though "RCC framed structure (not brick)" sits
    # right here, because the curated fact was invisible to the sentence-writer.
    project_facts: list[dict] = field(default_factory=list)
    # The resolved project's name (multi-project-per-tenant), or None for a single/tenant-wide
    # tenant. Carried to the grounded RESPONDER so the sentence-writer knows which project it is
    # speaking about and never applies a DIFFERENT project's policy, scope, pricing or
    # configurations to it — the cross-project contamination that retrieval scoping guards at the
    # data layer, backstopped here at the reasoning layer.
    project_name: str | None = None
    # Advisory custom instructions rendered for THIS turn — the subordinate TENANT
    # PREFERENCES block injected into the grounded RESPONDER (the only text the caller
    # hears). Scope-filtered (direction/state/intent/date) and capped upstream. None when
    # no advisory rows apply, so the responder prompt is byte-for-byte unchanged then.
    tenant_preferences: str | None = None
    # OUTBOUND only: the interest THIS customer registered with us — the project
    # (enquiry_project) and the unit (enquiry_interest, e.g. "2 BHK"). Carried out of the turn
    # context so the outbound identity gate (turn_runner) can have the purpose opener RECALL and
    # VERIFY that interest before pitching, rather than asserting a context the caller never
    # heard. Both None on inbound, or when no enquiry is on file.
    enquiry_interest: str | None = None
    enquiry_project: str | None = None
    # The fast understanding pre-pass (P1): intent, entities and the routing flags that
    # decided whether retrieval fired this turn. Recorded for the audit trace; empty when
    # the pre-pass was skipped (retry legs, which pass skip_retrieval=True).
    routing: dict = field(default_factory=dict)
    model: str = ""
    tokens: dict = field(default_factory=dict)
    # The fast understand() classifier's own token usage (P1, separate LLM call from the
    # main propose() below) -- captured on the Understanding dataclass but otherwise
    # dropped on the floor. Recorded here so the turn's total cost (turn_runner's
    # DecisionTrace.tokens) accounts for every model call the turn actually made, not just
    # the main one.
    understanding_tokens: dict = field(default_factory=dict)
    latency_ms: int = 0
    # Per-stage wall-clock breakdown of THIS plan() call, in ms (P4-0 profiler). Lets the
    # debug envelope attribute a slow turn to a specific stage — the understand+retrieve
    # overlap, the strong propose call, or the rule gate — instead of one opaque planner_ms.
    # "Profile before optimizing": this is what tells us which lever actually moves latency.
    stage_ms: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def band(self) -> str:
        return self.confidence.band if self.confidence else "low"


async def _persona(db: AsyncSession, tenant_id: uuid.UUID) -> str | None:
    row = (await db.execute(select(PromptTemplate).where(
        PromptTemplate.tenant_id == tenant_id, PromptTemplate.purpose == "persona",
        PromptTemplate.is_active == True))).scalars().first()  # noqa: E712
    return row.template if row else None


def _response_schema(tool_names: list[str]) -> dict:
    """Structured Outputs schema for step 6 (OpenAI's strict json_schema shape: every
    object needs additionalProperties=false and every property listed as required).

    The prompt has always asked for this exact shape; the gap was that plain JSON mode
    only guarantees parseable output, not that the model actually included every field
    or picked a real tool name — a "confirm" turn could name no tool, or a made-up one,
    and nothing before this caught it before the trace did. This makes the shape a
    decoding constraint instead of a request.

    "tool" is constrained to the real candidate list plus null, which is what stops a
    hallucinated or misremembered tool name from ever reaching the executor. "params"
    and "entities" travel as JSON-encoded strings (params_json / entities_json) because
    Structured Outputs has no way to express an object with arbitrary keys — every
    property of every object in a strict schema must be named up front, and a tool's
    params differ tool to tool.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["understood", "reasoning", "action", "tool", "params_json",
                     "response_draft", "missing_info", "self_confidence"],
        "properties": {
            "understood": {
                "type": "object",
                "additionalProperties": False,
                "required": ["intent", "sentiment", "entities_json",
                             "human_handoff_requested"],
                "properties": {
                    # Free strings, not an enum — prompts.py explicitly allows the model
                    # to use its own label when none of the suggested ones fit, and
                    # downstream code (NO_TRIAGE_SENTIMENTS, the escalation rules) keys
                    # off specific words like "frustrated" and "distressed" that a
                    # narrower enum would make the model unable to ever produce. This is
                    # also why the operator-editable turn policies (rental_enquiry,
                    # possession_date, pricing_query …) can key on intent at all.
                    "intent": {"type": ["string", "null"]},
                    "sentiment": {"type": ["string", "null"]},
                    # The one HANDOVER decision, made EXPLICITLY by the model rather than
                    # inferred by keyword-matching its free-form intent label. This is the
                    # authoritative signal the escalation rule keys on. It exists because
                    # inferring "wants a human" from a substring of the intent label read
                    # "I'll come by in person" as a handover request ("person") and
                    # transferred a caller mid-reschedule. True ONLY when the caller
                    # actually asks for a person/agent/manager — never for coming in
                    # person, an in-person visit, or naming the sales agent.
                    "human_handoff_requested": {"type": "boolean"},
                    "entities_json": {
                        "type": "string",
                        "description": "JSON-encoded object of extracted slot values, "
                                       "e.g. {\"bhk_type\": \"2 BHK\"}. Use \"{}\" if "
                                       "nothing was extracted.",
                    },
                },
            },
            "reasoning": {"type": "string"},
            "action": {"type": "string", "enum": list(VALID_ACTIONS)},
            "tool": {
                "type": ["string", "null"],
                "enum": [*tool_names, None],
                "description": "One of the candidate tool names, or null. Must be "
                               "non-null whenever action is invoke_tool or confirm — a "
                               "confirm with tool=null stages nothing, so the customer's "
                               "yes has nothing to release.",
            },
            "params_json": {
                "type": "string",
                "description": "JSON-encoded object of the tool's parameters, exactly "
                               "as the tool's schema expects. \"{}\" if tool is null.",
            },
            "response_draft": {"type": "string"},
            "missing_info": {"type": "array", "items": {"type": "string"}},
            "self_confidence": {"type": "number"},
        },
    }


def _parse_json_object(value: Any) -> dict:
    """params_json / entities_json arrive as JSON-encoded strings under Structured
    Outputs; a schema-less caller (the mock provider, tests) may still pass a plain
    dict directly. Accept either, and never let a malformed string reach the executor
    as though it were real data."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def banded_action(action: str, tool: str | None, band: str,
                  stt_confidence: float | None = None) -> str:
    """The action a proposal may take given its confidence band (see plan()'s step 7 note):
    a medium-confidence WRITE is demoted to a read-back ("confirm"), any low-confidence tool
    call escalates, and a read-only lookup is never demoted to "confirm" -- there is nothing
    to confirm, and demoting it only broke the lookup chain on noisy-audio voice turns.

    ``stt_confidence`` -- a WRITE heard below conf_stt_readback_below is read back whatever
    the band says: acting on a possibly-misheard instruction is the one thing low STT
    confidence must prevent. This is that safeguard scoped to actions only. Expressed instead
    as a tenant turn rule it blocked every tool or forced an unanswerable "what is your
    utterance?" ask on any low-confidence turn -- including correctly heard short answers
    ("11" at 0.32), which Deepgram's multi-language model scores low routinely."""
    if action != "invoke_tool":
        return action
    if band == "low":
        return "escalate"
    if tool in READ_ONLY_TOOLS:
        return action
    readback_below = get_settings().conf_stt_readback_below
    if band == "medium" or (stt_confidence is not None and stt_confidence < readback_below):
        return "confirm"
    return action


async def _apply_classifier_project(
        db: AsyncSession, *, tenant_id: uuid.UUID, tc: TurnContext,
        understanding: Understanding, search_query: str, focus: list[str] | None,
        as_of_date: date | None, speculative,
        session_embed_cache: "dict[str, list[float]] | None" = None):
    """The fast classifier picked a project from the offered options. Bind it to the turn.

    Mutates ``tc`` in place so the prompt built after this — its PROJECT FACTS block, its
    retrieval scope — reflects the resolved project, and sets ``project_signal='classifier'``
    so plan() knows to PIN it on the conversation for the rest of the call. Returns the
    retrieval result to use: the speculative one re-run SCOPED to the project when this turn
    wanted a document (its shared-only pass searched the wrong pool), else the speculative
    result untouched. DB-only — no model call. Degrades to leaving the turn ambiguous on any
    lookup failure rather than binding the wrong project.
    """
    from app.db.models import Project
    from app.engine.rag.synonyms import normalize_synonyms
    try:
        idx = tc.project_candidates.index(understanding.project_choice)
        pid = tc.project_candidate_ids[idx]
    except (ValueError, IndexError):
        return speculative
    project = (await db.execute(select(Project).where(
        Project.tenant_id == tenant_id, Project.id == pid))).scalar_one_or_none()
    if project is None:
        return speculative

    tc.project_id = project.id
    tc.project_name = project.name
    tc.retrieval_project_id = project.id
    tc.project_ambiguous = False
    tc.project_signal = "classifier"
    proj_facts = [f for f in (project.facts or [])
                  if isinstance(f, dict) and f.get("label") and f.get("value")]
    if proj_facts:                       # else keep [] — a project may configure none
        tc.project_facts = proj_facts
    proj_syn = normalize_synonyms(project.synonyms)
    if proj_syn:                         # else keep the tenant-wide groups already on tc
        tc.retrieval_synonyms = proj_syn

    # Nothing to retrieve for this turn (a bare "the second one" routes away from documents),
    # or the speculative pass already scoped correctly — keep what we have.
    if not understanding.do_retrieve:
        return speculative
    try:
        async with db.begin_nested():
            return await retrieve(
                db, tenant_id=tenant_id, query=search_query, on_date=as_of_date,
                focus_terms=focus, synonyms=tc.retrieval_synonyms,
                project_id=tc.retrieval_project_id,
                session_embed_cache=session_embed_cache,
                granularity=getattr(tc, "retrieval_granularity", None))
    except Exception:                    # noqa: BLE001 — degrade to the speculative pool
        return speculative


async def _understand_and_retrieve(
        db: AsyncSession, *, tenant_id: uuid.UUID, tc: TurnContext, user_input: str,
        active_entities: dict | None, as_of_date: date | None,
        session_embed_cache: "dict[str, list[float]] | None" = None,
        skip_understanding: bool = False,
        asr_hints: list | None = None,
        awaiting_confirmation: str | None = None,
        ) -> tuple[Understanding | None, list[dict], dict]:
    """Steps 2+3 fused: classify the turn and, if it might want a document, retrieve —
    with the two overlapped so the classification hides behind the search (see plan()).

    Returns ``(understanding, knowledge, retrieval_meta)``. Never raises: a retrieval
    failure degrades to no knowledge (the agent says it will find out) and a classifier
    failure degrades to "retrieve anyway", so a turn is never lost to either.

    ``skip_understanding`` (P4-1, understanding_concurrent_with_propose): do the RETRIEVAL
    ONLY and return ``understanding=None``, with the knowledge ALWAYS kept (the do_retrieve
    routing is not known yet — the caller runs understand() concurrently with propose() and
    applies the routing afterwards). Only used on a non-ambiguous turn, so the project
    disambiguation leg below is never needed in this mode.
    """
    # The QUERY TEXT and the RANKING signal are now decided separately, because folding the
    # call's subject into the query text is only ever safe for a turn that has no subject of
    # its own.
    #
    # A bare follow-up ("give me both cost breakdowns", "the cheaper one") carries no subject:
    # searching it alone returned whichever cost sheet ranks highest rather than the one under
    # discussion, so the carried focus is folded into its query text. But a turn that NAMES its
    # own subject ("is a daycare available at the clubhouse?") must be searched clean — folding
    # in the standing unit turned it into "3 BHK … daycare", and the cost sheets then buried the
    # amenities chunk, so a booked caller was told "no daycare details" while an unknown caller
    # (no carried unit) got the right answer from the same documents. `needs_context_anchor`
    # draws that line with generic English anaphora, no domain nouns.
    _anchor_query = needs_context_anchor(user_input)
    search_query = contextual_query(active_entities, user_input) if _anchor_query else user_input

    # The RANKING signal is always supplied: focus_terms can only REORDER the window (it is
    # non-evicting in retrieval), so on a clean-searched turn it still lifts the right variant
    # when a cost sheet is genuinely in the results, and on a bare follow-up it settles which
    # cost sheet leads. The standing unit preference joins it, placed first so it outranks the
    # drifting in-flight terms.
    _focus = focus_terms(active_entities)
    _pref = getattr(tc, "active_preference", None)
    if _pref:
        if _pref not in _focus:
            _focus = [_pref, *_focus]
        # Fold the preference into the query TEXT only on an anchored (subject-thin) turn, and
        # only when the utterance did not already name it — same rule as the carried focus.
        if _anchor_query and _pref.lower() not in search_query.lower():
            search_query = f"{_pref} {search_query}".strip()

    # A voice transcript's likely-misheard words (app/engine/voice/asr_hints.py): search for
    # the term they probably were too. "when is the position" alone ranks no possession
    # chunk; with "possession" added, the keyword and vector arms both find it. Additive only
    # -- the transcript's own words stay in the query, so a hint that was wrong costs nothing.
    for _h in asr_hints or []:
        if _h.term.lower() not in search_query.lower():
            search_query = f"{search_query} {_h.term}".strip()

    async def _retrieve() -> tuple[Any, str | None]:
        """Speculative retrieval. Wrapped so an exception becomes a value gather() can
        return alongside the understanding result rather than cancelling it. The SAVEPOINT
        keeps a retrieval failure from poisoning the session and losing the message."""
        try:
            async with db.begin_nested():
                rag = await retrieve(
                    db, tenant_id=tenant_id, query=search_query, on_date=as_of_date,
                    # Not just padding the query — telling retrieval which variant this call
                    # is about, as a ranking signal separate from the query text. The standing
                    # preference is folded in above so it outranks the drifting focus terms.
                    focus_terms=_focus,
                    # Tenant synonym groups bridge the caller's words to the documents'
                    # (daycare↔creche, generator↔power backup) so the keyword arm stops
                    # missing chunks it should match.
                    synonyms=tc.retrieval_synonyms,
                    # Scope to the resolved project (multi-project-per-tenant): its own chunks
                    # plus tenant-wide ones. None for a single/tenant-wide tenant, and the
                    # shared-only sentinel while the project is still ambiguous.
                    project_id=tc.retrieval_project_id,
                    session_embed_cache=session_embed_cache,
                    # R5.03: resolved once per turn (tenant override or global) — no extra query.
                    granularity=getattr(tc, "retrieval_granularity", None))
            return rag, None
        except Exception as e:  # noqa: BLE001 — any failure degrades to "no knowledge"
            return None, f"{e.__class__.__name__}: {e}"

    # The classifier (LLM, no DB) and the speculative retrieval (DB, no LLM) touch disjoint
    # resources, so they are safe to run concurrently on the one session. The classifier is
    # on the FAST model; retrieval is the ~300 ms it would have cost regardless.
    #
    # When the project is still ambiguous, the SAME classifier call is also asked to pick which
    # offered project the caller means (candidates=...). This is the second disambiguation tier
    # — it reuses the call that already runs, so it adds NO extra model call — and it only earns
    # its keep for a free-form answer the deterministic selectors in project_resolution missed.
    if skip_understanding:
        # Retrieval only — understand() runs later, concurrently with propose(). Keep the
        # result regardless of routing (unknown here); the caller applies do_retrieve after.
        understanding = None
        rag, rag_error = await _retrieve()
    else:
        understanding, (rag, rag_error) = await asyncio.gather(
            understand(user_input, history=tc.history, model="fast",
                       candidates=tc.project_candidates if tc.project_ambiguous else None,
                       asr_note=prompts.asr_hint_note(asr_hints),
                       awaiting_confirmation=awaiting_confirmation),
            _retrieve())

    # The classifier resolved the ambiguous project from a free-form answer. Apply it to the
    # context NOW — before the prompt and its PROJECT FACTS block are built from tc — pin it for
    # the rest of the call, and re-run retrieval scoped to it if this turn actually wanted a
    # document (the speculative pass ran shared-only, so its chunks are the wrong pool).
    if understanding is not None and tc.project_ambiguous and understanding.project_choice:
        rag = await _apply_classifier_project(
            db, tenant_id=tenant_id, tc=tc, understanding=understanding,
            search_query=search_query, focus=_focus, as_of_date=as_of_date,
            speculative=rag, session_embed_cache=session_embed_cache)

    base_meta = {"attempted": True, "query": search_query, "raw_input": user_input,
                 "as_of": str(as_of_date) if as_of_date else None,
                 "routing": understanding.as_trace() if understanding else {},
                 # Which project this turn resolved to, and how — so "why did it pick that
                 # project / why didn't it ask which?" is answered from the debug envelope.
                 "project": {"id": str(tc.project_id) if tc.project_id else None,
                             "name": tc.project_name, "signal": tc.project_signal,
                             "ambiguous": tc.project_ambiguous,
                             "candidates": tc.project_candidates,
                             "retrieval_scope": (str(tc.retrieval_project_id)
                                                 if tc.retrieval_project_id else None)}}

    # Understanding routed the turn away from the knowledge base (scheduling, a status
    # question, or nothing a document could answer). The speculative search still ran — that
    # is the point — but its result is dropped so the responder is not handed chunks about a
    # subject the customer did not raise.
    # skip_understanding path: routing is not known here, so KEEP the knowledge and let the
    # caller apply do_retrieve after understand() finishes (concurrently with propose).
    if understanding is not None and not understanding.do_retrieve:
        # Record the discarded search's latency too — the whole point of the speculative
        # design is that this cost is hidden behind the classifier, and the profiler can
        # only confirm that if it can see what the wasted search actually took.
        base_meta.update({"returned": 0, "used": False,
                          "latency_ms": getattr(rag, "latency_ms", 0) if rag else 0,
                          "skipped_reason": "understanding routed away from knowledge"})
        return understanding, [], base_meta

    if rag is None:
        base_meta.update({"returned": 0, "used": False, "error": rag_error})
        return understanding, [], base_meta

    knowledge = rag.as_context()
    base_meta.update({"vector_hits": rag.vector_hits, "text_hits": rag.text_hits,
                      "returned": len(knowledge), "latency_ms": rag.latency_ms,
                      "excluded_expired": rag.excluded_expired, "used": True,
                      "error": rag.error,
                      # R5.08: query intent + metadata lifts/filter, for the turn trace.
                      "metadata": getattr(rag, "metadata", None) or {},
                      # R5.02: which HNSW scan settings served this turn, and whether the
                      # index returned fewer rows than actually matched. getattr: diagnostics
                      # are optional — a trace field must never be able to fail a turn.
                      "vector": (rag.vector_diagnostics()
                                 if hasattr(rag, "vector_diagnostics") else None),
                      # R5.01: did the cross-encoder run, how long it took, and what it moved
                      # into / out of the window. None when reranking is disabled.
                      "rerank": (rag.rerank_diagnostics()
                                 if hasattr(rag, "rerank_diagnostics") else None),
                      # R5.03: search granularity + child->parent expansion counters.
                      "parent_child": (rag.parent_child_diagnostics()
                                       if hasattr(rag, "parent_child_diagnostics") else None)})
    return understanding, knowledge, base_meta


async def plan(db: AsyncSession, *, tenant_id: uuid.UUID, conversation: Conversation,
               user_input: str, stt_confidence: float | None = None,
               identity_verified: bool = False, pending_action: dict | None = None,
               as_of_date: date | None = None, model: str | None = None,
               skip_retrieval: bool = False, active_entities: dict | None = None,
               offered_slots: list | None = None,
               booking_summary: str | None = None,
               awaiting_project_choice: bool = False,
               session_embed_cache: "dict[str, list[float]] | None" = None,
               asr_hints: list | None = None,
               awaiting_confirmation: str | None = None) -> Proposal:
    """model  — override the planner model. Continuation steps inside one turn are a
                narrower decision than the opening one and do not need the strong model.
       skip_retrieval — the documents were already fetched earlier in this turn; fetching
                them again for the same customer sentence is pure latency.
       asr_hints — likely-misheard words in a voice transcript (app/engine/voice/asr_hints.py).
                Shown to the classifier and the planner, and folded into the retrieval query,
                so a misheard term is read -- and searched -- as the term it probably was.
       awaiting_confirmation — the question the agent just asked to confirm a parked action.
                Given only while one is parked; the understanding call then also returns the
                caller's consent read (consent.py), on proposal.routing["consent"]."""
    started = time.perf_counter()
    # Per-stage timings accumulate here and ride out on proposal.stage_ms (P4-0 profiler).
    stage_ms: dict = {}
    _t_ctx = time.perf_counter()

    # 1 — context
    tc: TurnContext = await build_turn_context(
        db, tenant_id=tenant_id, conversation=conversation, user_input=user_input,
        stt_confidence=stt_confidence, identity_verified=identity_verified,
        pending_action=pending_action, active_entities=active_entities,
        offered_slots=offered_slots, booking_summary=booking_summary,
        awaiting_project_choice=awaiting_project_choice,
        # So a visit booked for a simulated future date still reads as upcoming when the
        # test harness stands in a different "today".
        as_of_date=as_of_date)

    # 2 — understanding  +  3 — knowledge retrieval, run TOGETHER (P1).
    #
    # The fast understanding call decides SEMANTICALLY whether a document could answer this
    # turn — the judgement the deleted `needs_knowledge`/`SCHEDULING` regexes only ever
    # approximated. To keep it off the critical path (focus.py's rule: never add a serial
    # model call to the retrieval path), it runs CONCURRENTLY with a speculative retrieval
    # on the same query the old gate would have used. If understanding then routes the turn
    # to the scheduler or the customer's record — or says no document is needed — the
    # speculative result is discarded. A wasted embedding is nearly free; the hidden latency
    # is what matters. Skipped wholesale on retry legs (skip_retrieval=True), which have
    # already retrieved once this turn.
    stage_ms["context"] = int((time.perf_counter() - _t_ctx) * 1000)
    understanding: Understanding | None = None
    retrieval_meta = {"attempted": False}
    tc.knowledge = []
    # P4-1: overlap understand() with propose() instead of with the (~15ms) retrieval, so the
    # ~1.3s classifier stops being a serial cost. Only when enabled, retrieval is happening
    # (not a retry leg) and the project is already resolved — an ambiguous project needs the
    # classifier's project_choice BEFORE the prompt is built, so it stays on the serial path.
    _concurrent_understand = (get_settings().understanding_concurrent_with_propose
                              and not skip_retrieval and not tc.project_ambiguous)
    if not skip_retrieval:
        _t_ur = time.perf_counter()
        understanding, tc.knowledge, retrieval_meta = await _understand_and_retrieve(
            db, tenant_id=tenant_id, tc=tc, user_input=user_input,
            active_entities=active_entities, as_of_date=as_of_date,
            session_embed_cache=session_embed_cache,
            # Concurrent mode: retrieve only now (result always kept); understand() is launched
            # alongside propose() below and its routing applied afterwards.
            skip_understanding=_concurrent_understand,
            asr_hints=asr_hints, awaiting_confirmation=awaiting_confirmation)
        # Wall time of this leg. In concurrent mode it is retrieval ALONE (~15ms); understand's
        # own time is recorded after the gather below. In serial mode it is max(understand,
        # retrieve). The individual legs are recorded too, so an outlier can be pinned.
        stage_ms["understand_retrieve"] = int((time.perf_counter() - _t_ur) * 1000)
        stage_ms["understanding"] = understanding.latency_ms if understanding else 0
        stage_ms["retrieval"] = (retrieval_meta or {}).get("latency_ms") or 0
        # The fast classifier resolved the ambiguous project (see _apply_classifier_project):
        # PIN it on the conversation, the same durable write the deterministic resolver makes
        # in build_turn_context, so the rest of the call scopes to it without re-resolving.
        # (Only reachable on the serial path — concurrent mode requires a resolved project.)
        if tc.project_signal == "classifier" and tc.project_id is not None:
            if getattr(conversation, "project_id", None) != tc.project_id:
                conversation.project_id = tc.project_id

    # 4 — candidate actions
    candidates = await registry.candidate_tools(
        db, tenant_id, current_state=tc.current_state, principal_kind="agent")
    tool_specs = [registry.tool_summary(t) for t in candidates]

    # 5 — rule gate, pass 1: rules that depend only on customer/state/time.
    #     Turn-level rules keyed on intent or sentiment cannot fire yet — they are
    #     re-evaluated in pass 2 once the model has told us what it understood.
    _t_rules = time.perf_counter()
    session_ctx = {"identity_verified": identity_verified, "channel": tc.channel,
                   "direction": tc.direction, "turn_index": tc.turn_index}
    rule_ctx = await build_rule_context(
        db, tenant_id, evaluation_point="turn", customer=tc.customer,
        session=session_ctx,
        turn={"sentiment": None, "intent": None, "stt_confidence": stt_confidence})
    decision = await rule_engine.evaluate_point(db, tenant_id, evaluation_point="turn",
                                                ctx=rule_ctx)
    obligations = [f"{r.body.get('reason')} (satisfy '{r.body.get('condition')}' first)"
                   for r in decision.requirements]
    if decision.blocks:
        obligations += [f"BLOCKED: {b.reason}" for b in decision.blocks]
    stage_ms["rules_pass1"] = int((time.perf_counter() - _t_rules) * 1000)

    # 6 — propose
    # Advisory custom instructions for the PLANNER prompt: loaded here (before the model
    # runs, so intent is not yet known) with only direction/state/date scope, so the DRAFT
    # is tone-aware too. The responder still gets the full, intent-aware set in step 6b —
    # it is the mandatory surface. Best-effort: any failure omits the block.
    _planner_prefs = None
    try:
        from app.custom_instructions.loader import load_active_advisory
        from app.custom_instructions.render import render_advisory_block
        from app.custom_instructions.scope import TurnScope
        _planner_prefs = render_advisory_block(await load_active_advisory(
            db, tenant_id, project_id=getattr(tc, "project_id", None),
            turn=TurnScope(direction=tc.direction, state=tc.current_state,
                           intent=None, today=as_of_date or date.today())))
    except Exception:
        logger.exception("advisory load for planner prompt failed — continuing without it")

    # Load the tenant intent vocabulary ONCE here (best-effort). Its labels nudge the planner
    # to emit a tenant's custom canonical intent directly (build_system_prompt below), and its
    # aliases normalise the planner's free-form label onto that intent for the pass-2 rule
    # gate further down — so a Phase-2 routing rule keyed on e.g. loan_enquiry fires reliably.
    # One query, reused for both; a failure leaves behaviour exactly as before.
    _tenant_vocab = None
    _extra_aliases = None
    _extra_intent_labels: list[str] = []
    try:
        from app.custom_instructions.intents import extra_aliases, load_vocabulary, names
        _tenant_vocab = await load_vocabulary(db, tenant_id)
        _extra_aliases = extra_aliases(_tenant_vocab)
        _extra_intent_labels = [e.name for e in _tenant_vocab if e.source == "tenant"]
    except Exception:
        logger.exception("intent vocabulary load failed — using built-in aliases only")

    # Vocal Backend Tasksheet P0, Generic Sprint 1: which industry pack supplies
    # BASE_SYSTEM's domain-specific sections and default persona for this tenant.
    # A failure leaves behaviour exactly as before (real-estate-India default).
    _pack = None
    try:
        from app.packs import PackRegistry
        _pack = await PackRegistry.for_tenant(db, tenant_id)
    except Exception:
        logger.exception("industry pack resolution failed — using default pack")

    system = prompts.build_system_prompt(tc, tool_specs, persona=await _persona(db, tenant_id),
                                         rule_obligations=obligations,
                                         tenant_preferences=_planner_prefs,
                                         extra_intent_labels=_extra_intent_labels,
                                         pack=_pack)
    user = prompts.build_user_prompt(tc, asr_hints=asr_hints)

    proposal = Proposal(action="clarify", candidates=[t.name for t in candidates],
                        rules_evaluated=decision.evaluated, memory_used=memory_reference(tc),
                        knowledge_used=tc.knowledge, retrieval=retrieval_meta,
                        existing_appointments=tc.existing_appointments,
                        retrieval_synonyms=tc.retrieval_synonyms,
                        project_facts=tc.project_facts,
                        project_name=getattr(tc, "project_name", None),
                        enquiry_interest=getattr(tc, "enquiry_interest", None),
                        enquiry_project=getattr(tc, "enquiry_project", None),
                        routing=understanding.as_trace() if understanding else {},
                        stage_ms=stage_ms,
                        understanding_tokens=(understanding.tokens or {}) if understanding else {})
    _t_propose = time.perf_counter()
    try:
        # 900 was the default and far more than this contract needs. A planner call costs
        # about 1300 ms of fixed overhead plus generation time, so output length is one of
        # only two levers that exist — the other being how many calls a turn makes. The
        # contract asks for a short reasoning note, a two-sentence draft and a params
        # object; 400 tokens covers that with room to spare, and anything longer was the
        # model padding at the customer's expense.
        # `model` here is plan()'s own caller-supplied override (None from every real call
        # site today); None flows straight through to each provider's own main tier.
        # When PLANNER_LLM_MODEL_TIER=fast the planner routes to the fast-tier model
        # (e.g. qwen-3.8-27b) while the responder stays on main (gpt-oss-120b).
        _plan_model = model if model is not None else (
            "fast" if get_settings().planner_llm_model_tier == "fast" else None)
        _propose = get_llm().complete_json(
            system=system, user=user, model=_plan_model,
            max_tokens=400, schema=_response_schema([t.name for t in candidates]),
            schema_name="planner_response")
        if _concurrent_understand:
            # P4-1: understand() runs HERE, in parallel with propose(), so its ~1.3s hides
            # behind the (longer) propose call instead of being serial. propose already has the
            # retrieved knowledge in its prompt; understand's routing is applied just below to
            # decide what the RESPONDER sees — so responder behaviour is unchanged.
            understanding, result = await asyncio.gather(
                understand(user_input, history=tc.history, model="fast", candidates=None,
                           asr_note=prompts.asr_hint_note(asr_hints),
                           awaiting_confirmation=awaiting_confirmation),
                _propose)
            stage_ms["understanding"] = understanding.latency_ms if understanding else 0
            proposal.routing = understanding.as_trace() if understanding else {}
            proposal.understanding_tokens = (understanding.tokens or {}) if understanding else {}
            # Apply the routing to the RESPONDER's material only (propose already saw it): a turn
            # understand routed away from documents must not hand the responder off-topic chunks,
            # exactly as the serial path drops them. propose keeping them is the intended change.
            if understanding is not None and not understanding.do_retrieve:
                proposal.knowledge_used = []
                tc.knowledge = []
                retrieval_meta.update({"used": False, "returned": 0,
                                       "skipped_reason": "understanding routed away from "
                                                         "knowledge (concurrent mode)"})
                proposal.retrieval = retrieval_meta
        else:
            result = await _propose
        stage_ms["propose"] = int((time.perf_counter() - _t_propose) * 1000)
        data = result.data
        proposal.model = result.model
        proposal.tokens = result.tokens
        raw_understood = data.get("understood") or {}
        # THE BUG THIS FIXES: this used to be JUST this turn's own entities_json — nothing
        # carried from earlier turns. Asked "I'm looking at 2 BHK", the model's reasoning
        # correctly said "customer specified interest in 2 BHK", but the SAME turn's
        # entities_json came back empty (no tool was being called yet, just an
        # acknowledgment), so the responder's framing block saw no established subject at
        # all and asked the customer to specify the configuration they had just named.
        # session["active_entities"] (tc.active_entities here) already carries this
        # correctly turn-to-turn for tool params via the same merge_entities() used in
        # focus.py — the responder's own "what do they mean" view just never drew on it.
        # Merging keeps this turn's own extraction authoritative (a changed answer wins)
        # while never losing what was already established a moment earlier.
        proposal.understood = {
            **raw_understood,
            "entities": merge_entities(
                tc.active_entities,
                _parse_json_object(
                    raw_understood.get("entities_json", raw_understood.get("entities"))),
                user_input),
        }
        proposal.reasoning = str(data.get("reasoning", ""))[:300]
        proposal.response_draft = str(data.get("response_draft", ""))[:2000]
        proposal.missing_info = list(data.get("missing_info") or [])
        action = str(data.get("action", "clarify"))
        proposal.action = action if action in VALID_ACTIONS else "clarify"
        tool = data.get("tool")
        # The model may only name a tool from the candidate list. This has to run for
        # BOTH invoke_tool and confirm, not invoke_tool alone.
        #
        # THE BUG THIS FIXES: prompts.py tells the model that a "confirm" turn MUST also
        # carry the tool and params it will run once the customer agrees — that is the
        # whole mechanism the confirmation gate depends on. The model regularly did
        # exactly that: {"action": "confirm", "tool": "book_appointment", "params": {...}}
        # — and this block silently threw the tool and params away, because it only ever
        # read them when action was "invoke_tool". Every genuinely correct confirm then
        # looked identical to a genuinely broken one (tool=None either way), and
        # turn_runner's "planner named no tool" corrective retry fired on it — costing a
        # full extra model call on nearly every booking and reschedule in a call, and
        # occasionally failing outright when the retry ALSO used action=confirm.
        if proposal.action in ("invoke_tool", "confirm"):
            if tool is None:
                pass  # a genuine no-tool turn; turn_runner's corrective retry handles it
            elif tool in proposal.candidates:
                proposal.tool = tool
                proposal.params = _parse_json_object(
                    data.get("params_json", data.get("params")))
            else:
                # Say WHY, and say it to the customer.
                #
                # This silently turned the turn into a clarify and told nobody. The model
                # had asked to book a visit, the engine refused because the tool is not
                # permitted in that journey stage, and the model — given no reason — kept
                # trying: a vague reply, then the same question, then a claim that the
                # booking existed. Four turns of a customer being handled around a wall
                # nobody mentioned.
                #
                # A refusal the customer can act on is worth more than a silent one.
                proposal.action = "clarify"
                proposal.error = (f"Model proposed '{tool}', which is not available while "
                                  f"the customer is in state '{tc.current_state}'. "
                                  f"Available: {sorted(proposal.candidates)}")
                if not (proposal.response_draft or "").strip():
                    proposal.response_draft = (
                        "That's not something I can do from here — let me put you through "
                        "to a colleague who can.")
                    proposal.action = "escalate"
                proposal.response_draft = proposal.response_draft or \
                    "Let me check that for you — one moment."
        self_conf = float(data.get("self_confidence", 0.5) or 0.5)
    except LLMError as e:
        stage_ms["propose"] = int((time.perf_counter() - _t_propose) * 1000)
        proposal.error = str(e)
        proposal.action = "escalate"
        proposal.response_draft = ("I'm having trouble on my side — let me get a colleague "
                                   "to help you.")
        self_conf = 0.0

    # 5b — rule gate, pass 2: now that intent and sentiment are known.
    #
    # Reusing pass 1's context rather than rebuilding it. build_rule_context costs six
    # database round trips — tenant, two do-not-contact lookups and three call counters —
    # and NONE of them depend on intent or sentiment. Only the "turn" key changes between
    # the two passes, so the second build was fetching the same six answers a few hundred
    # milliseconds after the first. Everything else about the customer is identical; the
    # model telling us what it understood does not change their call history.
    understood = proposal.understood or {}
    # Reuse the intent vocabulary loaded once for the prompt above (spec §4.3): the aliases
    # normalise the planner's free-form label onto a tenant's custom canonical intent (e.g.
    # loan_enquiry) that a Phase-2 routing rule keys on. Loading it here again would be a
    # second identical query on the hot path for no new information.
    _turn_intent = normalize_intent(understood.get("intent"), _extra_aliases)
    # Safety net for the producer/consumer gap (spec §4.3): the model's label is context-
    # sensitive and, mid-pitch on an OUTBOUND call, a "what are the loan options?" turn can be
    # labelled a product/pricing question and answered — so turn.intent never becomes
    # loan_enquiry and the routing rule silently does not fire. If the model's label did not
    # already resolve to a base safety intent, and the CALLER'S OWN WORDS name a tenant intent
    # (its aliases), key the rule on that — so the rule fires on what the caller asked,
    # inbound or outbound, regardless of the model's labelling. Best-effort; never fails a turn.
    if _tenant_vocab and _turn_intent not in INTENT_ALIASES:
        try:
            from app.custom_instructions.intents import detect_from_text
            _txt_intent = detect_from_text(user_input, _tenant_vocab)
            if _txt_intent and _txt_intent != _turn_intent:
                _turn_intent = _txt_intent
        except Exception:
            logger.exception("tenant-intent text detection failed — using model label")
    # The conversation's captured information, for slot-based `require` gates ("collect budget
    # before offering a visit"): the entities merged across the call so far (active_entities)
    # plus anything the model extracted THIS turn — so a caller who states their budget in the
    # very turn they ask to book satisfies the requirement immediately. Values-only, non-empty.
    _slots: dict = {k: v for k, v in (active_entities or {}).items()
                    if v not in (None, "", [], {})}
    try:
        _te = understood.get("entities")
        if not isinstance(_te, dict):
            import json as _json
            _ej = understood.get("entities_json")
            _te = _json.loads(_ej) if isinstance(_ej, str) and _ej.strip() else {}
        for k, v in (_te or {}).items():
            if v not in (None, "", [], {}):
                _slots[k] = v
    except Exception:
        pass
    rule_ctx2 = {**rule_ctx, "slots": _slots,
                 "turn": {"sentiment": understood.get("sentiment"),
                          "intent": _turn_intent,
                          # The model's explicit handover decision, keyed on directly by
                          # the escalation rule — no keyword inference stands between the
                          # caller asking for a person and the rule firing.
                          "human_handoff": bool(understood.get("human_handoff_requested")),
                          "stt_confidence": stt_confidence}}
    decision2 = await rule_engine.evaluate_point(db, tenant_id, evaluation_point="turn",
                                                 ctx=rule_ctx2)
    seen = {(e.get("rule_set"), e.get("rule")) for e in decision.evaluated if e.get("matched")}
    for e in decision2.evaluated:
        if e.get("matched") and (e.get("rule_set"), e.get("rule")) not in seen:
            decision.evaluated.append(e)
    decision.blocks += decision2.blocks
    decision.escalations += decision2.escalations
    decision.requirements += decision2.requirements
    if decision2.blocks:
        decision.allowed = False
    proposal.rules_evaluated = decision.evaluated

    # 6b — advisory custom instructions. Loaded HERE, after the understanding step, because
    # scope matching may key on this turn's resolved intent (not known at build_system_prompt
    # time). Deterministic metadata filter, tiny row count (per-tenant caps). Best-effort:
    # any failure leaves tenant_preferences None and the turn proceeds exactly as before —
    # an advisory nudge is never worth failing a live turn over.
    try:
        from app.custom_instructions.loader import load_active_advisory
        from app.custom_instructions.render import render_advisory_block
        from app.custom_instructions.scope import TurnScope
        _adv_rows = await load_active_advisory(
            db, tenant_id, project_id=getattr(tc, "project_id", None),
            turn=TurnScope(direction=tc.direction, state=tc.current_state,
                           intent=(proposal.understood or {}).get("intent"),
                           today=as_of_date or date.today()))
        proposal.tenant_preferences = render_advisory_block(_adv_rows)
    except Exception:
        logger.exception("advisory instruction load failed — continuing without it")
        proposal.tenant_preferences = None

    # 7 — confidence
    proposal.confidence = score_confidence(
        self_confidence=self_conf, intent=(proposal.understood or {}).get("intent"),
        stt_confidence=stt_confidence, knowledge_count=len(tc.knowledge),
        missing_info=proposal.missing_info, recent_actions=tc.recent_actions)

    # rule-forced escalation overrides whatever the model wanted — except for
    # compliance actions, which must always be honoured first. The escalation is
    # still recorded and handled immediately afterwards.
    if decision.escalations:
        # Act on the MOST important escalation when several fire: lowest rule priority wins,
        # so a compliance escalation (band 0..99) always precedes a tenant routing one
        # (clamped to >=100). Ties keep insertion order (the first-evaluated rule set).
        e = min(decision.escalations, key=lambda v: getattr(v, "priority", 100))
        proposal.forced_escalation = {"type": e.body.get("type", "immediate"),
                                      "reason": e.reason, "rule": e.rule_id,
                                      # Targeted escalation: the transfer-target key a routing
                                      # instruction named (spec §4.4). None for a generic
                                      # escalation; resolved against the registry in turn_runner.
                                      "target": e.body.get("target")}
        honouring_optout = (proposal.action == "invoke_tool"
                            and proposal.tool in ALWAYS_HONOUR_TOOLS)
        if not honouring_optout:
            proposal.action = "escalate"
        else:
            proposal.escalate_after_tool = True
    elif decision.blocks:
        b = decision.blocks[0]
        proposal.action = "respond" if proposal.action == "invoke_tool" else proposal.action
        proposal.tool = None
        proposal.error = proposal.error or f"Blocked by rule {b.rule_id}: {b.reason}"
        # THE BUG THIS FIXES: every block used to set this, including an ordinary tool-scoped
        # business rule (a tenant's "no site visits on Sundays"). That put the responder into
        # no-pitch "policy_blocked" mode and stripped any "Would Saturday work?" alternative --
        # right for DNC, wrong for a rule that only closes off one option. Scoped to the
        # verdicts that actually mean "stop contacting/selling to this caller entirely."
        if is_no_further_contact_block(b):
            proposal.rule_blocked = {"rule": b.rule_id, "reason": b.reason}

    # rule-required information: "collect X before doing Y". A `require` verdict fired (its
    # `when` matched, e.g. intent == book_appointment) and the required slot is still missing
    # from the captured entities — so the caller is trying to do the gated action without the
    # information the tenant said to collect first. Record it; turn_runner enforces the ask (and
    # the anti-trap counter, which needs the session). Skipped when escalating — handing the
    # caller to a person is more urgent than collecting a field. Detection only; no session here.
    if not proposal.forced_escalation and not decision.escalations:
        _unmet = rule_engine.unmet_requirements(decision, rule_ctx2)
        if _unmet:
            r = _unmet[0]
            proposal.forced_requirement = {
                "condition": r.body.get("condition"),
                # The effective slot — explicit, or derived from the condition for a rule
                # authored before the slot field existed — so the ask can name it.
                "slot": rule_engine.require_slot(r.body),
                "reason": r.body.get("reason"), "ask": r.body.get("ask"),
                "rule": r.rule_id}

    # confidence banding governs whether the action may proceed.
    #
    # A READ-ONLY lookup is never demoted to "confirm": it changes nothing, so there is
    # nothing for the caller to confirm, and demoting it only stops the turn mid-task.
    # THE BUG THIS FIXES: a voice turn heard at STT confidence < conf_stt_threshold carries a
    # 0.85 penalty, which puts an ordinary 0.9 plan at 0.765 -- "medium". Every chain re-plan
    # (turn_runner) re-scored the next lookup the same way, turned locate_centre -> search_slots
    # into "confirm", and broke the chain after ONE step: the caller heard "let me check the
    # slots..." twice, never heard a time, and their next "okay" booked a slot nobody had read
    # out. Chat sends no STT confidence, so it never hit this. Writes still band exactly as
    # before (that is where a read-back matters), and "low" still escalates any action.
    proposal.action = banded_action(proposal.action, proposal.tool, proposal.band,
                                    stt_confidence=stt_confidence)

    proposal.latency_ms = int((time.perf_counter() - started) * 1000)
    return proposal