"""Post-call memory pipeline: Observe -> Extract -> Summarise -> Update -> Audit.

Runs after a conversation ends. One cheap-model call produces both the episodic
summary and the semantic facts; deterministic code then merges them with supersede
semantics so nothing is ever silently overwritten.

Design rules:
- Facts are append-only. A changed value supersedes the old row and links back to it.
- Low-confidence facts are stored as 'candidate' for human review, not discarded.
- Profile fields are only overwritten above a higher confidence bar.
- Every write is attributable to the conversation it came from.
"""
import logging
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import (AnalyticsEvent, Conversation, Customer, CustomerFact,
                           CustomerProductInterest, Message)
from app.observability import StageTimer
from app.providers.llm.base import LLMError
from app.providers.llm.router import get_llm

FACT_CATEGORIES = ("pain_point", "objection", "preference", "buying_intent",
                   "decision_maker", "product_interest")

BUSINESS_DAY_START = time(9, 0)
BUSINESS_DAY_END = time(21, 0)

log = logging.getLogger(__name__)

OUTCOMES = ("booked", "rescheduled", "cancelled", "callback_requested", "not_interested",
            "info_provided", "complaint_raised", "escalated", "opted_out", "no_outcome")

ANALYSIS_SYSTEM = """You analyse a finished customer service phone call and return structured data.

Return JSON with exactly these keys:
{
  "summary": "<2-3 sentences: what the customer wanted, what happened, what is outstanding>",
  "outcome": "booked|rescheduled|cancelled|callback_requested|not_interested|info_provided|complaint_raised|escalated|opted_out|no_outcome",
  "sentiment": "positive|neutral|negative|angry",
  "intent_labels": ["<short snake_case labels for what the call was about>"],
  "next_step": "<what should happen next, or empty string>",
  "facts": [
    {"category": "pain_point|objection|preference|buying_intent|decision_maker|product_interest",
     "key": "<short stable snake_case identifier, e.g. preferred_time, price_concern>",
     "value": "<the fact in a few words>",
     "confidence": <0.0-1.0>}
  ],
  "profile_updates": {
    "preferred_language": null,
    "preferred_contact_start": null,
    "preferred_contact_end": null,
    "email": null,
    "location": null
  },
  "product_interests": [{"product": "<name>", "level": "interested|purchased|rejected"}]
}

RULES:
1. Only record what the customer actually said or clearly implied. Never infer personality,
   financial status, health, or anything they did not state.
2. Extract EVERY durable fact the customer revealed — objections, preferences, constraints,
   who decides, what they are interested in. Include them even if they also appear in the
   summary or in profile_updates; the summary is prose, facts are the queryable record.
   A call where the customer objects to price AND names a decision maker AND states a
   calling preference must produce at least three facts.
3. "key" must be stable across calls so the same fact updates rather than duplicates:
   use preferred_time, price_concern, decision_maker, competitor, budget_constraint, etc.
4. confidence reflects how explicit the customer was: stated outright = 0.9+,
   implied = 0.5-0.7, guessed = below 0.5.
5. profile_updates fields stay null unless the customer stated them explicitly.
   Times are "HH:MM" 24-hour. preferred_contact_start is the EARLIEST time they may be
   called; preferred_contact_end is the LATEST. Examples:
     "only call me after 6 in the evening"  -> start "18:00", end "21:00"
     "call before 10 in the morning"        -> start "09:00", end "10:00"
     "between 2 and 5 please"               -> start "14:00", end "17:00"
     "mornings are better"                  -> start "09:00", end "12:00"
6. Do not restate the appointment itself as a fact — the booking record already holds it."""


@dataclass
class MemoryReport:
    conversation_id: uuid.UUID
    summary: str | None = None
    outcome: str | None = None
    sentiment: str | None = None
    facts_added: list[dict] = field(default_factory=list)
    facts_superseded: list[dict] = field(default_factory=list)
    facts_unchanged: int = 0
    candidates: int = 0
    profile_updated: dict = field(default_factory=dict)
    product_interests: list[dict] = field(default_factory=list)
    skipped: str | None = None
    model: str = ""
    tokens: dict = field(default_factory=dict)
    # Per-stage latency of the post-call run (P1.08 instrumentation). The post-call
    # pipeline runs as a FastAPI BackgroundTask on the SAME event loop as live turns, so
    # its cost is not free — a slow stage here competes with in-flight calls. Timed so a
    # regression (e.g. the analysis LLM getting slower, or fact writes multiplying DB
    # round-trips) is visible in the logs instead of silently taxing the loop.
    timing: dict = field(default_factory=dict)


def _render_transcript(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        if m.role == "system":
            continue
        speaker = {"customer": "Customer", "agent": "Agent", "tool": "[system note]"}.get(m.role, m.role)
        lines.append(f"{speaker}: {m.content}")
    return "\n".join(lines)


def _parse_time(v: Any) -> time | None:
    if not v or not isinstance(v, str):
        return None
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", v.strip())
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    return time(h, mi) if 0 <= h <= 23 and 0 <= mi <= 59 else None


def _value_of(v: Any) -> str:
    if isinstance(v, dict):
        return str(v.get("text", json.dumps(v, sort_keys=True)))
    return str(v)


# An outcome the model read off the transcript is a claim, not a fact. The analysis LLM
# reported outcome="booked" on calls where book_appointment never ran — because the agent
# had SAID "shall I book that" and, once, "done, you're booked". That claim then became
# an episodic summary, which became context on the next call, and the agent opened by
# telling the customer about a booking that did not exist. A wrong number in a dashboard
# is bad; a wrong belief in memory compounds.
#
# So every action-implying outcome is checked against what actually executed.
OUTCOME_REQUIRES_TOOL = {
    "booked": ("book_appointment",),
    "rescheduled": ("reschedule_appointment",),
    "cancelled": ("cancel_appointment",),
    "callback_requested": ("schedule_callback",),
    "opted_out": ("record_do_not_contact",),
}


async def _verify_outcome(db: AsyncSession, conv, claimed: str) -> str:
    """Downgrade an outcome the tool record does not support."""
    required = OUTCOME_REQUIRES_TOOL.get(claimed)
    if not required:
        return claimed  # escalated / complaint_raised / info_provided need no tool

    from app.db.models import ToolExecution
    succeeded = set((await db.execute(select(ToolExecution.tool_name).where(
        ToolExecution.tenant_id == conv.tenant_id,
        ToolExecution.conversation_id == conv.id,
        ToolExecution.status == "success"))).scalars().all())

    if any(tool in succeeded for tool in required):
        return claimed

    log.warning("post_call: outcome %r claimed for conversation %s but %s never succeeded "
                "— recording no_outcome instead", claimed, conv.id, " or ".join(required))
    return "no_outcome"


async def run_post_call(db: AsyncSession, *, tenant_id: uuid.UUID,
                        conversation_id: uuid.UUID) -> MemoryReport:
    s = get_settings()
    report = MemoryReport(conversation_id=conversation_id)
    # Stage timing for the post-call run (P1.08). Records into report.timing so
    # run_post_call_task can log where the background work went; costs only perf_counter
    # deltas. Early returns below simply leave later stages absent from the dict.
    _t = StageTimer()

    async with _t.stage("load"):
        conv = (await db.execute(select(Conversation).where(
            Conversation.tenant_id == tenant_id,
            Conversation.id == conversation_id))).scalar_one_or_none()
    if conv is None:
        report.skipped = "conversation not found"
        report.timing = _t.as_dict()
        return report

    async with _t.stage("load"):
        messages = (await db.execute(select(Message).where(
            Message.tenant_id == tenant_id, Message.conversation_id == conversation_id)
            .order_by(Message.created_at))).scalars().all()
    customer_turns = [m for m in messages if m.role == "customer"]
    if len(customer_turns) < 1:
        report.skipped = "no customer turns to analyse"
        report.timing = _t.as_dict()
        return report

    transcript = _render_transcript(messages)

    try:
        # The dominant post-call cost — a full analysis-model call. Timed on its own so it
        # is never confused with the DB work around it.
        async with _t.stage("analyze_llm"):
            result = await get_llm().complete_json(
                system=ANALYSIS_SYSTEM,
                user=f"## TRANSCRIPT\n{transcript}\n\nReturn the JSON object only.",
                model=s.llm_model_summary, temperature=0.1, max_tokens=2000)
        data = result.data
        report.model, report.tokens = result.model, result.tokens
    except LLMError as e:
        report.skipped = f"analysis failed: {e}"
        report.timing = _t.as_dict()
        return report

    # ---- episodic: the conversation summary ----
    conv.summary = str(data.get("summary") or "")[:4000] or None
    outcome = str(data.get("outcome") or "").strip()
    outcome = outcome if outcome in OUTCOMES else "no_outcome"
    conv.outcome = await _verify_outcome(db, conv, outcome)
    conv.sentiment = str(data.get("sentiment") or "").strip() or None
    labels = data.get("intent_labels") or []
    conv.intent_labels = [str(x)[:60] for x in labels][:8] if isinstance(labels, list) else []
    report.summary, report.outcome, report.sentiment = conv.summary, conv.outcome, conv.sentiment

    if conv.customer_id is None:
        await db.flush()
        report.timing = _t.as_dict()
        return report

    customer = (await db.execute(select(Customer).where(
        Customer.id == conv.customer_id))).scalar_one_or_none()
    if customer is None:
        await db.flush()
        report.timing = _t.as_dict()
        return report

    # ---- semantic: facts, with supersede semantics ----
    incoming = [f for f in (data.get("facts") or []) if isinstance(f, dict)][:s.max_facts_per_call]
    existing = (await db.execute(select(CustomerFact).where(
        CustomerFact.tenant_id == tenant_id, CustomerFact.customer_id == customer.id,
        CustomerFact.status.in_(["active", "candidate"])))).scalars().all()
    by_key = {(f.category, f.key): f for f in existing}

    # One flush for the whole fact merge, not one per fact (P1.08 latency fix).
    #
    # The mid-loop flush only ever existed to populate new_fact.id so a superseded prior
    # could point at its replacement (superseded_by). That made the DB round-trips scale
    # with the number of facts a call produced — up to max_facts_per_call of them, each a
    # separate await on the shared event loop. Instead the (prior, new_fact) pairs are
    # collected and the ids are wired after a SINGLE flush below.
    #
    # This is provably equivalent to the per-fact version: by_key is built once from the
    # pre-existing rows and is never updated with the new facts, so no iteration reads back
    # a row a previous iteration wrote — there is no dependency on an id being present yet.
    # prior.value is never mutated, so the "was" value captured for the report during the
    # loop is still the old value. Only superseded_by is deferred, and it needs the flush.
    _pending_supersede: list[tuple[CustomerFact, CustomerFact]] = []
    _added_any = False
    async with _t.stage("facts_merge"):
        for raw in incoming:
            category = str(raw.get("category") or "").strip()
            key = str(raw.get("key") or "").strip()[:100]
            if category not in FACT_CATEGORIES or not key:
                continue
            try:
                confidence = float(raw.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))
            value_text = _value_of(raw.get("value"))[:500]
            if not value_text:
                continue

            status = "active" if confidence >= s.fact_confidence_threshold else "candidate"
            prior = by_key.get((category, key))

            if prior is not None and _value_of(prior.value) == value_text:
                if confidence > float(prior.confidence):
                    prior.confidence = confidence
                report.facts_unchanged += 1
                continue

            new_fact = CustomerFact(
                tenant_id=tenant_id, customer_id=customer.id, category=category, key=key,
                value={"text": value_text}, confidence=confidence,
                source_conversation_id=conversation_id, status=status)
            db.add(new_fact)
            _added_any = True

            if prior is not None:
                # Mark the prior superseded now; link it to its replacement's id after the
                # single flush below, once the id exists.
                prior.status = "superseded"
                _pending_supersede.append((prior, new_fact))
                report.facts_superseded.append({"category": category, "key": key,
                                                "was": _value_of(prior.value), "now": value_text})
            else:
                report.facts_added.append({"category": category, "key": key,
                                           "value": value_text, "confidence": confidence,
                                           "status": status})
            if status == "candidate":
                report.candidates += 1

        if _added_any:
            await db.flush()                       # one round-trip populates every new id
            for _prior, _new in _pending_supersede:
                _prior.superseded_by = _new.id

    # ---- profile promotion (higher bar, explicit statements only) ----
    updates = data.get("profile_updates") or {}
    if isinstance(updates, dict):
        start = _parse_time(updates.get("preferred_contact_start"))
        end = _parse_time(updates.get("preferred_contact_end"))
        # A one-sided window is ambiguous ("after 6pm" often arrives as end=18:00).
        # Complete it with the business day rather than storing something misleading.
        if start is not None and end is None:
            end = BUSINESS_DAY_END
        elif end is not None and start is None:
            start = BUSINESS_DAY_START if end > BUSINESS_DAY_START else BUSINESS_DAY_START
        if start is not None and end is not None and start >= end:
            # inverted or degenerate window: treat the single stated time as the start
            start, end = start, BUSINESS_DAY_END
        if start is not None and end is not None:
            # a preference narrows the legal calling window, never widens it
            start = max(start, BUSINESS_DAY_START)
            end = min(end, BUSINESS_DAY_END)
            if start >= end:
                start, end = BUSINESS_DAY_START, BUSINESS_DAY_END
            customer.preferred_contact_start = start
            customer.preferred_contact_end = end
            report.profile_updated["preferred_contact_start"] = start.strftime("%H:%M")
            report.profile_updated["preferred_contact_end"] = end.strftime("%H:%M")
        lang = updates.get("preferred_language")
        if isinstance(lang, str) and 2 <= len(lang) <= 10:
            customer.preferred_language = lang.lower()[:10]
            report.profile_updated["preferred_language"] = customer.preferred_language
        email = updates.get("email")
        if isinstance(email, str) and "@" in email and len(email) <= 255:
            customer.email = email
            report.profile_updated["email"] = email
        loc = updates.get("location")
        if isinstance(loc, str) and 2 <= len(loc) <= 255:
            customer.location = loc
            report.profile_updated["location"] = loc

    # ---- product interests ----
    for pi in (data.get("product_interests") or [])[:8]:
        if not isinstance(pi, dict):
            continue
        product = str(pi.get("product") or "").strip()[:200]
        level = str(pi.get("level") or "interested").strip()
        if not product or level not in ("interested", "purchased", "rejected"):
            continue
        # Scope interest to the project the call resolved to (multi-project-per-tenant), so the
        # same unit in two projects stays two distinct interests.
        _proj_id = getattr(conv, "project_id", None)
        dup = (await db.execute(select(CustomerProductInterest).where(
            CustomerProductInterest.tenant_id == tenant_id,
            CustomerProductInterest.customer_id == customer.id,
            CustomerProductInterest.project_id == _proj_id,
            CustomerProductInterest.product_ref == product))).scalars().first()
        if dup is not None:
            dup.interest_level = level
        else:
            db.add(CustomerProductInterest(
                tenant_id=tenant_id, customer_id=customer.id, project_id=_proj_id,
                product_ref=product, interest_level=level, noted_at=datetime.now(timezone.utc),
                source_conversation_id=conversation_id))
        report.product_interests.append({"product": product, "level": level})

    # ---- audit ----
    async with _t.stage("audit"):
        db.add(AnalyticsEvent(tenant_id=tenant_id, event_type="post_call_analysed",
                              customer_id=customer.id, conversation_id=conversation_id,
                              properties={"outcome": conv.outcome, "sentiment": conv.sentiment,
                                          "facts_added": len(report.facts_added),
                                          "facts_superseded": len(report.facts_superseded),
                                          "profile_updated": list(report.profile_updated)}))
        await db.flush()
    report.timing = {**_t.as_dict(), "total_ms": _t.total_ms()}
    return report


async def run_post_call_task(tenant_id: uuid.UUID, conversation_id: uuid.UUID) -> None:
    """Background entry point — opens its own session (the request's is already closed).

    This is the seam where Celery takes over in the infrastructure stage: the call site
    changes, the pipeline does not.
    """
    from app.db.base import get_session_factory
    async with get_session_factory()() as db:
        try:
            report = await run_post_call(db, tenant_id=tenant_id, conversation_id=conversation_id)
            await db.commit()
            # Post-call runs on the live event loop, so its cost is worth watching. Logged
            # at DEBUG (it is background housekeeping, not an incident) with the per-stage
            # timing — analyze_llm dominates; a spike in load/facts_merge/audit points at
            # DB latency or an unexpectedly fact-heavy call.
            if report.timing:
                log.debug("post_call timing for %s: %s%s", conversation_id, report.timing,
                          f" (skipped: {report.skipped})" if report.skipped else "")
        except Exception:
            await db.rollback()
            log.exception("post_call failed for conversation %s", conversation_id)
