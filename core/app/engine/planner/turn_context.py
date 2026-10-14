"""Step 1: assemble everything the planner needs for this turn."""
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (Appointment, Call, Conversation, Customer, CustomerFact, Message,
                           Tenant)
from app.engine.rag.synonyms import normalize_synonyms
from app.engine.state_machine.machine import load_definition
from app.services.spoken import relative_day_prefix

logger = logging.getLogger(__name__)

RECENT_TURNS = 12
RECENT_SUMMARIES = 5

# Site visits are spoken in the branch's local zone. The store persists a timezone-aware
# timestamp, but — depending on which write path booked it — the offset it carries may be
# IST or UTC, so it is normalised here before rendering. Mirrors scheduling_tools._spoken;
# kept in step with it so the "your visit is on file" read-back names the same hour the
# booking confirmation did.
_IST = ZoneInfo("Asia/Kolkata")

# Statuses that mean the customer CURRENTLY has a visit on the books. A cancelled,
# completed or no-show visit is history and must never be reported as "scheduled".
_LIVE_APPT_STATUSES = ("booked", "rescheduled")

# How many upcoming visits to surface. A caller almost always has one; a short cap keeps a
# pathological account from flooding the prompt while still covering the ordinary case.
_MAX_UPCOMING = 3


def _spoken_local(dt: datetime | None) -> str | None:
    """One canonical spoken rendering of an appointment time, e.g. "Wednesday 19 August at
    5:00 PM", in the branch's local (IST) zone.

    Hand-formats the 12-hour clock rather than using ``%-I`` / ``%l`` so it is identical on
    Windows and Linux — the platform-specific strftime flags silently break the read-back on
    one of them.
    """
    if dt is None:
        return None
    local = dt.astimezone(_IST) if dt.tzinfo else dt.replace(tzinfo=_IST)
    hour12 = local.hour % 12 or 12
    ampm = "AM" if local.hour < 12 else "PM"
    # A relative day the caller thinks in ("tomorrow, …"), computed deterministically so the
    # read-back is not left to the model's shaky date arithmetic (see app.services.spoken).
    return f"{relative_day_prefix(local)}{local:%A %d %B} at {hour12}:{local.minute:02d} {ampm}"


async def load_upcoming_appointments(db: AsyncSession, *, tenant_id: uuid.UUID,
                                     customer_id: uuid.UUID,
                                     as_of: date | None = None) -> list[dict]:
    """The customer's live visits, read from Postgres — the durable answer to "is my visit
    scheduled?".

    THE BUG THIS FIXES (Voice Agent Feedback #8): booking status was being answered from the
    in-flight session ``BookingTask``, which only knows about a booking being made on THIS
    call. A returning caller whose visit was booked on an earlier call therefore heard "your
    visit is not booked yet" — flatly contradicting the appointment sitting in the database.
    The database is the record of truth, so status is read from it here, at context assembly,
    for every identified caller, and rendered into the prompt as its own block.

    Only genuinely live, not-yet-past visits are returned (``booked`` / ``rescheduled``); a
    cancelled or completed visit is history. ``as_of`` lets the test harness simulate "today"
    so a visit booked for a simulated future date still counts as upcoming; without it, real
    wall-clock time in IST is used.
    """
    rows = (await db.execute(select(Appointment).where(
        Appointment.tenant_id == tenant_id,
        Appointment.customer_id == customer_id,
        Appointment.status.in_(_LIVE_APPT_STATUSES))
        .order_by(Appointment.starts_at.asc()))).scalars().all()

    # "Not past" is judged on the local calendar day, not the exact instant, so a visit
    # earlier today still reads as the customer's current visit rather than vanishing at
    # its start time. The simulated date, when supplied, stands in for today.
    cutoff = as_of or datetime.now(_IST).date()
    out: list[dict] = []
    for a in rows:
        local = a.starts_at.astimezone(_IST) if a.starts_at.tzinfo else a.starts_at
        if local.date() < cutoff:
            continue
        out.append({
            "appointment_id": str(a.id),
            "centre_id": str(a.centre_id) if a.centre_id else None,
            # The short reference the customer is read out elsewhere, so both surfaces agree.
            "reference": str(a.id).replace("-", "")[:8].upper(),
            "starts_at": a.starts_at.isoformat(),
            "when": _spoken_local(a.starts_at),
            "status": a.status,
            # What the visit is FOR, bound at book time. This is the authoritative referent
            # for "the flat I selected" — it does not drift the way the retrieval hint does.
            "unit": a.unit_type,
        })
        if len(out) >= _MAX_UPCOMING:
            break
    return out


@dataclass
class TurnContext:
    tenant_id: uuid.UUID
    conversation_id: uuid.UUID
    customer: Customer | None
    channel: str
    direction: str
    turn_index: int
    user_input: str
    history: list[dict] = field(default_factory=list)        # working memory
    facts: list[dict] = field(default_factory=list)          # semantic memory
    summaries: list[dict] = field(default_factory=list)      # episodic memory
    knowledge: list[dict] = field(default_factory=list)      # retrieved chunks (Stage 8)
    current_state: str | None = None
    state_goal: str | None = None
    # Why an OUTBOUND call is being placed (marketing|service|feedback_survey|…). None on an
    # inbound call. Read by the prompt to set the outbound register and objective; the actual
    # identity precondition is enforced deterministically in turn_runner, not here.
    call_purpose: str | None = None
    identity_verified: bool = False
    pending_action: dict | None = None
    stt_confidence: float | None = None
    recent_actions: list[str] = field(default_factory=list)  # for repeat detection
    active_entities: dict = field(default_factory=dict)      # what the call is about
    # The times a slot search actually returned, exact strings ("Sunday 17 August at
    # 4:00 PM"). search_slots already writes this to session["offered_slots"], but
    # nothing rendered it into the prompt — the model had to reconstruct a date it was
    # told minutes ago from a truncated tool note or an evicted history turn, and it
    # guessed: "May 17th" when today was August. This is what lets it copy instead.
    offered_slots: list = field(default_factory=list)
    # A one-line summary of the in-progress BookingTask (Rule 6), rendered into the prompt
    # so "is it booked?" and every read-back are answered from state, not guessed. None
    # when no booking is in flight, or when the state machine is off.
    booking_summary: str | None = None
    # The identified caller's live visits, read straight from Postgres (the record of
    # truth), so "is my visit scheduled?" is answered from the database even for a visit
    # booked on a PREVIOUS call — which the in-flight booking_summary above never knows
    # about. Empty for unknown callers and for callers with no live visit.
    existing_appointments: list[dict] = field(default_factory=list)
    # Tenant-curated always-answerable facts (location, structure, RERA…), read from
    # tenant.settings["project_facts"]. Rendered into the STABLE prompt prefix so structured
    # answers the brochure buries in prose — "off Sarjapur Road, in Chikkanayakanahalli", "RCC
    # structure, not brick" — are on hand every turn without a retrieval round-trip that a
    # keyword search kept missing (Feedback #6, #7). Each item is {"label", "value"}.
    project_facts: list[dict] = field(default_factory=list)
    # Tenant-declared synonym groups (tenant.settings["retrieval_synonyms"]) that bridge the
    # caller's vocabulary to the documents' — daycare↔creche, generator↔power backup. Threaded
    # to the retrieval keyword arm; the engine itself carries no domain vocabulary. Normalised
    # here so the hot path never re-validates operator-edited JSON.
    retrieval_synonyms: list[list[str]] = field(default_factory=list)
    # R5.03: "flat" | "parent_child" — tenant.settings["retrieval_granularity"] validated against
    # the global RETRIEVAL_GRANULARITY, resolved ONCE per turn here and passed to retrieve(), so
    # per-tenant rollout costs the hot path no extra tenant query.
    retrieval_granularity: str | None = None
    # The caller's standing unit choice ("3 BHK"), from durable CRM memory rather than the
    # in-flight focus bag. Anchors retrieval so "details of the flat I selected" fetches the
    # right cost sheet even after the transient focus has drifted, and gives the prompt an
    # authoritative referent for it. None for unknown callers or before any unit is stated;
    # once a visit is booked its bound unit (existing_appointments) is the stronger record.
    active_preference: str | None = None
    # The project this call resolved to (multi-project-per-tenant). None for a tenant with no
    # projects (tenant-wide, unchanged behaviour) or while ambiguous. project_facts and
    # retrieval_synonyms above are taken FROM this project when resolved.
    project_id: uuid.UUID | None = None
    project_name: str | None = None
    # OUTBOUND only: what THIS customer registered interest in with us — the project
    # (enquiry_project, e.g. "DSR Highland Greenz") and the unit (enquiry_interest, e.g.
    # "2 BHK"). Loaded DURABLY for the whole call (not tied to which project-resolution rule won
    # this turn), so the agent can (a) recall and VERIFY it in the opener before pitching and
    # (b) state it back whenever the caller asks "what was I looking for?" — instead of asking
    # them to remind it. Both None on inbound (the load is skipped) or when no enquiry is on file.
    enquiry_interest: str | None = None
    enquiry_project: str | None = None
    # What to pass to retrieve(): the resolved project, the shared-only sentinel when
    # ambiguous (so no single project's chunks leak into a mixed answer), or None tenant-wide.
    retrieval_project_id: uuid.UUID | None = None
    # True only when the tenant has >1 active project and none was selected — the agent must
    # ask which project before answering product/price questions. Names offered to the caller.
    project_ambiguous: bool = False
    project_candidates: list[str] = field(default_factory=list)
    # The ids of the offered candidates, in the SAME order as project_candidates — so a
    # classifier-resolved choice (by label) can be mapped back to a project id without a
    # second lookup. Parallel arrays, not a dict, because the label carries the location and
    # two projects could in principle share a name.
    project_candidate_ids: list[uuid.UUID] = field(default_factory=list)
    # Which resolution rule won (mention|pinned|dialled_number|caller_history|tenant_default|
    # only_project|ambiguous|tenant_wide) — surfaced in the turn debug so "why did it pick that
    # project / why didn't it ask?" is a one-glance answer, not a guess.
    project_signal: str = "tenant_wide"


async def build_turn_context(db: AsyncSession, *, tenant_id: uuid.UUID,
                             conversation: Conversation, user_input: str,
                             stt_confidence: float | None = None,
                             identity_verified: bool = False,
                             pending_action: dict | None = None,
                             active_entities: dict | None = None,
                             offered_slots: list | None = None,
                             booking_summary: str | None = None,
                             awaiting_project_choice: bool = False,
                             call_purpose: str | None = None,
                             as_of_date: date | None = None) -> TurnContext:
    customer = None
    if conversation.customer_id:
        customer = (await db.execute(select(Customer).where(
            Customer.id == conversation.customer_id))).scalar_one_or_none()

    # Why an outbound call was placed — read from the linked Call so the prompt speaks in the
    # right register. Only on outbound (an inbound call has no purpose); the caller passes it
    # explicitly in tests, otherwise it comes from the Call row.
    if call_purpose is None and conversation.direction == "outbound":
        call_purpose = (await db.execute(
            select(Call.purpose).where(Call.conversation_id == conversation.id)
            .order_by(Call.created_at.desc()).limit(1))).scalars().first()

    msgs = (await db.execute(select(Message).where(
        Message.tenant_id == tenant_id, Message.conversation_id == conversation.id)
        .order_by(Message.created_at.desc()).limit(RECENT_TURNS))).scalars().all()
    history = [{"role": m.role, "content": m.content} for m in reversed(msgs)]
    turn_index = len([m for m in msgs if m.role == "customer"])

    facts: list[dict] = []
    summaries: list[dict] = []
    existing_appointments: list[dict] = []
    active_preference: str | None = None
    enquiry_interest: str | None = None
    enquiry_project: str | None = None
    state_goal = None
    current_state = None

    # Tenant configuration — the curated facts and synonym groups. Loaded for EVERY turn,
    # identified caller or not (unknown callers ask "where is it?" too), from the single
    # tenants row. A tenant with neither key configured simply gets empty lists and the
    # retrieval/prompt behaviour is exactly as before.
    tenant = (await db.execute(select(Tenant).where(
        Tenant.id == tenant_id))).scalar_one_or_none()
    tsettings = (tenant.settings if tenant and isinstance(tenant.settings, dict) else {}) or {}
    # Tenant-wide fallback: what a single-project tenant has always used. A resolved project
    # below overrides these with its own facts/synonyms; a tenant with no projects keeps them.
    project_facts = [f for f in (tsettings.get("project_facts") or [])
                     if isinstance(f, dict) and f.get("label") and f.get("value")]
    retrieval_synonyms = normalize_synonyms(tsettings.get("retrieval_synonyms"))
    from app.config import get_settings as _get_settings
    from app.engine.rag.parent_child import resolve_granularity
    retrieval_granularity = resolve_granularity(tsettings.get("retrieval_granularity"),
                                                _get_settings().retrieval_granularity)

    # Resolve WHICH project this call is about (multi-project-per-tenant). Local import keeps
    # turn_context — imported very early — free of a module cycle. For a tenant with no
    # projects this returns "tenant_wide" and nothing below changes.
    from app.engine.planner.project_resolution import project_choice_label, resolve_project
    resolution = await resolve_project(
        db, tenant_id=tenant_id, conversation=conversation, customer=customer,
        user_input=user_input, awaiting_project_choice=awaiting_project_choice)
    project_id = resolution.project_id
    project_name = resolution.project.name if resolution.project else None
    retrieval_project_id = resolution.retrieval_project_id
    project_ambiguous = resolution.ambiguous
    project_signal = resolution.signal
    # Offer each project with its PLACE, so the caller can tell them apart by location, not
    # just by name ("DSR Highland Greenz (off Sarjapur Road)" vs "Lemon Tree (Whitefield)").
    project_candidates = [project_choice_label(p) for p in resolution.candidates]
    project_candidate_ids = [p.id for p in resolution.candidates]

    if resolution.project is not None:
        # Answer from THIS project's curated facts/synonyms; fall back to the tenant-wide set
        # only where the project has not configured its own (so a half-configured project is
        # never worse than the old single-project behaviour).
        proj_facts = [f for f in (resolution.project.facts or [])
                      if isinstance(f, dict) and f.get("label") and f.get("value")]
        if proj_facts:
            project_facts = proj_facts
        proj_syn = normalize_synonyms(resolution.project.synonyms)
        if proj_syn:
            retrieval_synonyms = proj_syn
    elif project_ambiguous:
        # Which project is unknown: assert NO project's facts, so "what do you offer?" cannot
        # be answered from one project's sheet while the caller may mean another.
        project_facts = []

    # Dynamic knowledge (time-bounded facts: seasonal offers, floor-rise pricing) merged
    # into the SAME PROJECT FACTS block the grounded responder already trusts as verified —
    # so the agent can cite them and even ground a "yes" on them, which an advisory nudge
    # could never do (the responder strips a fact it cannot ground). Just a date-range query,
    # no embedding, no re-index; a row self-expires when valid_until passes. project_id is
    # None when resolution is ambiguous, so only tenant-wide offers surface then — never one
    # project's offer while the caller may mean another. Best-effort: a failure leaves the
    # static facts untouched. See app/custom_instructions/loader.py.
    try:
        from app.custom_instructions.loader import load_active_dynamic_facts
        _today = as_of_date or datetime.now(_IST).date()
        _dynamic = await load_active_dynamic_facts(
            db, tenant_id, project_id=project_id, today=_today)
        if _dynamic:
            project_facts = project_facts + _dynamic
    except Exception:
        logger.exception("dynamic facts load failed — continuing with static facts only")

    # Pin the resolution to the conversation when it rests on real caller evidence, so the
    # rest of the call scopes to the same project without re-resolving from scratch. Written on
    # the shared conversation row and committed with the turn (mirrors the current_state heal).
    if resolution.pin and project_id and getattr(conversation, "project_id", None) != project_id:
        conversation.project_id = project_id

    if customer is not None:
        current_state = customer.current_state
        # The durable booking status, read from the database of record (Feedback #8). Cheap,
        # indexed by (tenant, customer); loaded for every identified caller so the answer to
        # "is my visit scheduled?" is on hand before the customer even asks.
        existing_appointments = await load_upcoming_appointments(
            db, tenant_id=tenant_id, customer_id=customer.id, as_of=as_of_date)
        # The standing unit choice, read from durable memory. Local import avoids a module
        # cycle (preference → models; turn_context is imported very early).
        from app.engine.planner.preference import load_active_preference
        active_preference = await load_active_preference(
            db, tenant_id=tenant_id, customer_id=customer.id)

        # OUTBOUND only: the interest THIS customer registered with us (project + unit), read
        # durably so the agent can recall/verify it in the opener AND state it back at any point
        # the caller asks "what was I looking for?". Gated on direction so an INBOUND turn never
        # runs this query — inbound cost and behaviour are byte-for-byte unchanged. Degrades to
        # None on any failure (the opener/prompt simply omit the recall).
        if conversation.direction == "outbound":
            from app.engine.planner.project_resolution import load_registered_interest
            try:
                _reg = await load_registered_interest(
                    db, tenant_id=tenant_id, customer_id=customer.id)
            except Exception:
                _reg = None
            if _reg is not None:
                enquiry_project, enquiry_interest = _reg

        defn = await load_definition(db, tenant_id)

        # Reconcile a STALE booking state against the source of truth. current_state is a
        # denormalised cache; the appointments table (existing_appointments, just loaded) is
        # authoritative. When the state says "site_visit_scheduled" but NO live/upcoming visit
        # exists — the visit was cancelled or a no-show and the state never reverted — the caller
        # is deadlocked: book_appointment is refused ("Tool not allowed in state
        # 'site_visit_scheduled'") for a visit that no longer exists, and there is nothing live
        # to reschedule or cancel either. Heal it to "qualified", EXACTLY the transition the
        # state machine already defines for a cancellation (site_visit_scheduled
        # --tool:cancel_appointment--> qualified). Persisted on the shared customer row (same
        # session, committed with the turn), so booking is unblocked now and the CRM stops
        # claiming a visit the caller does not have. Only fires when the truth is empty, so a
        # caller who genuinely has a live visit is untouched.
        if current_state == "site_visit_scheduled" and not existing_appointments:
            current_state = "qualified"
            customer.current_state = current_state

        state_goal = (defn.get("states", {}).get(current_state, {}) or {}).get("goal")

        rows = (await db.execute(select(CustomerFact).where(
            CustomerFact.tenant_id == tenant_id, CustomerFact.customer_id == customer.id,
            CustomerFact.status == "active").order_by(
            CustomerFact.created_at.desc()).limit(25))).scalars().all()
        facts = [{"id": str(f.id), "category": f.category, "key": f.key,
                  "value": f.value, "confidence": float(f.confidence)} for f in rows]

        convs = (await db.execute(select(Conversation).where(
            Conversation.tenant_id == tenant_id, Conversation.customer_id == customer.id,
            Conversation.id != conversation.id, Conversation.summary.isnot(None))
            .order_by(Conversation.created_at.desc()).limit(RECENT_SUMMARIES))).scalars().all()
        summaries = [{"id": str(c.id), "summary": c.summary, "outcome": c.outcome,
                      "sentiment": c.sentiment, "at": c.created_at.isoformat()
                      if c.created_at else None} for c in convs]

    recent_actions = [m.content for m in reversed(msgs)
                      if m.role == "system" and m.content.startswith("action:")][-3:]

    return TurnContext(
        tenant_id=tenant_id, conversation_id=conversation.id, customer=customer,
        channel=conversation.channel, direction=conversation.direction,
        turn_index=turn_index, user_input=user_input, history=history,
        facts=facts, summaries=summaries, current_state=current_state,
        state_goal=state_goal, call_purpose=call_purpose,
        identity_verified=identity_verified,
        pending_action=pending_action, stt_confidence=stt_confidence,
        recent_actions=recent_actions, active_entities=dict(active_entities or {}),
        offered_slots=list(offered_slots or []), booking_summary=booking_summary,
        existing_appointments=existing_appointments, project_facts=project_facts,
        retrieval_synonyms=retrieval_synonyms, active_preference=active_preference,
        retrieval_granularity=retrieval_granularity,
        project_id=project_id, project_name=project_name,
        enquiry_interest=enquiry_interest, enquiry_project=enquiry_project,
        retrieval_project_id=retrieval_project_id, project_ambiguous=project_ambiguous,
        project_candidates=project_candidates,
        project_candidate_ids=project_candidate_ids, project_signal=project_signal)


def memory_reference(tc: TurnContext) -> dict[str, Any]:
    """What the audit trail records as 'memory consulted this turn'."""
    return {"fact_ids": [f["id"] for f in tc.facts],
            "summary_conversation_ids": [s["id"] for s in tc.summaries],
            "appointment_ids": [a["appointment_id"] for a in tc.existing_appointments],
            "history_turns": len(tc.history)}