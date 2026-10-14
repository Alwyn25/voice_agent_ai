"""Outbound call gating — the single path every automated dial must pass through.

Campaigns, retries, reminders and manual triggers all call place_outbound_call().
There is deliberately no other way to create an outbound Call row, so the DNC and
calling-hours rules cannot be bypassed by a new feature forgetting to check them.
"""
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import AnalyticsEvent, Call, Customer, Escalation
from app.engine.rules import engine as rule_engine
from app.engine.rules.context import build_context
from app.services.events import publish

# The time-of-day rules across BOTH active calling policies — the DSR set
# (dsr_calling_policy.yaml) and the generic `calling_policy`. Only these are lifted by the
# OUTBOUND_IGNORE_CALLING_HOURS test flag; every other block (DNC, escalation, consent, caps,
# broker ownership …) still applies. Both `business_calling_hours` and `legal_calling_hours`
# emit "Outside permitted calling hours (09:00-21:00)", so both must be here.
_CALLING_HOURS_RULE_IDS = frozenset({
    # dsr_calling_policy
    "legal_calling_hours", "lead_preferred_hours",
    "no_marketing_before_ten", "no_marketing_after_eight",
    # generic calling_policy
    "business_calling_hours", "customer_preferred_hours",
})

# Same idea, for the attempt/answer/lifetime counters instead of the clock — how many times a
# customer has already been dialled, not when. Lifted by OUTBOUND_IGNORE_FREQUENCY_CAPS alone;
# consent (DNC, opt-out, legal dispute, open escalation) and every segment/business rule are
# untouched by either flag.
_FREQUENCY_CAP_RULE_IDS = frozenset({
    # dsr_calling_policy
    "daily_attempt_cap", "marketing_daily_cap", "stop_after_repeated_no_answer",
    "back_off_after_two_no_answer", "failed_interaction_ceiling", "lifetime_contact_ceiling",
    "new_lead_one_attempt_a_day",
    # generic calling_policy
    "back_off_after_three_no_answer", "new_lead_cooling_period",
})


@dataclass
class DialDecision:
    allowed: bool
    call_id: uuid.UUID | None = None
    blocked_by: str | None = None          # rule id
    reason: str | None = None
    rules_evaluated: list[dict] = field(default_factory=list)


async def check_can_call(db: AsyncSession, *, tenant_id: uuid.UUID, customer: Customer,
                         purpose: str = "marketing") -> DialDecision:
    """Evaluate the calling policy without dialling. Used by campaign preview too."""
    # An open escalation means a human owns this customer — no automated contact.
    open_escalation = (await db.execute(select(Escalation.id).where(
        Escalation.tenant_id == tenant_id, Escalation.customer_id == customer.id,
        Escalation.status.in_(["pending", "accepted"])))).first()
    if open_escalation is not None:
        return DialDecision(allowed=False, blocked_by="open_escalation",
                            reason="A human is handling this customer; automated contact "
                                   "is suspended until the escalation is resolved")

    ctx = await build_context(db, tenant_id, evaluation_point="outbound_call",
                              customer=customer,
                              session={"direction": "outbound", "purpose": purpose})
    decision = await rule_engine.evaluate_point(db, tenant_id,
                                                evaluation_point="outbound_call", ctx=ctx)
    if not decision.allowed:
        blocks = decision.blocks
        # A block's `scope` (set in its `then: [{block: {scope: ...}}]`) says which kind of
        # outbound attempt it applies to -- "all"/"outbound_call" always applies, but a rule
        # scoped "marketing" (18 of the ~34 rules in dsr_calling_policy: opt-outs, promo-hour
        # windows, "don't re-pitch someone with a scheduled visit", etc.) must NOT block a
        # "service" call (e.g. confirming that same visit) -- that purpose split is the entire
        # reason this function takes a `purpose` argument. The rule engine itself is
        # evaluation-point-agnostic and never reads `scope` (other evaluation points use it for
        # unrelated things, e.g. a turn-level rule's scope: "tool:book_appointment"), so this
        # call path is where that filtering has to happen.
        if purpose != "marketing":
            blocks = [b for b in blocks if b.body.get("scope") != "marketing"]
        # TEST-ONLY: when OUTBOUND_IGNORE_CALLING_HOURS is set, drop a block that came from a
        # time-of-day rule so a dial can be exercised outside 09:00-21:00. Deliberately scoped
        # to the hours rules by id — every other block (DNC, open escalation, consent, daily /
        # lifetime caps, broker ownership …) still stands, so the gate's real invariants hold.
        if get_settings().outbound_ignore_calling_hours:
            blocks = [b for b in blocks if b.rule_id not in _CALLING_HOURS_RULE_IDS]
        # TEST-ONLY, same shape: when OUTBOUND_IGNORE_FREQUENCY_CAPS is set, drop a block that
        # came from an attempt/answer/lifetime counter, so repeat test dials to the same few
        # seeded customers aren't spent hitting "one call a day" instead of exercising the
        # call. Consent and segment/business rules are untouched by this flag too.
        if get_settings().outbound_ignore_frequency_caps:
            blocks = [b for b in blocks if b.rule_id not in _FREQUENCY_CAP_RULE_IDS]
        if blocks:
            block = blocks[0]
            return DialDecision(allowed=False, blocked_by=block.rule_id, reason=block.reason,
                                rules_evaluated=decision.evaluated)
    return DialDecision(allowed=True, rules_evaluated=decision.evaluated)


async def place_outbound_call(db: AsyncSession, *, tenant_id: uuid.UUID, customer: Customer,
                              purpose: str = "marketing", campaign_id: uuid.UUID | None = None,
                              retry_of: uuid.UUID | None = None,
                              retry_reason: str | None = None,
                              actor: str = "system") -> DialDecision:
    """Create the Call row if policy allows. Actual dialling happens in the telephony stage;
    until then the call sits in 'queued' with every gate already applied."""
    decision = await check_can_call(db, tenant_id=tenant_id, customer=customer, purpose=purpose)

    if not decision.allowed:
        db.add(AnalyticsEvent(tenant_id=tenant_id, event_type="call_blocked",
                              customer_id=customer.id,
                              properties={"rule": decision.blocked_by, "reason": decision.reason,
                                          "purpose": purpose,
                                          "campaign_id": str(campaign_id) if campaign_id else None}))
        await db.flush()
        publish(tenant_id, "call.blocked", customer_id=str(customer.id),
                rule=decision.blocked_by, reason=decision.reason)
        return decision

    call = Call(tenant_id=tenant_id, customer_id=customer.id, to_number=customer.phone_normalized,
                status="queued", provider=None, purpose=purpose, campaign_id=campaign_id,
                retry_of_call_id=retry_of, retry_reason=retry_reason)
    db.add(call)
    db.add(AnalyticsEvent(tenant_id=tenant_id, event_type="call_queued",
                          customer_id=customer.id,
                          properties={"purpose": purpose, "actor": actor,
                                      "campaign_id": str(campaign_id) if campaign_id else None}))
    await db.flush()
    decision.call_id = call.id
    publish(tenant_id, "call.queued", call_id=str(call.id), customer_id=str(customer.id),
            phone=customer.phone_normalized, purpose=purpose)
    return decision
