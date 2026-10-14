"""Rule evaluation: gather applicable rule sets, evaluate in priority order,
accumulate verdicts. A block can never be overridden.
"""
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import RuleSet
from app.engine.rules import schema
from app.engine.rules.conditions import ConditionError, evaluate


@dataclass
class Verdict:
    kind: str            # block | require | escalate | set
    rule_id: str
    rule_set: str
    body: dict[str, Any] = field(default_factory=dict)
    # Priority of the rule that produced this verdict (lower = more important). Lets a
    # consumer prefer a compliance verdict over a lower-ranked tenant one when several fire
    # on the same turn — e.g. picking which escalation to act on.
    priority: int = 100

    @property
    def reason(self) -> str:
        return self.body.get("reason", "")


@dataclass
class RuleDecision:
    allowed: bool
    blocks: list[Verdict] = field(default_factory=list)
    requirements: list[Verdict] = field(default_factory=list)
    escalations: list[Verdict] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)
    evaluated: list[dict] = field(default_factory=list)   # full trace for audit

    @property
    def block_reason(self) -> str | None:
        return self.blocks[0].reason if self.blocks else None


async def active_rule_sets(db: AsyncSession, tenant_id: uuid.UUID) -> list[RuleSet]:
    return list((await db.execute(select(RuleSet).where(
        RuleSet.tenant_id == tenant_id,
        RuleSet.is_active == True))).scalars().all())  # noqa: E712


async def evaluate_point(db: AsyncSession, tenant_id: uuid.UUID, *,
                         evaluation_point: str, ctx: dict) -> RuleDecision:
    decision = RuleDecision(allowed=True)
    for rs in await active_rule_sets(db, tenant_id):
        compiled = rs.compiled or {}
        if not schema.applies(compiled, evaluation_point):
            continue
        for rule in compiled.get("rules", []):
            rid = rule.get("id", "?")
            try:
                matched = evaluate(rule.get("when"), ctx)
                error = None
            except ConditionError as e:
                matched, error = False, str(e)
            entry = {"rule_set": compiled.get("rule_set", rs.name), "rule": rid,
                     "priority": rule.get("priority", 100), "matched": matched}
            if error:
                entry["error"] = error
            if matched:
                applied = []
                for verdict in rule.get("then", []):
                    kind = next(iter(verdict))
                    body = verdict[kind] or {}
                    v = Verdict(kind=kind, rule_id=rid,
                                rule_set=compiled.get("rule_set", rs.name), body=body,
                                priority=rule.get("priority", 100))
                    if kind == "block":
                        decision.blocks.append(v)
                        decision.allowed = False
                    elif kind == "require":
                        decision.requirements.append(v)
                    elif kind == "escalate":
                        decision.escalations.append(v)
                    elif kind == "set":
                        decision.flags.update(body)
                    applied.append({"verdict": kind, **body})
                entry["applied"] = applied
            decision.evaluated.append(entry)
            if matched and rule.get("terminal"):
                break
    return decision


# Suffixes a `require` condition uses to say "this info has been gathered" — stripped to
# recover the underlying slot when a rule predates the explicit `slot` field, so an older
# instruction whose condition is "budget_collected" still gates on the captured `budget`.
_SLOT_CONDITION_SUFFIXES = ("_collected", "_provided", "_known", "_confirmed", "_captured",
                            "_given", "_present", "_supplied", "_set", "_done", "_received")

# A slot requirement stands aside after this many asks (anti-trap), so a caller who will not
# give the value still completes the action. A flag requirement (identity_verified) never waives.
WAIVE_REQUIREMENT_AFTER = 2


def require_slot(body: dict) -> str | None:
    """The information slot a `require` verdict gates on. Prefers the explicit ``slot`` field;
    falls back to deriving it from the ``condition`` name (``budget_collected`` → ``budget``)
    so an instruction compiled before the ``slot`` field existed still enforces against real
    captured data rather than a flag nothing sets."""
    slot = str(body.get("slot") or "").strip()
    if slot:
        return slot
    cond = str(body.get("condition") or "").strip().lower()
    for suf in _SLOT_CONDITION_SUFFIXES:
        if cond.endswith(suf) and len(cond) > len(suf):
            return cond[: -len(suf)]
    # No explicit slot and no "collected"-style suffix: this is a FLAG requirement
    # (identity_verified, supervisor_approved, …), NOT a captured-info slot. Returning None
    # keeps it on the flag-satisfaction path only — it is never treated as a waivable slot,
    # so an existing gate like identity can never be waived by the anti-trap counter.
    return None


def unmet_requirements(decision: RuleDecision, ctx: dict) -> list[Verdict]:
    """The requirements from ``decision`` that are NOT yet satisfied by ``ctx``.

    A ``require`` is satisfied when EITHER:
      * its named ``condition`` flag is truthy on session/customer — the original path, used by
        hand-authored rules with real flags (e.g. ``identity_verified``, ``supervisor_approved``); OR
      * it carries a ``slot`` and that piece of information has been captured — a non-empty value
        for that key in ``ctx['slots']`` (the accumulated conversation entities). This is what
        lets a custom instruction like "collect budget before offering a visit" gate on real,
        captured data instead of a flag nothing sets — and generalises to any "collect X first".
    """
    unmet = []
    slots = ctx.get("slots") or {}
    asks = ctx.get("require_asks") or {}
    for req in decision.requirements:
        name = req.body.get("condition")
        satisfied = bool(ctx.get("session", {}).get(name) or ctx.get("customer", {}).get(name))
        if not satisfied:
            slot = require_slot(req.body)
            if slot:
                val = slots.get(slot)
                satisfied = val is not None and str(val).strip() != ""
                # Anti-trap: a SLOT requirement stands aside once it has been asked enough
                # times, so a caller who will not provide the value is not stranded. Only slot
                # requirements waive — a flag gate like identity_verified never does.
                if not satisfied and int(asks.get(name, 0)) >= WAIVE_REQUIREMENT_AFTER:
                    satisfied = True
        if not satisfied:
            unmet.append(req)
    return unmet
