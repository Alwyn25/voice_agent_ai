"""Routing-rule grounding + validation (Phase 2).

A routing custom instruction OWNS a generated RuleSet and reuses the existing rule engine
end to end (spec §2.3 / §4). The compiler LLM emits only a single rule's ``when`` / ``then``;
this module:

  * wraps it into a full RuleSet applied at the ``turn`` point;
  * validates it through the SAME ``schema.compile_rules`` a hand-authored rule goes through
    (a rule that does not compile is never stored);
  * GROUNDS it — every ``field`` must be a real context field, every ``turn.intent`` value
    must be in the tenant's intent vocabulary, every ``escalate.target`` must be a known
    transfer target. An ungroundable rule returns problems, not a guess.

Nothing here is on the hot turn path; it runs at author time.
"""
from __future__ import annotations

import re
from typing import Any

import yaml

from app.config import get_settings
from app.engine.rules.schema import RuleCompileError, VERDICT_KINDS, compile_rules

# The context fields a routing condition may address (from app/engine/rules/context.py).
# proposal.params.* is open (tool-defined), so it is matched by prefix below.
ALLOWED_FIELDS: set[str] = {
    "context.event", "context.evaluation_point",
    "turn.intent", "turn.sentiment", "turn.stt_confidence", "turn.human_handoff",
    "customer.id", "customer.current_state", "customer.tags", "customer.preferred_language",
    "customer.region_code", "customer.priority_score", "customer.lead_source",
    "customer.on_dnc_list", "customer.on_dnc_marketing", "customer.has_contact_window",
    "now.local_time", "now.weekday", "now.date", "now.datetime", "now.is_weekend",
    "counters.attempts_today", "counters.consecutive_no_answer",
    "counters.failed_interactions", "counters.total_calls",
    "session.identity_verified", "session.channel", "session.direction",
    "session.purpose", "session.campaign_id", "session.turn_index",
    "proposal.tool",
}
_ALLOWED_PREFIXES = ("proposal.params.", "policy.")

_SLUG = re.compile(r"[^a-z0-9]+")


def rule_set_name(instruction_id) -> str:
    """A deterministic, unique RuleSet name owned by this instruction."""
    return f"ci_{str(instruction_id).replace('-', '')[:12]}"


def _field_ok(path: str) -> bool:
    return path in ALLOWED_FIELDS or any(path.startswith(p) for p in _ALLOWED_PREFIXES)


def _walk_conditions(node: Any):
    """Yield every leaf condition dict ({field, op, value}) in a when-tree."""
    if not isinstance(node, dict):
        return
    for comb in ("all", "any"):
        if comb in node and isinstance(node[comb], list):
            for child in node[comb]:
                yield from _walk_conditions(child)
    if "not" in node:
        yield from _walk_conditions(node["not"])
    if "field" in node:
        yield node


def ground(when: Any, then: Any, *, vocab_names: set[str],
           target_keys: set[str]) -> list[str]:
    """Grounding checks beyond structural validity. Returns a list of problems (empty = ok)."""
    problems: list[str] = []

    for leaf in _walk_conditions(when):
        field = str(leaf.get("field", ""))
        if not _field_ok(field):
            problems.append(f"condition references unknown field '{field}'")
            continue
        if field.endswith("turn.intent") or field == "turn.intent":
            values = leaf.get("value")
            values = values if isinstance(values, list) else [values]
            for v in values:
                # A $-reference to another field is not an intent literal — skip it.
                if isinstance(v, str) and not v.startswith("$") and v not in vocab_names:
                    problems.append(
                        f"intent '{v}' is not in the intent vocabulary — add it first so "
                        f"the planner can emit it")

    for verdict in (then or []):
        if not isinstance(verdict, dict) or len(verdict) != 1:
            continue
        kind = next(iter(verdict))
        body = verdict[kind] or {}
        if kind not in VERDICT_KINDS:
            problems.append(f"verdict '{kind}' is not whitelisted "
                            f"({'/'.join(VERDICT_KINDS)})")
        if kind == "escalate":
            target = body.get("target")
            if target and str(target) not in target_keys:
                problems.append(
                    f"escalate target '{target}' is not a known transfer target — add it "
                    f"to the transfer-target registry first")
    return problems


def _has_require(then: Any) -> bool:
    return any(isinstance(v, dict) and "require" in v for v in (then or []))


def build_and_validate(*, instruction_id, when: Any, then: Any, priority: int,
                       vocab_names: set[str], target_keys: set[str],
                       gate_tools: list[str] | None = None,
                       tool_names: set[str] | None = None
                       ) -> tuple[dict | None, str, list[str]]:
    """Wrap when/then into a RuleSet, structurally validate (compile_rules) AND ground it.

    Returns ``(compiled, yaml_source, problems)``. ``compiled`` is the stored JSONB shape
    (or None when there are problems). ``yaml_source`` is the human-readable rule for the
    review gate and the RuleSet row.

    ``gate_tools`` scopes a ``require`` rule to the tool(s) that perform the gated action, so
    the requirement is enforced at the tool gate — the same layer as built-in preconditions
    like identity/registration — rather than as a turn-level nudge. Escalate/block/set rules
    stay turn-level.
    """
    problems: list[str] = []
    if not isinstance(when, dict) or not when:
        problems.append("routing rule has no 'when' condition")
    if not isinstance(then, list) or not then:
        problems.append("routing rule has no 'then' verdicts")
    if problems:
        return None, "", problems

    # Clamp into the tenant band so a routing instruction can never be authored at a
    # priority that would let it precede a compliance rule (spec §7 / Q4).
    floor = get_settings().ci_routing_priority_min
    banded = max(int(priority or 100), floor)

    # Where the rule is evaluated. A `require` gates an ACTION, so it must run at that action's
    # tool point(s); everything else keys on the turn. A require with no valid gate tool cannot
    # be enforced, so that is a grounding gap rather than a silently inert rule.
    gate_tools = [t for t in (gate_tools or []) if not tool_names or t in tool_names]
    applies_to = ["turn"]
    rule_when = when
    if _has_require(then):
        if gate_tools:
            applies_to = [f"tool:{t}" for t in gate_tools]
            # At a tool gate turn.intent is the raw model label and may not equal the compiled
            # intent, so keying the require on intent could silently miss. The tool scope IS the
            # trigger ("before this action"), so gate on the tool itself — it fires whenever one
            # of the gated tools is attempted, which is exactly "before Y happens".
            rule_when = {"field": "proposal.tool", "op": "in", "value": gate_tools}
        else:
            problems.append(
                "this 'collect X before Y' instruction does not name a known action (tool) to "
                "gate — say which step it must happen before (e.g. before a site visit)")

    name = rule_set_name(instruction_id)
    rule = {"id": f"{name}_r1", "priority": banded,
            "when": rule_when, "then": then}
    ruleset = {"rule_set": name, "applies_to": applies_to,
               "description": "Generated from a custom instruction (routing).",
               "rules": [rule]}
    yaml_source = yaml.safe_dump(ruleset, sort_keys=False, allow_unicode=True)

    # Ground first (clearer messages), then the engine's own structural compile. Ground the
    # rule's ACTUAL when (tool-gated for a require, else the model's condition).
    problems += ground(rule_when, then, vocab_names=vocab_names, target_keys=target_keys)
    try:
        compiled = compile_rules(yaml_source)
    except RuleCompileError as e:
        problems += e.problems
        return None, yaml_source, problems

    if problems:
        return None, yaml_source, problems
    return compiled, yaml_source, []


def validate_intent_bridge(proposed_intents: list[dict]) -> list[str]:
    """The producer/consumer contract check (spec §4.3), run deterministically with no LLM.

    A routing rule keys on ``turn.intent == <name>``, but the planner never emits ``<name>``
    directly — it free-labels, and ``normalize_intent`` maps that label onto the canonical one
    via the intent's aliases. If the aliases do not cover the labels the planner will actually
    produce, the rule compiles perfectly and then NEVER fires. This asserts every proposed
    intent's own ``example_labels`` normalise back to it under its proposed aliases, so that
    silent failure is caught at author time instead of in production.
    """
    from app.engine.planner.confidence import normalize_intent
    notes: list[str] = []
    for it in (proposed_intents or []):
        name = it.get("name")
        aliases = it.get("aliases") or []
        if not name:
            continue
        amap = {name: list(aliases)}
        for label in (it.get("example_labels") or []):
            if normalize_intent(label, amap) != name:
                notes.append(
                    f"the agent may not recognise phrasing like '{label}' as '{name}' — "
                    f"broaden the aliases so it always routes")
    return notes


def _summarise_then(then: Any) -> list[str]:
    out = []
    for v in (then or []):
        if not isinstance(v, dict) or len(v) != 1:
            continue
        kind = next(iter(v))
        body = v[kind] or {}
        if kind == "escalate":
            tgt = f" to {body['target']}" if body.get("target") else ""
            out.append(f"escalate{tgt} ({body.get('reason', '')})")
        elif kind == "require":
            out.append(f"require '{body.get('condition')}' ({body.get('reason', '')})")
        elif kind == "block":
            out.append(f"block ({body.get('reason', '')})")
        elif kind == "set":
            out.append("set " + ", ".join(f"{k}={val}" for k, val in body.items()))
    return out


def _satisfying_ctx(when: Any) -> dict:
    """Build the smallest context that makes ``when`` true for eq/in leaf conditions — enough
    to deterministically simulate the common routing rule (turn.intent == X). Ops we cannot
    trivially satisfy (gt, matches, between, not) are left unset, so the rule simply may not
    fire and the simulation says so honestly."""
    ctx: dict = {}
    for leaf in _walk_conditions(when):
        field = str(leaf.get("field", ""))
        op, value = leaf.get("op"), leaf.get("value")
        if op == "in" and isinstance(value, list) and value:
            value = value[0]
        elif op not in ("eq", "contains"):
            continue
        parts = field.split(".")
        cur = ctx
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = [value] if op == "contains" else value
    return ctx


def simulate(compiled: dict | None) -> dict | None:
    """Deterministically 'run' the compiled routing rule against a synthesised trigger, so
    the review gate can show the outcome ("when this intent is detected, the agent escalates
    to finance_desk"). No LLM, no DB — reuses the engine's own condition evaluator."""
    if not compiled or not compiled.get("rules"):
        return None
    from app.engine.rules.conditions import ConditionError, evaluate
    rule = compiled["rules"][0]
    when, then = rule.get("when"), rule.get("then")
    ctx = _satisfying_ctx(when)
    try:
        fired = bool(evaluate(when, ctx))
    except ConditionError:
        fired = False
    return {"trigger_context": ctx, "fires": fired,
            "actions": _summarise_then(then) if fired else []}
