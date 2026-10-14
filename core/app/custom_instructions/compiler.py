"""Author-time compiler: plain English -> classification + compiled artifact.

ONE structured-output LLM call does everything (spec §2.2 / §2.3):

  * classifies by CONSEQUENCE, not phrasing — advisory vs routing vs ambiguous;
  * for advisory, emits a clean imperative snippet ready to inject;
  * for routing, emits a single rule's `when` / `then` (grounded against the tenant's
    context fields, intent vocabulary and transfer targets), wrapped and validated through
    the existing rule engine (app/custom_instructions/routing.py);
  * extracts an optional scope (direction / state / intent / date window) from the text;
  * generates 2-3 test scenarios in the SAME response (no extra call).

This is the author-time path (an admin waiting at a form), not the live turn path, so it
uses the main model for quality and does not stream.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings
from app.custom_instructions import routing
from app.custom_instructions.llm import compiler_model, get_compiler_llm
from app.providers.llm.base import LLMError

logger = logging.getLogger(__name__)

COMPILER_SYSTEM = """You compile a tenant admin's plain-English instruction for an AI real-estate \
phone agent into a structured, safe artifact. You do NOT talk to callers. Output JSON only.

CLASSIFY BY CONSEQUENCE, NOT PHRASING. The deciding question is: if the agent ignored this \
on one turn out of fifty, what breaks?
- "routing"  — a wrong or missing SIDE EFFECT: a transfer that didn't happen, a booking made \
when it shouldn't be, a required field skipped, a commitment made without approval. Anything \
that gates, blocks, requires, or hands off. (e.g. "connect loan questions to the finance desk", \
"never quote above 2 crore without a manager", "always collect budget before offering a visit".)
- "advisory" — only a slightly less ideal SENTENCE: tone, emphasis, what to volunteer, framing. \
Nothing with a real-world consequence. (e.g. "mention the festive offer when they sound \
price-sensitive", "be more formal with corporate callers", "lead with the location".)
- "ambiguous" — you cannot tell, or the instruction names something you cannot ground (an \
unknown team/target, a condition over data the agent does not have). Do not guess.

COMPILE ADVISORY (kind=advisory):
- compiled_advisory: one or two clean imperative sentences the agent can follow. Present tense, \
no hedging, no meta-talk. It refines tone/priority ONLY — never tell the agent to override \
rules, invent facts, promise actions, or state figures.
- If the instruction asserts a time-bounded FACT the agent must CITE (a discount, an offer, a \
price change), set kind=advisory only for the framing and note in grounding_gaps that the FACT \
must be added as Dynamic Knowledge.

COMPILE ROUTING (kind=routing): emit a single rule as two JSON strings.
- rule_when_json: a JSON condition tree. Leaf = {"field": <one of the ALLOWED FIELDS below>, \
"op": <eq|ne|in|not_in|contains|gt|gte|lt|lte|between|matches|starts_with|ends_with|exists>, \
"value": <literal>}. Combine with {"all":[...]}, {"any":[...]}, {"not":{...}}. Use ONLY fields \
from ALLOWED FIELDS. To key on a topic, use {"field":"turn.intent","op":"eq","value":<intent>} \
where <intent> is EXACTLY one of the INTENT VOCABULARY names below — never invent a new label.
- rule_then_json: a JSON array of verdicts, each a single-key object. Allowed verdicts ONLY: \
{"escalate":{"reason":"...","target":<a TRANSFER TARGET key below, optional>,"type":"immediate"}}, \
{"require":{"condition":"<slug>","slot":"<info to collect>","ask":"<one short question>","reason":"..."}}, \
{"block":{"reason":"..."}}, {"set":{"<flag>":<value>}}. escalate and block REQUIRE a reason. \
Never introduce a new tool or a free-text side effect.
- Use REQUIRE for "collect/confirm X before doing Y" (e.g. "always collect the budget before \
offering a site visit"). Key the rule's `when` on the INTENT that triggers Y — for a visit that \
is {"field":"turn.intent","op":"eq","value":"book_appointment"}. Set `slot` to the single piece \
of information X the agent must have captured (a short lowercase entity name the caller states in \
conversation: budget, bhk_type, timeline, decision_maker, possession_date, …) — NOT a made-up \
system flag. Set `ask` to the exact question the agent should speak to collect it. The agent will \
ask (at most twice) until the caller provides X, then proceed. Do not use require for tone.
- For a REQUIRE, ALSO set the top-level `gate_tools` to the TOOL name(s) from the TOOLS list \
that perform action Y — this is what makes the requirement gate the action itself (e.g. a site \
visit is offered/booked by search_slots and book_appointment, so gate_tools = those). Pick ONLY \
tools that exist in the TOOLS list; if none fits, leave gate_tools empty and put the gap in \
grounding_gaps. gate_tools is empty for advisory, escalate, block and set.
- rule_priority: an integer (lower = evaluated first); default 100.

SELF-PROVISIONING (the admin is non-technical and does not know these registries exist — do \
NOT dead-end them). When the instruction needs an intent or a target that is not registered \
yet, still classify it kind=routing, still emit a compilable rule keyed on a NEW name you \
choose, and DESCRIBE the missing piece so the system can create it during the admin's review:
- Missing intent: pick a canonical lowercase snake_case name (e.g. loan_enquiry), key the rule \
on it, and add an entry to proposed_intents: {"name", "description", "aliases": the surface \
words a caller actually says (e.g. loan, home loan, emi, financing), "example_labels": 2-4 \
short snake_case labels the planner is LIKELY to emit for this topic (e.g. loan, loan_question, \
financing), "examples": caller lines}. The aliases MUST cover every example_label, or the rule \
will never fire.
- Missing target: pick a stable key (e.g. finance_desk), key the escalate verdict on it, and \
add an entry to proposed_targets: {"key", "display_name", "destination_kind": one of \
phone|sip|queue|user}. Never invent a phone number — the admin supplies it at review.
- Use kind=ambiguous ONLY when you cannot ground even by proposing: a condition over data the \
agent does not have (no matching ALLOWED FIELD), a side effect outside escalate/require/block/ \
set, or a genuinely unclear instruction. Put the blocker in grounding_gaps. For advisory, all \
rule_* fields are null and proposed_intents/proposed_targets are empty.

SCOPE (advisory mainly): extract only what the text clearly implies; leave the rest null.
- direction "inbound"|"outbound"|null; state a lifecycle state name or null; intent a short \
lowercase topic label or null; valid_from/valid_until ISO dates (YYYY-MM-DD) or null.

TEST SCENARIOS: 2-3 short realistic caller lines with the behaviour a correct agent would show.
GROUNDING GAPS: anything the admin must clarify before activation. Empty list if none.
routing_note: when kind=routing, one sentence naming the gate/handoff. null otherwise."""

COMPILER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "reasoning", "consequence_if_missed", "compiled_advisory",
                 "rule_when_json", "rule_then_json", "rule_priority", "gate_tools",
                 "proposed_intents", "proposed_targets",
                 "suggested_scope", "grounding_gaps", "test_scenarios", "routing_note"],
    "properties": {
        "kind": {"type": "string", "enum": ["advisory", "routing", "ambiguous"]},
        "reasoning": {"type": "string"},
        "consequence_if_missed": {"type": "string"},
        "compiled_advisory": {"type": ["string", "null"]},
        "rule_when_json": {"type": ["string", "null"]},
        "rule_then_json": {"type": ["string", "null"]},
        "rule_priority": {"type": ["integer", "null"]},
        "gate_tools": {"type": "array", "items": {"type": "string"}},
        "proposed_intents": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "description", "aliases", "example_labels", "examples"],
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "example_labels": {"type": "array", "items": {"type": "string"}},
                    "examples": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "proposed_targets": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["key", "display_name", "destination_kind"],
                "properties": {
                    "key": {"type": "string"},
                    "display_name": {"type": "string"},
                    "destination_kind": {
                        "type": ["string", "null"],
                        "enum": ["phone", "sip", "queue", "user", None]},
                },
            },
        },
        "suggested_scope": {
            "type": "object",
            "additionalProperties": False,
            "required": ["direction", "state", "intent", "valid_from", "valid_until"],
            "properties": {
                "direction": {"type": ["string", "null"], "enum": ["inbound", "outbound", None]},
                "state": {"type": ["string", "null"]},
                "intent": {"type": ["string", "null"]},
                "valid_from": {"type": ["string", "null"]},
                "valid_until": {"type": ["string", "null"]},
            },
        },
        "grounding_gaps": {"type": "array", "items": {"type": "string"}},
        "test_scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["caller_says", "expected_behaviour"],
                "properties": {
                    "caller_says": {"type": "string"},
                    "expected_behaviour": {"type": "string"},
                },
            },
        },
        "routing_note": {"type": ["string", "null"]},
    },
}


@dataclass
class GroundingContext:
    """The artifacts the compiler grounds a routing rule against (spec §2.2)."""
    vocab_names: set[str] = field(default_factory=set)
    target_keys: set[str] = field(default_factory=set)
    tool_names: set[str] = field(default_factory=set)
    vocab_block: str = ""
    targets_block: str = ""
    tools_block: str = ""

    def prompt_suffix(self) -> str:
        fields = "\n".join(f"- {f}" for f in sorted(routing.ALLOWED_FIELDS)) + \
            "\n- proposal.params.<name> (tool parameters)"
        targets = self.targets_block or "(none registered — do not use escalate.target)"
        vocab = self.vocab_block or "(none)"
        tools = self.tools_block or "(none — do not use gate_tools)"
        return (f"\n\n## ALLOWED FIELDS\n{fields}"
                f"\n\n## INTENT VOCABULARY (use these exact names for turn.intent)\n{vocab}"
                f"\n\n## TRANSFER TARGETS (use these exact keys for escalate.target)\n{targets}"
                f"\n\n## TOOLS (use these exact names for a require's gate_tools — the action Y "
                f"a 'collect X before Y' instruction gates)\n{tools}")


@dataclass
class CompileResult:
    kind: str                                  # advisory | routing | ambiguous
    reasoning: str
    consequence_if_missed: str
    compiled_prompt: str | None                # advisory snippet, else None
    scope: dict
    grounding_gaps: list[str] = field(default_factory=list)
    test_scenarios: list[dict] = field(default_factory=list)
    routing_note: str | None = None
    # routing artifacts (None/empty for advisory)
    rule_when: dict | None = None
    rule_then: list | None = None
    rule_priority: int = 100
    gate_tools: list[str] = field(default_factory=list)  # tools a `require` gates (applies_to)
    compiled_ruleset: dict | None = None       # validated JSONB shape, None if not groundable
    rule_yaml: str | None = None               # human-readable rule for the review gate
    dry_run: dict | None = None                # deterministic simulation of the compiled rule
    # Self-provisioning (spec §4.3/§4.4): the intents/targets this routing rule needs that do
    # not exist yet. The review gate creates them atomically on activation. A target entry
    # carries needs_destination=True when the admin must supply the actual number/SIP.
    provisioning_plan: dict = field(default_factory=lambda: {"intents": [], "targets": []})
    bridge_notes: list[str] = field(default_factory=list)  # alias-coverage warnings (deterministic)
    model: str = ""
    tokens: dict = field(default_factory=dict)

    @property
    def needs_destination(self) -> bool:
        """True when activation is blocked pending a human-supplied transfer destination."""
        return any(t.get("needs_destination")
                   for t in self.provisioning_plan.get("targets", []))

    @property
    def provisioning_required(self) -> bool:
        plan = self.provisioning_plan
        return bool(plan.get("intents") or plan.get("targets"))

    @property
    def activatable(self) -> bool:
        """Ready to activate with NO further admin input. A routing rule that only needs a new
        intent is activatable (the intent is auto-provisioned); one that needs a brand-new
        transfer destination is not, until the admin types the number at the review gate."""
        if self.grounding_gaps:
            return False
        if self.kind == "advisory":
            return bool(self.compiled_prompt)
        if self.kind == "routing":
            return self.compiled_ruleset is not None and not self.needs_destination
        return False

    def as_preview(self) -> dict:
        return {"kind": self.kind, "reasoning": self.reasoning,
                "consequence_if_missed": self.consequence_if_missed,
                "compiled_prompt": self.compiled_prompt, "scope": self.scope,
                "grounding_gaps": self.grounding_gaps,
                "test_scenarios": self.test_scenarios, "routing_note": self.routing_note,
                "rule_when": self.rule_when, "rule_then": self.rule_then,
                "rule_priority": self.rule_priority, "gate_tools": self.gate_tools,
                "rule_yaml": self.rule_yaml,
                "dry_run": self.dry_run,
                "provisioning_plan": self.provisioning_plan,
                "provisioning_required": self.provisioning_required,
                "needs_destination": self.needs_destination,
                "bridge_notes": self.bridge_notes,
                "model": self.model, "activatable": self.activatable}


def _clean_scope(raw: dict | None) -> dict:
    raw = raw or {}
    out: dict[str, Any] = {}
    for key in ("direction", "state", "intent", "valid_from", "valid_until"):
        val = raw.get(key)
        if isinstance(val, str):
            val = val.strip() or None
        if val:
            out[key] = val
    return out


def _parse_json(value) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


# Some OpenAI-compatible providers (notably Groq's gpt-oss models) intermittently fail
# strict structured-output generation on a larger schema, returning a 400 "Failed to
# generate JSON". The compile step is author-time and low-volume, so a couple of quick
# retries turn a flaky provider into a reliable authoring experience at negligible cost.
_COMPILE_MAX_ATTEMPTS = 3


async def _complete_with_retry(*, system: str, user: str, model: str):
    last: LLMError | None = None
    max_tokens = get_settings().ci_compile_max_tokens
    for attempt in range(_COMPILE_MAX_ATTEMPTS):
        try:
            return await get_compiler_llm().complete_json(
                system=system, user=user, model=model, temperature=0.1, max_tokens=max_tokens,
                schema=COMPILER_SCHEMA, schema_name="custom_instruction_compile")
        except LLMError as e:
            last = e
            logger.warning("compile attempt %d/%d failed: %s",
                           attempt + 1, _COMPILE_MAX_ATTEMPTS, e)
    raise last if last else LLMError("compile failed")


async def compile_instruction(raw_text: str, *,
                              grounding: GroundingContext | None = None,
                              model: str | None = None) -> CompileResult:
    """Run the single classify+compile call. Raises LLMError on model failure.

    ``grounding`` supplies the intent vocabulary + transfer targets a routing rule is
    grounded against; without it a routing classification cannot be validated and is
    returned as a grounding gap.
    """
    text = (raw_text or "").strip()
    if not text:
        raise LLMError("empty instruction text")

    system = COMPILER_SYSTEM + (grounding.prompt_suffix() if grounding else "")
    result = await _complete_with_retry(
        system=system, user=f"Instruction to compile:\n\n{text}",
        model=model or compiler_model())

    data = result.data or {}
    kind = data.get("kind")
    kind = kind if kind in ("advisory", "routing", "ambiguous") else "ambiguous"

    compiled_prompt = data.get("compiled_advisory")
    compiled_prompt = (compiled_prompt.strip()
                       if isinstance(compiled_prompt, str) and compiled_prompt.strip() else None)
    if kind != "advisory":
        compiled_prompt = None

    gaps = [str(g).strip() for g in (data.get("grounding_gaps") or []) if str(g).strip()]
    scenarios = [
        {"caller_says": str(s.get("caller_says", "")).strip(),
         "expected_behaviour": str(s.get("expected_behaviour", "")).strip()}
        for s in (data.get("test_scenarios") or []) if isinstance(s, dict)
    ]

    res = CompileResult(
        kind=kind, reasoning=str(data.get("reasoning") or "").strip(),
        consequence_if_missed=str(data.get("consequence_if_missed") or "").strip(),
        compiled_prompt=compiled_prompt, scope=_clean_scope(data.get("suggested_scope")),
        grounding_gaps=gaps, test_scenarios=scenarios,
        routing_note=(str(data.get("routing_note")).strip() if data.get("routing_note") else None),
        rule_priority=int(data.get("rule_priority") or 100),
        model=result.model, tokens=result.tokens or {})

    if kind == "routing":
        _compile_routing(res, data, grounding)
    return res


def _norm_key(raw: str) -> str:
    """Canonicalise a proposed intent name / target key the way the admin API does, so a name
    the compiler proposes matches the row that gets created (idempotent by this form)."""
    import re
    return re.sub(r"[^a-z0-9]+", "_", (raw or "").strip().lower()).strip("_")


def _clean_proposed_intents(raw: Any) -> list[dict]:
    out = []
    for it in (raw or []):
        if not isinstance(it, dict):
            continue
        name = _norm_key(str(it.get("name", "")))
        if not name:
            continue
        out.append({
            "name": name,
            "description": str(it.get("description") or "").strip(),
            "aliases": sorted({str(a).strip().lower()
                               for a in (it.get("aliases") or []) if str(a).strip()}),
            "example_labels": [str(x).strip() for x in (it.get("example_labels") or [])
                               if str(x).strip()],
            "examples": [str(x).strip() for x in (it.get("examples") or []) if str(x).strip()],
        })
    return out


def _clean_proposed_targets(raw: Any) -> list[dict]:
    out = []
    for tg in (raw or []):
        if not isinstance(tg, dict):
            continue
        key = _norm_key(str(tg.get("key", "")))
        if not key:
            continue
        kind = tg.get("destination_kind")
        out.append({
            "key": key,
            "display_name": str(tg.get("display_name") or key).strip(),
            "destination_kind": kind if kind in ("phone", "sip", "queue", "user") else "phone",
        })
    return out


def _compile_routing(res: CompileResult, data: dict,
                     grounding: GroundingContext | None) -> None:
    """Parse, wrap and validate the routing rule onto ``res`` (mutates it).

    Missing intents/targets are not dead-ends: the model proposes them, we validate the rule
    against the AUGMENTED grounding (existing ∪ proposed) so it compiles now, and record a
    provisioning plan the review gate creates atomically on activation."""
    when = _parse_json(data.get("rule_when_json"))
    then = _parse_json(data.get("rule_then_json"))
    res.rule_when, res.rule_then = when, then
    proposed_intents = _clean_proposed_intents(data.get("proposed_intents"))
    proposed_targets = _clean_proposed_targets(data.get("proposed_targets"))

    if grounding is None:
        res.grounding_gaps.append(
            "routing rules need the tenant grounding context to validate — none was provided")
        return
    if when is None or then is None:
        # No compilable rule emitted — the model should have populated grounding_gaps; add a
        # generic one if it did not, so the row is never activatable without a rule.
        if not res.grounding_gaps:
            res.grounding_gaps.append(
                "could not compile a routing rule from this instruction — clarify the "
                "condition and the action")
        return

    # Validate against the SAME rule engine a hand-authored rule uses, but with the proposed
    # intents/targets folded into the vocabulary/target set so a rule that keys on a
    # to-be-created intent still compiles. A temp id is fine for preview; the owned RuleSet is
    # rebuilt with the real instruction id at activation, re-validated against LIVE grounding.
    aug_vocab = set(grounding.vocab_names) | {p["name"] for p in proposed_intents}
    aug_targets = set(grounding.target_keys) | {p["key"] for p in proposed_targets}
    gate_tools = [_norm_key(str(t)) for t in (data.get("gate_tools") or []) if str(t).strip()]
    res.gate_tools = [t for t in gate_tools if t in grounding.tool_names]
    compiled, yaml_source, problems = routing.build_and_validate(
        instruction_id=uuid.uuid4(), when=when, then=then, priority=res.rule_priority,
        vocab_names=aug_vocab, target_keys=aug_targets,
        gate_tools=gate_tools, tool_names=grounding.tool_names)
    res.rule_yaml = yaml_source or None
    if problems:
        res.grounding_gaps.extend(problems)
        return

    res.compiled_ruleset = compiled
    # Deterministically simulate the rule so the review gate shows how the agent behaves
    # ("when this intent is detected → escalate to finance_desk"). No LLM, no DB.
    res.dry_run = routing.simulate(compiled)

    # Only propose what does not already exist (the model may re-propose a base intent).
    plan_intents = [p for p in proposed_intents if p["name"] not in grounding.vocab_names]
    plan_targets = [dict(p, needs_destination=(p["key"] not in grounding.target_keys))
                    for p in proposed_targets if p["key"] not in grounding.target_keys]
    res.provisioning_plan = {"intents": plan_intents, "targets": plan_targets}

    # Robustness gate: prove (deterministically, no LLM) that each proposed intent's aliases
    # actually normalise the labels the planner would emit — catching the "rule compiled but
    # never fires" silent failure before it can ship.
    res.bridge_notes = routing.validate_intent_bridge(plan_intents)
