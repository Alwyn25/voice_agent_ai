"""Rule set YAML parsing, validation and compilation.

Authored form (YAML):

    rule_set: calling_policy
    applies_to: [outbound_call, "tool:*"]
    rules:
      - id: dnc_absolute
        priority: 0
        terminal: true
        when:
          any:
            - {field: customer.on_dnc_list, op: eq, value: true}
        then:
          - block: {scope: all, reason: "Customer is on Do-Not-Contact list"}

Compiled form is the same structure, validated and priority-sorted, stored as JSONB.
"""
import yaml

from app.engine.rules.conditions import validate_condition

VERDICT_KINDS = ["block", "require", "escalate", "set"]
EVAL_POINTS_STATIC = ["outbound_call", "turn", "conversation_start"]


class RuleCompileError(Exception):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def parse_yaml(source: str) -> dict:
    try:
        data = yaml.safe_load(source)
    except yaml.YAMLError as e:
        raise RuleCompileError([f"YAML parse error: {e}"])
    if not isinstance(data, dict):
        raise RuleCompileError(["Rule set must be a YAML mapping"])
    return data


def validate(data: dict) -> list[str]:
    problems: list[str] = []
    if not data.get("rule_set"):
        problems.append("missing 'rule_set' name")
    applies = data.get("applies_to")
    if not applies or not isinstance(applies, list):
        problems.append("'applies_to' must be a non-empty list")
    else:
        for a in applies:
            if not isinstance(a, str):
                problems.append(f"applies_to entry must be a string: {a}")
            elif not (a in EVAL_POINTS_STATIC or a.startswith("tool:")):
                problems.append(f"unknown evaluation point '{a}' "
                                f"(valid: {', '.join(EVAL_POINTS_STATIC)}, tool:<name>, tool:*)")
    rules = data.get("rules")
    if not isinstance(rules, list) or not rules:
        problems.append("'rules' must be a non-empty list")
        return problems

    seen_ids = set()
    for i, r in enumerate(rules):
        where = f"rules[{i}]"
        if not isinstance(r, dict):
            problems.append(f"{where}: must be a mapping")
            continue
        rid = r.get("id")
        if not rid:
            problems.append(f"{where}: missing 'id'")
        elif rid in seen_ids:
            problems.append(f"{where}: duplicate rule id '{rid}'")
        else:
            seen_ids.add(rid)
        if "priority" in r and not isinstance(r["priority"], int):
            problems.append(f"{where}: 'priority' must be an integer")
        problems += validate_condition(r.get("when"), f"{where}.when")
        then = r.get("then")
        if not isinstance(then, list) or not then:
            problems.append(f"{where}: 'then' must be a non-empty list of verdicts")
            continue
        for j, v in enumerate(then):
            if not isinstance(v, dict) or len(v) != 1:
                problems.append(f"{where}.then[{j}]: each verdict is a single-key mapping "
                                f"({'/'.join(VERDICT_KINDS)})")
                continue
            kind = next(iter(v))
            if kind not in VERDICT_KINDS:
                problems.append(f"{where}.then[{j}]: unknown verdict '{kind}'")
            body = v[kind] or {}
            if kind in ("block", "escalate") and not body.get("reason"):
                problems.append(f"{where}.then[{j}]: '{kind}' requires a 'reason'")
            if kind == "require" and not body.get("condition"):
                problems.append(f"{where}.then[{j}]: 'require' needs a 'condition'")
    return problems


def compile_rules(source: str) -> dict:
    """Parse + validate + normalise. Raises RuleCompileError with all problems."""
    data = parse_yaml(source)
    problems = validate(data)
    if problems:
        raise RuleCompileError(problems)
    rules = sorted(data["rules"], key=lambda r: r.get("priority", 100))
    for r in rules:
        r.setdefault("priority", 100)
        r.setdefault("terminal", False)
    return {"rule_set": data["rule_set"], "applies_to": data["applies_to"],
            "description": data.get("description", ""), "rules": rules}


def applies(compiled: dict, evaluation_point: str) -> bool:
    for a in compiled.get("applies_to", []):
        if a == evaluation_point:
            return True
        if a == "tool:*" and evaluation_point.startswith("tool:"):
            return True
    return False
