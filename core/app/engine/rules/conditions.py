"""Condition grammar for rules: all / any / not combinators over leaf operators.

A leaf condition is:  {field: "customer.tags", op: "contains", value: "vip"}
Combinators:          {all: [...]}, {any: [...]}, {not: {...}}

Field paths address the frozen RuleContext namespace (see context.py).
Evaluation is pure and deterministic — no LLM, no side effects.
"""
import re
from datetime import time
from typing import Any

OPERATORS = ["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "contains",
             "not_contains", "matches", "between", "exists", "starts_with", "ends_with"]

COMBINATORS = ["all", "any", "not"]


class ConditionError(Exception):
    pass


def get_field(ctx: dict, path: str) -> Any:
    cur: Any = ctx
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _as_time(v: Any) -> time | None:
    if isinstance(v, time):
        return v
    if isinstance(v, str) and re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", v):
        parts = [int(p) for p in v.split(":")]
        return time(parts[0], parts[1], parts[2] if len(parts) > 2 else 0)
    return None


def _compare(a: Any, b: Any) -> tuple[Any, Any]:
    """Coerce a pair for comparison: times as times, numbers as floats."""
    ta, tb = _as_time(a), _as_time(b)
    if ta is not None and tb is not None:
        return ta, tb
    if isinstance(a, (int, float)) and isinstance(b, str):
        try:
            return float(a), float(b)
        except ValueError:
            return a, b
    if isinstance(b, (int, float)) and isinstance(a, str):
        try:
            return float(a), float(b)
        except ValueError:
            return a, b
    return a, b


def resolve_value(v: Any, ctx: dict) -> Any:
    """Values may reference the context: "$customer.preferred_contact_start"."""
    if isinstance(v, str) and v.startswith("$"):
        return get_field(ctx, v[1:])
    if isinstance(v, list):
        return [resolve_value(x, ctx) for x in v]
    return v


def eval_leaf(cond: dict, ctx: dict) -> bool:
    field = cond.get("field")
    op = cond.get("op", "eq")
    expected = resolve_value(cond.get("value"), ctx)
    if field is None:
        raise ConditionError(f"Leaf condition missing 'field': {cond}")
    if op not in OPERATORS:
        raise ConditionError(f"Unknown operator '{op}'")

    actual = get_field(ctx, field)

    if op == "exists":
        return (actual is not None) == bool(expected if expected is not None else True)
    if actual is None:
        # A missing value only satisfies negative operators
        return op in ("ne", "not_in", "not_contains")

    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op in ("gt", "gte", "lt", "lte"):
        a, b = _compare(actual, expected)
        try:
            return {"gt": a > b, "gte": a >= b, "lt": a < b, "lte": a <= b}[op]
        except TypeError:
            return False
    if op == "in":
        return actual in (expected or [])
    if op == "not_in":
        return actual not in (expected or [])
    if op == "contains":
        if isinstance(actual, (list, tuple, set)):
            return expected in actual
        return str(expected) in str(actual)
    if op == "not_contains":
        if isinstance(actual, (list, tuple, set)):
            return expected not in actual
        return str(expected) not in str(actual)
    if op == "matches":
        return re.search(str(expected), str(actual)) is not None
    if op == "starts_with":
        return str(actual).startswith(str(expected))
    if op == "ends_with":
        return str(actual).endswith(str(expected))
    if op == "between":
        if not isinstance(expected, (list, tuple)) or len(expected) != 2:
            raise ConditionError("'between' expects a two-item list")
        lo, hi = expected
        a, l = _compare(actual, lo)
        _, h = _compare(actual, hi)
        try:
            if l <= h:
                return l <= a <= h
            return a >= l or a <= h          # window crossing midnight
        except TypeError:
            return False
    return False


def evaluate(cond: Any, ctx: dict) -> bool:
    """Recursive evaluation of a condition tree."""
    if cond is None:
        return True
    if not isinstance(cond, dict):
        raise ConditionError(f"Condition must be a mapping, got {type(cond).__name__}")
    if "all" in cond:
        return all(evaluate(c, ctx) for c in cond["all"])
    if "any" in cond:
        return any(evaluate(c, ctx) for c in cond["any"])
    if "not" in cond:
        return not evaluate(cond["not"], ctx)
    return eval_leaf(cond, ctx)


def validate_condition(cond: Any, path: str = "when") -> list[str]:
    """Static validation at compile time. Returns list of problems."""
    problems: list[str] = []
    if cond is None:
        return problems
    if not isinstance(cond, dict):
        return [f"{path}: must be a mapping"]
    keys = set(cond)
    combo = keys & set(COMBINATORS)
    if combo:
        if len(combo) > 1:
            problems.append(f"{path}: use only one of all/any/not")
        key = combo.pop()
        child = cond[key]
        if key == "not":
            problems += validate_condition(child, f"{path}.not")
        else:
            if not isinstance(child, list):
                problems.append(f"{path}.{key}: must be a list")
            else:
                for i, c in enumerate(child):
                    problems += validate_condition(c, f"{path}.{key}[{i}]")
        return problems
    if "field" not in cond:
        problems.append(f"{path}: leaf condition needs 'field'")
    op = cond.get("op", "eq")
    if op not in OPERATORS:
        problems.append(f"{path}: unknown operator '{op}' (valid: {', '.join(OPERATORS)})")
    if op == "between" and not isinstance(cond.get("value"), list):
        problems.append(f"{path}: 'between' needs a two-item list value")
    return problems
