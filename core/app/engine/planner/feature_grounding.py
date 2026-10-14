"""Pure helpers for the feature-grounding check (the semantic judge lives in grounding_check).

Two small, dependency-free pieces the LLM verifier is built on and degrades to:

  - ``grounding_text`` flattens the retrieved passages + this turn's tool result into the one
    block the verifier reasons over (and the string the fallback searches).
  - ``cited_verbatim`` is the DEGRADED-MODE check only: if the semantic verifier call itself
    fails, a reply that quoted the source word-for-word is still demonstrably grounded, so it
    is allowed through rather than punished for an infrastructure error. It is NOT the primary
    decision — meaning is (grounding_check) — because verbatim presence cannot tell whether the
    quoted words actually support the CLAIM ("west facing" is present, but does not support a
    "park-facing" flat). That judgement is the model's job.

Stdlib only, so it unit-tests without a database — ``python -m app.engine.planner.feature_grounding``.
"""
from __future__ import annotations

import json
import re

_WS = re.compile(r"\s+")


def _norm(text: str | None) -> str:
    """Lowercase and collapse whitespace, so a citation that spans a line break still matches."""
    return _WS.sub(" ", (text or "")).strip().lower()


def grounding_text(knowledge: list[dict] | None, tool_output: dict | None,
                   project_facts: list[dict] | None = None) -> str:
    """Everything the reply was allowed to draw on, flattened to one normalised string: the
    retrieved passages, this turn's tool result, and the curated project facts.

    ``project_facts`` is included because the responder may now ground an answer on a curated
    fact ("RCC framed structure, not brick"); the degraded-mode verbatim check must see the
    same material the reply was allowed to quote, or it would reject a legitimate citation."""
    parts: list[str] = []
    for k in (knowledge or []):
        content = k.get("content") if isinstance(k, dict) else None
        if content:
            parts.append(str(content))
    for f in (project_facts or []):
        if isinstance(f, dict) and f.get("value"):
            parts.append(f"{f.get('label') or ''} {f['value']}")
    if tool_output:
        parts.append(json.dumps(tool_output, default=str))
    return _norm(" ".join(parts))


def cited_verbatim(evidence: str | None, grounding: str) -> bool:
    """True when the model's cited phrase is present in the material word-for-word (substring on
    normalised text). Degraded-mode only — see the module docstring."""
    e = _norm(evidence)
    return bool(e) and e in grounding


# ---------------------------------------------------------------------------------------
# Built-in self-test — runs without pytest.
#   python -m app.engine.planner.feature_grounding
# ---------------------------------------------------------------------------------------
def _selftest() -> int:
    failures: list[str] = []

    def check(name, cond):
        if not cond:
            failures.append(name)

    KB = [
        {"content": "## Unit 1 — 1,239 sq ft, West facing."},
        {"content": "Infrastructure: STP, WTP and 100 percent power backup."},
    ]
    g = grounding_text(KB, None)
    check("grounding_lowercased", "west facing" in g and "power backup" in g)
    check("grounding_collapses_ws", "\n" not in g)

    # verbatim degraded-mode check
    check("verbatim_present", cited_verbatim("west facing", g))
    check("verbatim_present_multiword", cited_verbatim("power  backup", g))  # ws-normalised
    check("verbatim_absent", not cited_verbatim("park facing", g))
    check("verbatim_none", not cited_verbatim(None, g))
    check("verbatim_empty", not cited_verbatim("", g))

    # tool-output grounding
    gt = grounding_text(None, {"configs": ["2 BHK", "3 BHK"]})
    check("tool_output_included", cited_verbatim("2 bhk", gt))
    check("empty_grounding", grounding_text(None, None) == "")

    if failures:
        print(f"feature_grounding self-test: {len(failures)} FAILED -> {failures}")
        return 1
    print("feature_grounding self-test: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
