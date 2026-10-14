#!/usr/bin/env python3
"""Tool-selection accuracy, scored with RAGAS's ToolCallAccuracy.

Needs the eval-only dependencies -- same as eval_retrieval_ragas.py, install and run
inside a throwaway container so the persistent core-1 container is never touched:

    docker compose run --rm core sh -c \\
        "pip install -q -r requirements-eval.txt && python scripts/eval_tool_selection_ragas.py"

Reuses the SAME ground truth as scripts/run_conversation_tests.py -- the scripted
scenarios in tests/conversations.py, specifically every turn with an `expect_tool`
assertion (a single, definite expected tool for that turn; `expect_any`-only turns are not
scored here, since RAGAS's reference_tool_calls wants one specific answer, not a set of
acceptable ones). Each scenario is driven through the real HTTP API exactly like the
scripted test runner -- same session, same reflex handling -- so what gets scored is the
tool the live planner actually chose, not a mock.

ToolCallAccuracy compares tool NAME and ARGS against a reference call. This ground truth
only records the expected tool name (matching what conversations.py already asserts), so
args are left empty on both sides -- this is deliberately a name-match test, not an
argument-correctness test (that is what the postcondition work on individual tools covers).

forbid_tools is also checked on every turn along the way (not just expect_tool ones) --
deterministically, no LLM judge needed for "did a forbidden tool run" -- and reported
alongside the RAGAS score for a fuller picture.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_conversation_tests as _rct  # noqa: E402
from conversations import CONVERSATIONS  # noqa: E402
from run_conversation_tests import call, find_lead, login, reflex_for, tools_of  # noqa: E402

# run_conversation_tests.BASE is 127.0.0.1:8000 -- correct when that script runs inside the
# same container as the app (docker compose exec core ...), which is how it's normally
# invoked. This eval deliberately runs in a SEPARATE throwaway container instead (to keep
# ragas/langchain off the persistent one, see the module docstring), where 127.0.0.1 refers
# to that throwaway container itself -- nothing is listening there.
#
# The exact persistent container name, not the "core" service alias: `docker compose run`
# joins this throwaway container to the SAME service, so DNS for the bare service name can
# round-robin between it and the real one -- and the throwaway container has nothing
# listening on 8000 either. The container name is unambiguous.
_rct.BASE = "http://ai-voice-agent-backend-core-1:8000"


async def _drive_and_score(convo, metric):
    """Runs one scripted conversation for real, scoring every expect_tool turn and
    checking forbid_tools on every turn. Returns (scored_rows, forbidden_hits)."""
    from ragas.messages import AIMessage, HumanMessage, ToolCall

    if convo.get("lead"):
        lead = find_lead(convo["lead"])
        if lead is None:
            return [], [f"lead {convo['lead']!r} not found"]
        body = {"customer_id": lead["id"], "channel": "chat", "direction": "inbound"}
    else:
        body = {"channel": "chat", "direction": "inbound"}
        if convo.get("caller_phone"):
            body["phone"] = convo["caller_phone"]

    session = call("POST", "/sessions", body)
    sid = session["session_id"]
    rows, forbidden_hits = [], []

    consumed: set[int] = set()
    for idx, spec in enumerate(convo["turns"]):
        if idx in consumed:
            continue
        said = spec["say"]
        turn = call("POST", f"/sessions/{sid}/turns", {"text": said})
        ran = tools_of(turn)

        for forbidden in spec.get("forbid_tools", []):
            if forbidden in ran:
                forbidden_hits.append(f"{convo['name']!r}: '{said}' ran forbidden {forbidden}")

        if spec.get("expect_tool"):
            user_input = [
                HumanMessage(content=said),
                AIMessage(content=turn.get("response_text") or "",
                         tool_calls=[ToolCall(name=t, args={}) for t in ran]),
            ]
            reference = [ToolCall(name=spec["expect_tool"], args={})]
            result = await metric.ascore(user_input=user_input, reference_tool_calls=reference)
            rows.append({"scenario": convo["name"], "said": said,
                        "expected": spec["expect_tool"], "ran": ran, "score": result.value})

        for _ in range(2):
            follow = reflex_for(turn.get("response_text"))
            if not follow or turn.get("end_call") or follow == said:
                break
            nxt = next((j for j in range(idx + 1, len(convo["turns"]))
                        if j not in consumed), None)
            if nxt is not None and \
                    convo["turns"][nxt]["say"].strip().lower() == follow.strip().lower():
                consumed.add(nxt)
            turn = call("POST", f"/sessions/{sid}/turns", {"text": follow})
            said = follow

    return rows, forbidden_hits


async def main() -> int:
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ToolCallAccuracy
    from openai import AsyncOpenAI

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.config import get_settings

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url or None)
    judge = llm_factory(model=settings.openai_model_main, provider="openai", client=client)
    metric = ToolCallAccuracy(llm=judge)

    login()

    scenarios = [c for c in CONVERSATIONS if any(t.get("expect_tool") for t in c["turns"])]
    print(f"\n  Driving {len(scenarios)} scenario(s) with an expect_tool assertion "
          f"through the live API\n")

    all_rows, all_forbidden = [], []
    for i, convo in enumerate(scenarios, start=1):
        try:
            rows, forbidden = await _drive_and_score(convo, metric)
        except Exception as e:
            print(f"  [{i}/{len(scenarios)}] {convo['name'][:50]:50} ERROR: "
                  f"{e.__class__.__name__}: {e}")
            continue
        all_rows += rows
        all_forbidden += forbidden
        for r in rows:
            mark = "OK  " if r["score"] >= 0.99 else "MISS"
            print(f"  [{i}/{len(scenarios)}] {mark}  {r['said'][:44]:44} "
                  f"expected={r['expected']:22} ran={r['ran']}")

    n = len(all_rows)
    avg = sum(r["score"] for r in all_rows) / n if n else 0.0
    print(f"\n  {'=' * 74}")
    print(f"  TOOL SELECTION (expect_tool turns): {n} scored, avg ToolCallAccuracy={avg:.3f}")
    misses = [r for r in all_rows if r["score"] < 0.99]
    if misses:
        print(f"\n  {len(misses)} miss(es):")
        for r in misses:
            print(f"    [{r['scenario']}] '{r['said']}' -> expected {r['expected']}, ran {r['ran']}")

    print(f"\n  FORBIDDEN-TOOL COMPLIANCE (every turn, not just expect_tool ones): "
          f"{len(all_forbidden)} violation(s)")
    for f in all_forbidden:
        print(f"    - {f}")
    print()
    return 0 if not misses and not all_forbidden else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
