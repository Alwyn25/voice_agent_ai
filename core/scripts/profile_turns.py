#!/usr/bin/env python3
"""Where the time goes on a turn.

    docker compose exec core python scripts/profile_turns.py
    docker compose exec core python scripts/profile_turns.py --runs 5

Run it before and after a change. "It feels faster" is not a measurement, and the stages
do not cost what you would guess: the planner and responder are two separate model calls
in series, so a turn that runs no tool at all can still be the slowest thing in the call.

Reports per turn shape: planner, tools, responder, and the gap between their sum and the
wall clock (session load, rule evaluation, trace write — everything not separately timed).
"""
import argparse
import json
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "http://127.0.0.1:8000"
EMAIL = "rahul@advora.ai"
PASSWORD = "rahul12345"
TOKEN = None
BUDGET_MS = 650          # the Voice Pipeline doc's budget for the whole brain stage

# Deliberately different shapes. The cheapest turn in the call still pays for two model
# calls, which is the point the numbers need to make.
SHAPES = [
    ("chit-chat, no tool, no retrieval", "Rohan Mehta", "hello"),
    ("knowledge question (retrieval)", "Rohan Mehta", "what configurations do you have"),
    ("own-record lookup (one tool)", "Aditya Verma", "what's my site visit booked for"),
    ("slot search (chained tools)", "Nisha Pillai", "I want to visit on Thursday evening"),
]


def call(method, path, payload=None):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(payload).encode() if payload is not None else None)
    req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=120) as res:
        body = res.read().decode()
        return json.loads(body) if body else None


def baseline(runs: int) -> None:
    """What a bare model call costs today, with nothing of ours around it.

    Without this the profile cannot tell you anything. Two runs of this script an hour
    apart showed chit-chat — a single call on a code path that had not changed — go from
    1670 ms to 2482 ms. That is the API having a slower afternoon, and it is large enough
    to swamp any change we make. Measure the floor in the same run as the engine, or do
    not compare runs at all.
    """
    import asyncio
    import statistics as st
    import time as _t

    from app.config import get_settings
    from app.providers.llm.base import LLMError
    from app.providers.llm.router import get_llm

    cfg = get_settings()
    llm = get_llm()
    # Roughly the shape of a real planner prompt: the system block is the expensive part,
    # and a trivial prompt would flatter the numbers badly.
    filler = ("You are a phone agent. Reply with JSON only: {\"ok\": true}\n"
              + "Background you must consider but need not use.\n" * 120)

    print("\nAPI BASELINE — a bare model call, no engine\n")
    print(f"  {'tier':22} {'small prompt':>14} {'planner-sized':>15}")
    # `model` is an ABSTRACT tier request to get_llm() (None -> main, "fast" -> fast) --
    # see openai_provider._resolve_model's docstring. `label` is just for this printout.
    for label, model in (("main", None), ("fast", "fast")):
        row = []
        for system in ('Reply with JSON only: {"ok": true}', filler):
            times = []
            for _ in range(max(2, runs // 2)):
                t0 = _t.perf_counter()
                try:
                    asyncio.run(llm.complete_json(system=system, user="Say ok.",
                                                  model=model, max_tokens=20))
                    times.append(int((_t.perf_counter() - t0) * 1000))
                except LLMError as e:
                    print(f"  {label:22}  failed: {e}")
                    times = []
                    break
            row.append(int(st.median(times)) if times else 0)
        print(f"  {label:22} {row[0]:>14} {row[1]:>15}")
    print("\n  This is the floor. No amount of engine work gets below it.")


def wait_for_api(seconds: int = 90) -> bool:
    """The container runs alembic before uvicorn, so a build finishes well before the API
    answers. Chaining the commands with >> means this script fires into a closed port and
    dies on a connection error that looks like a bug and is not."""
    import time as _t
    deadline = _t.time() + seconds
    while _t.time() < deadline:
        try:
            call("GET", "/health")
            return True
        except Exception:                                        # noqa: BLE001
            _t.sleep(2)
    return False


def find_lead(name):
    for c in call("GET", f"/customers?search={urllib.parse.quote(name)}")["items"]:
        if c["full_name"].lower() == name.lower():
            return c
    return None


def one_turn(lead, text):
    session = call("POST", "/sessions", {"customer_id": lead["id"],
                                         "channel": "chat", "direction": "inbound"})
    sid = session["session_id"]
    try:
        turn = call("POST", f"/sessions/{sid}/turns", {"text": text})
    finally:
        try:
            call("POST", f"/sessions/{sid}/end", {"outcome": "profile", "analyse": "skip"})
        except Exception:                                        # noqa: BLE001
            pass
    t = dict(((turn.get("debug") or {}).get("timing")) or {})
    t["retrieval_ms"] = ((turn.get("debug") or {}).get("retrieval") or {}).get("latency_ms") or 0
    t["tools"] = len(turn.get("executed_tools") or [])
    return t


def main() -> int:
    global TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5, help="samples per shape")
    ap.add_argument("--baseline-only", action="store_true",
                    help="just measure the API, skip the engine")
    args = ap.parse_args()
    if not wait_for_api():
        print("\nThe API never came up. Check: docker compose logs core\n")
        return 2
    TOKEN = call("POST", "/auth/login", {"email": EMAIL, "password": PASSWORD})["access_token"]

    baseline(args.runs)
    if args.baseline_only:
        return 0

    print(f"\n{args.runs} run(s) per shape. Milliseconds: median (min-max).\n")
    print(f"  {'turn shape':34} {'wall':>6} {'plan':>6} {'x':>3} {'rag':>6} "
          f"{'tools':>6} {'resp':>6} {'other':>6}")
    print("  " + "-" * 80)

    grand = []
    for label, lead_name, text in SHAPES:
        lead = find_lead(lead_name)
        if lead is None:
            print(f"  {label:36}  lead {lead_name!r} not found")
            continue
        samples = [one_turn(lead, text) for _ in range(args.runs)]
        med = lambda k: int(statistics.median(s.get(k) or 0 for s in samples))  # noqa: E731
        walls = sorted(s.get("wall_ms") or 0 for s in samples)
        wall, plan, tools, resp = med("wall_ms"), med("planner_ms"), med("tool_ms"), med("responder_ms")
        calls, rag = med("planner_calls"), med("retrieval_ms")
        other = max(0, wall - plan - tools - resp)
        grand += [s.get("wall_ms") or 0 for s in samples]
        print(f"  {label:34} {wall:>6} {plan:>6} {calls:>3} {rag:>6} "
              f"{tools:>6} {resp:>6} {other:>6}   ({walls[0]}-{walls[-1]})")

    if grand:
        grand.sort()
        p50 = int(statistics.median(grand))
        p95 = grand[min(len(grand) - 1, int(len(grand) * 0.95))]
        print(f"\n  overall  p50 {p50} ms   p95 {p95} ms   budget {BUDGET_MS} ms"
              f"   ({p50 / BUDGET_MS:.1f}x over at p50)")

    print("""
Reading this:
  plan / x  total planner time, and HOW MANY planner calls the turn made. A turn that
            chains tools re-plans between them; two or three calls is normal, and it is
            the single biggest thing on the clock.
  rag       knowledge retrieval, inside the planner figure — embedding call plus the
            pgvector and text search.
  resp      the phrasing call. Should sit well below plan; if not, check it is on the
            fast model.
  tools     database work for the tools themselves.
  other     session load, rule evaluation, trace write. Should now be small — anything
            large here is real DB round trips, not hidden model calls.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())