#!/usr/bin/env python3
"""Where the BRIDGE time goes — P1.08.

The benchmark report attributes ~60 ms of every turn to a "bridge" stage (frame relay +
WS hop) and suspects a synchronous call on the asyncio path. This script answers two
questions the old profile_turns.py could not:

  1. WHAT is the bridge actually made of? (session load, message persist, customer load,
     trace write, session save, commit) — read straight from the new bridge_* sub-stage
     timings now on debug.timing.
  2. Is anything BLOCKING the event loop while it runs? — read from the loop monitor via
     /health/loop, and, in --cprofile mode, from a cProfile of process_turn itself.

Two modes
---------
HTTP mode (default) — drives the live API exactly like profile_turns.py, so it measures
the real containerised stack (real Redis, real Railway Postgres RTT):

    docker compose exec core python scripts/profile_bridge.py
    docker compose exec core python scripts/profile_bridge.py --runs 8

cProfile mode — profiles process_turn IN-PROCESS with the LLM mocked (LLM_PROVIDER=mock),
so the model API's network time does not swamp the picture and what remains is OUR code:
the bridge I/O and the engine around it. This is the mode that names a synchronous hotspot
by function.

    docker compose exec core python scripts/profile_bridge.py --cprofile 20

    NOTE: cProfile mode runs real turns against a real conversation and therefore WRITES
    to the database (messages, decision traces) just as a live call would. It picks the
    most recent conversation unless --conversation <uuid> is given. Use a throwaway/demo
    conversation; do not point it at production data you care about.
"""
import argparse
import json
import os
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# All credentials come from the environment — never hard-code a token/API key in a file
# that lives in git. Overridable so the same script drives any environment:
#   BRIDGE_PROFILE_BASE      API base URL (default local dev)
#   BRIDGE_PROFILE_TOKEN     a JWT access token OR a raw API key (skips the login step)
#   BRIDGE_PROFILE_EMAIL     demo login used only when no token is supplied
#   BRIDGE_PROFILE_PASSWORD  demo login password
# When TOKEN is empty, http_mode logs in with EMAIL/PASSWORD to obtain a JWT. The defaults
# below are the seeded demo login, valid only against a seed_demo.py database.
BASE = os.getenv("BRIDGE_PROFILE_BASE", "http://127.0.0.1:8000")
EMAIL = os.getenv("BRIDGE_PROFILE_EMAIL", "rahul@advora.ai")
PASSWORD = os.getenv("BRIDGE_PROFILE_PASSWORD", "rahul12345")
TOKEN = os.getenv("BRIDGE_PROFILE_TOKEN", "")

# The bridge sub-stages, in the order they run within a turn, so the report reads as a
# waterfall. Kept in sync with the async with _bridge.stage("…") calls in turn_runner.py.
BRIDGE_STAGES = ("session_load", "input_persist", "customer_load",
                 "trace_persist", "session_save", "commit")

# A few deliberately different turn shapes — a bridge cost that only shows up on the
# tool-running shape (extra flush/commit work) is worth seeing separately from chit-chat.
SHAPES = [
    ("chit-chat, no tool", "Rohan Mehta", "hello"),
    ("pricing question", "Rohan Mehta", "what does a 2 BHK cost"),
    ("own-record lookup", "Aditya Verma", "what's my site visit booked for"),
]


# --------------------------------------------------------------------------- HTTP client
def _auth_header(token: str) -> tuple[str, str]:
    """Pick the right auth header for whatever credential was supplied.

    The API accepts two forms (see app/security/deps.py):
      - a JWT access token via ``Authorization: Bearer <jwt>`` — three dot-separated
        base64 segments, which is what /auth/login returns;
      - a raw API key via ``X-API-Key: <key>`` — no dots (e.g. 64 hex chars).
    Sending an API key as a Bearer token makes the server try to JWT-decode it and 401,
    so the header is chosen by the token's shape rather than assumed.
    """
    if token.count(".") == 2:
        return "Authorization", f"Bearer {token}"
    return "X-API-Key", token


def call(method, path, payload=None):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(payload).encode() if payload is not None else None)
    req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header(*_auth_header(TOKEN))
    with urllib.request.urlopen(req, timeout=120) as res:
        body = res.read().decode()
        return json.loads(body) if body else None


def wait_for_api(seconds: int = 90) -> bool:
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
    # Encode the search term — names contain spaces (and could contain other characters
    # urllib rejects in a raw URL), so the query value must be percent-encoded.
    q = urllib.parse.quote(name)
    for c in call("GET", f"/customers?search={q}")["items"]:
        if c["full_name"].lower() == name.lower():
            return c
    return None


def one_turn_timing(lead, text) -> dict:
    """Run a single turn through the API and return its debug.timing dict."""
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
    return dict(((turn.get("debug") or {}).get("timing")) or {})


def _median(values):
    return int(statistics.median(values)) if values else 0


def _p95(values):
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return int(ordered[idx])


def http_mode(runs: int) -> int:
    global TOKEN
    if not wait_for_api():
        print("\nThe API never came up. Check: docker compose logs core\n")
        return 2
    # Only log in when no credential was pre-set at the top of the file. If TOKEN is
    # already filled (a pasted JWT or API key), use it as-is — logging in would overwrite
    # it, and the demo EMAIL/PASSWORD below only exist in a seeded demo DB, so on any other
    # database that login 401s before the supplied token is ever tried.
    if not TOKEN:
        try:
            TOKEN = call("POST", "/auth/login",
                         {"email": EMAIL, "password": PASSWORD})["access_token"]
        except urllib.error.HTTPError as e:
            print(f"\n  Login failed ({e.code}). The demo credentials in this script "
                  f"({EMAIL}) do not exist in this database.\n"
                  "  Either set a valid EMAIL/PASSWORD at the top of the script, or paste a\n"
                  "  token/API key into TOKEN. Or skip auth entirely with:\n"
                  "      python scripts/profile_bridge.py --cprofile 20\n")
            return 2

    # Auth preflight — fail with a clear message instead of a stack trace deep in the run.
    try:
        call("GET", "/customers?search=x")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            hdr = _auth_header(TOKEN)[0] if TOKEN else "none"
            print(f"\n  Credential rejected ({e.code}) — the token was sent via {hdr} but "
                  "the server did not accept it.\n"
                  "  Check the token is a valid API key / access token for THIS database,\n"
                  "  or skip auth entirely with:\n"
                  "      python scripts/profile_bridge.py --cprofile 20\n")
            return 2
        raise

    print(f"\nBRIDGE BREAKDOWN — {runs} run(s) per shape. Milliseconds: median (p95).")
    print("  bridge = sum of the six sub-stages (real DB/Redis overhead).")
    print("  unattr = time in the turn we did NOT instrument (e.g. an auxiliary model "
          "call);\n           a large value points at a stage to wrap, not at bridge I/O.\n")
    header = f"  {'turn shape':22} {'total':>10} {'bridge':>10} {'unattr':>10}  " + \
             "  ".join(f"{s:>13}" for s in BRIDGE_STAGES)
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Accumulate every run so the closing summary can rank stages across all shapes.
    pooled: dict[str, list[int]] = {k: [] for k in ("total_ms", "bridge_ms", "unattributed_ms")}
    pooled.update({f"bridge_{s}_ms": [] for s in BRIDGE_STAGES})

    for label, lead_name, text in SHAPES:
        lead = find_lead(lead_name)
        if not lead:
            print(f"  {label:22}  (lead '{lead_name}' not found — run seed_demo.py)")
            continue
        rows: dict[str, list[int]] = {k: [] for k in pooled}
        for _ in range(runs):
            t = one_turn_timing(lead, text)
            for k in rows:
                if k in t:
                    rows[k].append(int(t[k]))
                    pooled[k].append(int(t[k]))

        def cell(key):
            return f"{_median(rows[key]):>4} ({_p95(rows[key]):>4})"

        line = f"  {label:22} {cell('total_ms'):>10} {cell('bridge_ms'):>10} " + \
               f"{cell('unattributed_ms'):>10}  " + \
               "  ".join(f"{cell(f'bridge_{s}_ms'):>13}" for s in BRIDGE_STAGES)
        print(line)

    # Rank the sub-stages by their share of the bridge, so the biggest lever is obvious.
    print("\n  Bridge sub-stages, ranked (median across all shapes):")
    ranked = sorted(((s, _median(pooled[f"bridge_{s}_ms"])) for s in BRIDGE_STAGES),
                    key=lambda kv: kv[1], reverse=True)
    bridge_total = max(1, _median(pooled["bridge_ms"]))
    for stage, ms in ranked:
        bar = "#" * int(round(20 * ms / bridge_total))
        print(f"    {stage:16} {ms:>4} ms  {bar}")

    # Loop health: if the monitor is on, a non-zero spike count here is the smoking gun
    # for a blocking call — the bridge is slow AND the loop stalled while it ran.
    try:
        loop = call("GET", "/health/loop")
        if loop.get("enabled"):
            # These are CUMULATIVE stats since the server (monitor) started — not just this
            # run. A spike from cold-start (first DB connect, first statement compile, first
            # embeddings) is counted here even when this run was clean, which is why the
            # numbers can look alarming right after a restart. Re-run once warm and watch
            # whether `spikes` GROWS: growth = a real in-run stall; steady = old cold-start.
            print(f"\n  Event loop (cumulative since server start): "
                  f"max_lag={loop['max_lag_ms']}ms, spikes={loop['spikes']} "
                  f"over {loop['samples']} samples.")
            if loop["spikes"]:
                print("    ^ The loop stalled at least once since startup. Check the log "
                      "timestamps against this run:\n      a cold-start spike (DB connect / "
                      "first compile) is one-time; a spike whose count grows on a WARM\n"
                      "      run is a real blocker — find the callsite with LOOP_MONITOR_DEEP="
                      "true, then turn DEEP\n      back OFF (asyncio debug mode taxes every "
                      "await and inflates the bridge numbers).")
        else:
            print("\n  Event loop monitor is OFF. Enable it (LOOP_MONITOR_ENABLED=true) to "
                  "catch blocking calls.")
    except Exception:                                            # noqa: BLE001
        pass

    print("\n  The bridge is mostly network RTT (Redis + Postgres). A sub-stage far above "
          "the ~15 ms\n  base RTT, or any loop spike, is where to look. A large 'unattr' "
          "means the turn spent\n  time in a stage we don't yet time (usually an extra "
          "model call) — instrument that stage\n  before drawing bridge conclusions.\n")
    return 0


# ----------------------------------------------------------------------- cProfile mode
def cprofile_mode(iterations: int, conversation_id: str | None) -> int:
    """Profile process_turn in-process with the LLM mocked, to name sync hotspots.

    The mock provider (LLM_PROVIDER=mock) returns a canned proposal with no network call,
    so the profile is dominated by OUR code and the DB/Redis round-trips — exactly the
    surface P1.08 cares about. cProfile counts every Python call, so a synchronous helper
    burning CPU on the hot path rises to the top of the ``tottime`` ranking; an awaited I/O
    call shows as cumulative time in the coroutine machinery, not tottime.
    """
    import asyncio
    import cProfile
    import io
    import os
    import pstats

    # Must be set before app.config caches settings and app.providers.llm.router caches
    # the provider — hence the top-of-function import order below.
    os.environ["LLM_PROVIDER"] = "mock"

    from sqlalchemy import select

    from app.db.base import get_session_factory
    from app.db.models import Conversation
    from app.engine.planner.turn_runner import process_turn

    async def _pick_conversation(db):
        if conversation_id:
            conv = (await db.execute(select(Conversation).where(
                Conversation.id == conversation_id))).scalar_one_or_none()
            if conv is None:
                raise SystemExit(f"conversation {conversation_id} not found")
            return conv
        # Most recent conversation — good enough for a profiling target.
        conv = (await db.execute(select(Conversation)
                                 .order_by(Conversation.created_at.desc())
                                 .limit(1))).scalar_one_or_none()
        if conv is None:
            raise SystemExit("no conversations in the database — run seed_demo.py first")
        return conv

    async def _run():
        factory = get_session_factory()
        async with factory() as db:
            conv = await _pick_conversation(db)
            print(f"Profiling {iterations} turns against conversation {conv.id} "
                  f"(tenant {conv.tenant_id}), LLM mocked.\n")
            # Warm up once so first-call import/connection costs do not pollute the profile.
            await process_turn(db, tenant_id=conv.tenant_id, conversation=conv,
                               user_input="hello")

            profiler = cProfile.Profile()
            profiler.enable()
            for i in range(iterations):
                await process_turn(db, tenant_id=conv.tenant_id, conversation=conv,
                                   user_input="hello there, just checking in")
            profiler.disable()
            return profiler

    profiler = asyncio.run(_run())

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)

    # tottime = time in the function itself, excluding sub-calls. This is where a
    # synchronous CPU blocker shows up. cumtime ranks the call tree (useful for I/O).
    print("=" * 78)
    print("TOP 20 BY tottime — a synchronous blocker on the async path ranks HERE")
    print("=" * 78)
    ps.sort_stats("tottime").print_stats(20)
    print(s.getvalue())

    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2)
    print("=" * 78)
    print("TOP 20 BY cumtime — the overall call tree (I/O waits included)")
    print("=" * 78)
    ps2.sort_stats("cumulative").print_stats(20)
    print(s2.getvalue())

    print("How to read this: entries high in the FIRST list that are not obviously async\n"
          "I/O (e.g. json, regex, schema validation, a rules loop) are candidates to move\n"
          "off the event loop with asyncio.to_thread, or to cache. Entries only high in the\n"
          "SECOND list are usually awaited round-trips (Redis/Postgres) — latency, not a\n"
          "loop block.\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Profile the per-turn bridge overhead (P1.08).")
    ap.add_argument("--runs", type=int, default=5, help="HTTP mode: samples per shape")
    ap.add_argument("--cprofile", type=int, metavar="N", default=0,
                    help="cProfile mode: profile N in-process turns (LLM mocked)")
    ap.add_argument("--conversation", default=None,
                    help="cProfile mode: target this conversation id (default: most recent)")
    args = ap.parse_args()

    if args.cprofile:
        return cprofile_mode(args.cprofile, args.conversation)
    return http_mode(args.runs)


if __name__ == "__main__":
    raise SystemExit(main())
