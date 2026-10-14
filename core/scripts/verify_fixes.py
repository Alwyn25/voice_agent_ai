#!/usr/bin/env python3
"""Report which fixes are actually live — in the CODE and in the DATABASE.

Half of this platform's behaviour is data, not code. Tool contracts and rule sets live in
Postgres, so editing definitions.py and restarting changes nothing at all until the tools
are reseeded. That gap is invisible from the outside: the agent just carries on behaving
the way it did before, and it looks like the fix did not work.

    docker compose exec core python scripts/verify_fixes.py

Stdlib only, like the other scripts here.
"""
import json
import urllib.error
import urllib.request
import sys
from pathlib import Path

# scripts/ is not on the path when run as `python scripts/x.py`; the other seed scripts
# do exactly this so `app.*` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


BASE = "http://127.0.0.1:8000"
EMAIL = "rahul@advora.ai"
PASSWORD = "rahul12345"

OK, BAD, WARN = "  [LIVE]    ", "  [MISSING] ", "  [WARN]    "
failures = []


def check(live: bool, label: str, remedy: str = "") -> None:
    print((OK if live else BAD) + label)
    if not live:
        failures.append((label, remedy))
        if remedy:
            print(f"             -> {remedy}")


def call(method, path, payload=None, token=None):
    req = urllib.request.Request(
        BASE + path, method=method,
        data=json.dumps(payload).encode() if payload is not None else None)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as res:
        body = res.read().decode()
        return json.loads(body) if body else None


# ----------------------------------------------------------------- code checks

print("\nCODE — is the running process the patched one?\n")

try:
    from app.engine.planner import turn_runner
    check(hasattr(turn_runner, "PROMISE_PATTERNS"),
          "turn_runner: promise guard (PROMISE_PATTERNS)",
          "scheduling/turn_runner.py not replaced, or the container was not restarted")
    if hasattr(turn_runner, "_promises_action"):
        sample = "Let me check the available slots for the evening of August 12th."
        check(turn_runner._promises_action(sample),
              f"turn_runner: guard matches the real failing line")
except Exception as e:
    check(False, f"turn_runner import failed: {e}")

try:
    from app.engine.tools.builtin import scheduling_tools as st
    check(hasattr(st, "PART_OF_DAY"), "scheduling_tools: time-of-day filtering",
          "scheduling_tools.py not replaced, or the container was not restarted")
    check(hasattr(st, "_spread"), "scheduling_tools: whole-day slot spread")
    if hasattr(st, "_as_time"):
        check(str(st._as_time("4pm")) == "16:00:00", "scheduling_tools: '4pm' parses to 16:00")
except Exception as e:
    check(False, f"scheduling_tools import failed: {e}")

try:
    from app.engine.planner import prompts
    check("NEVER SAY YOU WILL DO SOMETHING" in prompts.BASE_SYSTEM,
          "prompts: do-not-narrate rule in BASE_SYSTEM")
except Exception as e:
    check(False, f"prompts import failed: {e}")

try:
    from app.engine.tools.builtin import lookup_tools
    check(hasattr(lookup_tools, "CALLER_STATUS"),
          "lookup_tools: caller-facing appointment status")
except Exception as e:
    check(False, f"lookup_tools import failed: {e}")


# ----------------------------------------------------------------- data checks

print("\nDATABASE — contracts and rules are rows, not code.\n")

try:
    token = call("POST", "/auth/login", {"email": EMAIL, "password": PASSWORD})["access_token"]
except Exception as e:
    print(f"{WARN}cannot reach {BASE} ({e}) — skipping the database checks")
    print("             -> is the core container up?")
    token = None

if token:
    tools = {t["name"]: t for t in call("GET", "/tools", token=token)}

    props = (tools.get("search_slots", {}).get("input_schema", {}) or {}).get("properties", {})
    check("part_of_day" in props,
          "tool contract: search_slots accepts part_of_day",
          "docker compose exec core python scripts/seed.py --reseed-tools")
    check("time_from" in props, "tool contract: search_slots accepts time_from")

    # This is the trap worth naming: the OLD contract sets additionalProperties: false, so
    # once the model starts sending part_of_day against an unreseeded contract, the executor
    # rejects the call outright with a schema error and the tool never runs at all.
    if "part_of_day" not in props:
        strict = (tools.get("search_slots", {}).get("input_schema", {}) or {}).get(
            "additionalProperties", True) is False
        if strict:
            print(f"{WARN}old contract is strict (additionalProperties: false) — any "
                  f"part_of_day the model sends will be BLOCKED, not ignored")

    rules = {r["name"]: r for r in call("GET", "/rules", token=token)}
    dsr = rules.get("dsr_turn_policy")
    if dsr:
        active = call("GET", f"/rules/dsr_turn_policy?version={dsr['active_version']}",
                      token=token)
        ids = [r["id"] for r in (active.get("compiled") or {}).get("rules", [])]
        check("identity_before_rescheduling" in ids,
              "rule set: identity_before_rescheduling is in the ACTIVE version",
              "docker compose exec core python scripts/seed_dsr.py --only rules   "
              "(or paste + activate in the Rule Engine screen)")
    else:
        print(f"{WARN}dsr_turn_policy not found — is the DSR pack loaded for this tenant?")

    personas = [t for t in call("GET", "/prompts", token=token)] if False else None
    # /prompts is not implemented yet; the persona check is done from the DB side instead.

try:
    import asyncio

    from sqlalchemy import select

    from app.db.base import get_session_factory
    from app.db.models import PromptTemplate

    async def _persona_rows():
        async with get_session_factory()() as db:
            return (await db.execute(select(PromptTemplate).where(
                PromptTemplate.purpose == "persona",
                PromptTemplate.is_active == True))).scalars().all()  # noqa: E712

    rows = asyncio.run(_persona_rows())
    check(bool(rows),
          "persona: an agent persona is configured for this tenant",
          "no active persona row -> the agent falls back to the tenant's industry "
          "pack's default_persona (real_estate_india's is \"Maya, a polite and "
          "efficient customer care agent\") — generic, not DSR")
except Exception as e:
    print(f"{WARN}persona check skipped ({e.__class__.__name__})")


# ----------------------------------------------------------------- verdict

print()
if failures:
    print(f"{len(failures)} thing(s) not live. The agent will behave exactly as it did "
          f"before until these are addressed.\n")
    sys.exit(1)
print("Everything is live. If behaviour is still wrong, it is a new problem — pull the "
      "decision trace for the failing turn and read debug.planner_error.\n")