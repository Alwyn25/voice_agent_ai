#!/usr/bin/env python3
"""Three calls, one file you can put in front of people.

    docker compose exec core python scripts/showcase.py
    docker compose exec core python scripts/showcase.py --out /mnt/user-data/outputs/demo.txt

Deliberately small: three conversations, sixteen customer turns, roughly forty model calls.
Enough to show the engine doing real work without spending an afternoon's quota on it.

Writes as it goes, so if a call fails halfway you still have everything up to that point.
Plain text, no colour codes — it will paste into a document or a slide unchanged.

The three callers are chosen to make three different arguments:
  1. a new lead    — it answers from documents, then books, and waits to be told yes
  2. an existing customer — it verifies who they are BEFORE it changes their booking
  3. an opted-out lead    — it refuses, and no amount of asking changes that
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "http://127.0.0.1:8000"
EMAIL, PASSWORD = "rahul@advora.ai", "rahul12345"
TOKEN = None
LLM_DOWN = "having trouble on my side"
NOISE_RULES = {"turn_policy_evaluated", "calling_policy_evaluated", "hindi_response"}

CALLS = [
    {"lead": "Rohan Mehta",
     "headline": "A new enquiry, start to finish",
     "claim": "It answers from the project documents, finds a real slot, and does not "
              "book anything until he says so.",
     "says": [
         "hi, I saw the project online — what sizes do you have?",
         "what are the extra charges on top of the flat cost?",
         "ok, can I come and see it Friday afternoon?",
         "yes please, book that",
         "my budget is about 1.4 crore and my wife decides",
     ]},

    {"lead": "Aditya Verma",
     "headline": "Changing a booking, safely",
     "claim": "Moving a booking changes a record, so it asks who he is FIRST — not after "
              "he has already agreed.",
     "says": [
         "what's my site visit booked for?",
         "I need to move it to Friday afternoon",
         "yes please",
         "0007",
     ]},

    {"lead": "Kavya Menon",
     "headline": "The one it will not do",
     "claim": "She asked us to stop contacting her. The agent refuses its own operator, "
              "and asking again does not help.",
     "says": [
         "hello",
         "I want to book a site visit",
         "come on, just book it for me",
     ]},
]


def call(method, path, payload=None):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(payload).encode() if payload is not None else None)
    req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=180) as res:
        body = res.read().decode()
        return json.loads(body) if body else None


def find_lead(name):
    for c in call("GET", f"/customers?search={urllib.parse.quote(name)}")["items"]:
        if c["full_name"].lower() == name.lower():
            return c
    return None


class Out:
    """Console and file at once, flushed every line so a crash keeps what ran."""

    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, line=""):
        print(line)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def run_call(spec, out, totals) -> bool:
    lead = find_lead(spec["lead"])
    if lead is None:
        out(f"  ! {spec['lead']} not found — run seed_dsr.py")
        return True

    out("")
    out("=" * 78)
    out(f"  {spec['headline'].upper()}")
    out(f"  Caller: {lead['full_name']}   ·   CRM stage: {lead['current_state']}")
    out("")
    for line in _wrap(spec["claim"], 74):
        out(f"  {line}")
    out("=" * 78)
    out("")

    session = call("POST", "/sessions", {"customer_id": lead["id"],
                                         "channel": "chat", "direction": "inbound"})
    sid = session["session_id"]
    if session.get("opening_line"):
        out(f"   AGENT   {session['opening_line']}")
        out("")

    for said in spec["says"]:
        out(f"   CALLER  {said}")
        turn = call("POST", f"/sessions/{sid}/turns", {"text": said})
        reply = turn.get("response_text") or ""
        for i, line in enumerate(_wrap(reply, 66)):
            out(f"   {'AGENT  ' if i == 0 else '       '} {line}")

        ran = turn.get("executed_tools") or []
        ok = [t["tool_name"] for t in ran if t.get("status") in ("success", "skipped_duplicate")]
        blocked = [t["tool_name"] for t in ran if t.get("status") == "blocked"]
        dbg = turn.get("debug") or {}
        fired = sorted({r.get("rule") for r in (dbg.get("rules_fired") or [])
                        if r.get("rule") and r["rule"] not in NOISE_RULES})
        ms = (dbg.get("timing") or {}).get("wall_ms") or 0

        notes = []
        if ok:
            notes.append("ran: " + " -> ".join(ok))
        if blocked:
            notes.append("BLOCKED: " + ", ".join(blocked))
        if fired:
            notes.append("rule: " + ", ".join(fired))
        if turn.get("pending_confirmation"):
            notes.append("waiting for consent")
        if turn.get("escalation"):
            notes.append("handed to a person")
        if turn.get("state_changed_to"):
            notes.append(f"stage -> {turn['state_changed_to']}")
        notes.append(f"{ms/1000:.1f}s")
        out(f"           [ {'  |  '.join(notes)} ]")
        out("")

        totals["turns"] += 1
        totals["tools"] += len(ok)
        totals["ms"] += ms
        totals["names"].update(ok)
        totals["rules"].update(fired)

        if LLM_DOWN in reply.lower():
            out("   ** Stopped: the model call failed. Run scripts/check_llm.py. **")
            return False

    try:
        call("POST", f"/sessions/{sid}/end", {"outcome": "showcase", "analyse": "skip"})
    except Exception:                                            # noqa: BLE001
        pass
    return True


def _wrap(text, width):
    words, lines, cur = (text or "").split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


ALL_TOOLS = ["locate_centre", "search_slots", "book_appointment", "reschedule_appointment",
             "cancel_appointment", "my_appointments", "my_tickets", "create_ticket",
             "knowledge_lookup", "verify_identity", "record_customer_fact",
             "update_customer_profile", "schedule_callback", "record_do_not_contact"]


def main() -> int:
    global TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/user-data/outputs/agent-demo.txt")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    TOKEN = call("POST", "/auth/login", {"email": EMAIL, "password": PASSWORD})["access_token"]

    # A demo that shows the wrong thing is worse than one that does not run.
    expected = {"Rohan Mehta": "lead", "Aditya Verma": "site_visit_scheduled",
                "Kavya Menon": "do_not_contact"}
    stale = [f"{n}: is {(find_lead(n) or {}).get('current_state')}, should be {w}"
             for n, w in expected.items()
             if find_lead(n) and find_lead(n).get("current_state") != w]
    if stale:
        print("\n  The demo data has drifted:")
        for line in stale:
            print(f"    {line}")
        print("\n    docker compose exec core python scripts/reset_visits.py\n")
        return 2

    out = Out(args.out)
    totals = {"turns": 0, "tools": 0, "ms": 0, "names": set(), "rules": set()}
    started = time.perf_counter()

    out("=" * 78)
    out("  DSR HIGHLAND GREENZ  ·  AI VOICE SALES & LIFECYCLE AGENT")
    out(f"  Live transcript  ·  {datetime.now():%d %B %Y, %H:%M}")
    out("")
    for line in _wrap("Three calls, unscripted on the agent's side. Under each reply is "
                      "what the engine actually did: which tools it ran, which policy "
                      "rules fired, and how long it took.", 74):
        out(f"  {line}")
    out("=" * 78)

    for spec in CALLS:
        if not run_call(spec, out, totals):
            break

    used = [t for t in ALL_TOOLS if t in totals["names"]]
    out("")
    out("=" * 78)
    out("  WHAT THIS RUN SHOWED")
    out("")
    out(f"    {totals['turns']} customer turns across {len(CALLS)} calls, "
        f"{int(time.perf_counter() - started)}s total")
    out(f"    {totals['tools']} tool executions covering {len(used)} of {len(ALL_TOOLS)} tools")
    out(f"    {len(totals['rules'])} policy rules fired")
    out("")
    out("    tools used:  " + ", ".join(used))
    if totals["rules"]:
        out("    rules fired: " + ", ".join(sorted(totals["rules"])))
    out("")
    for line in _wrap("Every turn above wrote a decision trace to the database: what it "
                      "understood, which tools were available, which rules were evaluated "
                      "and why, what it ran, and what it said. Any of them can be opened "
                      "in the Call Logs screen.", 72):
        out(f"    {line}")
    out("=" * 78)
    out.close()

    print(f"\n  Saved to {args.out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())