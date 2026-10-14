#!/usr/bin/env python3
"""A demo you can run in front of people.

    docker compose exec core python scripts/demo.py            # pause between scenes
    docker compose exec core python scripts/demo.py --auto     # no pauses
    docker compose exec core python scripts/demo.py --only dnc,price
    docker compose exec core python scripts/demo.py --list

Each scene shows the conversation AND what the engine did underneath it — which tools ran,
which rules fired, how long the turn took. The transcript on its own is the least
interesting part: any chatbot can produce plausible sentences. What is worth showing a room
is that the agent could not have said otherwise — that a rule blocked it, a contract
validated it, and a decision trace recorded it.

Scenes are ordered to build an argument, not to cover features:
  1. it does the job          2. it will not be talked past
  3. it knows what it cannot do   4. it refuses to guess
  5. it stops itself lying
"""
import argparse
import json
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "http://127.0.0.1:8000"
EMAIL, PASSWORD = "rahul@advora.ai", "rahul12345"
TOKEN = None

BOLD, DIM, GREEN, RED, YELLOW, CYAN, OFF = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m")

# Fires on every turn as a marker, not a decision. Showing it in a demo buries the rules
# that actually did something under a word nobody can interpret.
NOISE_RULES = {"turn_policy_evaluated", "calling_policy_evaluated", "hindi_response"}

SCENES = [
    {"key": "book",
     "title": "It does the job",
     "point": "Three tools, one turn, one confirmation. Notice she is never asked "
              "to repeat herself.",
     "lead": "Rohan Mehta",
     "says": ["I'd like to come and see the project",
              "Friday afternoon if you have anything",
              "yes please"],
     "watch": "locate_centre and search_slots run in the SAME turn. The booking waits "
              "for her word."},

    {"key": "dnc",
     "title": "It will not be talked past",
     "point": "Kavya asked us to stop calling. Watch the agent refuse its own operator.",
     "lead": "Kavya Menon",
     "says": ["hello", "I want to book a site visit"],
     "watch": "dnc_state_stops_everything is terminal. No tool runs. This is the "
              "compliance story in one screen."},

    {"key": "price",
     "title": "It knows what it is not allowed to do",
     "point": "A discount request, and then a refund. Two different refusals, both "
              "from rules a non-engineer can edit.",
     "lead": "Vikram Singh",
     "says": ["can you give me a discount on the 3 BHK",
              "then I want a refund on what I already paid"],
     "watch": "cannot_negotiate_price blocks. A refund escalates outright — the agent "
              "may not settle money at all."},

    {"key": "identity",
     "title": "It asks before it changes anything",
     "point": "Moving a booking is a change to a record, so identity comes first — "
              "before the slot hunt, not after she has agreed.",
     "lead": "Aditya Verma",
     "says": ["I need to move my site visit",
              "Friday afternoon",
              "yes please",
              "0007"],
     "watch": "verify_identity, then reschedule_appointment, in one turn once she "
              "answers."},

    {"key": "knowledge",
     "title": "It refuses to guess",
     "point": "Two questions: one the documents answer, one they do not.",
     "lead": "Sanjay Gupta",
     "says": ["what configurations do you have",
              "is the monsoon offer still running",
              "what does a 2 BHK cost"],
     "watch": "Expired offers are not quoted as live. There is no 2 BHK cost sheet "
              "loaded — watch it say so instead of inventing a number."},

    {"key": "journey",
     "title": "The whole thing, one call",
     "point": "A single conversation from first contact to a booked visit, a logged "
              "complaint, a recorded budget and a callback. Fourteen tools exist; this "
              "call uses ten of them.",
     "lead": "Sanjay Gupta",
     "says": [
         "hi, I saw your project online and wanted to know more",
         "what configurations do you have and how big are they",
         "what are the extra charges on top of the flat cost",
         "where exactly is the site",
         "ok I'd like to visit — anything Friday afternoon?",
         "yes please",
         "my budget is about 1.4 crore and my wife makes the final call",
         "one thing — nobody called me back after I enquired last week, can you log that",
         "yes please",
         "also could someone call me tomorrow at 6am to discuss the payment plan",
         "and can you check what I have booked now",
         "that's everything, thanks",
     ],
     "watch": "knowledge_lookup, locate_centre, search_slots, book_appointment, "
              "record_customer_fact, create_ticket, schedule_callback, my_appointments — "
              "and the 6am callback quietly moved into legal calling hours."},

    {"key": "human",
     "title": "It gets out of the way",
     "point": "Asked for a person, it finds out what for — once — then transfers.",
     "lead": "Rohan Mehta",
     "says": ["can I speak to someone",
              "no, I would rather talk to a person"],
     "watch": "One question, then it goes. An unhappy caller is never made to explain "
              "themselves twice."},
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


def find_lead(name):
    for c in call("GET", f"/customers?search={urllib.parse.quote(name)}")["items"]:
        if c["full_name"].lower() == name.lower():
            return c
    return None


def wrap(text, indent="        "):
    return textwrap.fill(text, width=88, initial_indent=indent, subsequent_indent=indent)


def play(scene, auto: bool) -> dict:
    lead = find_lead(scene["lead"])
    if lead is None:
        print(f"{RED}  lead {scene['lead']!r} not found — run seed_dsr.py{OFF}")
        return {}

    print(f"\n{BOLD}{'─' * 90}{OFF}")
    print(f"{BOLD}  {scene['title']}{OFF}   {DIM}· {scene['lead']} · {lead['current_state']}{OFF}")
    print(wrap(scene["point"], "  "))
    print(f"{DIM}{wrap('Watch for: ' + scene['watch'], '  ')}{OFF}")
    print(f"{BOLD}{'─' * 90}{OFF}\n")
    if not auto:
        input(f"{DIM}  [enter]{OFF}")

    session = call("POST", "/sessions", {"customer_id": lead["id"],
                                         "channel": "chat", "direction": "inbound"})
    sid = session["session_id"]
    opening = session.get("opening_line")
    if opening:
        print(f"  {CYAN}agent{OFF}  {opening}\n")

    totals = {"turns": 0, "tools": 0, "ms": 0, "rules": set(), "names": set()}
    for said in scene["says"]:
        print(f"  {BOLD}lead {OFF}  {said}")
        turn = call("POST", f"/sessions/{sid}/turns", {"text": said})
        reply = turn.get("response_text") or ""
        print(f"  {CYAN}agent{OFF}  {reply}")

        ran = [t for t in (turn.get("executed_tools") or [])]
        ok = [t["tool_name"] for t in ran if t.get("status") in ("success", "skipped_duplicate")]
        blocked = [(t["tool_name"], t.get("error") or t.get("failed_check")) for t in ran
                   if t.get("status") == "blocked"]
        dbg = turn.get("debug") or {}
        fired = [r.get("rule") for r in (dbg.get("rules_fired") or [])
                 if r.get("rule") and r["rule"] not in NOISE_RULES]
        ms = (dbg.get("timing") or {}).get("wall_ms")

        bits = []
        if ok:
            bits.append(f"{GREEN}✓ {' → '.join(ok)}{OFF}")
        for name, why in blocked:
            bits.append(f"{RED}✗ {name} blocked{OFF}" + (f" {DIM}({why}){OFF}" if why else ""))
        if fired:
            bits.append(f"{YELLOW}rule: {', '.join(sorted(set(fired))[:3])}{OFF}")
        if turn.get("pending_confirmation"):
            bits.append(f"{DIM}awaiting her yes{OFF}")
        if turn.get("escalation"):
            bits.append(f"{YELLOW}→ handed to a person{OFF}")
        if ms:
            bits.append(f"{DIM}{ms} ms{OFF}")
        if bits:
            print(f"         {DIM}│{OFF} " + f"  {DIM}·{OFF} ".join(bits))
        print()

        totals["turns"] += 1
        totals["tools"] += len(ok)
        totals["ms"] += ms or 0
        totals["rules"] |= set(fired)
        totals["names"] |= set(ok)

    try:
        call("POST", f"/sessions/{sid}/end", {"outcome": "demo", "analyse": "skip"})
    except Exception:                                            # noqa: BLE001
        pass
    return totals


def main() -> int:
    global TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true", help="no pauses between scenes")
    ap.add_argument("--only", default="", help="comma-separated scene keys")
    ap.add_argument("--list", action="store_true", help="list the scenes and exit")
    args = ap.parse_args()

    if args.list:
        print()
        for sc in SCENES:
            print(f"  {sc['key']:10} {sc['title']}")
        print()
        return 0

    TOKEN = call("POST", "/auth/login", {"email": EMAIL, "password": PASSWORD})["access_token"]

    # Every scene assumes the leads are where seed_dsr put them. Run the suite first and
    # Rohan is already booked, Aditya is qualified rather than scheduled, and the demo
    # quietly shows the wrong thing to the one audience that matters.
    expected = {"Rohan Mehta": "lead", "Kavya Menon": "do_not_contact",
                "Vikram Singh": "negotiation", "Aditya Verma": "site_visit_scheduled",
                "Sanjay Gupta": "qualified"}
    stale = [f"{n}: {(find_lead(n) or {}).get('current_state')} (want {w})"
             for n, w in expected.items()
             if find_lead(n) and find_lead(n).get("current_state") != w]
    if stale:
        print(f"\n{RED}  The demo data has drifted — scenes will show the wrong thing.{OFF}")
        for line in stale:
            print(f"    {line}")
        print(f"\n    docker compose exec core python scripts/reset_visits.py\n")
        return 2

    wanted = {k.strip() for k in args.only.split(",") if k.strip()}
    scenes = [s for s in SCENES if not wanted or s["key"] in wanted]

    print(f"\n{BOLD}  DSR Highland Greenz — AI voice agent{OFF}")
    print(f"{DIM}  Every line below is live. Nothing is scripted on the agent's side.{OFF}")

    grand = {"turns": 0, "tools": 0, "ms": 0, "rules": set(), "names": set()}
    started = time.perf_counter()
    for scene in scenes:
        t = play(scene, args.auto)
        for k in ("turns", "tools", "ms"):
            grand[k] += t.get(k, 0)
        grand["rules"] |= t.get("rules", set())
        grand["names"] |= t.get("names", set())

    print(f"\n{BOLD}{'─' * 90}{OFF}")
    all_tools = ["locate_centre", "search_slots", "book_appointment",
                 "reschedule_appointment", "cancel_appointment", "my_appointments",
                 "my_tickets", "create_ticket", "knowledge_lookup", "verify_identity",
                 "record_customer_fact", "update_customer_profile", "schedule_callback",
                 "record_do_not_contact"]
    used = [t for t in all_tools if t in grand["names"]]
    print(f"{BOLD}  {grand['turns']} turns · {grand['tools']} tool executions · "
          f"{len(used)}/{len(all_tools)} tools · "
          f"{len(grand['rules'])} distinct rules fired · "
          f"{int(time.perf_counter() - started)}s{OFF}\n")
    print(f"  {GREEN}used{OFF}     " + ", ".join(used))
    unused = [t for t in all_tools if t not in grand["names"]]
    if unused:
        print(f"  {DIM}not used{OFF} " + ", ".join(unused))
    print()
    print(wrap("Every one of those turns wrote a decision trace: what it understood, which "
               "tools were considered, which rules were evaluated and why, what it ran, and "
               "what it said. Open any of them in Call Logs.", "  "))
    print(f"{BOLD}{'─' * 90}{OFF}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())