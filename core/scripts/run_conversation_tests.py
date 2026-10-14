#!/usr/bin/env python3
"""Run the conversation tests against the live engine.

    docker compose exec core python scripts/run_conversation_tests.py
    docker compose exec core python scripts/run_conversation_tests.py --only evening,cancel
    docker compose exec core python scripts/run_conversation_tests.py --explore
    docker compose exec core python scripts/run_conversation_tests.py --explore --goals 3

Two modes.

SCRIPTED is the regression gate. Fixed customer lines, assertions about behaviour rather
than phrasing: which tool ran, which must not have, what state resulted, whether consent
was taken. Deterministic enough to block a release on.

EXPLORE (--explore) puts an LLM in the customer's chair with a goal and a temperament, and
lets it talk until it gets what it came for or runs out of patience. It finds what scripts
cannot — the drift into unrelated tools, the "ok / ok / ok" loop, the reply that answers a
different question. It is NOT pass/fail: it reports what happened and flags the same
behavioural faults the scripted mode asserts. Run it nightly, read it yourself.

Both talk to the HTTP API exactly as the portal does, so they exercise auth, rules and
serialisation too — not just the engine in isolation. Stdlib only, like the other scripts.
"""
import argparse
import urllib.parse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from conversations import CONVERSATIONS, REFLEXES  # noqa: E402

BASE = "http://127.0.0.1:8000"
EMAIL = "rahul@advora.ai"
PASSWORD = "rahul12345"
TOKEN = None

GREEN, RED, YELLOW, DIM, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def call(method, path, payload=None, timeout=120):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(payload).encode() if payload is not None else None)
    req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            body = res.read().decode()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:300]
        raise RuntimeError(f"{method} {path} -> {e.code} {detail}") from e


def login():
    global TOKEN
    TOKEN = call("POST", "/auth/login", {"email": EMAIL, "password": PASSWORD})["access_token"]


def find_lead(name):
    for c in call("GET", f"/customers?search={urllib.parse.quote(name)}")["items"]:
        if c["full_name"].lower() == name.lower():
            return c
    return None


def tools_of(turn):
    """Only tools that actually did something. A blocked tool did not run."""
    return [t["tool_name"] for t in (turn.get("executed_tools") or [])
            if t.get("status") in ("success", "skipped_duplicate")]


def reflex_for(reply):
    low = (reply or "").lower()
    for r in REFLEXES:
        if re.search(r["if_reply_matches"], low):
            return r["say"]
    return None


# --------------------------------------------------------------------- scripted

def check_turn(spec, turn, ran):
    """Every assertion this turn failed, as readable lines."""
    bad = []
    reply = (turn.get("response_text") or "").lower()

    if spec.get("expect_tool") and spec["expect_tool"] not in ran:
        bad.append(f"expected {spec['expect_tool']} to run; ran {ran or 'nothing'}")

    if spec.get("expect_any") and not any(t in ran for t in spec["expect_any"]):
        bad.append(f"expected one of {spec['expect_any']}; ran {ran or 'nothing'}")

    for forbidden in spec.get("forbid_tools", []):
        if forbidden in ran:
            bad.append(f"{forbidden} MUST NOT run here")

    for needle in spec.get("reply_has", []):
        if needle.lower() not in reply:
            bad.append(f"reply is missing {needle!r}")

    for needle in spec.get("reply_lacks", []):
        if needle.lower() in reply:
            bad.append(f"reply contains {needle!r}")

    if spec.get("expect_state") and turn.get("state_changed_to") != spec["expect_state"]:
        bad.append(f"state is {turn.get('state_changed_to')!r}, expected {spec['expect_state']!r}")

    if spec.get("expect_action") and turn.get("action") != spec["expect_action"]:
        bad.append(f"action is {turn.get('action')!r}, expected {spec['expect_action']!r}")

    if spec.get("pending"):
        asked = bool(re.search(r"shall i|would you like|want me to|do you want me to|"
                               r"should i|confirm|proceed with|how does that (?:work|sound)|"
                               r"does that (?:work|suit)|is that ok", reply))
        if not (turn.get("pending_confirmation") or asked or turn.get("action") == "confirm"):
            bad.append("expected consent to be sought before acting; it was not")

    if spec.get("expect_knowledge") and not (turn.get("debug") or {}).get("knowledge_used"):
        bad.append("expected knowledge retrieval; nothing was retrieved")

    took = ((turn.get("debug") or {}).get("timing") or {}).get("wall_ms")
    if spec.get("max_ms") and took and took > spec["max_ms"]:
        bad.append(f"took {took} ms, budget {spec['max_ms']} ms")

    return bad


def run_scripted(convo):
    # An anonymous scenario (lead is None/empty) opens a session with NO customer_id — a
    # genuinely unidentified caller, the way an inbound call with no matching number arrives.
    # This is what exercises the "must not invent a booking for an unknown caller" guard.
    if convo.get("lead"):
        lead = find_lead(convo["lead"])
        if lead is None:
            return [f"lead {convo['lead']!r} not found — reseed the demo"], []
        body = {"customer_id": lead["id"], "channel": "chat", "direction": "inbound"}
    else:
        body = {"channel": "chat", "direction": "inbound"}
        # An unrecognised caller whose NUMBER we still have — the normal inbound case, where
        # the carrier hands us a caller ID that matches no customer on file. The API stores
        # it as session["caller_phone"], and registration must then ask only for a name.
        # Without a way to express this the harness could only ever test the anonymous
        # console caller, which is why the "never ask for a number you already have"
        # regression went unnoticed.
        if convo.get("caller_phone"):
            body["phone"] = convo["caller_phone"]

    session = call("POST", "/sessions", body)
    sid = session["session_id"]
    failures, log, ran_anywhere = [], [], []

    consumed: set[int] = set()
    for idx, spec in enumerate(convo["turns"]):
        if idx in consumed:
            continue                      # a reflex already sent this line
        said = spec["say"]
        turn = call("POST", f"/sessions/{sid}/turns", {"text": said})
        ran = tools_of(turn)
        ran_anywhere += ran
        log.append((said, turn.get("response_text"), ran,
                    (turn.get("debug") or {}).get("triage")))

        for line in check_turn(spec, turn, ran):
            failures.append(f"  '{said}' -> {line}")

        # Answer whatever the agent asked, wherever it asked it. Without this, moving the
        # identity question one turn earlier — an improvement — would fail the test.
        # A scenario about what the caller says IN REPLY to a read-back ("what time?" to
        # "shall I book?") must not have the reflex answer "yes please" for them first.
        for _ in range(0 if convo.get("no_reflex") else 2):
            follow = reflex_for(turn.get("response_text"))
            if not follow or turn.get("end_call"):
                break
            if follow == said:
                break
            # Do not let a reflex duplicate the line the script is about to send. The next
            # line is the next one not already consumed, which may be several ahead.
            nxt = next((j for j in range(idx + 1, len(convo["turns"]))
                        if j not in consumed), None)
            if nxt is not None and \
                    convo["turns"][nxt]["say"].strip().lower() == follow.strip().lower():
                consumed.add(nxt)
            turn = call("POST", f"/sessions/{sid}/turns", {"text": follow})
            ran = tools_of(turn)
            ran_anywhere += ran
            log.append((f"{follow}  {DIM}(reflex){OFF}", turn.get("response_text"), ran,
                        (turn.get("debug") or {}).get("triage")))
            said = follow

    want_state = convo.get("expect_customer_state")
    if want_state:
        fresh = call("GET", f"/customers/{lead['id']}")
        actual = (fresh or {}).get("current_state")
        if actual != want_state:
            failures.append(f"  customer ended in state {actual!r}, expected {want_state!r}")

    for want in convo.get("must_run_somewhere", []):
        if want not in ran_anywhere:
            failures.append(f"  {want} never ran in this conversation (ran: {ran_anywhere})")
    for never in convo.get("forbid_anywhere", []):
        if never in ran_anywhere:
            failures.append(f"  {never} ran and must not have")

    try:
        call("POST", f"/sessions/{sid}/end", {"outcome": "test", "analyse": "skip"})
    except Exception:                                            # noqa: BLE001
        pass
    return failures, log


# ---------------------------------------------------------------------- explore

GOALS = [
    {"lead": "Aditya Verma", "temperament": "polite but in a hurry",
     "goal": "find out when your site visit is, then move it to any afternoon later this week"},
    {"lead": "Rohan Mehta", "temperament": "curious, asks follow-up questions",
     "goal": "find out what configurations and prices are available, then book a visit"},
    {"lead": "Nisha Pillai", "temperament": "vague about timing, says 'ok' a lot",
     "goal": "book a site visit, but you only ever want evening times"},
    {"lead": "Vikram Singh", "temperament": "impatient and increasingly annoyed",
     "goal": "get a discount on a 3 BHK; if refused, demand a human"},
    {"lead": "Meera Iyer", "temperament": "blunt",
     "goal": "get taken off their calling list for good"},
]

CUSTOMER_SYSTEM = """You are role-playing a CUSTOMER on a phone call to a property developer.

You are {name}. You are {temperament}.
What you want: {goal}

Rules:
- Reply with ONE short line, the way someone speaks on the phone. No stage directions.
- Never break character and never mention that this is a test.
- If the agent asks for the last four digits of your mobile, they are {digits}.
- If it has clearly done what you wanted, say a brief thanks and nothing more.
- If it stalls, repeats itself, or answers something you did not ask, react like a real
  person would — press it, or get shorter and more irritated.

Reply as JSON only: {{"say": "...", "satisfied": true|false, "annoyed": true|false}}"""

# Behavioural faults worth flagging in a transcript no human has read yet.
SMELLS = [
    (r"let me (check|find|look)|i'll (check|find|look)|get back to you", "promised an action"),
    (r"haven't been able to complete|have not been able to complete", "canned failure reply"),
    (r"anything else (i can help|you'd like)", "closing phrase used as a way out"),
    (r"i'm having trouble on my side", "LLM error fallback"),
]


def run_explore(spec, max_turns=14):
    from app.providers.llm.base import LLMError
    from app.providers.llm.router import get_llm
    import asyncio

    lead = find_lead(spec["lead"])
    if lead is None:
        print(f"  lead {spec['lead']!r} not found")
        return

    digits = (lead.get("phone") or "")[-4:]
    system = CUSTOMER_SYSTEM.format(name=spec["lead"], temperament=spec["temperament"],
                                    goal=spec["goal"], digits=digits)
    session = call("POST", "/sessions", {"customer_id": lead["id"],
                                         "channel": "chat", "direction": "inbound"})
    sid = session["session_id"]
    history, flags, all_tools = [], [], []
    reply = session.get("opening_line") or "Hello, how can I help?"
    print(f"  {DIM}agent{OFF}  {reply}")

    llm = get_llm()
    for _ in range(max_turns):
        try:
            out = asyncio.run(llm.complete_json(
                system=system,
                user="Conversation so far:\n" + "\n".join(history[-12:]) +
                     f"\nAgent just said: {reply}\nYour next line?",
                temperature=0.8, max_tokens=120)).data
        except LLMError as e:
            print(f"  {RED}customer model failed: {e}{OFF}")
            break

        said = (out.get("say") or "").strip()
        if not said:
            break
        print(f"  {DIM}lead {OFF} {said}")
        history.append(f"Customer: {said}")

        turn = call("POST", f"/sessions/{sid}/turns", {"text": said})
        reply = turn.get("response_text") or ""
        ran = tools_of(turn)
        all_tools += ran
        history.append(f"Agent: {reply}")
        took = ((turn.get("debug") or {}).get("timing") or {}).get("wall_ms")
        print(f"  {DIM}agent{OFF}  {reply}   {DIM}[{', '.join(ran) or 'no tool'} · {took} ms]{OFF}")

        for pattern, label in SMELLS:
            if re.search(pattern, reply.lower()):
                flags.append(f"{label}: {reply[:70]}")

        if turn.get("end_call") or turn.get("escalation"):
            print(f"  {YELLOW}-- {'escalated' if turn.get('escalation') else 'call ended'} --{OFF}")
            break
        if out.get("satisfied"):
            print(f"  {GREEN}-- customer got what they came for --{OFF}")
            break

    if flags:
        print(f"  {YELLOW}worth a look:{OFF}")
        for f in dict.fromkeys(flags):
            print(f"    - {f}")
    try:
        call("POST", f"/sessions/{sid}/end", {"outcome": "test", "analyse": "skip"})
    except Exception:                                            # noqa: BLE001
        pass


# ------------------------------------------------------------------------- main

def preflight():
    """One real turn before the suite starts.

    The first run of this script pointed at /sessions/{id}/turn instead of /turns and
    produced twenty-two identical 404s — the same wiring error reported once per test,
    which buries the one line that matters. A single probe up front turns that into one
    clear message.
    """
    lead = None
    for name in ("Aditya Verma", "Rohan Mehta"):
        lead = find_lead(name)
        if lead:
            break
    if lead is None:
        print(f"{RED}No demo leads found.{OFF} Run scripts/seed_dsr.py first.\n")
        return False

    session = call("POST", "/sessions", {"customer_id": lead["id"],
                                         "channel": "chat", "direction": "inbound"})
    sid = session["session_id"]
    try:
        turn = call("POST", f"/sessions/{sid}/turns", {"text": "hello"})
    except RuntimeError as e:
        print(f"{RED}The turn endpoint is not answering.{OFF}\n  {e}")
        print("  Check the route in app/api/sessions.py and the path in this script.\n")
        return False
    finally:
        try:
            call("POST", f"/sessions/{sid}/end", {"outcome": "preflight", "analyse": "skip"})
        except Exception:                                        # noqa: BLE001
            pass

    if not (turn.get("response_text") or "").strip():
        print(f"{RED}The engine answered with nothing.{OFF} Check scripts/check_llm.py.\n")
        return False

    # The suite MUTATES the demo data — it opts Meera out, cancels Aditya's visit, books
    # new ones. Run it twice without resetting and tests start failing for reasons that
    # have nothing to do with the code: a lead already on the do-not-contact list hits the
    # terminal dnc_state_stops_everything rule, which blocks every tool including the one
    # the test is trying to exercise. That looks exactly like a regression and is not one.
    # All eight, not two. Four separate rounds of debugging ended at leftover demo data
    # rather than at code: a lead still opted out, an appointment already cancelled, a
    # fossil callback, a lead moved out of the state its test needs. Every one of them
    # looked like a regression and cost a round trip to find.
    stale = []
    for name, want in (("Vikram Singh", "negotiation"), ("Rohan Mehta", "lead"),
                       ("Kavya Menon", "do_not_contact"), ("Priya Nair", "site_visit_done"),
                       ("Sanjay Gupta", "qualified"), ("Meera Iyer", "dormant"),
                       ("Aditya Verma", "site_visit_scheduled"), ("Nisha Pillai", "contacted")):
        lead = find_lead(name)
        if lead and lead.get("current_state") != want:
            stale.append(f"{name} is {lead.get('current_state')!r}, should be {want!r}")
    if stale:
        print(f"{RED}The demo data is left over from a previous run.{OFF}")
        for line in stale:
            print(f"  {line}")
        print("\n  docker compose exec core python scripts/reset_visits.py\n")
        return False
    took = ((turn.get("debug") or {}).get("timing") or {}).get("wall_ms")
    print(f"{DIM}engine responding, {took} ms on a trivial turn{OFF}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="substring match on test names")
    ap.add_argument("--explore", action="store_true", help="LLM plays the customer")
    ap.add_argument("--goals", type=int, default=0, help="how many explore goals to run")
    ap.add_argument("--verbose", action="store_true", help="print every turn")
    args = ap.parse_args()

    login()
    if not preflight():
        return 2

    if args.explore:
        goals = GOALS[:args.goals] if args.goals else GOALS
        print(f"\nEXPLORATORY — {len(goals)} simulated caller(s). Not pass/fail; read it.\n")
        for spec in goals:
            print(f"{spec['lead']} — {spec['goal']}")
            run_explore(spec)
            print()
        return 0

    wanted = [w.strip().lower() for w in args.only.split(",") if w.strip()]
    convos = [c for c in CONVERSATIONS
              if not wanted or any(w in c["name"].lower() for w in wanted)]

    print(f"\n{len(convos)} conversation(s)\n")
    failed = errors = 0
    for convo in convos:
        try:
            failures, log = run_scripted(convo)
        except Exception as e:                                    # noqa: BLE001
            print(f"{RED}ERROR{OFF}  {convo['name']}\n  {e}\n")
            failed += 1
            errors += 1
            if errors >= 3:
                print(f"{RED}Three conversations failed the same way — stopping.{OFF}")
                print("  This is an environment or wiring problem, not agent behaviour.\n")
                return 2
            continue

        if failures:
            failed += 1
            print(f"{RED}FAIL{OFF}   {convo['name']}")
            print(f"       {DIM}{convo['why']}{OFF}")
            for line in failures:
                print(f"{RED}{line}{OFF}")
            for _, _, _, dbg in [x for x in log if len(x) > 3]:
                if dbg:
                    print(f"       {DIM}triage: {dbg}{OFF}")
            for said, reply, ran, _dbg in log:
                print(f"       {DIM}> {said}{OFF}")
                print(f"       {DIM}  {reply}   [{', '.join(ran) or 'no tool'}]{OFF}")
            print()
        else:
            print(f"{GREEN}pass{OFF}   {convo['name']}")
            if args.verbose:
                for said, reply, ran, _dbg in log:
                    print(f"       {DIM}> {said}{OFF}")
                    print(f"       {DIM}  {reply}   [{', '.join(ran) or 'no tool'}]{OFF}")

    print(f"\n{len(convos) - failed} passed, {failed} failed\n")
    if failed:
        print("Conversations mutate the demo data. Before re-running:")
        print("  docker compose exec core python scripts/reset_visits.py\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())