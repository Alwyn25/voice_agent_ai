#!/usr/bin/env python3
"""Replay the SLOW tool-calling turns from the golden run and show WHERE the time goes.

    docker compose exec core python scripts/replay_outliers.py
    docker compose exec core python scripts/replay_outliers.py --runs 5
    docker compose exec core python scripts/replay_outliers.py --only reschedule
    docker compose exec core python scripts/replay_outliers.py --baseline

Why this script exists
----------------------
The golden eval (golden_eval_final.xlsx) showed that a handful of tool-calling turns cost
2-3x a normal turn. The investigation found the cost is NOT the tool: `tool_ms` is ~11ms
median. It is the number of *sequential model calls* a tool turn makes:

    understand -> propose -> [corrective retry?] -> (understand -> propose) x N -> responder

The re-plan chain (turn_runner: MAX_TOOL_STEPS=3) and the confirm-without-tool corrective
retry stack extra planner calls, and each is a full round-trip to the provider. On the
OpenAI golden run the fired turns averaged ~2.9x the single-plan baseline in planner_ms.

The xlsx exported planner_ms/responder_ms/tool_ms/wall_ms but DROPPED two fields that
prove the cause: `planner_calls` (how many planner round-trips) and `stages` (context /
understand+retrieve / propose split). This script replays each outlier live and prints
BOTH, next to the number recorded in the golden run, so you can compare on your own OpenAI
key and see which turns re-plan.

Reading the output
------------------
    plan / x   total planner time, and HOW MANY planner calls (x). x>1 == it re-planned or
               paid a corrective retry. This is the single biggest thing on a tool turn.
    resp       the phrasing call (main model, streamed). Big on knowledge_lookup answers.
    tool       DB work for the tools themselves. Should stay tiny.
    cache%     share of this turn's PROMPT tokens the provider served from its prefix cache
               (planner + responder + understanding, summed). The Tier-1 lever: the prompt is
               deliberately prefix-ordered (prompts.build_system_prompt), so on the 2nd+ turn
               of a call this should climb above 0 on OpenAI. 0 everywhere = caching is NOT
               firing (a stable block drifted below a varying one, or the model does not report
               cached_tokens) -- a regression to chase.
    stages     last plan() leg only: context / u+r (understand+retrieve) / propose.
    REC        what the golden OpenAI run recorded for the same utterance, for comparison.

Every run prints the LATENCY-RELEVANT CONFIG it measured (provider, models, and the four
off-by-default flags) so a before/after comparison is self-labelling -- you always know which
config produced which numbers.

Run `--baseline` first: it measures a bare model call so you can tell a "slow afternoon"
at the provider (which moves every number) apart from the structural re-plan cost.

Testing each lever (flip ONE thing in .env, re-run, diff the columns)
--------------------------------------------------------------------
    P4-1    concurrent understand UNDERSTANDING_CONCURRENT_WITH_PROPOSE=true -> watch plan AND
                                  wall drop on ~every turn: understand (~1.3s) stops being serial
                                  and hides behind propose. Detail line: understand_retrieve
                                  collapses to ~retrieval-only (~15ms) while u= stays ~1.3s (it
                                  now runs during propose). VALIDATE: no scheduling/status turn
                                  should start answering from a document.
    Tier 1  prefix caching        watch cache% (no flag -- verify it is already >0 on turn 2+).
    Tier 3  responder streaming   STREAM_RESPOND_NO_TOOL_ENABLED=true  -> watch resp/wall on the
                                  responder-dominated (knowledge_lookup) turns.
            semantic endpointing  SEMANTIC_ENDPOINTING_ENABLED=true    -> a VAD/TTFA lever, NOT
                                  in these planner/responder numbers; measure on a real call.
                                  Printed in the flag block so a run documents whether it was on.
            smart turn            SMART_TURN_ENABLED=true               -> same, VAD-side.
    P2.05   parallel precond.     PARALLEL_PRECONDITIONS_ENABLED=true   -> watch tool on the
                                  booking/reschedule turns (book_appointment precondition fan-out).
    re-plan chain                 no flag -- watch the x column and the '*' turns.
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

# A fixed "today" so relative days ("Sunday", "this evening", "push it by a week") resolve
# the same way every run. Override with --as-of. Kept in the window a seeded visit sits in.
DEFAULT_AS_OF = "2025-09-15"   # a Monday

# The outliers, lifted from golden_eval_final.xlsx (slowest fired tool turns + a couple of
# slow no-tool responders for contrast). `rec` is the golden OpenAI run's own numbers, in
# ms, so the live replay has something to be compared against. `lead` picks a seeded persona
# that actually carries the state the turn needs -- a reschedule needs someone WITH a visit
# (Aditya Verma is site_visit_scheduled); a pure knowledge turn can use any lead (Rohan).
#   Aditya Verma   -> site_visit_scheduled (has an appointment: reschedule/cancel/my-visit)
#   Priya Nair     -> site_visit_done
#   Rohan Mehta    -> new lead, clean qualify/quote/book path (knowledge, slot search)
#   Sanjay Gupta   -> Hindi speaker, out of city
OUTLIERS = [
    # ---- planner-dominated: scheduling / booking chains (the re-plan tail) ----
    dict(id="push_a_week",   dir="outbound", lead="Aditya Verma", lang="en",
         text="Something came up, push it by a week.",
         rec=dict(wall=5850, plan=5080, resp=722, tool=32)),
    dict(id="move_sun_mon",  dir="outbound", lead="Aditya Verma", lang="en",
         text="Can we move my Sunday visit to Monday evening?",
         rec=dict(wall=5511, plan=4792, resp=684, tool=22)),
    dict(id="resched_missed", dir="outbound", lead="Aditya Verma", lang="en",
         text="Reschedule the one I missed.",
         rec=dict(wall=4610, plan=3906, resp=672, tool=19)),
    dict(id="sat_instead",   dir="outbound", lead="Aditya Verma", lang="en",
         text="Is Saturday free instead?",
         rec=dict(wall=4501, plan=3781, resp=686, tool=21)),
    dict(id="show_evening",  dir="outbound", lead="Rohan Mehta", lang="en",
         text="Can someone show me around this evening?",
         rec=dict(wall=4685, plan=3969, resp=682, tool=21)),
    dict(id="setup_weekend", dir="outbound", lead="Rohan Mehta", lang="en",
         text="Sure, set something up for the weekend.",
         rec=dict(wall=4117, plan=3460, resp=621, tool=23)),
    dict(id="evening_hi",    dir="outbound", lead="Sanjay Gupta", lang="hi",
         text="aaj shaam ko koi dikha sakta hai kya",
         rec=dict(wall=4880, plan=4014, resp=840, tool=11)),
    dict(id="redo_lastweek_hi", dir="outbound", lead="Sanjay Gupta", lang="hi",
         text="pichle hafte reh gaya tha, dobara kar sakte hain",
         rec=dict(wall=4474, plan=4121, resp=311, tool=29)),

    # ---- record lookups (one tool, but slow) ----
    dict(id="my_visit_again", dir="outbound", lead="Aditya Verma", lang="en",
         text="When is my visit again?",
         rec=dict(wall=5114, plan=1128, resp=2046, tool=11)),
    dict(id="whos_meeting",  dir="outbound", lead="Aditya Verma", lang="en",
         text="Who's meeting me at the site?",
         rec=dict(wall=5057, plan=2878, resp=1195, tool=975)),
    dict(id="have_details",  dir="outbound", lead="Aditya Verma", lang="en",
         text="Do you have my details already?",
         rec=dict(wall=6188, plan=1300, resp=1558, tool=20)),
    dict(id="cancel_visit",  dir="outbound", lead="Aditya Verma", lang="en",
         text="I want to cancel my site visit.",
         rec=dict(wall=5543, plan=925, resp=1345, tool=13)),

    # ---- responder-dominated: knowledge_lookup long answers ----
    dict(id="registration_cost", dir="outbound", lead="Rohan Mehta", lang="en",
         text="How much is registration going to cost?",
         rec=dict(wall=7220, plan=1841, resp=2588, tool=284)),
    dict(id="dnd_registered", dir="outbound", lead="Rohan Mehta", lang="en",
         text="Is this registered under DND rules?",
         rec=dict(wall=6241, plan=1526, resp=2612, tool=308)),
    dict(id="dnd_registered_hi", dir="inbound", lead="Sanjay Gupta", lang="hi",
         text="ye DND rules ke under registered hai kya",
         rec=dict(wall=5057, plan=4000, resp=786, tool=257)),
    dict(id="banks_approved", dir="outbound", lead="Rohan Mehta", lang="en",
         text="Which banks have approved the project?",
         rec=dict(wall=4108, plan=2610, resp=1068, tool=421)),
    dict(id="east_3bhk_hi",  dir="outbound", lead="Rohan Mehta", lang="hi",
         text="east facing teen BHK abhi bacha hai kya",
         rec=dict(wall=6414, plan=1582, resp=1690, tool=25)),

    # ---- no-tool responders, for contrast (should NOT re-plan; x should be 1) ----
    dict(id="budget_crore",  dir="outbound", lead="Rohan Mehta", lang="en",
         text="My budget is under a crore, is that possible here?",
         rec=dict(wall=6071, plan=1655, resp=682, tool=8)),
    dict(id="paying_250",    dir="outbound", lead="Rohan Mehta", lang="en",
         text="What am I paying 250 a foot for?",
         rec=dict(wall=5359, plan=1254, resp=1620, tool=662)),
    dict(id="modular_kitchen", dir="outbound", lead="Rohan Mehta", lang="en",
         text="Is the kitchen modular?",
         rec=dict(wall=4037, plan=1266, resp=1490, tool=25)),
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
    try:
        items = call("GET", f"/customers?search={urllib.parse.quote(name)}")["items"]
    except Exception:                                            # noqa: BLE001
        return None
    for c in items:
        if (c.get("full_name") or "").lower() == name.lower():
            return c
    return items[0] if items else None


def baseline(runs: int) -> None:
    """A bare model call, no engine -- the floor every number sits on.

    Two runs an hour apart can differ by 800ms purely because the API had a slower
    afternoon. Measure the floor in the SAME run as the replay, or the comparison to the
    golden numbers means nothing.
    """
    import asyncio
    import statistics as st
    import time as _t

    from app.config import get_settings
    from app.providers.llm.base import LLMError
    from app.providers.llm.router import get_llm

    cfg = get_settings()
    llm = get_llm()
    filler = ("You are a phone agent. Reply with JSON only: {\"ok\": true}\n"
              + "Background you must consider but need not use.\n" * 120)

    print(f"\nAPI BASELINE -- bare model call, no engine   (provider={cfg.llm_provider})\n")
    print(f"  {'tier':26} {'small prompt':>14} {'planner-sized':>15}")
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
                    print(f"  {label:26}  failed: {e}")
                    times = []
                    break
            row.append(int(st.median(times)) if times else 0)
        print(f"  {label:26} {row[0]:>14} {row[1]:>15}")
    print("\n  This is the floor. A tool turn pays this once PER planner call (the x column).")


def print_flags() -> None:
    """Label the run with the exact latency-relevant config it measured, so before/after
    comparisons are self-documenting. Read straight from the same get_settings() the engine
    uses, so what prints here is what actually ran this turn -- not what a .env claims."""
    from app.config import get_settings
    cfg = get_settings()
    # Each provider now owns its own *_model_main/*_model_fast setting -- print whichever
    # pair actually applies to the active LLM_PROVIDER.
    _main, _fast = {
        "cerebras": (cfg.cerebras_model_main, cfg.cerebras_model_fast),
        "groq": (cfg.groq_model_main, cfg.groq_model_fast),
    }.get(cfg.llm_provider, (cfg.openai_model_main, cfg.openai_model_fast))
    print("\n  CONFIG UNDER TEST (what these numbers were produced with)")
    print(f"    provider={cfg.llm_provider}  main={_main}  fast={_fast}")
    # The off-by-default latency flags, each mapped to which column it should move.
    flags = [
        ("understanding_concurrent_with_propose", "plan/wall on ALL turns (u hidden behind propose)"),
        ("stream_respond_no_tool_enabled", "resp/wall on knowledge turns"),
        ("semantic_endpointing_enabled", "VAD/TTFA -- not in these numbers"),
        ("smart_turn_enabled", "VAD/TTFA -- not in these numbers"),
        ("parallel_preconditions_enabled", "tool on booking turns"),
        ("filler_audio_cache_enabled", "TTFA on filler/ack lines"),
    ]
    for name, moves in flags:
        val = getattr(cfg, name, None)
        mark = "ON " if val else "off"
        print(f"    [{mark}] {name:35} -> {moves}")
    print("    (flip ONE in .env, re-run, diff the column named above.)")


def one_turn(spec, as_of, enforce, skip_identity=False):
    lead = find_lead(spec["lead"])
    if lead is None:
        return {"error": f"lead {spec['lead']!r} not found -- run scripts/seed_dsr.py"}
    start = {"customer_id": lead["id"], "channel": "chat", "direction": spec["dir"],
             "as_of_date": as_of}
    if spec["dir"] == "outbound":
        start["enforce_outbound_policy"] = enforce
    try:
        session = call("POST", "/sessions", start)
    except urllib.error.HTTPError as e:
        return {"error": f"session start {e.code}: {e.read().decode()[:160]}"}
    sid = session["session_id"]
    # On outbound the agent speaks first with "am I speaking with X?" -- so an unwarmed first
    # turn is consumed as the IDENTITY reply and the intended tool never runs (the golden rows
    # had identity already established). Clear the gate with an untimed confirmation turn, then
    # measure the real utterance against the tool path. Pass --raw-outbound to skip this and
    # measure the cold identity turn instead.
    if spec["dir"] == "outbound" and not skip_identity:
        first = ((lead.get("full_name") or "").split() or [""])[0]
        try:
            call("POST", f"/sessions/{sid}/turns",
                 {"text": f"Yes, speaking — this is {first}."})
        except Exception:                                        # noqa: BLE001
            pass
    try:
        turn = call("POST", f"/sessions/{sid}/turns", {"text": spec["text"]})
    except urllib.error.HTTPError as e:
        return {"error": f"turn {e.code}: {e.read().decode()[:160]}"}
    finally:
        try:
            call("POST", f"/sessions/{sid}/end", {"outcome": "profile", "analyse": "skip"})
        except Exception:                                        # noqa: BLE001
            pass
    dbg = turn.get("debug") or {}
    t = dict(dbg.get("timing") or {})
    t["retrieval_ms"] = (dbg.get("retrieval") or {}).get("latency_ms") or t.get("retrieval_ms") or 0
    t["stages"] = t.get("stages") or {}
    t["tools"] = [x.get("tool_name") if isinstance(x, dict) else x
                  for x in (turn.get("executed_tools") or [])]
    t["action"] = turn.get("action")
    t["response"] = (turn.get("response_text") or "")[:90]
    # Prefix-cache proof (Tier 1). debug.usage.tokens carries per-stage token dicts
    # (understanding / planner / responder), each now including "cached" -- how many prompt
    # tokens the provider served from its prefix cache. Sum across the stages so cache% is a
    # per-turn figure; 0 when the provider/model does not report cached_tokens.
    _tok = (dbg.get("usage") or {}).get("tokens") or {}
    def _tsum(field: str) -> int:
        return sum(int((_tok.get(stage) or {}).get(field) or 0)
                   for stage in ("understanding", "planner", "responder"))
    t["prompt_tokens"] = _tsum("prompt")
    t["cached_tokens"] = _tsum("cached")
    return t


def med(samples, key):
    vals = [s.get(key) or 0 for s in samples if isinstance(s.get(key), (int, float))]
    return int(statistics.median(vals)) if vals else 0


def main() -> int:
    global TOKEN
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=3, help="samples per outlier (median reported)")
    ap.add_argument("--only", default="", help="substring filter on id or utterance")
    ap.add_argument("--lead", default="", help="force ALL turns onto this seeded lead name")
    ap.add_argument("--as-of", default=DEFAULT_AS_OF, help="simulated 'today' (YYYY-MM-DD)")
    ap.add_argument("--enforce-outbound", action="store_true",
                    help="run outbound through the real dial gate (may 403 on DNC/off-window)")
    ap.add_argument("--raw-outbound", action="store_true",
                    help="do NOT clear the identity gate first -- measures the cold opener turn")
    ap.add_argument("--baseline", action="store_true", help="also measure the bare-model floor")
    ap.add_argument("--json", default="", help="write full per-run results to this path")
    args = ap.parse_args()

    if not wait_for_api():
        print("\nThe API never came up. Check: docker compose logs core\n")
        return 2
    TOKEN = call("POST", "/auth/login", {"email": EMAIL, "password": PASSWORD})["access_token"]

    if args.baseline:
        baseline(args.runs)

    print_flags()

    specs = OUTLIERS
    if args.only:
        q = args.only.lower()
        specs = [s for s in specs if q in s["id"].lower() or q in s["text"].lower()]
    if args.lead:
        specs = [dict(s, lead=args.lead) for s in specs]
    if not specs:
        print("No outliers matched --only", args.only)
        return 1

    print(f"\n{args.runs} run(s) per outlier, as_of={args.as_of}. Milliseconds: live median vs "
          f"golden REC.\n")
    hdr = (f"  {'id':18} {'dir':4} {'wall':>5} {'plan':>5} {'x':>2} {'resp':>5} {'tool':>5} "
           f"{'cache%':>6} | {'REC wall':>8} {'REC plan':>8} {'REC resp':>8} | tools")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    dump = []
    rows = []
    for spec in specs:
        samples = [one_turn(spec, args.as_of, args.enforce_outbound, args.raw_outbound)
                   for _ in range(args.runs)]
        errs = [s.get("error") for s in samples if s.get("error")]
        if errs:
            print(f"  {spec['id']:18} {spec['dir']:4} ERROR: {errs[0]}")
            dump.append({"id": spec["id"], "error": errs[0]})
            continue
        _prompt = med(samples, "prompt_tokens")
        _cached = med(samples, "cached_tokens")
        r = dict(id=spec["id"], dir=spec["dir"],
                 wall=med(samples, "wall_ms"), plan=med(samples, "planner_ms"),
                 calls=med(samples, "planner_calls") or 1, resp=med(samples, "responder_ms"),
                 tool=med(samples, "tool_ms"), rag=med(samples, "retrieval_ms"),
                 prompt_tokens=_prompt, cached_tokens=_cached,
                 cache_pct=(round(100 * _cached / _prompt) if _prompt else 0),
                 rec=spec["rec"], tools=samples[-1].get("tools") or [],
                 action=samples[-1].get("action"),
                 stages=samples[-1].get("stages") or {}, response=samples[-1].get("response"))
        rows.append(r)
        dump.append({"spec": spec, "samples": samples, "median": r})

    rows.sort(key=lambda r: -r["wall"])
    for r in rows:
        flag = "*" if r["calls"] > 1 else " "
        rec = r["rec"]
        _cache = f"{r['cache_pct']}%" if r["prompt_tokens"] else "-"
        print(f"  {r['id']:18} {r['dir']:4} {r['wall']:>5} {r['plan']:>5} "
              f"{r['calls']:>1}{flag} {r['resp']:>5} {r['tool']:>5} {_cache:>6} "
              f"| {rec['wall']:>8} {rec['plan']:>8} {rec['resp']:>8} "
              f"| {','.join(r['tools']) or '-'}")

    # Per-turn detail: the last-leg stage split + what it actually did.
    #
    # understand_retrieve is the CONCURRENT leg = max(understanding, retrieval). The (u=.. r=..)
    # breakdown after it tells you WHICH half dominates -- decisive for the provider split:
    # `understanding` runs on the CHAT provider (Cerebras in prod) but `retrieval`'s embedding
    # has no Cerebras API and stays on OpenAI (config.py). So if understanding dominates here on
    # OpenAI, that cost collapses on Cerebras (retrieval hides it); if retrieval dominates, it is
    # a real production cost you keep paying regardless of the chat provider.
    print("\n  detail (last run) -- stage split of the FINAL plan() leg only")
    print("     (understand_retrieve = max(u,r) run concurrently; u=understanding r=retrieval):")
    for r in rows:
        st = r["stages"]
        sstr = "  ".join(f"{k}={int(v)}" for k, v in st.items()
                         if k in ("context", "understand_retrieve", "propose", "rules_pass1"))
        _u, _rg = int(st.get("understanding") or 0), int(st.get("retrieval") or 0)
        _split = f"  (u={_u} r={_rg})" if (_u or _rg) else ""
        print(f"  * {r['id']:18} calls={r['calls']} action={r['action']}  {sstr}{_split}")
        print(f"      say: {r['response']!r}")

    plan_heavy = [r for r in rows if r["plan"] >= r["resp"]]
    resp_heavy = [r for r in rows if r["resp"] > r["plan"]]
    reran = [r for r in rows if r["calls"] > 1]
    print(f"\n  {len(rows)} turns  |  planner-dominated: {len(plan_heavy)}  "
          f"responder-dominated: {len(resp_heavy)}  |  re-planned (x>1): {len(reran)}")
    print("  '*' next to x marks a turn that made more than one planner call -- the re-plan "
          "chain\n  or a corrective retry. That is the latency the tool turns pay and the "
          "no-tool ones do not.")
    # Tier-1 caching verdict for the whole run: aggregate cached/prompt across every turn that
    # reported token usage, so "is prefix caching actually firing?" gets one honest number.
    _tp = sum(r["prompt_tokens"] for r in rows)
    _tc = sum(r["cached_tokens"] for r in rows)
    if _tp:
        _pct = round(100 * _tc / _tp)
        _verdict = ("caching IS firing" if _pct > 0
                    else "caching NOT firing (0 cached) -- check prompt-prefix ordering / model")
        print(f"  prefix cache: {_tc}/{_tp} prompt tokens served from cache ({_pct}%) -- {_verdict}")
    else:
        print("  prefix cache: no token usage reported (mock provider, or usage disabled) -- "
              "cache% is '-'")

    if args.json:
        Path(args.json).write_text(json.dumps(dump, indent=2, default=str))
        print(f"\n  full results -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
