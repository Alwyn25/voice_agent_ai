#!/usr/bin/env python3
"""End-to-end RAG answer-accuracy evaluation on a FROZEN 50-question golden set.

The questions and their grading rubric are hardcoded in rag_accuracy_golden.py (sampled once from
highland_greenz_golden_dataset.xlsx; see that module for how). Every run asks the live agent the
same 50 questions -- POST /sessions/{id}/turns, one fresh session each, so the real routing ->
retrieval -> responder path runs -- and grades against the same rubric. Run it before and after a
change and use --compare to see whether the change actually helped.

Why the number can be trusted
-----------------------------
  * Fixed questions + fixed, hand-reviewed rubric: a score change can only come from the agent.
  * The LLM judge (temperature 0, strict JSON schema, never the agent's own model) does NOT
    decide pass/fail. It only marks each rubric fact present / missing / contradicted and each
    forbidden claim made / not made. The verdict is computed in code (see verdict_for).
    ACCURACY = answered with correct KB information and said nothing false or invented:
        forbidden claim made (invented)   -> fail (forbidden_violation)
        any fact contradicted / wrong no. -> fail (incorrect)
        no core fact given at all         -> fail (no_answer / wrongful_abstention)
        otherwise                         -> PASS (correct / correct_core / correct_partial)
    An incomplete but true answer passes; how complete answers are is reported separately as
    COVERAGE (all core facts present). Wording, language and phrasing never matter.
  * Independent LLM-free numeric checks. Where the code is unambiguous -- a price/area (100+)
    credited as "present" but absent from a response that states figures in digits -- it
    overrules the judge (numeric_guard, logged as AUTO-CORRECTED). Every other disagreement is
    only flagged needs_review.
  * The judge sees the KB passages the turn retrieved: a claim the KB supports is never scored
    as false, even when the short golden reference doesn't mention it.
  * Infra errors are retried, never scored as agent failures, and excluded from the denominator;
    above --max-error-rate the run is marked INVALID.
  * Uncertainty is reported (Wilson 95% CI). --compare runs an exact McNemar test on the paired
    per-question results, so "76% -> 80%" is labelled as a real improvement or as noise.
  * Failure attribution: failed facts are checked against the chunks the turn actually
    retrieved -> retrieval_miss (never reached the model) vs generation_miss (reached it, lost).
  * Style is reported separately and does NOT affect accuracy: reply language matches the
    caller's, and the word limit from answer_constraints.
  * Questions whose golden answer contradicts the agent's own knowledge base are flagged
    (kb_conflict in rag_accuracy_golden.py) and a second accuracy excluding them is reported,
    so a stale dataset row is never silently counted as an agent failure.

Usage -- via Docker Compose (stack up: docker compose up -d). Runs in a throwaway container off
the core image so the live core-1 container is never modified; it talks to the running API at
http://core:8000, and reports land in the host's eval_results/ through the repo bind mount:

    docker compose run --rm -e EVAL_BASE_URL=http://core:8000 core sh -c \\
        "pip install -q openpyxl && python scripts/eval_rag_accuracy.py"

  Extra flags go after the script name:
    ... python scripts/eval_rag_accuracy.py --compare ../eval_results/rag_accuracy/<previous_run>"
    ... python scripts/eval_rag_accuracy.py --trials 3          # agent run-to-run variance
    ... python scripts/eval_rag_accuracy.py --variants          # also ask romanised spelling variants
    ... python scripts/eval_rag_accuracy.py --ids EN011 HI020   # debug specific questions

  (Paths inside the container are relative to /app/core, hence ../eval_results for --compare.)
  From the host instead: python core/scripts/eval_rag_accuracy.py [flags] (needs openpyxl, openai).

Env: OPENAI_API_KEY (judge; from the compose env_file / .env), EVAL_JUDGE_MODEL (default gpt-4o),
EVAL_JUDGE_API_KEY / EVAL_JUDGE_BASE_URL for another judge endpoint, EVAL_BASE_URL,
EVAL_EMAIL / EVAL_PASSWORD for the app login (default: the seeded dev account).
Exit codes: 0 ok, 1 setup failure, 2 run invalid (too many errors), 3 below --fail-under.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    from openai import OpenAI
except ImportError as e:
    sys.exit(f"Missing dependency ({e.name}). pip install openpyxl openai")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from rag_accuracy_golden import GOLDEN, GOLDEN_VERSION  # noqa: E402

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

GREEN, RED, YELLOW, DIM, BOLD, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
PASS_VERDICTS = ("correct", "correct_core", "correct_partial")
COVERAGE_VERDICTS = ("correct", "correct_core")
# Bump whenever verdict_for's pass rule changes: scores under different rules are not comparable.
SCORING_VERSION = "4-no-false-info-kb-evidence-numguard"
LANG_NAMES = {"en": "English", "hi": "Hindi (romanised)", "hinglish": "Hinglish (romanised)"}


def rubric(q: dict) -> list[dict]:
    """Flatten a golden question into numbered facts: core, then any_of items, then detail."""
    facts = [{"fact": f, "kind": "core"} for f in q["core"]]
    if q["any_of"]:
        facts += [{"fact": f, "kind": "any_of"} for f in q["any_of"]["items"]]
    return facts + [{"fact": f, "kind": "detail"} for f in q["detail"]]


# ─────────────────────────────────────────────────────────────── app client

class AppClient:
    def __init__(self, base: str, email: str, password: str, timeout: float):
        self.base, self.email, self.password, self.timeout = base.rstrip("/"), email, password, timeout
        self._token: str | None = None
        self._lock = threading.Lock()

    def _raw(self, method, path, payload=None, auth=True):
        req = urllib.request.Request(self.base + path, method=method,
                                     data=json.dumps(payload).encode() if payload is not None else None)
        req.add_header("Content-Type", "application/json")
        if auth and self._token:
            req.add_header("Authorization", f"Bearer {self._token}")
        with urllib.request.urlopen(req, timeout=self.timeout) as res:
            body = res.read().decode()
            return json.loads(body) if body else None

    def login(self):
        with self._lock:
            self._token = self._raw("POST", "/auth/login",
                                    {"email": self.email, "password": self.password}, auth=False)["access_token"]

    def call(self, method, path, payload=None, attempts=4):
        """Retry transient failures (network, 5xx, 429) with backoff; re-login once on 401."""
        relogged = False
        for i in range(attempts):
            try:
                return self._raw(method, path, payload)
            except urllib.error.HTTPError as e:
                if e.code == 401 and not relogged:
                    relogged = True
                    self.login()
                    continue
                if e.code < 500 and e.code != 429 or i == attempts - 1:
                    raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {e.read().decode()[:300]}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if i == attempts - 1:
                    raise RuntimeError(f"{method} {path} -> {e.__class__.__name__}: {e}") from e
            time.sleep(min(2 ** i, 10) + random.random())

    def ask(self, text: str) -> dict:
        """One question in a brand-new session, so no earlier turn can leak into the answer."""
        sid = self.call("POST", "/sessions", {"channel": "chat", "direction": "inbound"})["session_id"]
        t0 = time.perf_counter()
        try:
            # Not retried: a turn is not idempotent within its session. A failed turn fails the trial.
            turn = self.call("POST", f"/sessions/{sid}/turns", {"text": text}, attempts=1)
        finally:
            try:
                self.call("POST", f"/sessions/{sid}/end", {"outcome": "eval", "analyse": "skip"}, attempts=2)
            except Exception:
                pass
        turn["_client_ms"] = round((time.perf_counter() - t0) * 1000)
        turn["_session_id"] = sid
        return turn


def retrieved_passages(turn: dict) -> list[str]:
    """Every chunk the turn put in front of the model: pre-retrieval plus knowledge_lookup calls."""
    debug = turn.get("debug") or {}
    out = [c.get("content") or "" for c in (debug.get("knowledge_used") or []) if isinstance(c, dict)]
    for t in turn.get("executed_tools") or []:
        if not isinstance(t, dict) or "knowledge_lookup" not in json.dumps(t)[:400]:
            continue
        stack = [t]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "passages" and isinstance(v, list):
                        out += [p if isinstance(p, str) else json.dumps(p, ensure_ascii=False) for p in v]
                    else:
                        stack.append(v)
            elif isinstance(node, list):
                stack.extend(node)
    seen, uniq = set(), []
    for p in out:
        if p and p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


# ─────────────────────────────────────────────────────────────── judge

def _schema(name, props):
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "properties": props, "required": list(props)}}}


def _check_indices(items: list[dict], n: int):
    """Exactly one mark per numbered item. Some models number from 0 despite a 1-based list; a
    consistent 0..n-1 is unambiguous, so normalise it rather than burn retries on it."""
    idx = sorted(i["index"] for i in items)
    if idx == list(range(n)):
        for i in items:
            i["index"] += 1
    elif idx != list(range(1, n + 1)):
        raise ValueError(f"indices {idx} != 1..{n}")


GRADE_SYSTEM = """You are a strict, impartial grader for a real-estate voice agent (DSR Highland
Greenz, Bengaluru). Callers speak English, Hindi or Hinglish; Hindi is romanised (Latin script).
The agent's reply may be in any of these -- judge MEANING, never wording or language. The reply
was spoken aloud, so numbers may be words ("teen" = 3, "saade paanch lakh" = 5.5 lakh).

You get the question, the reference from the official documents, numbered FACTS and numbered
FORBIDDEN claims, and the agent's response.

For EACH fact, status:
- "present": the response conveys it with the same value/meaning. Equivalent forms count:
  digits vs words, Indian vs western grouping (1,54,32,011 = 15432011), Rs / rupees / ₹,
  sq ft / sft / square feet, lakh/crore conversions (5.5 lakh = 5,50,000), hedges like "about".
  A rounded figure counts only if correctly rounded with at least 3 significant digits
  (1.54 crore for 1,54,32,011 is present; 1.5 crore is not). A fact that contains a number is
  present only if the response states that number (in any equivalent form) -- naming the thing
  without its number is "missing". For behaviour facts ("says it does not have X", "offers to
  connect to sales") judge whether the response actually does that.
- "contradicted": the response states a DIFFERENT value, name or yes/no for the SAME thing
  (e.g. 1,210 where the fact says 1,211; "modular kitchen included" where it says not included).
  Naming a different item (e.g. "multipurpose hall" where the fact is "banquet hall") is
  "missing", not "contradicted".
- "missing": not stated and not contradicted.

For EACH forbidden claim: judge it LITERALLY. "made" = true only if the response asserts exactly
that claim (hedged assertions like "probably next year" count). A related but different statement
is not a violation (saying which towers have a configuration is not a per-tower unit count;
calling a figure the "total" is not calling it the base cost). Declining, or saying the
information is not available, is never making the claim.

"contradictions": claims in the response that directly conflict with a value or fact stated in
the reference or the facts. Information the reference does not mention -- even if you cannot
verify it -- is NOT a contradiction; the reference is not an exhaustive list of what is true.

KB PASSAGES, when given, are the knowledge base the agent answers from -- the source of truth.
A claim the passages support is never a contradiction and never a forbidden claim, even if the
reference does not mention it (e.g. an amenity the passages list). Use passages ONLY to clear a
claim; never mark a rubric fact present because the passages contain it -- only the response
counts for that.
"abstained": true only if the response declines to give the requested information at all (does
not have it, will check, hands off to sales / a callback) instead of answering.
"response_language": the language the response is in -- "en", "hi" (Hindi syntax and lexicon),
"hinglish" (Hindi matrix with substantial English), or "other".
Quote the relevant part of the response as evidence (empty string if none)."""

GRADE_FORMAT = _schema("grade", {
    "facts": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"index": {"type": "integer"},
                       "status": {"type": "string", "enum": ["present", "missing", "contradicted"]},
                       "evidence": {"type": "string"}},
        "required": ["index", "status", "evidence"]}},
    "forbidden": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"index": {"type": "integer"}, "made": {"type": "boolean"},
                       "evidence": {"type": "string"}},
        "required": ["index", "made", "evidence"]}},
    "contradictions": {"type": "array", "items": {"type": "string"}},
    "abstained": {"type": "boolean"},
    "response_language": {"type": "string", "enum": ["en", "hi", "hinglish", "other"]},
    "rationale": {"type": "string"},
})

SUPPORT_SYSTEM = """You check whether retrieved document passages contain given facts. For each
numbered fact, "supported" is true only if the passages (read together) explicitly state it with
the same value. Equivalent forms count (digits vs words, digit grouping, unit spellings,
lakh/crore conversions). List the numbers of the supporting passages."""

SUPPORT_FORMAT = _schema("support", {"facts": {"type": "array", "items": {
    "type": "object", "additionalProperties": False,
    "properties": {"index": {"type": "integer"}, "supported": {"type": "boolean"},
                   "passages": {"type": "array", "items": {"type": "integer"}}},
    "required": ["index", "supported", "passages"]}}})


class Judge:
    def __init__(self, model: str, api_key: str, base_url: str | None):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url or None, max_retries=6, timeout=120)
        # Reasoning models reject temperature/seed; everything else runs greedy for repeatability.
        self._sampling = ({} if re.match(r"^(o\d|gpt-5)", model) else {"temperature": 0, "seed": 20260924})

    def _json(self, system: str, user: str, fmt: dict, validate, attempts=3) -> dict:
        err = None
        for _ in range(attempts):
            resp = self.client.chat.completions.create(
                model=self.model, response_format=fmt, **self._sampling,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            try:
                data = json.loads(resp.choices[0].message.content or "")
                validate(data)
                return data
            except (ValueError, KeyError, TypeError) as e:
                err = e
        raise RuntimeError(f"judge returned invalid output {attempts}x: {err}")

    def grade(self, q: dict, facts: list[dict], question: str, response: str,
              passages: list[str] = ()) -> dict:
        # The grader is NOT told which facts are core / any_of / detail, so it can't lean on that.
        def ok(d):
            _check_indices(d["facts"], len(facts))
            _check_indices(d["forbidden"], len(q["forbidden"]))
        kb = "\n\n".join(f"[{i}] {p[:2000]}" for i, p in enumerate(passages[:16], 1))
        user = (f"Question ({LANG_NAMES[q['lang']]}): {question}\n"
                f"Reference: {q['reference']}\n"
                "Facts:\n" + "\n".join(f"{i}. {f['fact']}" for i, f in enumerate(facts, 1)) +
                "\nForbidden claims:\n" +
                ("\n".join(f"{i}. {f}" for i, f in enumerate(q["forbidden"], 1)) or "(none)") +
                (f"\n\nKB passages retrieved for this turn (source of truth, only to clear claims):\n{kb}"
                 if kb else "") +
                f"\n\nAgent response:\n<<<\n{response}\n>>>")
        return self._json(GRADE_SYSTEM, user, GRADE_FORMAT, ok)

    def support(self, facts: list[str], passages: list[str]) -> list[bool]:
        def ok(d):
            _check_indices(d["facts"], len(facts))
        ctx = "\n\n".join(f"[{i}] {p[:3000]}" for i, p in enumerate(passages[:24], 1))
        user = ("Facts:\n" + "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))
                + f"\n\nPassages:\n{ctx}")
        d = self._json(SUPPORT_SYSTEM, user, SUPPORT_FORMAT, ok)
        return [s["supported"] for s in sorted(d["facts"], key=lambda s: s["index"])]


# ─────────────────────────────────────────────────────────────── deterministic checks

_NUM_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
_NUM_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
                   "eighty": 80, "ninety": 90, "hundred": 100})
# Romanised Hindi ("teen clubhouses"). "do" (2) is left out: it collides with the English verb.
_NUM_WORDS.update({"ek": 1, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhe": 6,
                   "saat": 7, "aath": 8, "nau": 9, "das": 10, "sau": 100})
_MULT = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7, "cr": 1e7}


def numbers_in(text: str) -> set[float]:
    t = re.sub(r"(?<=\d),(?=\d)", "", (text or "").lower())
    vals: set[float] = set()
    # No trailing \b on the number itself: "80cm" and "1.5BHK" must still yield 80 and 1.5.
    for m in re.finditer(r"(\d+(?:\.\d+)?)(?:\s*(lakhs?|lacs?|crores?|cr)\b)?", t):
        v = float(m.group(1))
        vals.add(v)
        if m.group(2):
            vals.add(round(v * _MULT[m.group(2)], 2))
    for w, v in _NUM_WORDS.items():
        if re.search(rf"\b{w}\b", t):
            vals.add(float(v))
    return vals


def _sig_digits(x: float) -> int:
    return len(("%.15g" % abs(x)).replace(".", "").lstrip("0").rstrip("0")) or 1


def _num_match(n: float, resp: set[float]) -> bool:
    """Exact, or -- as the grading rubric allows -- n correctly rounded to the response figure's
    own precision with at least 3 significant digits (1.79 crore for 1,79,38,440). 1,211 is not a
    rounding of 1,210, so it does not match."""
    for r in resp:
        if r == n:
            return True
        sig = _sig_digits(r)
        if sig >= 3 and n and float(f"{n:.{sig}g}") == r:
            return True
    return False


def numeric_crosscheck(facts: list[dict], statuses: list[str], response: str) -> list[str]:
    """LLM-free sanity check; any disagreement with the judge is flagged, never auto-corrected.
    Only digit-bearing facts are checked (Hindi number words are left to the judge)."""
    resp = numbers_in(response)
    flags = []
    for i, (fact, status) in enumerate(zip(facts, statuses), 1):
        need = {n for n in numbers_in(fact["fact"]) if re.search(r"\d", fact["fact"])}
        if not need:
            continue
        lacking = sorted(n for n in need if not _num_match(n, resp))
        if status == "present" and lacking:
            flags.append(f"fact {i} judged present but response lacks {lacking}")
        # Single digits ("wing 2", "3 BHK") collide too easily to prove a fact was stated.
        elif status != "present" and not lacking and any(n >= 10 for n in need):
            flags.append(f"fact {i} judged {status} but all its numbers appear in the response")
    return flags


def numeric_guard(facts: list[dict], g: dict, response: str) -> list[str]:
    """Overrule the judge in the one case the code is unambiguous: it marked a fact "present"
    but a figure of 100+ in that fact (a price, an area) is nowhere in a response that does
    state figures in digits -- e.g. "3.2 lakh" credited for Rs.3,25,000. Downgrades to
    "missing" in place and returns what it changed. Small numbers and responses with no digits
    at all (numbers spoken as Hindi words) are left to the judge."""
    if not re.search(r"\d", response or ""):
        return []
    resp = numbers_in(response)
    fixed = []
    for f in g["facts"]:
        fact = facts[f["index"] - 1]["fact"]
        big = {n for n in numbers_in(fact) if n >= 100 and re.search(r"\d", fact)}
        lacking = sorted(n for n in big if not _num_match(n, resp))
        if f["status"] == "present" and lacking:
            f["status"] = "missing"
            fixed.append(f"AUTO-CORRECTED fact {f['index']} present -> missing: response lacks {lacking}")
    return fixed


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def language_ok(expected: str, got: str) -> bool:
    # hi vs hinglish is a fuzzy line even for humans (see the workbook README); either counts as
    # "replied in the caller's Hindi register". English callers must get English.
    return got == "en" if expected == "en" else got in ("hi", "hinglish")


def verdict_for(q: dict, facts: list[dict], g: dict) -> tuple[str, float]:
    """(verdict, score in [0,1]) derived from the judge's per-item marks -- never from its opinion.

    Accuracy = the agent answered from the KB and said nothing false or invented:
      forbidden_violation   made a forbidden claim (invented policy/date/price...)      FAIL
      incorrect             a wrong value / contradiction of the KB                    FAIL
      no_answer             no core fact at all (incl. "I don't know" on an answerable
                            question -> wrongful_abstention)                           FAIL
      correct               every fact, incl. optional detail                          PASS
      correct_core          every core fact (+ any_of minimum)                         PASS
      correct_partial       some core facts, some left out -- incomplete, not wrong    PASS
    Completeness is reported separately (coverage = correct + correct_core).
    """
    if any(f["made"] for f in g["forbidden"]):
        return "forbidden_violation", 0.0
    status = [f["status"] for f in sorted(g["facts"], key=lambda f: f["index"])]
    if "contradicted" in status or g["contradictions"]:
        return "incorrect", 0.0
    present = status.count("present")
    score = round(present / len(facts), 4)
    answered = any(s == "present" for s, f in zip(status, facts) if f["kind"] != "detail")
    if not answered:
        return ("wrongful_abstention" if g["abstained"] else "no_answer"), 0.0
    core_ok = all(s == "present" for s, f in zip(status, facts) if f["kind"] == "core")
    list_hits = sum(s == "present" for s, f in zip(status, facts) if f["kind"] == "any_of")
    list_ok = q["any_of"] is None or list_hits >= q["any_of"]["k"]
    if core_ok and list_ok:
        return ("correct" if present == len(facts) else "correct_core"), score
    return "correct_partial", score


# ─────────────────────────────────────────────────────────────── one evaluation

@dataclass
class Result:
    q: dict
    trial: str
    asked: str = ""
    response: str | None = None
    verdict: str | None = None
    score: float | None = None
    fact_status: list[dict] = field(default_factory=list)
    forbidden_made: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    rationale: str = ""
    abstained: bool | None = None
    response_language: str | None = None
    language_ok: bool | None = None
    words: int | None = None
    words_ok: bool | None = None
    needs_review: list[str] = field(default_factory=list)
    attribution: str | None = None
    passages: int = 0
    action: str | None = None
    tools: str | None = None
    latency_ms: int | None = None
    session_id: str | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.verdict in PASS_VERDICTS


def evaluate(app: AppClient, judge: Judge, q: dict, trial: str, text: str, check_retrieval: bool) -> Result:
    res = Result(q=q, trial=trial, asked=text)
    facts = rubric(q)
    try:
        turn = app.ask(text)
    except Exception as e:
        res.error = f"agent: {e}"[:400]
        return res
    res.response = (turn.get("response_text") or "").strip()
    res.action, res.latency_ms, res.session_id = turn.get("action"), turn.get("_client_ms"), turn.get("_session_id")
    res.tools = ", ".join(str(t.get("tool") or t.get("tool_name") or t.get("name") or "?")
                          for t in (turn.get("executed_tools") or []) if isinstance(t, dict)) or None
    passages = retrieved_passages(turn)
    res.passages = len(passages)
    res.words = word_count(res.response)
    res.words_ok = None if q["max_words"] is None else res.words <= q["max_words"]
    if not res.response:
        res.verdict, res.score = "empty_response", 0.0
        return res
    try:
        g = judge.grade(q, facts, text, res.response, passages)
        guarded = numeric_guard(facts, g, res.response)
        res.verdict, res.score = verdict_for(q, facts, g)
        res.fact_status = sorted(g["facts"], key=lambda f: f["index"])
        res.forbidden_made = [q["forbidden"][f["index"] - 1] + (f" -- \"{f['evidence']}\"" if f["evidence"] else "")
                              for f in sorted(g["forbidden"], key=lambda f: f["index"]) if f["made"]]
        res.contradictions, res.rationale, res.abstained = g["contradictions"], g["rationale"], g["abstained"]
        res.response_language = g["response_language"]
        res.language_ok = language_ok(q["lang"], g["response_language"])
        res.needs_review = guarded + numeric_crosscheck(facts, [f["status"] for f in res.fact_status],
                                                        res.response)
        # Attribution only makes sense for facts that live in the documents -- not for escalation
        # rows whose rubric is behaviour ("says it doesn't have X").
        if check_retrieval and q["source"] != "escalation" and not res.passed:
            needed = [f["fact"] for f, s in zip(facts, res.fact_status)
                      if f["kind"] != "detail" and s["status"] != "present"]
            if not needed:
                res.attribution = "forbidden_or_contradiction_only"
            else:
                supported = judge.support(needed, passages) if passages else [False] * len(needed)
                res.attribution = "generation_miss" if all(supported) else "retrieval_miss"
    except Exception as e:
        res.verdict, res.score = None, None
        res.error = f"judge: {e.__class__.__name__}: {e}"[:400]
    return res


# ─────────────────────────────────────────────────────────────── statistics

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant pairs (b: pass->fail, c: fail->pass)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.1f}%"


def per_question(results: list[Result]) -> dict[str, dict]:
    """Collapse trials/variants: a question passes if it passed in the majority of its asks."""
    groups: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        if r.error is None:
            groups[r.q["id"]].append(r)
    return {qid: {"q": rs[0].q, "pass_frac": sum(r.passed for r in rs) / len(rs),
                  "passed": sum(r.passed for r in rs) / len(rs) >= 0.5,
                  "score": sum(r.score for r in rs) / len(rs), "asks": len(rs),
                  "stable": len({r.passed for r in rs}) == 1}
            for qid, rs in groups.items()}


def summarize(results: list[Result]) -> dict:
    ok = [r for r in results if r.error is None]
    pq = per_question(results)
    k, n = sum(v["passed"] for v in pq.values()), len(pq)
    lo, hi = wilson(k, n)
    attrib, verdicts = defaultdict(int), defaultdict(int)
    for r in ok:
        verdicts[r.verdict] += 1
        if r.attribution:
            attrib[r.attribution] += 1
    lang = [r.language_ok for r in ok if r.language_ok is not None]
    words = [r.words_ok for r in ok if r.words_ok is not None]
    esc = [r for r in ok if r.q["slice"] == "escalation" or r.q["source"] == "escalation"]
    clean = {qid: v for qid, v in pq.items() if not v["q"]["kb_conflict"]}
    ck, cn = sum(v["passed"] for v in clean.values()), len(clean)
    return {
        "questions_scored": n, "questions_passed": k,
        "accuracy": k / n if n else float("nan"), "accuracy_ci95": (lo, hi),
        "clean_scored": cn, "clean_passed": ck,
        "accuracy_clean": ck / cn if cn else float("nan"), "accuracy_clean_ci95": wilson(ck, cn),
        "kb_conflicts": {qid: v["passed"] for qid, v in sorted(pq.items()) if v["q"]["kb_conflict"]},
        "partial_credit": sum(v["score"] for v in pq.values()) / n if n else float("nan"),
        "complete_rate": sum(r.verdict == "correct" for r in ok) / len(ok) if ok else float("nan"),
        "coverage_rate": sum(r.verdict in COVERAGE_VERDICTS for r in ok) / len(ok) if ok else float("nan"),
        "false_info_rate": (sum(r.verdict in ("forbidden_violation", "incorrect") for r in ok) / len(ok)
                            if ok else float("nan")),
        "no_answer_rate": (sum(r.verdict in ("no_answer", "wrongful_abstention", "empty_response") for r in ok)
                           / len(ok) if ok else float("nan")),
        "forbidden_violation_rate": sum(r.verdict == "forbidden_violation" for r in ok) / len(ok) if ok else float("nan"),
        "no_info_questions_handled": (sum(r.passed for r in esc) / len(esc)) if esc else float("nan"),
        "no_info_n": len(esc),
        "language_match_rate": sum(lang) / len(lang) if lang else float("nan"),
        "word_limit_rate": sum(words) / len(words) if words else float("nan"), "word_limit_n": len(words),
        "asks_total": len(results), "asks_scored": len(ok), "errors": len(results) - len(ok),
        "error_rate": (len(results) - len(ok)) / len(results) if results else 0.0,
        "verdicts": dict(sorted(verdicts.items())), "attribution": dict(sorted(attrib.items())),
        "unstable_questions": sorted(q for q, v in pq.items() if v["asks"] > 1 and not v["stable"]),
        "needs_review": sum(1 for r in ok if r.needs_review),
        "per_question": {qid: {"passed": v["passed"], "pass_frac": v["pass_frac"], "score": v["score"]}
                         for qid, v in pq.items()},
    }


def breakdown(results: list[Result], key) -> list[tuple]:
    groups: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        groups[key(r.q)].append(r)
    out = []
    for g, rs in sorted(groups.items()):
        pq = per_question(rs)
        k, n = sum(v["passed"] for v in pq.values()), len(pq)
        lo, hi = wilson(k, n)
        out.append((g, n, k, _pct(k / n if n else float("nan")), f"{_pct(lo)} - {_pct(hi)}",
                    _pct(sum(v["score"] for v in pq.values()) / n if n else float("nan")),
                    sum(1 for r in rs if r.error)))
    return out


def compare(current: dict, baseline_dir: Path) -> dict:
    manifest = json.loads((baseline_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("golden_version") != GOLDEN_VERSION:
        raise SystemExit(f"--compare: baseline used golden set {manifest.get('golden_version')!r}, this run "
                         f"uses {GOLDEN_VERSION!r}; scores from different question sets are not comparable.")
    if manifest.get("scoring_version") != SCORING_VERSION:
        raise SystemExit(f"--compare: baseline was scored with rule {manifest.get('scoring_version')!r}, this "
                         f"run uses {SCORING_VERSION!r}; pass/fail meant different things, not comparable.")
    if manifest.get("rubric_fingerprint") != rubric_fingerprint():
        print(f"{YELLOW}WARNING: the rubric changed since the baseline run; part of any delta may be "
              f"the rubric, not the agent.{OFF}")
    base = manifest["summary"]["per_question"]
    cur = current["per_question"]
    common = sorted(set(base) & set(cur))
    fixed = [q for q in common if not base[q]["passed"] and cur[q]["passed"]]
    broke = [q for q in common if base[q]["passed"] and not cur[q]["passed"]]
    b_acc = sum(base[q]["passed"] for q in common) / len(common) if common else float("nan")
    c_acc = sum(cur[q]["passed"] for q in common) / len(common) if common else float("nan")
    return {"baseline": str(baseline_dir), "baseline_git": manifest.get("git"), "n": len(common),
            "baseline_accuracy": b_acc, "current_accuracy": c_acc, "fixed": fixed, "regressed": broke,
            "p_value": mcnemar_exact(len(broke), len(fixed))}


# ─────────────────────────────────────────────────────────────── report

HDR_FILL, HDR_FONT = PatternFill("solid", fgColor="1E3A5F"), Font(bold=True, color="FFFFFF")
FILLS = {"pass": PatternFill("solid", fgColor="D6F5D6"), "partial": PatternFill("solid", fgColor="FFF4CC"),
         "fail": PatternFill("solid", fgColor="FFD6D6")}


def _sheet(wb, title, header, rows, widths=None):
    ws = wb.create_sheet(title)
    ws.append(header)
    for c in ws[1]:
        c.font, c.fill, c.alignment = HDR_FONT, HDR_FILL, Alignment(wrap_text=True, vertical="top")
    for r in rows:
        ws.append(list(r))
    for i, h in enumerate(header):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i + 1)].width = (widths or {}).get(h, 15)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    return ws


def _fact_lines(r: Result) -> str:
    st = {f["index"]: f["status"] for f in r.fact_status}
    return "\n".join(f"[{st.get(i, '?')}] ({f['kind']}) {f['fact']}" for i, f in enumerate(rubric(r.q), 1))


def write_report(path: Path, s: dict, results: list[Result], manifest: dict, cmp: dict | None):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    kv = [("Status", manifest["status"]),
          ("RAG accuracy: answered from KB, nothing false/invented",
           f"{_pct(s['accuracy'])}  ({s['questions_passed']}/{s['questions_scored']})"),
          ("Gave false / invented information", _pct(s["false_info_rate"])),
          ("Did not answer (no correct fact)", _pct(s["no_answer_rate"])),
          ("Coverage (all core facts present)", _pct(s["coverage_rate"])),
          ("Accuracy 95% CI (Wilson)", f"{_pct(s['accuracy_ci95'][0])} - {_pct(s['accuracy_ci95'][1])}"),
          ("Accuracy excluding golden-vs-KB conflicts",
           f"{_pct(s['accuracy_clean'])}  ({s['clean_passed']}/{s['clean_scored']}), 95% CI "
           f"{_pct(s['accuracy_clean_ci95'][0])} - {_pct(s['accuracy_clean_ci95'][1])}"),
          ("Golden-vs-KB conflict questions (passed?)",
           ", ".join(f"{q}={'pass' if p else 'fail'}" for q, p in s["kb_conflicts"].items()) or "-"),
          ("Partial-credit score", _pct(s["partial_credit"])),
          ("Complete answers (every fact incl. detail)", _pct(s["complete_rate"])),
          ("Forbidden-claim violations", _pct(s["forbidden_violation_rate"])),
          ("Not-in-source questions handled correctly", f"{_pct(s['no_info_questions_handled'])} (n={s['no_info_n']})"),
          ("Reply-language match (style, not in accuracy)", _pct(s["language_match_rate"])),
          ("Within word limit (style, not in accuracy)", f"{_pct(s['word_limit_rate'])} (n={s['word_limit_n']})"),
          ("Verdicts", json.dumps(s["verdicts"])),
          ("Failure attribution", json.dumps(s["attribution"])),
          ("Asks scored / total", f"{s['asks_scored']} / {s['asks_total']}"),
          ("Errors (excluded from score)", f"{s['errors']} ({_pct(s['error_rate'])})"),
          ("Flagged needs_review", s["needs_review"]),
          ("Unstable across trials/variants", ", ".join(s["unstable_questions"]) or "-")]
    if cmp:
        kv += [("Compared with", f"{cmp['baseline']} (git {cmp['baseline_git']})"),
               ("Accuracy baseline -> now", f"{_pct(cmp['baseline_accuracy'])} -> {_pct(cmp['current_accuracy'])}"),
               ("Fixed (fail -> pass)", ", ".join(cmp["fixed"]) or "-"),
               ("Regressed (pass -> fail)", ", ".join(cmp["regressed"]) or "-"),
               ("McNemar exact p-value", f"{cmp['p_value']:.3f}"
                + ("  (significant)" if cmp["p_value"] < 0.05 else "  (not significant: could be noise)"))]
    kv += [(k, v if isinstance(v, (str, int, float)) else json.dumps(v)) for k, v in manifest.items() if k != "status"]
    _sheet(wb, "Summary", ["Metric", "Value"], kv, {"Metric": 44, "Value": 90})
    cols = ["Group", "Questions", "Passed", "Accuracy", "95% CI", "Partial credit", "Errors"]
    _sheet(wb, "By Language", cols, breakdown(results, lambda q: q["lang"]))
    _sheet(wb, "By Slice", cols, breakdown(results, lambda q: q["slice"]))
    _sheet(wb, "By Difficulty", cols, breakdown(results, lambda q: q["difficulty"]))
    _sheet(wb, "By Intent", cols, breakdown(results, lambda q: q["intent"]))

    header = ["ID", "Lang", "Slice", "Difficulty", "Trial", "Asked", "Reference", "Response", "Verdict",
              "Score", "Rubric [judge status]", "Forbidden claims made", "Contradictions", "Judge rationale",
              "Attribution", "Passages", "Reply lang", "Lang OK", "Words", "Word limit", "Action", "Tools",
              "Latency ms", "Needs review", "KB conflict", "Error", "Session"]
    rows = [(r.q["id"], r.q["lang"], r.q["slice"], r.q["difficulty"], r.trial, r.asked, r.q["reference"],
             r.response, r.verdict, r.score, _fact_lines(r), "\n".join(r.forbidden_made),
             "\n".join(r.contradictions), r.rationale, r.attribution, r.passages, r.response_language,
             r.language_ok, r.words, r.q["max_words"], r.action, r.tools, r.latency_ms,
             "\n".join(r.needs_review), r.q["kb_conflict"], r.error, r.session_id)
            for r in sorted(results, key=lambda r: (r.q["id"], r.trial))]
    wide = {"Asked": 30, "Reference": 40, "Response": 50, "Rubric [judge status]": 55, "Judge rationale": 40,
            "Forbidden claims made": 30, "Contradictions": 30, "Needs review": 30, "KB conflict": 40,
            "Error": 30}
    ws = _sheet(wb, "Results", header, rows, wide)
    vcol = header.index("Verdict")
    for row in ws.iter_rows(min_row=2):
        v = row[vcol].value
        row[vcol].fill = (FILLS["pass"] if v in PASS_VERDICTS else FILLS["partial"] if v == "partial"
                          else FILLS["fail"])
    _sheet(wb, "Needs Review", header,
           [x for x in rows if x[header.index("Needs review")] or x[header.index("Error")]], wide)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


# ─────────────────────────────────────────────────────────────── setup helpers

def _dotenv(key: str) -> str | None:
    for p in (REPO_ROOT / ".env", REPO_ROOT / "core" / ".env"):
        if p.exists():
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                m = re.match(rf"^\s*{re.escape(key)}\s*=\s*(.*?)\s*$", line)
                if m and m.group(1):
                    return m.group(1).strip("'\"")
    return None


def _git_state() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True,
                               text=True, timeout=10).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" or "unknown"
    except Exception:
        pass
    # The core image has no git binary; the repo is bind-mounted, so read HEAD directly.
    # Dirty state can't be known this way, hence the "?" suffix.
    try:
        git = REPO_ROOT / ".git"
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return f"{head[:7]}-?"
        ref = head[5:]
        if (git / ref).exists():
            return f"{(git / ref).read_text().strip()[:7]}-?"
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return f"{line.split()[0][:7]}-?"
    except Exception:
        pass
    return "unknown"


def rubric_fingerprint() -> str:
    return hashlib.sha256(json.dumps(GOLDEN, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=1, help="ask each question N times (agent variance)")
    ap.add_argument("--variants", action="store_true",
                    help="also ask each hi/hinglish question's alternate romanised spelling")
    ap.add_argument("--ids", nargs="+", help="only these question IDs (debugging; not a comparable score)")
    ap.add_argument("--compare", type=Path, help="previous run directory to compare against")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--base-url", default=os.getenv("EVAL_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--email", default=os.getenv("EVAL_EMAIL", "rahul@advora.ai"))
    ap.add_argument("--password", default=os.getenv("EVAL_PASSWORD", "rahul12345"))
    ap.add_argument("--turn-timeout", type=float, default=180)
    ap.add_argument("--judge-model", default=os.getenv("EVAL_JUDGE_MODEL", "gpt-4o"))
    ap.add_argument("--no-retrieval-check", action="store_true", help="skip failure attribution")
    ap.add_argument("--max-error-rate", type=float, default=0.05,
                    help="above this share of errored asks the run is INVALID (exit 2)")
    ap.add_argument("--fail-under", type=float, default=None, help="exit 3 if accuracy is below this (0-1)")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "eval_results" / "rag_accuracy")
    args = ap.parse_args()

    questions = GOLDEN
    if args.ids:
        unknown = set(args.ids) - {q["id"] for q in GOLDEN}
        if unknown:
            sys.exit(f"Unknown question IDs: {sorted(unknown)}")
        questions = [q for q in GOLDEN if q["id"] in set(args.ids)]

    api_key = os.getenv("EVAL_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY") or _dotenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("No judge API key: set OPENAI_API_KEY or EVAL_JUDGE_API_KEY.")
    judge = Judge(args.judge_model, api_key, os.getenv("EVAL_JUDGE_BASE_URL"))
    agent_model = os.getenv("LLM_MODEL_MAIN") or _dotenv("LLM_MODEL_MAIN") or _dotenv("OPENAI_MODEL_MAIN")
    if agent_model and agent_model == judge.model:
        print(f"{YELLOW}WARNING: judge model == agent model ({agent_model}); self-grading inflates scores.{OFF}")

    app = AppClient(args.base_url, args.email, args.password, args.turn_timeout)
    try:
        app.login()
    except Exception as e:
        print(f"Cannot log in to {args.base_url} as {args.email}: {e}\nIs the stack up (docker compose up)?")
        return 1

    started = datetime.now(timezone.utc)
    run_dir = args.out_dir / started.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "golden_version": GOLDEN_VERSION, "rubric_fingerprint": rubric_fingerprint(),
        "scoring_version": SCORING_VERSION,
        "questions": len(questions), "subset": bool(args.ids), "trials": args.trials, "variants": args.variants,
        "judge_model": judge.model, "agent_model": agent_model or "unknown", "app_base_url": args.base_url,
        "git": _git_state(), "started_at": started.isoformat(timespec="seconds"),
    }

    jobs = []
    for q in questions:
        for t in range(1, args.trials + 1):
            jobs.append((q, f"t{t}", q["question"]))
            if args.variants and q["variant"]:
                jobs.append((q, f"t{t}-variant", q["variant"]))

    print(f"\n{BOLD}RAG accuracy eval{OFF}  {GOLDEN_VERSION}  {len(questions)} questions, {len(jobs)} asks  "
          f"judge={judge.model}  agent={agent_model}  git={manifest['git']}\n")
    results: list[Result] = []
    raw_path = run_dir / "results.jsonl"
    with ThreadPoolExecutor(args.concurrency) as ex, raw_path.open("w", encoding="utf-8") as raw:
        futs = [ex.submit(evaluate, app, judge, q, t, text, not args.no_retrieval_check) for q, t, text in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results.append(r)
            raw.write(json.dumps({**{k: v for k, v in r.__dict__.items() if k != "q"}, "id": r.q["id"],
                                  "passed": r.passed}, ensure_ascii=False) + "\n")
            raw.flush()
            tag = (f"{RED}ERROR{OFF}" if r.error else f"{GREEN}PASS {OFF}" if r.passed
                   else f"{RED}FAIL {OFF}")
            extra = r.error or (f"{r.verdict} {r.score:.2f}" + (f" [{r.attribution}]" if r.attribution else "")
                                + (f" {YELLOW}review{OFF}" if r.needs_review else ""))
            print(f"  [{i:>3}/{len(jobs)}] {tag} {r.q['id']:6} {r.trial:10} {r.asked[:46]:46}  {extra}")

    s = summarize(results)
    cmp = compare(s, args.compare) if args.compare and not args.ids else None
    invalid = s["error_rate"] > args.max_error_rate
    manifest.update(status="INVALID (too many errors)" if invalid else ("SUBSET (not comparable)" if args.ids else "OK"),
                    finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    report = run_dir / "report.xlsx"
    write_report(report, s, results, manifest, cmp)
    (run_dir / "manifest.json").write_text(json.dumps({**manifest, "summary": s, "comparison": cmp},
                                                      indent=2, default=str), encoding="utf-8")

    lo, hi = s["accuracy_ci95"]
    print(f"\n  {'=' * 78}")
    print(f"  {BOLD}RAG ACCURACY  {_pct(s['accuracy'])}{OFF}  ({s['questions_passed']}/{s['questions_scored']} questions)"
          f"   95% CI {_pct(lo)} - {_pct(hi)}   (graded against the golden dataset)")
    if s["kb_conflicts"]:
        clo, chi = s["accuracy_clean_ci95"]
        print(f"  excl. golden-vs-KB conflicts {_pct(s['accuracy_clean'])}  ({s['clean_passed']}/{s['clean_scored']})"
              f"   95% CI {_pct(clo)} - {_pct(chi)}")
        print(f"  {DIM}conflict questions (golden answer contradicts the KB -- see rag_accuracy_golden.py): "
              + ", ".join(f"{q}={'pass' if p else 'fail'}" for q, p in s["kb_conflicts"].items()) + OFF)
    print(f"  {DIM}(accuracy = answered from the KB and said nothing false or invented){OFF}")
    print(f"  gave false / invented info {_pct(s['false_info_rate'])}")
    print(f"  did not answer            {_pct(s['no_answer_rate'])}")
    print(f"  coverage                  {_pct(s['coverage_rate'])}   (all core facts present -- completeness, not accuracy)")
    print(f"  partial credit            {_pct(s['partial_credit'])}")
    print(f"  not-in-source handled     {_pct(s['no_info_questions_handled'])}  (n={s['no_info_n']})")
    print(f"  verdicts                  {s['verdicts']}")
    print(f"  failure attribution       {s['attribution']}")
    print(f"  {DIM}style (not in accuracy):  reply language {_pct(s['language_match_rate'])}, "
          f"word limit {_pct(s['word_limit_rate'])} (n={s['word_limit_n']}){OFF}")
    print(f"\n  {'group':<14}{'n':>4}{'pass':>6}{'acc':>9}   95% CI")
    for title, key in (("language", lambda q: q["lang"]), ("slice", lambda q: q["slice"]),
                       ("difficulty", lambda q: q["difficulty"])):
        for g, n, k, acc, ci, _, _ in breakdown(results, key):
            print(f"  {g:<14}{n:>4}{k:>6}{acc:>9}   {ci}")
        print()
    if cmp:
        verdict = ("significant" if cmp["p_value"] < 0.05 else "NOT significant -- could be noise")
        print(f"  {BOLD}vs baseline{OFF} ({cmp['baseline_git']}): {_pct(cmp['baseline_accuracy'])} -> "
              f"{_pct(cmp['current_accuracy'])}   McNemar p={cmp['p_value']:.3f} ({verdict})")
        print(f"    fixed:     {', '.join(cmp['fixed']) or '-'}")
        print(f"    regressed: {', '.join(cmp['regressed']) or '-'}\n")
    if s["needs_review"]:
        print(f"  {YELLOW}{s['needs_review']} result(s) flagged needs_review -- see the 'Needs Review' sheet.{OFF}")
    if s["unstable_questions"]:
        print(f"  {YELLOW}Unstable across trials/variants: {', '.join(s['unstable_questions'])}{OFF}")
    print(f"  errors: {s['errors']} ({_pct(s['error_rate'])}), excluded from the score")
    if invalid:
        print(f"\n  {RED}RUN INVALID: error rate {_pct(s['error_rate'])} > {_pct(args.max_error_rate)}. "
              f"Fix the errors before trusting this number.{OFF}")
    print(f"\n  Report: {report}\n  Raw:    {raw_path}\n")
    if invalid:
        return 2
    if args.fail_under is not None and s["accuracy"] < args.fail_under:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
