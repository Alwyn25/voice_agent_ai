#!/usr/bin/env python3
"""R5.08 A/B: retrieval with chunk-metadata ranking OFF vs ON, paired per question.

Answers "did R5.08 improve retrieval precision, and what did it cost in latency?" without the
noise that makes two separate eval runs hard to compare:

  * PAIRED — every question is retrieved in BOTH modes in the same process, back to back.
  * SAME EMBEDDING — the query vector is cached after the first call (retrieval._QVEC_CACHE),
    so both modes rank from an identical vector. Separate runs re-embed, and OpenAI
    embeddings drift slightly between calls — enough to flip a near-tie on its own.
  * JUDGE ONLY WHAT CHANGED — when both modes return the same window, their precision and
    recall are equal by definition, so only the questions whose window differs are scored
    with RAGAS (context_precision / context_recall, as eval_retrieval_ragas.py). No judge
    noise enters the delta from unchanged questions, and the LLM cost is small.
  * LATENCY — each mode is timed over --repeats warm calls, alternating order.

Questions come from rag_eval_dataset.py (answerable only). That set is independent of the
golden set in rag_accuracy_golden.py, which R5.08's query analyser was tuned against — so this
is the held-out check.

Run in a throwaway container (eval-only deps; the live core-1 container is untouched):

    docker compose run --rm core sh -c \\
        "pip install -q -r requirements-eval.txt && python scripts/eval_metadata_ab.py"
    ... python scripts/eval_metadata_ab.py --mode-b filter      # compare off vs filter
    ... python scripts/eval_metadata_ab.py --no-judge           # diff + latency only, no LLM
    ... python scripts/eval_metadata_ab.py --judge groq --concurrency 2   # Groq judge (GROQ_API_KEY)

Most of the wall time is the `pip install` of requirements-eval.txt in a fresh container
(ragas pulls langchain + datasets). The judging itself runs concurrently (--concurrency).

Writes eval_results/r508/metadata_ab_<timestamp>.json (per question: both windows, intent,
lifts, scores, timings).
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db.base import get_session_factory  # noqa: E402
from app.db.models import Tenant  # noqa: E402
from app.engine.rag import retrieval  # noqa: E402
from rag_eval_dataset import DATASET  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "eval_results" / "r508"


class _TokenPacer:
    """Client-side tokens-per-minute budget for the judge.

    Free-tier Groq allows 8,000 TPM and a RAGAS judge call is ~1.5–4K tokens (the retrieved
    chunks ride along in every call), so only ~3 calls fit in a minute. RAGAS fires its calls
    in parallel; without pacing most are rejected with 429 and burn their retries (measured:
    89 of 126 requests rejected). This holds each request until its estimated tokens fit in
    a sliding 60 s window, so requests are sent at the rate the API will accept.

    Tokens are estimated from the request body (~3.5 chars/token) plus an output allowance —
    an estimate, so set the budget a little under the real limit (default 7,000 for 8,000).
    """

    def __init__(self, tpm: int, output_allowance: int = 600):
        self.tpm, self.output_allowance = tpm, output_allowance
        self._window: list[tuple[float, int]] = []     # (sent_at, estimated tokens)
        self._lock = asyncio.Lock()
        self.waited_s = 0.0

    async def __call__(self, request) -> None:          # httpx request event hook
        est = min(self.tpm, len(request.content or b"") // 3 + self.output_allowance)
        async with self._lock:
            while True:
                now = time.monotonic()
                self._window = [(t, n) for t, n in self._window if now - t < 60]
                if sum(n for _, n in self._window) + est <= self.tpm:
                    self._window.append((now, est))
                    return
                wait = 60 - (now - self._window[0][0]) + 0.05
                self.waited_s += wait
                await asyncio.sleep(wait)


async def _retrieve(db, tenant_id, question: str, mode: str):
    """One retrieval under ``mode``. The settings object is process-wide and cached, so the
    mode is switched on it directly; retrieve() reads it on every call."""
    get_settings().retrieval_metadata_mode = mode
    started = time.perf_counter()
    res = await retrieval.retrieve(db, tenant_id=tenant_id, query=question)
    return res, (time.perf_counter() - started) * 1000


async def main(args) -> int:
    settings = get_settings()
    original_mode = settings.retrieval_metadata_mode
    items = [d for d in DATASET if d["kind"] == "answerable"][: args.limit or None]

    precision_metric = recall_metric = None
    judge_label = "off"
    if not args.no_judge:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import ContextPrecisionWithReference, ContextRecall
        # Judge endpoint. Groq is OpenAI-compatible, so the same client works with its base
        # URL; keys come from settings (.env), never the command line. The judge only has to
        # be CONSISTENT between A and B — both modes are scored by the same model in the same
        # run — so any capable model gives a valid paired delta (absolute scores are not
        # comparable across judges).
        if args.judge == "groq":
            api_key, base_url = settings.groq_api_key, settings.groq_base_url
            model = args.judge_model or settings.groq_model_main
        else:
            api_key, base_url = settings.openai_api_key, settings.openai_base_url or None
            model = args.judge_model or settings.openai_model_main
        if not api_key:
            print(f"No API key for judge '{args.judge}' in settings/.env.")
            return 1
        # Generous retries: a 429 that slips past the pacer should back off and retry, not
        # lose the question.
        http_client = None
        pacer = None
        if args.judge_tpm:
            import httpx
            pacer = _TokenPacer(args.judge_tpm)
            http_client = httpx.AsyncClient(timeout=120, event_hooks={"request": [pacer]})
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=8,
                             http_client=http_client)
        judge = llm_factory(model=model, provider="openai", client=client)
        precision_metric = ContextPrecisionWithReference(llm=judge)
        recall_metric = ContextRecall(llm=judge)
        judge_label = f"{args.judge}:{model}" + (f" tpm<={args.judge_tpm}" if pacer else "")

    async def score(item, contexts):
        if not contexts:
            return 0.0, 0.0
        p, r = await asyncio.gather(
            precision_metric.ascore(user_input=item["question"], reference=item["ground_truth"],
                                    retrieved_contexts=contexts),
            recall_metric.ascore(user_input=item["question"], retrieved_contexts=contexts,
                                 reference=item["ground_truth"]))
        return p.value, r.value

    rows, times = [], {"A": [], "B": []}
    to_judge: list[tuple[dict, dict, list[str], list[str]]] = []
    mode_a, mode_b = "off", args.mode_b
    try:
        async with get_session_factory()() as db:
            tenant = (await db.execute(select(Tenant))).scalars().first()
            if tenant is None:
                print("No tenant. Run seed.py first.")
                return 1
            print(f"\n  R5.08 A/B  A={mode_a}  B={mode_b}  questions={len(items)}  "
                  f"judge={judge_label}  concurrency={args.concurrency}\n")
            print("  phase 1 — retrieval (both modes, same cached embedding)")
            for i, item in enumerate(items, 1):
                q = item["question"]
                await _retrieve(db, tenant.id, q, mode_a)          # warm the embedding cache
                for rep in range(args.repeats):                    # alternate order per repeat
                    order = (mode_a, mode_b) if rep % 2 == 0 else (mode_b, mode_a)
                    for mode in order:
                        _, ms = await _retrieve(db, tenant.id, q, mode)
                        times["A" if mode == mode_a else "B"].append(ms)
                res_a, _ = await _retrieve(db, tenant.id, q, mode_a)
                res_b, _ = await _retrieve(db, tenant.id, q, mode_b)
                ids_a = [c.chunk_id for c in res_a.chunks]
                ids_b = [c.chunk_id for c in res_b.chunks]
                row = {"question": q, "source": item["source"],
                       "intent": res_b.metadata.get("intent"),
                       "lifted_in_window": res_b.metadata.get("in_window_lifted", 0),
                       "window_a": [f"{c.title} | {c.heading}" for c in res_a.chunks],
                       "window_b": [f"{c.title} | {c.heading}" for c in res_b.chunks],
                       "same_set": set(ids_a) == set(ids_b), "same_order": ids_a == ids_b}
                tag = ("same" if row["same_order"] else
                       "reordered" if row["same_set"] else "CHANGED")
                if not row["same_set"] and not args.no_judge:
                    to_judge.append((row, item, [c.content for c in res_a.chunks],
                                     [c.content for c in res_b.chunks]))
                rows.append(row)
                print(f"  [{i:>2}/{len(items)}] {q[:56]:56} {tag}")
    finally:
        settings.retrieval_metadata_mode = original_mode

    # Phase 2 — judge only the changed windows, several questions at once. This was the slow
    # part when run serially (~18 LLM calls per changed question, one after another). Bounded
    # by --concurrency so the judge's rate limit is respected.
    if to_judge:
        print(f"\n  phase 2 — judging {len(to_judge)} changed window(s)")
        sem = asyncio.Semaphore(max(1, args.concurrency))

        async def judge_one(row, item, ctx_a, ctx_b):
            async with sem:
                try:
                    (row["a_precision"], row["a_recall"]), (row["b_precision"], row["b_recall"]) = \
                        await asyncio.gather(score(item, ctx_a), score(item, ctx_b))
                    msg = (f"precision {row['a_precision']:.2f}->{row['b_precision']:.2f}"
                           f"  recall {row['a_recall']:.2f}->{row['b_recall']:.2f}")
                except Exception as e:  # noqa: BLE001 — one judge failure must not end the run
                    row["judge_error"] = f"{e.__class__.__name__}: {e}"
                    msg = f"JUDGE ERROR {row['judge_error'][:80]}"
                print(f"    {row['question'][:56]:56} {msg}")

        await asyncio.gather(*(judge_one(*t) for t in to_judge))

    changed = [r for r in rows if not r["same_set"]]
    reordered = [r for r in rows if r["same_set"] and not r["same_order"]]
    judged = [r for r in changed if "b_precision" in r]
    n = len(rows)
    # Unchanged questions contribute a delta of exactly 0, so the per-set mean delta is the
    # judged delta summed over ALL questions — no judge call was needed to know that.
    # Questions whose judging FAILED have no known delta — they are excluded from the
    # denominator, not counted as zero (that would silently shrink the result toward "no change").
    failed = len(changed) - len(judged)
    n_scored = n - failed
    d_prec = (sum(r["b_precision"] - r["a_precision"] for r in judged) / n_scored
              if n_scored else 0.0)
    d_rec = sum(r["b_recall"] - r["a_recall"] for r in judged) / n_scored if n_scored else 0.0
    lat = {k: {"p50": round(statistics.median(v), 2),
               "mean": round(statistics.fmean(v), 2)} for k, v in times.items() if v}
    summary = {"mode_a": mode_a, "mode_b": mode_b, "questions": n,
               "window_changed": len(changed), "window_reordered_only": len(reordered),
               "judged": len(judged), "judge_failed": failed, "scored": n_scored,
               "mean_delta_context_precision": round(d_prec, 4),
               "mean_delta_context_recall": round(d_rec, 4),
               "latency_ms": lat}
    print(f"\n  {'=' * 74}")
    print(f"  window changed on {len(changed)}/{n} questions, reordered only on "
          f"{len(reordered)}, identical on {n - len(changed) - len(reordered)}")
    if judged:
        print(f"  context_precision  mean delta over {n_scored} scored: {d_prec:+.4f}")
        print(f"  context_recall     mean delta over {n_scored} scored: {d_rec:+.4f}")
        if failed:
            print(f"  ({failed} changed question(s) failed judging and are excluded)")
    for k, label in (("A", mode_a), ("B", mode_b)):
        if k in lat:
            print(f"  latency {label:6} p50={lat[k]['p50']:.2f} ms  mean={lat[k]['mean']:.2f} ms "
                  f"(warm, embedding cached — i.e. the DB + ranking cost)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"metadata_ab_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"  written: {out}\n")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode-b", choices=("boost", "filter"), default="boost",
                    help="the mode compared against 'off' (default: boost)")
    ap.add_argument("--limit", type=int, default=None, help="first N answerable questions only")
    ap.add_argument("--repeats", type=int, default=4, help="timed calls per mode per question")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip RAGAS scoring: window diff + latency only (no LLM calls)")
    ap.add_argument("--judge", choices=("openai", "groq"), default="openai",
                    help="judge endpoint; keys/base URL come from .env (default: openai)")
    ap.add_argument("--judge-model", default=None,
                    help="judge model id (default: OPENAI_MODEL_MAIN / GROQ_MODEL_MAIN)")
    ap.add_argument("--judge-tpm", type=int, default=None,
                    help="pace judge calls under this many tokens/minute (free-tier Groq: "
                         "7000). Default: no pacing")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="questions judged at once (lower it if the judge rate-limits; "
                         "Groq's free tier wants ~2)")
    sys.exit(asyncio.run(main(ap.parse_args())))
