#!/usr/bin/env python3
"""RAG retrieval-accuracy evaluation, scored with RAGAS.

Needs the eval-only dependencies, deliberately not part of the running app's image
(see requirements-eval.txt for why) -- install and run inside a throwaway container so
the persistent core-1 container's pinned versions are never touched:

    docker compose run --rm core sh -c \\
        "pip install -q -r requirements-eval.txt && python scripts/eval_retrieval_ragas.py"

    # cost/time-bounded first run:
    ... python scripts/eval_retrieval_ragas.py --limit 10

Scores each "answerable" question in rag_eval_dataset.py against what
app.engine.rag.retrieval.retrieve() actually returns for it -- the same hybrid search a
live turn runs, not a mock -- using two RAGAS metrics:

  context_precision  -- of what was retrieved, how much is actually relevant to the
                        question (a low score means retrieval is dragging in noise)
  context_recall     -- of what the ground-truth answer needed, how much made it into
                        the retrieved context (a low score means the right chunk is
                        missing, not just outranked)

Both are LLM-judged against OPENAI_MODEL_MAIN (this script talks to OpenAI directly, not
through the app's provider chain), so every question costs real LLM calls. --limit exists
to bound that.

"trap" and "known_conflict" questions are listed but not scored this way -- see
rag_eval_dataset.py's module docstring for why.
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db.base import get_session_factory  # noqa: E402
from app.db.models import Tenant  # noqa: E402
from app.engine.rag import retrieval  # noqa: E402
from rag_eval_dataset import DATASET  # noqa: E402


async def main(limit: int | None, top_k: int | None) -> int:
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextPrecisionWithReference, ContextRecall

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url or None)
    judge = llm_factory(model=settings.openai_model_main, provider="openai", client=client)
    precision_metric = ContextPrecisionWithReference(llm=judge)
    recall_metric = ContextRecall(llm=judge)

    answerable = [d for d in DATASET if d["kind"] == "answerable"]
    traps = [d for d in DATASET if d["kind"] == "trap"]
    conflicts = [d for d in DATASET if d["kind"] == "known_conflict"]
    if limit:
        answerable = answerable[:limit]

    async with get_session_factory()() as db:
        tenant = (await db.execute(select(Tenant))).scalars().first()
        if tenant is None:
            print("  No tenant. Run seed.py first.\n")
            return 1

        print(f"\n  Scoring {len(answerable)} answerable question(s) "
              f"(top_k={top_k or settings.retrieval_top_k}) against tenant {tenant.name!r}\n")

        rows = []
        for i, item in enumerate(answerable, start=1):
            res = await retrieval.retrieve(db, tenant_id=tenant.id, query=item["question"],
                                           top_k=top_k)
            contexts = [c.content for c in res.chunks]
            label = f"  [{i}/{len(answerable)}] {item['question'][:58]:58}"
            if not contexts:
                rows.append({**item, "precision": 0.0, "recall": 0.0, "chunks": 0})
                print(f"{label} NOTHING RETRIEVED")
                continue
            try:
                p = await precision_metric.ascore(user_input=item["question"],
                                                  reference=item["ground_truth"],
                                                  retrieved_contexts=contexts)
                r = await recall_metric.ascore(user_input=item["question"],
                                               retrieved_contexts=contexts,
                                               reference=item["ground_truth"])
            except Exception as e:  # a judge-call failure must not lose the whole run
                rows.append({**item, "precision": None, "recall": None, "chunks": len(contexts)})
                print(f"{label} JUDGE ERROR: {e.__class__.__name__}: {e}")
                continue
            rows.append({**item, "precision": p.value, "recall": r.value, "chunks": len(contexts)})
            print(f"{label} precision={p.value:.2f}  recall={r.value:.2f}")

    print()
    _report(rows, traps, conflicts)
    return 0


def _report(rows: list[dict], traps: list[dict], conflicts: list[dict]) -> None:
    scored = [r for r in rows if r["precision"] is not None]
    n = len(scored)
    avg_p = sum(r["precision"] for r in scored) / n if n else 0.0
    avg_r = sum(r["recall"] for r in scored) / n if n else 0.0
    print(f"  {'=' * 74}")
    print(f"  ANSWERABLE ({n}/{len(rows)} scored): "
          f"avg context_precision={avg_p:.3f}   avg context_recall={avg_r:.3f}")

    weak = [r for r in scored if r["precision"] < 0.5 or r["recall"] < 0.5]
    if weak:
        print(f"\n  {len(weak)} question(s) scored below 0.5 on either metric "
              f"-- worth a manual look:\n")
        for r in weak:
            print(f"    [{r['precision']:.2f} / {r['recall']:.2f}]  {r['question']}")
            print(f"      expected source: {r['source']}")

    if traps:
        print(f"\n  {'=' * 74}")
        print(f"  TRAP questions ({len(traps)}) -- deliberate absence, not scored:")
        for t in traps:
            print(f"    - {t['question']}")

    if conflicts:
        print(f"\n  {'=' * 74}")
        print(f"  KNOWN CONFLICTS ({len(conflicts)}) -- excluded, ground truth "
              f"disagrees with the seeded knowledge base:")
        for c in conflicts:
            print(f"    - {c['question']}")
            print(f"      {c['source']}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Score only the first N answerable questions (cost/time control).")
    ap.add_argument("--top-k", type=int, default=None,
                    help="Override retrieval top_k (defaults to the app's configured value).")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args.limit, args.top_k)))
