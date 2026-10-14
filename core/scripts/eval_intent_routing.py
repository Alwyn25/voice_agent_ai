#!/usr/bin/env python3
"""Intent-routing accuracy: scores app.engine.planner.understanding.understand() against a
labeled utterance set (intent_eval_dataset.py).

    docker compose exec core python scripts/eval_intent_routing.py

Scores the three boolean routing flags by exact match -- see intent_eval_dataset.py's
module docstring for why these, and not the free-text intent label, are what gets scored.
Each utterance still costs one real call to the configured fast model (get_llm()), so this
needs live LLM credit same as any other script here, just a much smaller prompt than a full
turn.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine.planner.understanding import understand  # noqa: E402
from intent_eval_dataset import DATASET  # noqa: E402

FLAGS = ("needs_knowledge", "is_scheduling", "is_status_question")


async def main() -> int:
    print(f"\n  Scoring {len(DATASET)} utterance(s) against understand()\n")

    per_flag_correct = {f: 0 for f in FLAGS}
    all_correct = 0
    rows = []

    for i, item in enumerate(DATASET, start=1):
        result = await understand(item["utterance"])
        got = {f: getattr(result, f) for f in FLAGS}
        matches = {f: got[f] == item[f] for f in FLAGS}
        for f in FLAGS:
            if matches[f]:
                per_flag_correct[f] += 1
        exact = all(matches.values())
        if exact:
            all_correct += 1

        mark = "OK  " if exact else "MISS"
        print(f"  [{i}/{len(DATASET)}] {mark}  {item['utterance'][:42]:42} "
              f"intent={result.intent or '-':18} (expected ~{item['expected_intent']})")
        if not exact:
            for f in FLAGS:
                if not matches[f]:
                    print(f"           {f}: got {got[f]}, expected {item[f]}")
        rows.append({**item, "got": got, "exact": exact, "source": result.source})

    n = len(DATASET)
    print(f"\n  {'=' * 74}")
    print(f"  EXACT MATCH (all 3 flags correct): {all_correct}/{n} ({all_correct / n:.1%})")
    for f in FLAGS:
        print(f"    {f:20} {per_flag_correct[f]}/{n} ({per_flag_correct[f] / n:.1%})")

    misses = [r for r in rows if not r["exact"]]
    if misses:
        print(f"\n  {len(misses)} miss(es):")
        for r in misses:
            print(f"    - {r['utterance']}  (source={r['source']})")
    print()
    return 0 if all_correct == n else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
