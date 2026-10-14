#!/usr/bin/env python3
"""What retrieval actually returns for a question.

    docker compose exec core python scripts/check_retrieval.py "i am looking for a flat to buy"

Runs the same hybrid search a live turn runs, and prints every chunk with its score and
which arm found it. When the agent gives a wrong answer, this separates the three causes
that look identical from the outside:

  the gate skipped the turn        -> "retrieval did not run"
  the search found nothing useful  -> chunks listed with poor scores, or none
  the search was fine              -> the right chunk is here and the model ignored it

Guessing between those has cost several rounds. One embedding call, no planner, no
conversation.
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import Tenant  # noqa: E402
from app.engine.rag import needs_knowledge, retrieval  # noqa: E402


async def main(query: str, top_k: int) -> int:
    print(f"\n  query: {query!r}\n")

    gate = needs_knowledge(query)
    print(f"  gate     {'RUNS' if gate else 'SKIPPED'} — needs_knowledge() said {gate}")
    if not gate:
        print("\n  Retrieval never happens for this turn, so the model answers from the\n"
              "  prompt alone. If that is wrong, the gate is what to change.\n")
        return 0

    async with get_session_factory()() as db:
        tenant = (await db.execute(select(Tenant))).scalars().first()
        if tenant is None:
            print("  No tenant. Run seed.py first.\n")
            return 1

        res = await retrieval.retrieve(db, tenant_id=tenant.id, query=query, top_k=top_k)

    print(f"  latency  {res.latency_ms} ms")
    if res.error:
        print(f"  error    {res.error}")
    if getattr(res, "excluded_expired", 0):
        print(f"  expired  {res.excluded_expired} chunk(s) outside their effective dates")

    if not res.chunks:
        print("\n  NOTHING RETURNED. The model had no document to answer from — anything it\n"
              "  said about this subject came from its own training.\n")
        return 0

    print(f"\n  {len(res.chunks)} chunk(s):\n")
    for c in res.chunks:
        arms = []
        if c.vector_rank is not None:
            arms.append(f"vec#{c.vector_rank}")
        if c.text_rank is not None:
            arms.append(f"text#{c.text_rank}")
        print(f"  [{c.relevance:>5.3f}] {c.title[:44]:46} {c.doc_type:9} {' '.join(arms)}")
        body = " ".join((c.content or "").split())
        print(f"           {body[:150]}{'…' if len(body) > 150 else ''}\n")

    print("  If the right document is in this list and the agent still answered wrongly,\n"
          "  the problem is the prompt or the history, not retrieval.\n")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()
    sys.exit(asyncio.run(main(" ".join(args.query), args.top_k)))