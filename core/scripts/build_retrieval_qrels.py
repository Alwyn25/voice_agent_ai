#!/usr/bin/env python3
"""Draft a retrieval evaluation set (questions + evidence anchors) for ANY tenant's corpus.

WHY. eval_parent_child.py --gate needs labelled questions: for each, the evidence that must reach
the model. The frozen DSR set (retrieval_qrels.py) only covers DSR. A new tenant, or a corpus
arriving from an ingestion service, needs its own set before R5.03 (or any retrieval change) can
be enabled for it with evidence rather than hope.

HOW. Samples chunks from the tenant's KB (spread across documents), and for each asks an LLM for
one realistic caller question the passage answers plus 1-3 SHORT VERBATIM phrases from the passage
that carry the answer. The LLM never decides what is true: every anchor is checked to be a literal
substring of its source chunk (same normalisation the eval uses), and invalid anchors are dropped —
an item with none left is discarded. Anchors are matched by CONTENT, so the set survives re-chunking
(parent-child, re-ingest) exactly like the hand-written DSR labels.

A GENERATED SET IS A DRAFT. The file is written with "reviewed": false and the eval warns until a
human has read it (fix or delete weak questions, then set "reviewed": true). Budget: ~600 tokens per
question with gpt-4o-mini — 40 questions is well under $0.01. Dry run by default.

    docker compose exec core python scripts/build_retrieval_qrels.py --tenant-id <uuid>            # plan + cost
    docker compose exec core python scripts/build_retrieval_qrels.py --tenant-id <uuid> --apply    # generate
    docker compose exec core python scripts/eval_parent_child.py --gate --qrels-file <written file>

Options: --n 40 (questions), --model (default OPENAI_MODEL_FAST), --seed 7, --out PATH,
--hinglish 0.4 (share of questions phrased in romanised Hindi, as callers speak).
Exit codes: 0 ok, 1 setup failure.
"""
import argparse
import asyncio
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = REPO_ROOT / "eval_results" / "qrels"
MIN_CHUNK_CHARS = 120          # shorter chunks rarely hold a question-worthy fact
MAX_ANCHOR_CHARS = 60
TOKENS_PER_ITEM = 600          # prompt (passage + instructions) + JSON answer, for the estimate
PRICE_PER_M = {"gpt-4o-mini": 0.30, "gpt-4o": 5.0}     # blended $/1M tokens, estimate only

PROMPT = """You help build a test set for a customer-service phone agent's knowledge base.

Passage (from the document "{title}"):
<<<
{passage}
>>>

Write ONE question a real customer calling the business might ask that THIS passage answers.
{language}
Then give 1 to 3 short phrases COPIED VERBATIM from the passage (each at most 60 characters) that
contain the answer — exact characters, same numbers and punctuation, no paraphrase.

Reply as JSON: {{"question": "...", "anchors": ["...", "..."]}}"""

LANG_EN = "Phrase it in plain spoken English, the way people talk on the phone."
LANG_HINGLISH = ("Phrase it in romanised Hindi-English (Hinglish), the way Indian callers speak, "
                 "e.g. \"2 BHK ka price kya hai?\".")


# ─────────────────────────────────────────────────────────────── file format (shared with eval)

def load_qrels_file(path: str) -> dict:
    """Load and validate a qrels file written by this script (or by hand in the same format):

    {"version": str, "tenant_id": str, "reviewed": bool,
     "items": [{"id": str, "question": str, "evidence": [[anchor, ...], ...] | null}, ...]}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: no items")
    seen = set()
    for it in items:
        if not it.get("id") or not it.get("question"):
            raise ValueError(f"{path}: every item needs an id and a question: {it!r}")
        if it["id"] in seen:
            raise ValueError(f"{path}: duplicate id {it['id']}")
        seen.add(it["id"])
        ev = it.get("evidence")
        if ev is not None and not (isinstance(ev, list) and all(
                isinstance(g, list) and g and all(isinstance(a, str) and a for a in g)
                for g in ev)):
            raise ValueError(f"{path}: {it['id']} evidence must be a list of non-empty anchor "
                             f"lists, or null")
    data.setdefault("version", Path(path).stem)
    data.setdefault("reviewed", False)
    return data


def valid_anchors(anchors, chunk_text: str) -> list[str]:
    """Keep only anchors that are literal (normalised) substrings of the chunk, deduplicated."""
    from retrieval_qrels import norm
    body = norm(chunk_text)
    out: list[str] = []
    for a in anchors if isinstance(anchors, list) else []:
        if not isinstance(a, str):
            continue
        a = a.strip().strip('"').strip()
        if 3 <= len(a) <= MAX_ANCHOR_CHARS and norm(a) in body and a not in out:
            out.append(a)
    return out


# ─────────────────────────────────────────────────────────────── sampling

def sample_chunks(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Round-robin across documents (so one long document cannot dominate), random within."""
    rng = random.Random(seed)
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
        if len(r["content"]) >= MIN_CHUNK_CHARS:
            by_doc.setdefault(str(r["document_id"]), []).append(r)
    for lst in by_doc.values():
        rng.shuffle(lst)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    picked: list[dict] = []
    while len(picked) < n and any(by_doc.values()):
        for d in docs:
            if by_doc[d] and len(picked) < n:
                picked.append(by_doc[d].pop())
    return picked


async def _generate(client, model: str, chunk: dict, hinglish: bool) -> dict | None:
    prompt = PROMPT.format(title=chunk["title"], passage=chunk["content"][:1800],
                           language=LANG_HINGLISH if hinglish else LANG_EN)
    resp = await client.chat.completions.create(
        model=model, temperature=0.3, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}])
    try:
        data = json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return None
    question = str(data.get("question") or "").strip()
    anchors = valid_anchors(data.get("anchors"), chunk["content"])
    if not question or not anchors:
        return None
    return {"question": question, "evidence": [anchors], "lang": "hinglish" if hinglish else "en",
            "source_chunk_id": str(chunk["id"]), "source_title": chunk["title"]}


async def main(args) -> int:
    from openai import AsyncOpenAI
    from sqlalchemy import text as sql_text

    from app.config import get_settings
    from app.db.base import get_session_factory

    s = get_settings()
    model = args.model or s.openai_model_fast
    async with get_session_factory()() as db:
        tenant_id = args.tenant_id or (await db.execute(sql_text(
            "SELECT tenant_id::text FROM knowledge_chunks GROUP BY 1 ORDER BY count(*) DESC "
            "LIMIT 1"))).scalar()
        if not tenant_id:
            print("no knowledge chunks — ingest documents first")
            return 1
        # Model-facing units only (standalone + parents): what a question should be answerable
        # from. Children are search slices of parents; their text is inside the parent's.
        rows = [dict(r) for r in (await db.execute(sql_text(
            "SELECT c.id, c.document_id, c.content, d.title FROM knowledge_chunks c "
            "JOIN knowledge_documents d ON d.id = c.document_id "
            "WHERE c.tenant_id = CAST(:t AS uuid) AND d.status IN ('ready', 'partial') "
            "AND c.chunk_role <> 'child' ORDER BY c.id"), {"t": tenant_id})).mappings().all()]
    kb_fp = hashlib.sha256("\n\x1e".join(sorted(r["content"] for r in rows)).encode()).hexdigest()[:16]
    picked = sample_chunks(rows, args.n, args.seed)
    est_tokens = len(picked) * TOKENS_PER_ITEM
    price = PRICE_PER_M.get(model)
    cost = f"~${est_tokens * price / 1e6:.4f}" if price else "(unknown model price)"
    print(f"\n  tenant={tenant_id}  chunks={len(rows)}  documents="
          f"{len({r['document_id'] for r in rows})}  sampled={len(picked)}  model={model}  "
          f"~{est_tokens:,} tokens {cost}")
    if not args.apply:
        for c in picked[:10]:
            print(f"    - {c['title'][:40]:42} {c['content'][:70]!r}")
        print("\n  Dry run. Re-run with --apply to generate (costs the estimate above).\n")
        return 0

    client = AsyncOpenAI(api_key=s.openai_api_key, base_url=s.openai_base_url or None)
    rng = random.Random(args.seed + 1)
    items, dropped = [], 0
    for i, c in enumerate(picked, start=1):
        try:
            it = await _generate(client, model, c, hinglish=rng.random() < args.hinglish)
        except Exception as e:  # noqa: BLE001 -- one failed call must not lose the whole set
            print(f"    [{i}/{len(picked)}] call failed: {e.__class__.__name__}: {e}")
            it = None
        if it is None:
            dropped += 1
            continue
        items.append({"id": f"G{len(items) + 1:03d}", **it})
        print(f"    [{i}/{len(picked)}] {it['question'][:70]!r}  anchors={it['evidence'][0]}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else OUT_DIR / f"{tenant_id}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": f"generated-{stamp}", "tenant_id": tenant_id, "model": model,
               "kb_fingerprint": kb_fp, "reviewed": False,
               "note": "DRAFT: review every item (fix or delete weak ones), then set reviewed=true",
               "items": items}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  {len(items)} items written, {dropped} dropped (no verbatim anchor / bad reply)")
    print(f"  -> {out}\n  Review it, set \"reviewed\": true, then:\n"
          f"  python scripts/eval_parent_child.py --gate --qrels-file {out}\n")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tenant-id")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--model")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hinglish", type=float, default=0.4)
    ap.add_argument("--out")
    ap.add_argument("--apply", action="store_true", help="call the LLM (default: dry run)")
    sys.exit(asyncio.run(main(ap.parse_args())))
