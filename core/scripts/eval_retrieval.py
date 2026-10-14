#!/usr/bin/env python3
"""Retrieval-only evaluation (R5.01): does the knowledge base put the right evidence in front of
the model — and does re-ranking change that? No agent turn, no LLM judge, no OpenAI chat cost.

What it runs
------------
For each of the 50 frozen golden questions (rag_accuracy_golden.py) it calls the SAME
retrieve() the planner calls on a cold turn, against the live Postgres KB, once per VARIANT
(e.g. rerank off / fuse / replace), and scores the returned top-k against hand-written evidence
labels (retrieval_qrels.py). Every variant sees the identical query embedding (cached), so a
difference between variants can only come from the ranking step.

Cost: one text-embedding call per question the FIRST time (~50 short strings, well under
$0.001 with text-embedding-3-small); vectors are cached on disk, so every later run is $0.

Metrics (per variant, over the questions that have KB evidence — escalations are excluded)
  Complete@k   all of a question's evidence is inside the top-k window the model reads. The
               headline number: below it, the agent CANNOT give the full golden answer.
  Recall@k     share of evidence pieces inside the top-k, averaged over questions.
  Hit@1        the first chunk the model reads carries evidence.
  MRR          1 / rank of the first evidence-bearing chunk (0 when none is returned).
  Rerank ms    p50 / p95 of the cross-encoder step, measured cold (score cache cleared per
               call), plus how many calls would have hit the production timeout.
Paired comparison vs the first variant: questions improved / worsened, with an exact sign test,
so "+2 questions" is labelled as signal or noise.

How this relates to the 88% end-to-end accuracy
-----------------------------------------------
Retrieval is necessary, not sufficient: the e2e eval (eval_rag_accuracy.py) also depends on
the responder model. This eval tells you whether R5.01 moved the evidence, cheaply and exactly.
It also prints the question ids whose retrieved window CHANGED — only those can change the e2e
result, so a confirmation run of eval_rag_accuracy.py needs just `--ids <those>`.

Usage — inside a throwaway container off the core image (needs the DB and app settings):
    docker compose run --rm core python scripts/eval_retrieval.py
    ... python scripts/eval_retrieval.py --variants off,fuse,replace          (default)
    ... python scripts/eval_retrieval.py --variants "off,fuse,fp32:retrieval_rerank_onnx_file=onnx/model.onnx"
    ... python scripts/eval_retrieval.py --ids EN011 HI014 --show             (per-chunk detail)
    ... python scripts/eval_retrieval.py --query "2.5 bhk ka price kya hai"   (ad-hoc, no labels)
    ... python scripts/eval_retrieval.py --spelling-variants                  (+ misspelt forms)
    ... python scripts/eval_retrieval.py --production-timeout                 (quality AS served)

Variant syntax: NAME or NAME:setting=value;setting=value. Built-in names: off (rerank disabled),
fuse, replace. Any app setting can be overridden, e.g.
    "k20:retrieval_rerank_candidates=20"   "w2:retrieval_rerank_weight=2.0"
    "cap3:retrieval_max_chunks_per_doc=3"  "raw:retrieval_rerank_translate=false"
    'gl:retrieval_rerank_glossary={"khel kood": "sports"}'      (dict settings take JSON)
Start a custom variant's name with "off" to keep rerank disabled, e.g. "off_cap3:...".
Reports land in ../eval_results/retrieval/<timestamp>/ (manifest.json + results.jsonl).
Exit codes: 0 ok, 1 setup failure, 2 stale labels (an anchor matches no chunk in the KB).
"""
import argparse
import asyncio
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CORE_DIR = SCRIPT_DIR.parent
REPO_ROOT = CORE_DIR.parent
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sqlalchemy import text as sql_text  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db.base import get_session_factory  # noqa: E402
from app.engine.rag import reranker, retrieval  # noqa: E402
from rag_accuracy_golden import GOLDEN, GOLDEN_VERSION  # noqa: E402
from retrieval_qrels import (FOR_GOLDEN_VERSION, QRELS, QRELS_VERSION, groups_in,  # noqa: E402
                             anchor_matches, norm)

GREEN, RED, YELLOW, DIM, BOLD, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
OUT_ROOT = REPO_ROOT / "eval_results" / "retrieval"
EMBED_CACHE = OUT_ROOT / "embedding_cache.json"
BUILTIN_VARIANTS = {
    "off": {"retrieval_rerank_enabled": False},
    "fuse": {"retrieval_rerank_enabled": True, "retrieval_rerank_mode": "fuse"},
    "replace": {"retrieval_rerank_enabled": True, "retrieval_rerank_mode": "replace"},
}


# ─────────────────────────────────────────────────────────────── variants / settings

def _split_top(text: str, sep: str) -> list[str]:
    """Split on ``sep`` outside {...}, so JSON values may contain commas / semicolons."""
    parts, depth, cur = [], 0, []
    for ch in text:
        depth += ch == "{"
        depth -= ch == "}"
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def parse_variants(spec: str) -> list[tuple[str, dict]]:
    """'off,fuse,x:key=val;key=val' -> [(name, overrides)]. A custom variant starts from the
    built-in of the same name if there is one, else from 'off' when its name starts with
    "off", else from 'fuse' (rerank on)."""
    out = []
    for part in _split_top(spec, ","):
        name, _, rest = part.partition(":")
        base = BUILTIN_VARIANTS.get(name) or (BUILTIN_VARIANTS["off"] if name.startswith("off")
                                              else BUILTIN_VARIANTS["fuse"])
        overrides = dict(base)
        for kv in _split_top(rest, ";"):
            key, _, raw = kv.partition("=")
            overrides[key.strip()] = raw.strip()
        out.append((name, overrides))
    return out


def apply_overrides(s, overrides: dict) -> dict:
    """Set overrides on the live settings object, coercing to each field's current type.
    Returns the previous values so they can be restored."""
    previous = {}
    for key, raw in overrides.items():
        if not hasattr(s, key):
            sys.exit(f"unknown setting in variant: {key}")
        current = getattr(s, key)
        previous[key] = current
        value = raw
        if isinstance(raw, str):
            if isinstance(current, dict):
                value = json.loads(raw)          # e.g. retrieval_rerank_glossary={"a": "b"}
            elif isinstance(current, bool):
                value = raw.lower() in ("1", "true", "yes", "on")
            elif isinstance(current, int):
                value = int(raw)
            elif isinstance(current, float):
                value = float(raw)
        setattr(s, key, value)
    return previous


# ─────────────────────────────────────────────────────────────── embedding disk cache

def load_embed_cache(model: str) -> int:
    """Pre-fill retrieval's in-process query-vector LRU from disk, so no question is embedded
    twice across runs. Keyed by (model, text) exactly like the LRU — a model change never
    serves a stale vector."""
    if not EMBED_CACHE.exists():
        return 0
    data = json.loads(EMBED_CACHE.read_text(encoding="utf-8"))
    n = 0
    for text, vec in (data.get(model) or {}).items():
        retrieval._QVEC_CACHE[(model, text)] = vec
        n += 1
    retrieval._QVEC_CACHE_MAX = max(retrieval._QVEC_CACHE_MAX, n + 1024)
    return n


def save_embed_cache(model: str) -> None:
    data = json.loads(EMBED_CACHE.read_text(encoding="utf-8")) if EMBED_CACHE.exists() else {}
    bucket = data.setdefault(model, {})
    for (m, text), vec in retrieval._QVEC_CACHE.items():
        if m == model:
            bucket[text] = vec
    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    EMBED_CACHE.write_text(json.dumps(data), encoding="utf-8")


# ─────────────────────────────────────────────────────────────── KB checks

async def resolve_tenant(db, tenant_id: str | None) -> str:
    if tenant_id:
        return tenant_id
    row = (await db.execute(sql_text(
        "SELECT tenant_id::text, count(*) FROM knowledge_chunks GROUP BY 1 ORDER BY 2 DESC LIMIT 2"
    ))).all()
    if not row:
        sys.exit("no knowledge chunks in this database — seed the KB first")
    if len(row) > 1:
        print(f"{YELLOW}several tenants have chunks; using the largest ({row[0][0]}). "
              f"Pass --tenant-id to choose.{OFF}")
    return row[0][0]


async def kb_snapshot(db, tenant_id: str) -> list[str]:
    rows = (await db.execute(sql_text(
        "SELECT c.content FROM knowledge_chunks c JOIN knowledge_documents d ON d.id=c.document_id "
        "WHERE c.tenant_id = CAST(:t AS uuid) AND d.status='ready' ORDER BY c.id"), {"t": tenant_id})).all()
    return [r[0] for r in rows]


def stale_anchors(kb: list[str], ids: list[str]) -> list[str]:
    """Every anchor must exist SOMEWHERE in the KB, else the label is stale (the document
    changed) and the question would be scored as a miss for a reason that is not retrieval."""
    texts = [norm(c) for c in kb]
    problems = []
    for qid in ids:
        for gi, group in enumerate(QRELS.get(qid) or []):
            if not any(anchor_matches(a, t) for a in group for t in texts):
                problems.append(f"{qid} evidence #{gi + 1} {group}: no chunk contains any anchor")
    return problems


# ─────────────────────────────────────────────────────────────── scoring

def score_question(chunks, groups: list[list[str]], k: int) -> dict:
    """Evidence metrics for one retrieval result against one question's groups."""
    per_chunk = [sorted(groups_in(c.content, groups)) for c in chunks]
    in_k = set().union(*[set(g) for g in per_chunk[:k]]) if per_chunk[:k] else set()
    in_3 = set().union(*[set(g) for g in per_chunk[:3]]) if per_chunk[:3] else set()
    first = next((i for i, g in enumerate(per_chunk, start=1) if g), None)
    return {"complete_k": len(in_k) == len(groups), "recall_k": len(in_k) / len(groups),
            "recall_3": len(in_3) / len(groups), "hit_1": bool(per_chunk and per_chunk[0]),
            "rr": (1.0 / first) if first else 0.0, "first_rank": first,
            "missing": [groups[i] for i in range(len(groups)) if i not in in_k],
            "chunk_groups": per_chunk}


def chunk_row(c, groups) -> dict:
    return {"chunk_id": c.chunk_id, "label": f"{c.title} > {(c.heading or '').split(' > ')[-1]}",
            "doc_type": c.doc_type, "evidence": sorted(groups_in(c.content, groups)) if groups else [],
            "vector_rank": c.vector_rank, "text_rank": c.text_rank, "base_rank": c.base_rank,
            "rerank_rank": c.rerank_rank,
            "rerank_score": round(c.rerank_score, 3) if c.rerank_score is not None else None}


def sign_test_p(improved: int, worsened: int) -> float:
    """Exact two-sided binomial sign test on the discordant pairs (ties dropped)."""
    n = improved + worsened
    if n == 0:
        return 1.0
    k = min(improved, worsened)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pct(values, q):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))]


# ─────────────────────────────────────────────────────────────── main run

async def run(args) -> int:
    s = get_settings()
    variants = parse_variants(args.variants)
    embed_model = s.embedding_model or ""
    cached = load_embed_cache(embed_model)
    before_keys = set(retrieval._QVEC_CACHE.keys())

    questions = [q for q in GOLDEN if not args.ids or q["id"] in args.ids]
    items = [(q["id"], q["question"], q) for q in questions]
    if args.spelling_variants:
        items += [(q["id"] + "~v", q["variant"], q) for q in questions if q.get("variant")]
    if args.query:
        items = [(f"Q{i + 1}", text, None) for i, text in enumerate(args.query)]

    factory = get_session_factory()
    async with factory() as db:
        tenant_id = await resolve_tenant(db, args.tenant_id)
        kb = await kb_snapshot(db, tenant_id)
    kb_fp = hashlib.sha256("\n\x1e".join(sorted(kb)).encode()).hexdigest()[:16]

    print(f"{BOLD}Retrieval eval{OFF}  tenant={tenant_id}  chunks={len(kb)}  kb={kb_fp}  "
          f"top_k={s.retrieval_top_k}  golden={GOLDEN_VERSION}  qrels={QRELS_VERSION}")
    if FOR_GOLDEN_VERSION != GOLDEN_VERSION:
        print(f"{YELLOW}qrels were written for {FOR_GOLDEN_VERSION}, golden is {GOLDEN_VERSION} "
              f"— re-check the labels.{OFF}")
    stale = [] if args.query else stale_anchors(kb, [q["id"] for q in questions])
    if stale:
        print(f"{RED}{BOLD}STALE LABELS — the KB no longer contains these anchors:{OFF}")
        for p in stale:
            print(f"  {RED}{p}{OFF}")
        if not args.allow_stale:
            print("Fix retrieval_qrels.py (or pass --allow-stale to score anyway).")
            return 2
    print(f"embedding cache: {cached} vectors on disk ({embed_model})\n")

    # Load every model a variant needs BEFORE timing anything — a load is not a rerank.
    production_timeout = s.retrieval_rerank_timeout_ms
    for name, overrides in variants:
        prev = apply_overrides(s, overrides)
        if s.retrieval_rerank_enabled:
            cfg = reranker.config_from_settings(s)
            print(f"loading reranker for '{name}': {cfg.label} ...", end=" ", flush=True)
            ok = await reranker.warm_up(cfg)
            print(f"{GREEN}ready{OFF}" if ok else f"{RED}FAILED{OFF} (see log; variant will fall back to RRF)")
        apply_overrides(s, prev)

    results: dict[str, dict[str, dict]] = {}     # qid -> variant -> record
    for qid, text, q in items:
        groups = QRELS.get(qid.split("~")[0]) if q else None
        results[qid] = {}
        for name, overrides in variants:
            run_overrides = dict(overrides)
            if not args.production_timeout:
                # Quality mode: let every rerank finish so the RANKING is what gets measured;
                # how many calls would have missed the real deadline is reported separately.
                run_overrides["retrieval_rerank_timeout_ms"] = 10000
            prev = apply_overrides(s, run_overrides)
            reranker.clear_score_cache()
            async with factory() as db:
                started = time.perf_counter()
                res = await retrieval.retrieve(db, tenant_id=tenant_id, query=text,
                                               on_date=date.today())
                wall = (time.perf_counter() - started) * 1000
            apply_overrides(s, prev)
            rec = {"query": text, "latency_ms": round(wall, 1), "error": res.error,
                   "rerank": res.rerank_diagnostics(),
                   "chunks": [chunk_row(c, groups) for c in res.chunks]}
            if groups:
                rec["metrics"] = score_question(res.chunks, groups, s.retrieval_top_k)
                rec["metrics"].pop("chunk_groups")
            results[qid][name] = rec

    new_vectors = len(set(retrieval._QVEC_CACHE.keys()) - before_keys)
    save_embed_cache(embed_model)

    if args.query:
        print_adhoc(results, variants)
        return 0
    return report(args, s, variants, results, questions, tenant_id, kb_fp, len(kb), new_vectors,
                  production_timeout)


# ─────────────────────────────────────────────────────────────── reporting

def print_adhoc(results, variants):
    for qid, per in results.items():
        first = next(iter(per.values()))
        print(f"{BOLD}{qid}: {first['query']}{OFF}")
        for name, _ in variants:
            rec = per[name]
            rr = rec["rerank"] or {}
            print(f"  {BOLD}{name}{OFF}  {rec['latency_ms']:.0f} ms  rerank={rr.get('status', 'off')} "
                  f"{rr.get('ms', '')}")
            for i, c in enumerate(rec["chunks"], start=1):
                ce = f"ce={c['rerank_score']:+.2f}" if c["rerank_score"] is not None else ""
                print(f"    {i}. {c['label'][:70]:<70} base#{c['base_rank'] or '-'} {ce}")
        print()


def summarize(results, qids, name, k):
    rows = [results[q][name]["metrics"] for q in qids if "metrics" in results[q][name]]
    n = len(rows) or 1
    rr_ms = [results[q][name]["rerank"]["ms"] for q in qids
             if (results[q][name]["rerank"] or {}).get("status") == "ok"
             and results[q][name]["rerank"].get("scored")]
    statuses = {}
    for q in qids:
        st = (results[q][name]["rerank"] or {}).get("status", "off")
        statuses[st] = statuses.get(st, 0) + 1
    return {"n": len(rows),
            "complete_k": sum(r["complete_k"] for r in rows) / n,
            "recall_k": sum(r["recall_k"] for r in rows) / n,
            "recall_3": sum(r["recall_3"] for r in rows) / n,
            "hit_1": sum(r["hit_1"] for r in rows) / n,
            "mrr": sum(r["rr"] for r in rows) / n,
            "retrieval_ms_p50": statistics.median([results[q][name]["latency_ms"] for q in qids]),
            "rerank_ms_p50": pct(rr_ms, 0.5), "rerank_ms_p95": pct(rr_ms, 0.95),
            "rerank_ms_max": max(rr_ms) if rr_ms else None, "rerank_statuses": statuses,
            "_rr_ms": rr_ms}


def qkey(m):
    """Per-question ordering used for improved/worsened: evidence first, then its rank."""
    return (m["recall_k"], m["rr"])


def report(args, s, variants, results, questions, tenant_id, kb_fp, kb_chunks, new_vectors,
           production_timeout) -> int:
    k = s.retrieval_top_k
    scored = [qid for qid in results if QRELS.get(qid.split("~")[0])]
    excluded = [qid for qid in results if not QRELS.get(qid.split("~")[0])]
    base_name = variants[0][0]
    summaries = {name: summarize(results, scored, name, k) for name, _ in variants}

    print(f"{BOLD}Scored {len(scored)} queries with KB evidence{OFF} "
          f"({len(excluded)} escalation queries excluded: {' '.join(excluded)})\n")
    hdr = f"{'variant':<12}{'Complete@' + str(k):>12}{'Recall@' + str(k):>11}{'Recall@3':>10}" \
          f"{'Hit@1':>8}{'MRR':>8}{'ret p50':>9}{'rerank p50/p95':>17}  statuses"
    print(hdr)
    print("-" * len(hdr))
    for name, _ in variants:
        m = summaries[name]
        rr = (f"{m['rerank_ms_p50']:.0f}/{m['rerank_ms_p95']:.0f} ms"
              if m["rerank_ms_p50"] is not None else "-")
        print(f"{name:<12}{m['complete_k']:>12.1%}{m['recall_k']:>11.1%}{m['recall_3']:>10.1%}"
              f"{m['hit_1']:>8.1%}{m['mrr']:>8.3f}{m['retrieval_ms_p50']:>7.0f}ms{rr:>17}  "
              f"{m['rerank_statuses']}")
        if m["_rr_ms"]:
            over = sum(1 for x in m["_rr_ms"] if x > production_timeout)
            if over:
                print(f"{'':<12}{YELLOW}{over}/{len(m['_rr_ms'])} reranks exceeded the production "
                      f"timeout ({production_timeout} ms) — live, those turns fall back to RRF.{OFF}")

    comparisons = {}
    for name, _ in variants[1:]:
        improved, worsened, changed_window, reordered = [], [], [], []
        for qid in results:
            a, b = results[qid][base_name], results[qid][name]
            ids_a, ids_b = [c["chunk_id"] for c in a["chunks"]], [c["chunk_id"] for c in b["chunks"]]
            if set(ids_a) != set(ids_b):
                changed_window.append(qid)      # a different set of chunks reached the model
            elif ids_a != ids_b:
                reordered.append(qid)           # same chunks, different order
            if qid in scored:
                ka, kb_ = qkey(a["metrics"]), qkey(b["metrics"])
                if kb_ > ka:
                    improved.append(qid)
                elif kb_ < ka:
                    worsened.append(qid)
        p = sign_test_p(len(improved), len(worsened))
        verdict = ("likely REAL" if p < 0.05 else "NOT significant (could be noise)")
        comparisons[name] = {"improved": improved, "worsened": worsened, "p_value": round(p, 4),
                             "changed_window": changed_window, "reordered_only": reordered}
        print(f"\n{BOLD}{name} vs {base_name}{OFF}: {GREEN}{len(improved)} improved{OFF}, "
              f"{RED}{len(worsened)} worsened{OFF}, sign-test p={p:.3f} -> {verdict}")
        for label, ids, colour in (("improved", improved, GREEN), ("worsened", worsened, RED)):
            for qid in ids:
                a, b = results[qid][base_name]["metrics"], results[qid][name]["metrics"]
                print(f"  {colour}{label:<9}{OFF}{qid:<8} recall@{k} {a['recall_k']:.0%}->{b['recall_k']:.0%}"
                      f"  first-evidence rank {a['first_rank'] or '-'}->{b['first_rank'] or '-'}"
                      f"  {DIM}{results[qid][name]['query'][:60]}{OFF}")
        print(f"  window: {len(changed_window)} queries got a different SET of chunks, "
              f"{len(reordered)} the same chunks in a different order, "
              f"{len(results) - len(changed_window) - len(reordered)} unchanged")
        # Worth paying the e2e judge for: a different set of chunks reached the model, or the
        # evidence metrics moved. A pure reorder of the same chunks rarely changes an answer.
        ids = sorted({q.split('~')[0] for q in changed_window + improved + worsened})
        if ids:
            print(f"  e2e confirmation shortlist ({len(ids)} questions):\n    {DIM}python "
                  f"scripts/eval_rag_accuracy.py --ids {' '.join(ids)}{OFF}")

    # Still-missing evidence under the LAST variant: what no ranking tweak here fixed.
    last = variants[-1][0]
    misses = [(qid, results[qid][last]["metrics"]["missing"]) for qid in scored
              if results[qid][last]["metrics"]["missing"]]
    if misses:
        print(f"\n{BOLD}Evidence still outside the top-{k} under '{last}':{OFF}")
        for qid, missing in misses:
            print(f"  {qid:<8}{DIM}{results[qid][last]['query'][:50]:<52}{OFF} missing {missing}")

    if args.show:
        for qid in results:
            print(f"\n{BOLD}{qid}: {results[qid][base_name]['query']}{OFF}")
            for name, _ in variants:
                rec = results[qid][name]
                m = rec.get("metrics")
                head = (f"recall {m['recall_k']:.0%} first-evidence #{m['first_rank'] or '-'}"
                        if m else "(no labels)")
                print(f"  {BOLD}{name}{OFF}  {head}  rerank={(rec['rerank'] or {}).get('status', 'off')}")
                for i, c in enumerate(rec["chunks"], start=1):
                    ev = f"{GREEN}evidence {[e + 1 for e in c['evidence']]}{OFF}" if c["evidence"] else ""
                    ce = f"ce={c['rerank_score']:+.2f} ce#{c['rerank_rank']}" if c["rerank_score"] is not None else ""
                    print(f"    {i}. {c['label'][:62]:<62} base#{c['base_rank'] or '-':<3}{ce:<16}{ev}")

    # ── persist
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / ts
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for qid, per in results.items():
            f.write(json.dumps({"id": qid, "variants": per}, ensure_ascii=False) + "\n")
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(), "golden_version": GOLDEN_VERSION,
        "qrels_version": QRELS_VERSION, "tenant_id": tenant_id, "kb_fingerprint": kb_fp,
        "kb_chunks": kb_chunks, "top_k": k, "variants": {n: o for n, o in variants},
        "quality_mode_timeout": not args.production_timeout,
        "production_timeout_ms": production_timeout,
        "new_embedding_calls": new_vectors,
        "summary": {n: {kk: vv for kk, vv in m.items() if not kk.startswith("_")}
                    for n, m in summaries.items()},
        "comparison_vs_" + base_name: comparisons,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print(f"\nnew embedding calls this run: {new_vectors}   report: {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="off,fuse,replace")
    ap.add_argument("--ids", nargs="*", help="only these golden question ids")
    ap.add_argument("--query", nargs="*", help="ad-hoc question(s): show rankings, no scoring")
    ap.add_argument("--spelling-variants", action="store_true",
                    help="also run each question's misspelt romanised form")
    ap.add_argument("--show", action="store_true", help="print every query's ranked chunks")
    ap.add_argument("--production-timeout", action="store_true",
                    help="use the configured rerank timeout (default: let reranks finish)")
    ap.add_argument("--tenant-id")
    ap.add_argument("--allow-stale", action="store_true")
    args = ap.parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 1


if __name__ == "__main__":
    sys.exit(main())
