#!/usr/bin/env python3
"""Retrieval-only A/B for R5.03 parent-child chunking — flat vs parent_child on the SAME data.

Why this is cheap AND trustworthy
---------------------------------
* No agent turn, no responder, no LLM judge: it calls the same retrieve() a live turn calls and
  scores what comes back against the frozen evidence labels in retrieval_qrels.py (content
  anchors like "1,54,32,011" — matched against chunk TEXT, so re-chunking cannot break them).
* The only paid call is one text-embedding per question, and vectors are cached on disk in
  eval_results/retrieval/embedding_cache.json (shared with the R5.01 eval). Re-runs cost $0.
* Both modes run over ONE database state: after scripts/backfill_parent_child.py, parents are
  exactly the old chunks (same text, same embeddings) and children sit beside them. "flat"
  searches standalone+parents (the pre-R5.03 index), "parent_child" searches standalone+children
  and returns parents. The model-facing text in both is parent text, so a difference in the
  scores can only come from WHICH parents were found and in what order — i.e. from R5.03.
* The Redis result cache is bypassed for the run, so no variant is served another's answer.

Metrics (over questions that have KB evidence; escalation questions are excluded)
  Complete@k  every evidence piece is inside the top-k window the model reads — below this the
              agent CANNOT give the full golden answer. The headline number.
  Recall@k    share of evidence pieces in the top-k, averaged.   Recall@3: same for the top 3.
  Hit@1       the first chunk the model reads carries evidence.  MRR: 1 / rank of the first one.
Paired per-question comparison vs the first variant (improved / worsened) with an exact sign
test, so "+2 questions" is labelled signal or noise. The question ids whose retrieved window
CHANGED are printed as a ready-made --ids list: only those can change the end-to-end (88%)
accuracy, so the confirming eval_rag_accuracy.py run needs only them.

Usage (inside the core container — needs the DB and app settings):
    docker compose exec core python scripts/eval_parent_child.py
    ... --variants "flat,parent_child,pc_nosib:retrieval_granularity=parent_child;retrieval_parent_sibling_weight=0"
    ... --ids EN011 HI014 --show              per-chunk detail, incl. which children matched
    ... --query "2.5 bhk ka price kya hai"    ad-hoc, no labels
    ... --spelling-variants                   also ask the golden set's misspelt forms
    ... --qrels-file eval_results/qrels/<tenant>.json   any tenant, with its own labels
Variant syntax: NAME or NAME:setting=value;setting=value (any app setting). Built-ins: flat,
parent_child. A custom name starting with "pc" starts from parent_child, anything else from flat.
Reports: eval_results/retrieval_r503/<timestamp>/{manifest.json,results.jsonl}
ENABLE GATE (--gate): a candidate passes only if Complete@k and Recall@k are >= the first
variant's AND no question's evidence LEFT the window (--max-evidence-loss N to tolerate N). Rank
moves inside the window are reported, not gated. Run it before setting RETRIEVAL_GRANULARITY.
Exit codes: 0 ok, 1 setup problem, 2 stale labels (an anchor matches no chunk in the KB),
3 --gate failed.
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
from app.engine.rag import retrieval  # noqa: E402
from rag_accuracy_golden import GOLDEN, GOLDEN_VERSION  # noqa: E402
from retrieval_qrels import (FOR_GOLDEN_VERSION, QRELS, QRELS_VERSION,  # noqa: E402
                             anchor_matches, groups_in, norm)

GREEN, RED, YELLOW, DIM, BOLD, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
OUT_ROOT = REPO_ROOT / "eval_results" / "retrieval_r503"
EMBED_CACHE = REPO_ROOT / "eval_results" / "retrieval" / "embedding_cache.json"
BUILTIN = {"flat": {"retrieval_granularity": "flat"},
           "parent_child": {"retrieval_granularity": "parent_child"}}


# ─────────────────────────────────────────────────────────────── variants / settings

def parse_variants(spec: str) -> list[tuple[str, dict]]:
    out = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        name, _, rest = part.partition(":")
        base = BUILTIN.get(name) or (BUILTIN["parent_child"] if name.startswith("pc")
                                     else BUILTIN["flat"])
        overrides = dict(base)
        for kv in [x for x in rest.split(";") if x.strip()]:
            key, _, raw = kv.partition("=")
            overrides[key.strip()] = raw.strip()
        out.append((name, overrides))
    return out


def apply_overrides(s, overrides: dict) -> dict:
    """Set overrides on the live settings object (coerced to each field's type); return the
    previous values so they can be restored."""
    previous = {}
    for key, raw in overrides.items():
        if not hasattr(s, key):
            sys.exit(f"unknown setting in variant: {key}")
        current = getattr(s, key)
        previous[key] = current
        value = raw
        if isinstance(raw, str) and not isinstance(current, str):
            if isinstance(current, bool):
                value = raw.lower() in ("1", "true", "yes", "on")
            elif isinstance(current, int):
                value = int(raw)
            elif isinstance(current, float):
                value = float(raw)
        setattr(s, key, value)
    return previous


# ─────────────────────────────────────────────────────────────── embedding disk cache

def load_embed_cache(model: str) -> int:
    """Pre-fill retrieval's query-vector LRU from disk (same file/format as the R5.01 eval)."""
    if not EMBED_CACHE.exists():
        return 0
    data = json.loads(EMBED_CACHE.read_text(encoding="utf-8"))
    bucket = data.get(model) or {}
    for text, vec in bucket.items():
        retrieval._QVEC_CACHE[(model, text)] = vec
    retrieval._QVEC_CACHE_MAX = max(retrieval._QVEC_CACHE_MAX, len(bucket) + 1024)
    return len(bucket)


def save_embed_cache(model: str) -> None:
    data = json.loads(EMBED_CACHE.read_text(encoding="utf-8")) if EMBED_CACHE.exists() else {}
    bucket = data.setdefault(model, {})
    for (m, text), vec in retrieval._QVEC_CACHE.items():
        if m == model:
            bucket[text] = vec
    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    EMBED_CACHE.write_text(json.dumps(data), encoding="utf-8")


def bypass_result_cache() -> None:
    """The Redis result cache keys on granularity but not on every tunable a variant may change
    (e.g. sibling weight) — so for an A/B it is switched off entirely."""
    async def _miss(*_a, **_k):
        return None

    async def _skip(*_a, **_k):
        return None
    retrieval._cache_get = _miss
    retrieval._cache_set = _skip


# ─────────────────────────────────────────────────────────────── KB checks

async def resolve_tenant(db, tenant_id: str | None) -> str:
    if tenant_id:
        return tenant_id
    rows = (await db.execute(sql_text(
        "SELECT tenant_id::text, count(*) FROM knowledge_chunks GROUP BY 1 ORDER BY 2 DESC LIMIT 2"
    ))).all()
    if not rows:
        sys.exit("no knowledge chunks in this database — seed the KB first")
    if len(rows) > 1:
        print(f"{YELLOW}several tenants have chunks; using the largest ({rows[0][0]}). "
              f"Pass --tenant-id to choose.{OFF}")
    return rows[0][0]


async def kb_state(db, tenant_id: str) -> tuple[list[str], dict[str, int]]:
    """Model-facing texts (standalone + parents — what either mode can put in the window) and
    the role counts."""
    rows = (await db.execute(sql_text(
        "SELECT c.content, c.chunk_role FROM knowledge_chunks c "
        "JOIN knowledge_documents d ON d.id = c.document_id "
        "WHERE c.tenant_id = CAST(:t AS uuid) AND d.status = 'ready' ORDER BY c.id"),
        {"t": tenant_id})).all()
    roles: dict[str, int] = {}
    for _, role in rows:
        roles[role] = roles.get(role, 0) + 1
    return [r[0] for r in rows if r[1] != "child"], roles


def stale_anchors(kb: list[str], ids: list[str]) -> list[str]:
    texts = [norm(c) for c in kb]
    problems = []
    for qid in ids:
        for gi, group in enumerate(QRELS.get(qid) or []):
            if not any(anchor_matches(a, t) for a in group for t in texts):
                problems.append(f"{qid} evidence #{gi + 1} {group}: no chunk contains any anchor")
    return problems


# ─────────────────────────────────────────────────────────────── scoring

def score_question(chunks, groups: list[list[str]], k: int) -> dict:
    per_chunk = [sorted(groups_in(c.content, groups)) for c in chunks]
    in_k = set().union(*[set(g) for g in per_chunk[:k]]) if per_chunk[:k] else set()
    in_3 = set().union(*[set(g) for g in per_chunk[:3]]) if per_chunk[:3] else set()
    first = next((i for i, g in enumerate(per_chunk, start=1) if g), None)
    return {"complete_k": len(in_k) == len(groups), "recall_k": len(in_k) / len(groups),
            "recall_3": len(in_3) / len(groups), "hit_1": bool(per_chunk and per_chunk[0]),
            "rr": (1.0 / first) if first else 0.0, "first_rank": first,
            "missing": [groups[i] for i in range(len(groups)) if i not in in_k]}


def chunk_row(c, groups) -> dict:
    return {"chunk_id": c.chunk_id,
            "label": f"{c.title} > {(c.heading or '').split(' > ')[-1]}",
            "evidence": sorted(groups_in(c.content, groups)) if groups else [],
            "vector_rank": c.vector_rank, "text_rank": c.text_rank,
            "relevance": c.relevance,
            "matched_children": [{"rank_v": m.get("vector_rank"), "rank_t": m.get("text_rank"),
                                  "text": (m.get("content") or "")[:140]}
                                 for m in (c.matched_children or [])]}


def sign_test_p(improved: int, worsened: int) -> float:
    """Exact two-sided binomial sign test on the discordant pairs (ties dropped)."""
    n = improved + worsened
    if n == 0:
        return 1.0
    k = min(improved, worsened)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarize(results, qids, name) -> dict:
    rows = [results[q][name]["metrics"] for q in qids]
    n = len(rows) or 1
    pc = [results[q][name]["parent_child"] for q in qids]
    return {"n": len(rows),
            "complete_k": sum(r["complete_k"] for r in rows) / n,
            "recall_k": sum(r["recall_k"] for r in rows) / n,
            "recall_3": sum(r["recall_3"] for r in rows) / n,
            "hit_1": sum(r["hit_1"] for r in rows) / n,
            "mrr": sum(r["rr"] for r in rows) / n,
            "retrieval_ms_p50": statistics.median(
                [results[q][name]["latency_ms"] for q in qids]) if qids else 0,
            "avg_child_hits": sum(p.get("child_hits", 0) for p in pc) / n,
            "avg_merged_siblings": sum(p.get("merged_siblings", 0) for p in pc) / n,
            "parents_missing": sum(p.get("parents_missing", 0) for p in pc),
            "parent_fetch_ms_p50": statistics.median(
                [p.get("fetch_ms", 0.0) for p in pc]) if pc else 0.0}


# ─────────────────────────────────────────────────────────────── main run

def use_qrels_file(args) -> None:
    """--qrels-file: evaluate against a tenant's own labelled set (build_retrieval_qrels.py or
    hand-written in the same format) instead of the frozen DSR golden set. Rebinds the module
    globals every scoring/report function reads, so nothing else needs to know."""
    global GOLDEN, GOLDEN_VERSION, QRELS, QRELS_VERSION, FOR_GOLDEN_VERSION
    from build_retrieval_qrels import load_qrels_file
    data = load_qrels_file(args.qrels_file)
    GOLDEN = [{"id": it["id"], "question": it["question"]} for it in data["items"]]
    QRELS = {it["id"]: it.get("evidence") for it in data["items"]}
    GOLDEN_VERSION = FOR_GOLDEN_VERSION = QRELS_VERSION = data["version"]
    if not args.tenant_id and data.get("tenant_id"):
        args.tenant_id = data["tenant_id"]
    if not data.get("reviewed"):
        print(f"{YELLOW}{BOLD}qrels file is NOT reviewed{OFF}{YELLOW} ({args.qrels_file}): "
              f"generated labels are a draft — review them and set \"reviewed\": true before "
              f"trusting a gate result.{OFF}")


async def run(args) -> int:
    if args.qrels_file:
        use_qrels_file(args)
    s = get_settings()
    variants = parse_variants(args.variants)
    model = s.embedding_model or ""
    cached = load_embed_cache(model)
    before = set(retrieval._QVEC_CACHE.keys())
    bypass_result_cache()

    questions = [q for q in GOLDEN if not args.ids or q["id"] in args.ids]
    items = [(q["id"], q["question"], q) for q in questions]
    if args.spelling_variants:
        items += [(q["id"] + "~v", q["variant"], q) for q in questions if q.get("variant")]
    if args.query:
        items = [(f"Q{i + 1}", text, None) for i, text in enumerate(args.query)]

    factory = get_session_factory()
    async with factory() as db:
        tenant_id = await resolve_tenant(db, args.tenant_id)
        kb, roles = await kb_state(db, tenant_id)
    kb_fp = hashlib.sha256("\n\x1e".join(sorted(kb)).encode()).hexdigest()[:16]

    print(f"{BOLD}R5.03 retrieval A/B{OFF}  tenant={tenant_id}  roles={roles}  kb={kb_fp}  "
          f"top_k={s.retrieval_top_k}  golden={GOLDEN_VERSION}  qrels={QRELS_VERSION}")
    if any(o.get("retrieval_granularity") == "parent_child" for _, o in variants) \
            and not roles.get("child"):
        print(f"{RED}No child chunks exist for this tenant, so parent_child == flat. Run "
              f"scripts/backfill_parent_child.py --apply first (or re-ingest with "
              f"CHUNK_PARENT_CHILD_ENABLED=true).{OFF}")
        return 1
    if FOR_GOLDEN_VERSION != GOLDEN_VERSION:
        print(f"{YELLOW}qrels were written for {FOR_GOLDEN_VERSION}; golden is "
              f"{GOLDEN_VERSION} — re-check the labels.{OFF}")
    stale = [] if args.query else stale_anchors(kb, [q["id"] for q in questions])
    if stale:
        print(f"{RED}{BOLD}STALE LABELS — the KB no longer contains these anchors:{OFF}")
        for p in stale:
            print(f"  {RED}{p}{OFF}")
        if not args.allow_stale:
            print("Fix retrieval_qrels.py (or pass --allow-stale to score anyway).")
            return 2
    print(f"embedding cache: {cached} query vectors on disk ({model})\n")

    results: dict[str, dict[str, dict]] = {}
    for qid, text, q in items:
        groups = QRELS.get(qid.split("~")[0]) if q else None
        results[qid] = {}
        for name, overrides in variants:
            prev = apply_overrides(s, overrides)
            try:
                async with factory() as db:
                    started = time.perf_counter()
                    res = await retrieval.retrieve(db, tenant_id=tenant_id, query=text,
                                                   on_date=date.today())
                    wall = (time.perf_counter() - started) * 1000
            finally:
                apply_overrides(s, prev)
            rec = {"query": text, "latency_ms": round(wall, 1), "error": res.error,
                   "parent_child": res.parent_child_diagnostics(),
                   "chunks": [chunk_row(c, groups) for c in res.chunks]}
            if groups:
                rec["metrics"] = score_question(res.chunks, groups, s.retrieval_top_k)
            results[qid][name] = rec

    new_vectors = len(set(retrieval._QVEC_CACHE.keys()) - before)
    save_embed_cache(model)
    if args.query:
        print_adhoc(results, variants)
        return 0
    return report(args, s, variants, results, tenant_id, kb_fp, roles, new_vectors)


# ─────────────────────────────────────────────────────────────── reporting

def print_chunks(rec, indent="    "):
    for i, c in enumerate(rec["chunks"], start=1):
        ev = f"{GREEN}evidence {c['evidence']}{OFF}" if c["evidence"] else ""
        print(f"{indent}{i}. {c['label'][:66]:<66} rel={c['relevance']:.2f} "
              f"v#{c['vector_rank'] or '-'} t#{c['text_rank'] or '-'} {ev}")
        for m in c["matched_children"][:3]:
            print(f"{indent}     {DIM}child v#{m['rank_v'] or '-'} t#{m['rank_t'] or '-'}: "
                  f"{m['text']!r}{OFF}")


def print_adhoc(results, variants):
    for qid, per in results.items():
        print(f"{BOLD}{qid}: {next(iter(per.values()))['query']}{OFF}")
        for name, _ in variants:
            rec = per[name]
            print(f"  {BOLD}{name}{OFF}  {rec['latency_ms']:.0f} ms  {rec['parent_child']}")
            print_chunks(rec)
        print()


def report(args, s, variants, results, tenant_id, kb_fp, roles, new_vectors) -> int:
    k = s.retrieval_top_k
    scored = [q for q in results if QRELS.get(q.split("~")[0])]
    excluded = [q for q in results if not QRELS.get(q.split("~")[0])]
    base = variants[0][0]
    summaries = {name: summarize(results, scored, name) for name, _ in variants}

    print(f"{BOLD}Scored {len(scored)} queries with KB evidence{OFF} "
          f"({len(excluded)} escalation queries excluded)\n")
    hdr = (f"{'variant':<16}{'Complete@' + str(k):>12}{'Recall@' + str(k):>11}{'Recall@3':>10}"
           f"{'Hit@1':>8}{'MRR':>8}{'ret p50':>10}{'child hits':>12}{'merged':>8}")
    print(hdr)
    print("-" * len(hdr))
    for name, _ in variants:
        m = summaries[name]
        print(f"{name:<16}{m['complete_k']:>12.1%}{m['recall_k']:>11.1%}{m['recall_3']:>10.1%}"
              f"{m['hit_1']:>8.1%}{m['mrr']:>8.3f}{m['retrieval_ms_p50']:>8.0f}ms"
              f"{m['avg_child_hits']:>12.1f}{m['avg_merged_siblings']:>8.1f}")
        if m["parents_missing"]:
            print(f"{'':<16}{RED}{m['parents_missing']} child hit(s) had no parent — run "
                  f"backfill_parent_child.py --verify{OFF}")

    comparisons, all_changed = {}, set()
    for name, _ in variants[1:]:
        improved, worsened, changed, evidence_lost, evidence_gained = [], [], [], [], []
        for qid in results:
            a, b = results[qid][base], results[qid][name]
            if {c["chunk_id"] for c in a["chunks"]} != {c["chunk_id"] for c in b["chunks"]}:
                changed.append(qid)
            if qid in scored:
                ka = (a["metrics"]["recall_k"], a["metrics"]["rr"])
                kb_ = (b["metrics"]["recall_k"], b["metrics"]["rr"])
                (improved if kb_ > ka else worsened if kb_ < ka else []).append(qid)
                # Evidence LEFT the window (the model can no longer answer) vs only moved rank.
                if b["metrics"]["recall_k"] < a["metrics"]["recall_k"]:
                    evidence_lost.append(qid)
                elif b["metrics"]["recall_k"] > a["metrics"]["recall_k"]:
                    evidence_gained.append(qid)
        p = sign_test_p(len(improved), len(worsened))
        verdict = ("likely REAL" if p < 0.05 else "NOT significant (could be noise)")
        colour = GREEN if len(improved) > len(worsened) else RED if worsened else DIM
        print(f"\n{BOLD}{name} vs {base}{OFF}: {colour}{len(improved)} improved, "
              f"{len(worsened)} worsened{OFF}, sign-test p={p:.3f} -> {verdict}")
        if improved:
            print(f"  improved: {' '.join(improved)}")
        if worsened:
            print(f"  {RED}worsened: {' '.join(worsened)}{OFF}")
        print(f"  evidence entered the window: {' '.join(evidence_gained) or '-'}   "
              f"{RED if evidence_lost else ''}evidence LEFT the window: "
              f"{' '.join(evidence_lost) or '-'}{OFF}   (other changes are rank-only)")
        print(f"  window changed on {len(changed)} question(s)")
        sb, sc = summaries[base], summaries[name]
        gate_fail = []
        if sc["complete_k"] < sb["complete_k"]:
            gate_fail.append(f"Complete@{k} {sc['complete_k']:.1%} < {sb['complete_k']:.1%}")
        if sc["recall_k"] < sb["recall_k"]:
            gate_fail.append(f"Recall@{k} {sc['recall_k']:.1%} < {sb['recall_k']:.1%}")
        if len(evidence_lost) > args.max_evidence_loss:
            gate_fail.append(f"{len(evidence_lost)} question(s) lost evidence "
                             f"(allowed {args.max_evidence_loss}): {' '.join(evidence_lost)}")
        if args.gate:
            verdict = f"{GREEN}PASS{OFF}" if not gate_fail else f"{RED}FAIL{OFF}"
            print(f"  GATE: {verdict}")
            for g in gate_fail:
                print(f"    - {g}")
        comparisons[name] = {"improved": improved, "worsened": worsened, "changed": changed,
                             "evidence_lost": evidence_lost, "evidence_gained": evidence_gained,
                             "sign_test_p": p, "gate_failures": gate_fail}
        all_changed.update(q.split("~")[0] for q in changed)

    if args.show:
        for qid in results:
            print(f"\n{BOLD}{qid}{OFF} {results[qid][base]['query']}")
            for name, _ in variants:
                m = results[qid][name].get("metrics") or {}
                print(f"  {BOLD}{name}{OFF} recall={m.get('recall_k', 0):.2f} "
                      f"first=#{m.get('first_rank') or '-'} missing={m.get('missing')}")
                print_chunks(results[qid][name])

    if all_changed:
        print(f"\n{BOLD}End-to-end confirmation{OFF} — only these questions' retrieved context "
              f"changed, so only they can move the e2e accuracy:\n"
              f"  python scripts/eval_rag_accuracy.py --ids {' '.join(sorted(all_changed))}")
    print(f"\nnew query embeddings this run: {new_vectors} "
          f"({'$0 — all served from the disk cache' if not new_vectors else 'well under $0.001'})")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / stamp
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"created_utc": stamp, "tenant_id": tenant_id, "kb_fingerprint": kb_fp,
                "roles": roles, "top_k": k, "golden": GOLDEN_VERSION, "qrels": QRELS_VERSION,
                "variants": dict(variants), "summaries": summaries, "comparisons": comparisons,
                "changed_ids": sorted(all_changed), "new_embeddings": new_vectors,
                "settings": {key: getattr(s, key) for key in (
                    "chunk_child_target_chars", "chunk_child_overlap_chars",
                    "chunk_child_list_group_size", "chunk_child_min_chars",
                    "retrieval_parent_sibling_weight", "retrieval_parent_sibling_cap",
                    "retrieval_parent_child_pool_factor", "retrieval_min_score",
                    "retrieval_pc_vector_scope", "retrieval_pc_text_scope",
                    "retrieval_pc_fusion")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                       encoding="utf-8")
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for qid, per in results.items():
            f.write(json.dumps({"id": qid, **per}, default=str) + "\n")
    print(f"report: {out.relative_to(REPO_ROOT)}\n")
    if args.gate and any(c["gate_failures"] for c in comparisons.values()):
        return 3
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="flat,parent_child")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--query", nargs="*", help="ad-hoc queries (no labels)")
    ap.add_argument("--tenant-id")
    ap.add_argument("--qrels-file",
                    help="a tenant's own labelled set (scripts/build_retrieval_qrels.py) instead "
                         "of the frozen DSR golden set")
    ap.add_argument("--show", action="store_true", help="per-question chunk detail")
    ap.add_argument("--spelling-variants", action="store_true")
    ap.add_argument("--allow-stale", action="store_true")
    ap.add_argument("--gate", action="store_true",
                    help="enable gate: every variant must have Complete@k and Recall@k >= the "
                         "first variant and lose evidence on at most --max-evidence-loss "
                         "questions; exit 3 otherwise")
    ap.add_argument("--max-evidence-loss", type=int, default=0)
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
