"""R5.02 — vector index health report and IVFFlat-vs-HNSW benchmark.

Answers, with measurements rather than estimates:
  1. What vector index is actually live, is it valid, what parameters, how big, and what would
     migration 0015 do to it (``--plan`` prints the exact SQL, for a manual maintenance window).
  2. On the LIVE tables: does production retrieval return every row an exact search returns
     (recall), and how long does the vector query take?
  3. At realistic scale (``--synthetic N``): how IVFFlat (the state the task sheet assumed),
     HNSW with pgvector defaults (what we had), and HNSW with R5.02 tuning compare on latency
     and on recall for a SMALL tenant in a big multi-tenant table (the filtered-search case).

READ-ONLY BY CONSTRUCTION. Everything runs in ONE transaction that is always rolled back. The
mirror/synthetic datasets are TEMP tables named knowledge_chunks / knowledge_documents: Postgres
searches the session's temp schema first, so the production VECTOR_SQL runs unmodified against
them while the real tables are never touched, locked or indexed.

Usage (from core/, or inside the container: docker exec ai-voice-agent-core-1 python ...):
  python scripts/vector_index_report.py                       # inventory + live-table check
  python scripts/vector_index_report.py --plan                # inventory + migration SQL only
  python scripts/vector_index_report.py --mirror              # + forced-index modes on a copy
  python scripts/vector_index_report.py --synthetic 20000     # + scale benchmark
  python scripts/vector_index_report.py --synthetic 50000 --tenants 20 --small-share 0.01 \
      --queries 40 --json ../eval_results/r502_vector_index.json

Reading the mode table:
  exact            seq scan, no index — ground truth for recall; also the no-index latency
  ivfflat_default  IVFFlat lists=N/1000, probes=1 (pgvector default), index forced
  hnsw_default     HNSW m=16/ef_c=64, ef_search=40, no iterative scan — pre-R5.02 behaviour
  hnsw_r502        HNSW with the R5.02 settings from config, index forced
  production       R5.02 settings, planner free to choose — what a live call does
  recall  = share of the exact top-`pool` rows the mode returned (1.000 = nothing lost)
  short   = queries that returned FEWER rows than exact did (the silent-underfill failure)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.engine.rag.retrieval import VECTOR_SQL, VECTOR_SQL_TUNED  # noqa: E402
from app.engine.rag.vector_index import (  # noqa: E402
    COLUMN, HNSW_EF_CONSTRUCTION, HNSW_M, INSPECT_INDEXES_SQL, OPCLASS, PGVECTOR_VERSION_SQL,
    TABLE, IndexInfo, fmt_version, parse_version, plan_index_migration, plan_scan_config,
    scan_gucs, supports_iterative,
)

# ---------------------------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------------------------


@dataclass
class Mode:
    name: str
    sql: object                       # VECTOR_SQL or VECTOR_SQL_TUNED (the production text)
    gucs: dict[str, str]              # transaction-local settings for this mode
    needs: str | None = None          # 'hnsw' | 'ivfflat' index that must exist


@dataclass
class ModeResult:
    mode: str
    queries: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    mean_ms: float = 0.0
    recall: float = 0.0
    short: int = 0
    avg_returned: float = 0.0
    plan: str = ""


# Every mode resets every knob, so no setting from a previous mode can bleed into the next.
_BASE_GUCS = {"enable_seqscan": "on", "enable_indexscan": "on", "jit": "on",
              "plan_cache_mode": "auto"}


def build_modes(version, pool: int, *, forced: bool) -> list[Mode]:
    s = get_settings()
    iterative = supports_iterative(version)
    tuned = plan_scan_config(s, pool=pool, version=version)
    tuned_gucs = scan_gucs(tuned)          # exactly what retrieval applies in production
    tuned_sql = VECTOR_SQL_TUNED if tuned.enabled else VECTOR_SQL
    off = {"hnsw.iterative_scan": "off", "ivfflat.iterative_scan": "off"} if iterative else {}
    # Forcing an index: enable_seqscan=off inflates the seq-scan cost to 1e10, which also trips
    # JIT compilation (~250 ms measured) — noise that has nothing to do with the index. JIT is
    # therefore off in forced modes only; exact/production keep the server's real JIT setting.
    force = {"enable_seqscan": "off", "jit": "off"}

    modes = [Mode("exact", VECTOR_SQL, {"enable_indexscan": "off", **off})]
    if forced:
        modes += [
            Mode("ivfflat_default", VECTOR_SQL, {**force, **off, "ivfflat.probes": "1"},
                 needs="ivfflat"),
            Mode("hnsw_default", VECTOR_SQL, {**force, **off, "hnsw.ef_search": "40"},
                 needs="hnsw"),
            Mode("hnsw_r502", tuned_sql, {**force, **tuned_gucs}, needs="hnsw"),
            # production settings but a GENERIC plan — what Postgres' default plan caching can
            # fall into after 5 executions of a prepared statement (forced here so it is
            # deterministic). Shows what the force_custom_plan setting protects against.
            Mode("prod_generic", tuned_sql,
                 {**tuned_gucs, "plan_cache_mode": "force_generic_plan"}, needs="btree"),
        ]
    modes.append(Mode("production", tuned_sql, dict(tuned_gucs),
                      needs="btree" if forced else None))
    return modes


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


async def set_gucs(conn, gucs: dict[str, str]) -> None:
    for name, value in {**_BASE_GUCS, **gucs}.items():
        await conn.execute(text("SELECT set_config(:n, :v, true)"), {"n": name, "v": value})


def _plan_nodes(plan: dict) -> list[str]:
    """Flatten an EXPLAIN (FORMAT JSON) plan into its scan nodes, e.g. 'Index Scan(ix_...)'."""
    out, node = [], plan
    stack = [node]
    while stack:
        n = stack.pop()
        nt = n.get("Node Type", "")
        if "Scan" in nt:
            target = n.get("Index Name") or n.get("Relation Name") or n.get("CTE Name") or "?"
            out.append(f"{nt}({target})")
        stack.extend(reversed(n.get("Plans", [])))
    return out


async def explain_nodes(conn, sql, params) -> str:
    raw = (await conn.execute(text("EXPLAIN (FORMAT JSON) " + sql.text), params)).scalar()
    doc = raw if isinstance(raw, list) else json.loads(raw)
    return " > ".join(_plan_nodes(doc[0]["Plan"]))


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))]


async def run_mode(conn, mode: Mode, queries: list[dict],
                   truth: dict[int, list[str]] | None
                   ) -> tuple[ModeResult, dict[int, list[str]]]:
    """Run every query under ``mode``. Returns the stats and the ids each query returned
    (the ids of the 'exact' mode become the ground truth for the others)."""
    await set_gucs(conn, mode.gucs)
    res = ModeResult(mode=mode.name, queries=len(queries))
    if not queries:
        return res, {}
    res.plan = await explain_nodes(conn, mode.sql, queries[0])
    for q in queries[: min(3, len(queries))]:          # warm the cache / graph pages
        await conn.execute(mode.sql, q)
    times, recalls, returned = [], [], []
    ids_by_q: dict[int, list[str]] = {}
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        rows = (await conn.execute(mode.sql, q)).all()
        times.append((time.perf_counter() - t0) * 1000)
        ids = [str(r[0]) for r in rows]
        ids_by_q[i] = ids
        returned.append(len(ids))
        if truth is not None:
            expected = truth[i]
            recalls.append(len(set(ids) & set(expected)) / len(expected) if expected else 1.0)
            if len(ids) < len(expected):
                res.short += 1
    res.p50_ms, res.p95_ms = round(pct(times, .5), 2), round(pct(times, .95), 2)
    res.mean_ms = round(statistics.fmean(times), 2)
    res.avg_returned = round(statistics.fmean(returned), 1)
    res.recall = round(statistics.fmean(recalls), 4) if recalls else 1.0
    return res, ids_by_q


async def run_modes(conn, modes: list[Mode], queries: list[dict], have: set[str],
                    truth: dict[int, list[str]] | None = None
                    ) -> tuple[list[ModeResult], dict[int, list[str]] | None]:
    """Run the modes whose required index exists. An 'exact' mode sets the ground truth;
    pass a previously computed ``truth`` to score modes run in a later phase against it."""
    results: list[ModeResult] = []
    for mode in modes:
        if mode.needs and mode.needs not in have:
            continue
        r, ids = await run_mode(conn, mode, queries, truth)
        if mode.name == "exact":
            truth = ids
        results.append(r)
    return results, truth


def print_modes(title: str, results: list[ModeResult]) -> None:
    print(f"\n  {title}")
    print(f"  {'mode':<16}{'p50 ms':>9}{'p95 ms':>9}{'recall':>9}{'short':>7}{'rows':>7}  plan")
    for r in results:
        print(f"  {r.mode:<16}{r.p50_ms:>9.2f}{r.p95_ms:>9.2f}{r.recall:>9.3f}"
              f"{r.short:>4}/{r.queries:<3}{r.avg_returned:>6.1f}  {r.plan}")


def query_params(vec_text: str, tenant_id, pool: int) -> dict:
    return {"qvec": vec_text, "tenant_id": str(tenant_id), "today": date.today(),
            "doc_type": None, "project_id": None, "limit": pool}


async def build_index(conn, kind: str, n_rows: int, build_mem: str) -> dict:
    """Build an index on the TEMP knowledge_chunks. Returns build time and size."""
    await conn.execute(text("SELECT set_config('maintenance_work_mem', :m, true)"),
                       {"m": build_mem})
    if kind == "hnsw":
        ddl = (f"CREATE INDEX r502_hnsw ON knowledge_chunks USING hnsw ({COLUMN} {OPCLASS}) "
               f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})")
    else:
        lists = max(1, n_rows // 1000)                 # pgvector's guidance up to 1M rows
        ddl = (f"CREATE INDEX r502_ivfflat ON knowledge_chunks USING ivfflat "
               f"({COLUMN} {OPCLASS}) WITH (lists = {lists})")
    t0 = time.perf_counter()
    await conn.execute(text(ddl))
    secs = time.perf_counter() - t0
    size = (await conn.execute(text(
        f"SELECT pg_relation_size('r502_{kind}')"))).scalar()
    await conn.execute(text("ANALYZE knowledge_chunks"))
    print(f"  built {kind:<8} in {secs:6.1f}s  size={size / 1048576:.1f} MB   ({ddl})")
    return {"kind": kind, "seconds": round(secs, 2), "size_mb": round(size / 1048576, 2)}


# ---------------------------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------------------------


async def inventory(conn, report: dict) -> tuple:
    version = parse_version((await conn.execute(PGVECTOR_VERSION_SQL)).scalar())
    rows = (await conn.execute(INSPECT_INDEXES_SQL,
                               {"table": TABLE, "column": COLUMN})).mappings().all()
    existing = [IndexInfo.from_row(r) for r in rows]
    plan = plan_index_migration(existing)
    counts = (await conn.execute(text(
        "SELECT tenant_id, count(*) AS n, count(embedding) AS embedded "
        "FROM knowledge_chunks GROUP BY tenant_id ORDER BY n DESC"))).mappings().all()
    s = get_settings()

    print("=" * 100)
    print("R5.02 VECTOR INDEX REPORT")
    print("=" * 100)
    print(f"pgvector {fmt_version(version)}  (iterative scan "
          f"{'supported' if supports_iterative(version) else 'NOT supported: needs >= 0.8.0'})")
    print(f"config: tuning_enabled={s.retrieval_hnsw_tuning_enabled} "
          f"ef_search={s.retrieval_hnsw_ef_search} iterative={s.retrieval_hnsw_iterative_scan} "
          f"max_scan_tuples={s.retrieval_hnsw_max_scan_tuples} top_k={s.retrieval_top_k}")
    print(f"\nindexes on {TABLE}.{COLUMN}:")
    for ix in existing:
        print(f"  {ix.name:<34} {ix.method:<8} valid={ix.valid!s:<5} opclass={ix.opclass} "
              f"options={list(ix.options) or 'defaults'} size={ix.size_bytes / 1024:.0f} kB "
              f"target={'yes' if ix.is_target() else 'NO'}")
    if not existing:
        print("  (none — vector search is a full table scan)")
    print(f"\nmigration 0015 plan: {plan.action.upper()} — {plan.reason}")
    for stmt in plan.statements:
        print(f"  {stmt};")
    print(f"\nchunks per tenant ({sum(c['n'] for c in counts)} total):")
    for c in counts:
        print(f"  {c['tenant_id']}  rows={c['n']:<8} embedded={c['embedded']}")

    report["inventory"] = {
        "pgvector": fmt_version(version),
        "indexes": [{k: v for k, v in asdict(ix).items()} for ix in existing],
        "plan": {"action": plan.action, "reason": plan.reason, "statements": plan.statements},
        "tenants": [{"tenant_id": str(c["tenant_id"]), "rows": c["n"]} for c in counts],
    }
    return version, counts


async def live_section(conn, version, counts, args, report: dict) -> None:
    """Production behaviour on the real tables: exact vs what a live call gets."""
    pool = max(get_settings().retrieval_top_k * 4, 20)
    print(f"\n{'-' * 100}\nLIVE TABLES (real data; production SQL; pool={pool})")
    out = []
    for c in counts:
        if not c["embedded"]:
            continue
        vecs = (await conn.execute(text(
            "SELECT embedding::text FROM knowledge_chunks WHERE tenant_id = :t "
            "AND embedding IS NOT NULL ORDER BY random() LIMIT :q"),
            {"t": c["tenant_id"], "q": args.queries})).scalars().all()
        queries = [query_params(v, c["tenant_id"], pool) for v in vecs]
        results, _ = await run_modes(conn, build_modes(version, pool, forced=False), queries,
                                     have=set())
        print_modes(f"tenant {c['tenant_id']} ({c['n']} chunks, {len(queries)} queries)",
                    results)
        out.append({"tenant_id": str(c["tenant_id"]), "rows": c["n"],
                    "modes": [asdict(r) for r in results]})
    report["live"] = out


async def _make_temp_tables(conn, dims: int | None = None) -> None:
    await conn.execute(text("DROP TABLE IF EXISTS pg_temp.knowledge_chunks"))
    await conn.execute(text("DROP TABLE IF EXISTS pg_temp.knowledge_documents"))
    # LIKE copies column types (incl. vector(1536)) and defaults, but NO indexes and NO FKs.
    await conn.execute(text("CREATE TEMP TABLE knowledge_documents "
                            "(LIKE public.knowledge_documents INCLUDING DEFAULTS)"))
    await conn.execute(text("CREATE TEMP TABLE knowledge_chunks "
                            "(LIKE public.knowledge_chunks INCLUDING DEFAULTS)"))
    if dims:
        await conn.execute(text(f"ALTER TABLE knowledge_chunks "
                                f"ALTER COLUMN embedding TYPE vector({int(dims)})"))


async def _bench_temp(conn, version, n_rows: int, groups: list[tuple[str, list[dict]]],
                      pool: int, args, key: str, report: dict) -> None:
    """Shared by mirror and synthetic: exact baseline, then IVFFlat, then HNSW modes."""
    modes = build_modes(version, pool, forced=True)
    builds: list[dict] = []
    results_by_group: dict[str, list[ModeResult]] = {g: [] for g, _ in groups}
    truth_by_group: dict[str, dict | None] = {g: None for g, _ in groups}

    async def phase(have: set[str], names: set[str]) -> None:
        """Run the named modes for every query group. Phases exist because only ONE vector
        index is present at a time (IVFFlat, then HNSW), so each is measured cleanly alone."""
        for gname, queries in groups:
            rs, truth_by_group[gname] = await run_modes(
                conn, [m for m in modes if m.name in names], queries, have,
                truth=truth_by_group[gname])
            results_by_group[gname] += rs

    await conn.execute(text("ANALYZE knowledge_chunks"))
    await conn.execute(text("ANALYZE knowledge_documents"))
    await phase(set(), {"exact"})                          # ground truth, no index at all
    builds.append(await build_index(conn, "ivfflat", n_rows, args.build_mem))
    await phase({"ivfflat"}, {"ivfflat_default"})
    await conn.execute(text("DROP INDEX r502_ivfflat"))
    builds.append(await build_index(conn, "hnsw", n_rows, args.build_mem))
    await phase({"hnsw"}, {"hnsw_default", "hnsw_r502"})
    # Production also has btree indexes on (tenant_id) and (tenant_id, project_id). They are
    # added only now: with them present, "force the index" could pick a btree instead of the
    # vector index and the forced rows above would stop measuring what they claim to.
    await conn.execute(text("CREATE INDEX r502_tenant ON knowledge_chunks (tenant_id)"))
    await conn.execute(text("CREATE INDEX r502_tenant_project "
                            "ON knowledge_chunks (tenant_id, project_id)"))
    await conn.execute(text("ANALYZE knowledge_chunks"))
    await phase({"hnsw", "btree"}, {"prod_generic", "production"})

    for gname, _ in groups:
        print_modes(gname, results_by_group[gname])
    report[key] = {"rows": n_rows, "pool": pool, "builds": builds,
                   "groups": {g: [asdict(r) for r in rs] for g, rs in results_by_group.items()}}


async def mirror_section(conn, version, counts, args, report: dict) -> None:
    """A temp copy of the live data, so the index modes can be forced without touching it."""
    pool = max(get_settings().retrieval_top_k * 4, 20)
    print(f"\n{'-' * 100}\nMIRROR (temp copy of live data; indexes forced; pool={pool})")
    await _make_temp_tables(conn)
    await conn.execute(text("INSERT INTO knowledge_documents SELECT * FROM public.knowledge_documents"))
    await conn.execute(text("INSERT INTO knowledge_chunks SELECT * FROM public.knowledge_chunks"))
    n = (await conn.execute(text("SELECT count(*) FROM knowledge_chunks"))).scalar()
    groups = []
    for c in counts:
        vecs = (await conn.execute(text(
            "SELECT embedding::text FROM knowledge_chunks WHERE tenant_id = :t "
            "AND embedding IS NOT NULL ORDER BY random() LIMIT :q"),
            {"t": c["tenant_id"], "q": args.queries})).scalars().all()
        if vecs:
            groups.append((f"tenant {c['tenant_id']} ({c['n']} chunks)",
                           [query_params(v, c["tenant_id"], pool) for v in vecs]))
    await _bench_temp(conn, version, n, groups, pool, args, "mirror", report)


async def synthetic_section(conn, version, args, report: dict) -> None:
    """N clustered vectors across T tenants: one SMALL tenant (the filtered-search worst case),
    one BIG tenant (~50%), the rest spread evenly. All tenants share the same semantic space —
    like real-estate tenants do — so the small tenant's rows are sparse among its neighbours."""
    n, dims, T = args.synthetic, args.dims, max(3, args.tenants)
    pool = max(get_settings().retrieval_top_k * 4, 20)
    tenants = [uuid.uuid4() for _ in range(T)]
    small, big = tenants[0], tenants[1]
    print(f"\n{'-' * 100}\nSYNTHETIC SCALE TEST  rows={n} dims={dims} tenants={T} "
          f"small_share={args.small_share:.1%} pool={pool}")
    t0 = time.perf_counter()
    await _make_temp_tables(conn, dims)
    await conn.execute(text(
        "CREATE TEMP TABLE r502_centroids AS "
        "SELECT k, ARRAY(SELECT (random() - 0.5)::real FROM generate_series(1, :dims) j "
        "                WHERE k >= 0) AS v "                 # WHERE k>=0 correlates: 1 per k
        "FROM generate_series(0, :K - 1) k"), {"dims": dims, "K": args.clusters})
    ndocs = max(1, n // 20)
    await conn.execute(text("""
        INSERT INTO knowledge_documents (id, tenant_id, title, doc_type, status, version, metadata)
        SELECT gen_random_uuid(),
               CASE WHEN d::float / :ndocs < :small THEN (CAST(:tenants AS uuid[]))[1]
                    WHEN d::float / :ndocs < :small + 0.5 THEN (CAST(:tenants AS uuid[]))[2]
                    ELSE (CAST(:tenants AS uuid[]))[3 + d % (:T - 2)] END,
               'synthetic doc ' || d, 'brochure', 'ready', 1, jsonb_build_object('doc_no', d)
        FROM generate_series(0, :ndocs - 1) d
    """), {"ndocs": ndocs, "small": args.small_share, "tenants": tenants, "T": T})
    await conn.execute(text("""
        INSERT INTO knowledge_chunks (id, tenant_id, document_id, chunk_index, content,
                                      metadata, embedding)
        SELECT gen_random_uuid(), d.tenant_id, d.id, g % 20, 'synthetic chunk ' || g, '{}',
               (SELECT array_agg(c.v[j] + ((random() - 0.5) * :noise)::real ORDER BY j)
                FROM generate_series(1, :dims) j)::vector
        FROM generate_series(0, :n - 1) g
        JOIN knowledge_documents d ON (d.metadata->>'doc_no')::int = g / 20
        JOIN r502_centroids c ON c.k = (g * 7919) % :K
    """), {"n": n, "dims": dims, "noise": args.noise, "K": args.clusters})
    per = (await conn.execute(text(
        "SELECT tenant_id, count(*) FROM knowledge_chunks GROUP BY 1"))).all()
    per = {str(t): c for t, c in per}
    print(f"  generated in {time.perf_counter() - t0:.1f}s: small tenant={per.get(str(small), 0)} "
          f"rows, big tenant={per.get(str(big), 0)} rows")

    async def qvecs(k: int, offset: int) -> list[str]:
        """k query vectors, each a fresh noisy sample around a centroid. Centroids are picked
        deterministically (exactly one per query) and spread across clusters by ``offset``."""
        return list((await conn.execute(text("""
            SELECT (SELECT array_agg(c.v[j] + ((random() - 0.5) * :noise)::real ORDER BY j)
                    FROM generate_series(1, :dims) j)::vector::text
            FROM generate_series(1, :q) g
            JOIN r502_centroids c ON c.k = (g * 31 + :off) % CAST(:K AS integer)
        """), {"q": k, "dims": dims, "noise": args.noise, "K": args.clusters,
               "off": offset})).scalars().all())

    groups = [
        (f"SMALL tenant ({per.get(str(small), 0)} rows = {args.small_share:.1%} of table)",
         [query_params(v, small, pool) for v in await qvecs(args.queries, 0)]),
        (f"BIG tenant ({per.get(str(big), 0)} rows)",
         [query_params(v, big, pool) for v in await qvecs(args.queries, 7)]),
    ]
    await _bench_temp(conn, version, n, groups, pool, args, "synthetic", report)
    report["synthetic"].update({"dims": dims, "tenants": T, "small_share": args.small_share,
                                "clusters": args.clusters, "noise": args.noise})


def goal_check(report: dict) -> None:
    """Compare against the task sheet: 'TTFA Δ -15 ms' (HNSW ~5-8 ms vs IVFFlat ~20-30 ms)."""
    syn = report.get("synthetic")
    print(f"\n{'=' * 100}\nGOAL CHECK — R5.02 target: -15 ms vector search vs IVFFlat, no recall loss")
    if not syn:
        print("  Run with --synthetic N to measure at the corpus size the task targets (10k-100k).")
        return
    for gname, rows in syn["groups"].items():
        by = {r["mode"]: r for r in rows}
        ivf, hd, r5, prod = (by.get(k) for k in
                             ("ivfflat_default", "hnsw_default", "hnsw_r502", "production"))
        print(f"  {gname}")
        if ivf and r5:
            print(f"    HNSW(R5.02) vs IVFFlat   p50 {r5['p50_ms'] - ivf['p50_ms']:+.2f} ms "
                  f"({ivf['p50_ms']:.2f} -> {r5['p50_ms']:.2f})   recall "
                  f"{ivf['recall']:.3f} -> {r5['recall']:.3f}   short "
                  f"{ivf['short']} -> {r5['short']}")
        if hd and r5:
            print(f"    R5.02 tuning vs defaults p50 {r5['p50_ms'] - hd['p50_ms']:+.2f} ms   "
                  f"recall {hd['recall']:.3f} -> {r5['recall']:.3f}   short "
                  f"{hd['short']} -> {r5['short']}")
        if prod:
            print(f"    production path          p50 {prod['p50_ms']:.2f} ms  recall "
                  f"{prod['recall']:.3f}  plan: {prod['plan']}")


# ---------------------------------------------------------------------------------------------


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="inventory + migration SQL only")
    ap.add_argument("--mirror", action="store_true", help="forced-index modes on a temp copy")
    ap.add_argument("--synthetic", type=int, default=0, metavar="N", help="synthetic rows")
    ap.add_argument("--dims", type=int, default=1536)
    ap.add_argument("--tenants", type=int, default=20)
    ap.add_argument("--small-share", type=float, default=0.01)
    ap.add_argument("--clusters", type=int, default=64)
    ap.add_argument("--noise", type=float, default=0.6)
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--build-mem", default="512MB", help="maintenance_work_mem for test builds")
    ap.add_argument("--json", help="write the full report as JSON to this path")
    args = ap.parse_args()

    engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
    report: dict = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(args)}
    async with engine.connect() as conn:
        # temp_buffers must be set before the session touches any temp table; the default 8 MB
        # would make temp-table benchmarks disk-bound and unrepresentative of shared_buffers.
        await conn.execute(text("SET temp_buffers = '1GB'"))   # session-level: survives commit
        await conn.commit()                                   # close the autobegun transaction
        trans = await conn.begin()
        try:
            version, counts = await inventory(conn, report)
            if not args.plan:
                await live_section(conn, version, counts, args, report)
                if args.mirror:
                    await mirror_section(conn, version, counts, args, report)
                if args.synthetic:
                    await synthetic_section(conn, version, args, report)
                goal_check(report)
        finally:
            await trans.rollback()            # read-only by construction: nothing persists
    await engine.dispose()

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nreport written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
