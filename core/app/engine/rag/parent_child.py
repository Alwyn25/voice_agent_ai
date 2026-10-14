"""R5.03 — parent-child ("small-to-big") retrieval.

THE PROBLEM. One ~900-char chunk is the right amount of text to SHOW the model, and the wrong
amount to SEARCH with. Its embedding is the average of every fact inside it: in the 3 BHK cost
sheet, "carpet area of the 1,390 sq ft unit" is a small share of a vector that equally means the
1,239 and 1,560 sq ft units, clubhouse charges and car parking. A less relevant but "cleaner"
chunk can then outrank it, and the specific fact the caller asked for (a price, an area, a unit)
falls out of the window. Making every chunk small fixes the ranking and breaks the answer: a
200-char fragment has lost its table header, its neighbouring sentences and its context.

THE FIX — search small, answer big:

    ingestion   chunk_text() ──► PARENT chunks (~900 chars)      role='parent'  (embedded)
                                   └─► split_children() ──► CHILD chunks (~200 chars)
                                                             role='child', parent_id=<parent>
    retrieval   hybrid search over CHILDREN (+ standalone) ──► RRF scores per child
                ──► collapse_to_parents(): group by parent, one entry per parent,
                    parent score = best child (+ small capped sibling bonus)
                ──► the unchanged boosts / per-document cap / min-score run on PARENTS
                ──► the model reads the parent's ~900 chars, deduplicated

WHY PARENTS ARE EMBEDDED TOO. The parent rows are exactly the chunks the system stored before
R5.03, so keeping their vectors means "flat" mode (search standalone + parents) is byte-for-byte
the old index, and "parent_child" mode (search standalone + children) is the new one — over the
SAME data. Turning R5.03 on, A/B-testing it, and rolling it back are all a config flip
(``RETRIEVAL_GRANULARITY``), never a re-ingestion. The price is one extra embedding per parent
at ingestion: fractions of a cent per document.

MIXED DATA IS SAFE. A 'standalone' chunk (anything ingested before R5.03, or too small to split)
is searched in both modes and is its own "parent", so a half-finished backfill never hides text.

Everything here except :func:`fetch_parents` is pure (no DB, no network) and unit-tested in
tests/test_parent_child.py.
"""
import logging
import time
import uuid
from dataclasses import dataclass, field

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.engine.rag.chunkers import Chunk, split_children

log = logging.getLogger(__name__)

ROLE_STANDALONE = "standalone"
ROLE_PARENT = "parent"
ROLE_CHILD = "child"

GRANULARITY_FLAT = "flat"
GRANULARITY_PARENT_CHILD = "parent_child"

# How many characters of each matched child are kept in the debug trace. The full child is at
# most ~chunk_child_target_chars; this only bounds a pathological one.
_TRACE_CHILD_CHARS = 300


GRANULARITIES = (GRANULARITY_FLAT, GRANULARITY_PARENT_CHILD)


def resolve_granularity(value, default: str) -> str:
    """A tenant's ``retrieval_granularity`` override, validated; ``default`` (the global setting)
    when it is unset or invalid. Tenant settings are operator-edited JSON: a typo must degrade
    to the global behaviour with a warning, never raise on the hot path."""
    if value is None or value == "":
        return default
    if value in GRANULARITIES:
        return value
    log.warning("rag.granularity invalid tenant override %r (expected one of %s); using %r",
                value, GRANULARITIES, default)
    return default


def excluded_role(granularity: str) -> str:
    """The chunk_role a retrieval mode must NOT search.

    flat          -> never search children (the pre-R5.03 index: standalone + parents)
    parent_child  -> never search parents  (standalone + children; parents are fetched after)
    Anything unrecognised falls back to flat — the safe, pre-R5.03 behaviour.
    """
    return ROLE_PARENT if granularity == GRANULARITY_PARENT_CHILD else ROLE_CHILD


# Per-arm search scope in parent_child mode (RETRIEVAL_PC_VECTOR_SCOPE / _TEXT_SCOPE). Standalone
# chunks are searched in every scope. Implemented through the same `chunk_role <> :exclude_role`
# predicate as the modes above; "both" excludes a role that cannot exist (ck_chunks_role).
SCOPE_CHILDREN = "children"
SCOPE_PARENTS = "parents"
SCOPE_BOTH = "both"
SCOPE_SEPARATE = "separate"   # vector arm only: children and parents as two ranked lists
_NO_ROLE = "-"

# FUSION ORDER in parent_child mode (RETRIEVAL_PC_FUSION):
#   parent_rank — rank PARENTS inside each arm, then RRF (fuse_parent_rank). The default.
#   child_rank  — RRF over raw child hits, collapse to parents afterwards (collapse_to_parents).
#                 The first R5.03 implementation, kept for A/B: measured on the DSR KB it let the
#                 5-8 children of one cost sheet take 5-8 top keyword ranks, pushing every other
#                 document down ("how many flats in TOTAL" lost the "900 flats" chunk to cost-sheet
#                 lines that all say "Total ...").
FUSION_PARENT_RANK = "parent_rank"
FUSION_CHILD_RANK = "child_rank"


def scope_exclusion(scope: str) -> str:
    """The chunk_role an arm must NOT search for a given scope (unknown -> children).
    "separate" -> the children list here; retrieve() runs the parents list itself."""
    if scope == SCOPE_BOTH:
        return _NO_ROLE
    if scope == SCOPE_PARENTS:
        return ROLE_CHILD
    return ROLE_PARENT


# ─────────────────────────────────────────────────────────────── ingestion plan (pure)

@dataclass
class ChunkFamily:
    """One chunk from chunk_text() plus the children it was split into (possibly none)."""
    parent: Chunk
    children: list[Chunk] = field(default_factory=list)

    @property
    def role(self) -> str:
        return ROLE_PARENT if self.children else ROLE_STANDALONE


def child_split_kwargs(s, title: str | None = None) -> dict:
    """split_children() keyword arguments from settings — ONE place, so ingestion and the
    backfill script can never cut children differently."""
    return {"target_chars": s.chunk_child_target_chars,
            "overlap_chars": s.chunk_child_overlap_chars,
            "list_group_size": s.chunk_child_list_group_size,
            "min_chars": s.chunk_child_min_chars,
            "split_min_chars": s.chunk_child_split_min_chars,
            "title": title if s.chunk_child_title_prefix else None,
            "record_split": s.chunk_child_record_split}


def plan_families(chunks: list[Chunk], *, enabled: bool, **split_kwargs) -> list[ChunkFamily]:
    """Decide, per chunk, whether it becomes a parent with children or stays standalone.

    ``enabled=False`` returns every chunk as standalone — the exact pre-R5.03 ingestion.
    ``split_kwargs`` go to split_children() (see child_split_kwargs).
    """
    if not enabled:
        return [ChunkFamily(c) for c in chunks]
    return [ChunkFamily(c, split_children(c, **split_kwargs)) for c in chunks]


# ─────────────────────────────────────────────────────────────── retrieval collapse (pure)

@dataclass
class CollapseStats:
    """What the child→parent step did on one retrieval — surfaced in the trace and logs."""
    granularity: str = GRANULARITY_FLAT
    fusion: str = FUSION_CHILD_RANK
    vector_scope: str = SCOPE_CHILDREN
    text_scope: str = SCOPE_CHILDREN
    hits: int = 0                # search hits going in (children + standalone [+ parents])
    child_hits: int = 0          # ... of which were child rows
    parents: int = 0             # distinct parents those child hits expanded to
    merged_siblings: int = 0     # child hits folded into a parent another child already brought
    parents_missing: int = 0     # child hits whose parent could not be loaded (served as-is)
    units: int = 0               # entries coming out (parents + standalone + orphaned children)
    fetch_ms: float = 0.0        # time spent loading parent rows
    fetch_error: str | None = None

    def as_dict(self) -> dict:
        return {"granularity": self.granularity, "fusion": self.fusion,
                "vector_scope": self.vector_scope, "text_scope": self.text_scope,
                "hits": self.hits,
                "child_hits": self.child_hits, "parents": self.parents,
                "merged_siblings": self.merged_siblings,
                "parents_missing": self.parents_missing, "units": self.units,
                "fetch_ms": round(self.fetch_ms, 2), "fetch_error": self.fetch_error}


def parent_score(member_scores: list[float], *, sibling_weight: float,
                 sibling_cap: float) -> float:
    """Score of a parent from its matching members' fused (RRF) scores.

    best + sibling_weight x (sum of the others), with that bonus capped at
    sibling_cap x best. The best child dominates by design — the whole point of R5.03 is that
    ONE precise match should win; siblings only break near-ties in favour of a parent several
    children agree on. The cap stops a long parent with many weak matches outvoting a short
    document with one strong one.
    """
    if not member_scores:
        return 0.0
    ordered = sorted(member_scores, reverse=True)
    best, rest = ordered[0], ordered[1:]
    bonus = min(sibling_weight * sum(rest), sibling_cap * best)
    return best + bonus


def _min_rank(values) -> int | None:
    ranks = [v for v in values if v is not None]
    return min(ranks) if ranks else None


def collapse_to_parents(scores: dict[str, float], rows_by_id: dict[str, dict],
                        parent_rows: dict[str, dict], *, sibling_weight: float,
                        sibling_cap: float) -> tuple[dict[str, float], dict[str, dict],
                                                     CollapseStats]:
    """Replace child hits by their parents, one entry per parent.

    Input is what retrieve() has after RRF fusion: ``scores`` and ``rows_by_id`` keyed by the
    HIT's chunk id; ``parent_rows`` maps parent id → the parent's row (from fetch_parents).
    Output is the same two structures, keyed by the UNIT id (the parent's id for child hits,
    the chunk's own id otherwise), so every step after this in retrieve() runs unchanged.

    Each output row carries:
      * the parent's content / metadata / title / doc_type / document_id (what the model reads),
      * vector_rank / text_rank = the best rank any member achieved (for the trace),
      * ``matched_children`` = the child hits that pulled the parent in, best first — the
        "which sentence actually matched?" answer when debugging a retrieval.

    A child whose parent is missing (should never happen: the FK cascades and re-ingestion is
    atomic — but a stale read or a manual edit could cause it) is kept as its own unit rather
    than dropped, and counted in ``parents_missing`` so it is visible.
    """
    stats = CollapseStats(granularity=GRANULARITY_PARENT_CHILD, hits=len(scores))
    groups: dict[str, list[str]] = {}            # unit id -> member hit ids
    for hit_id in scores:
        row = rows_by_id[hit_id]
        pid = row.get("parent_id")
        if row.get("chunk_role") == ROLE_CHILD and pid is not None:
            stats.child_hits += 1
            pid = str(pid)
            if pid in parent_rows:
                groups.setdefault(pid, []).append(hit_id)
                continue
            stats.parents_missing += 1
            row["parent_missing"] = True
        groups.setdefault(hit_id, []).append(hit_id)

    out_scores: dict[str, float] = {}
    out_rows: dict[str, dict] = {}
    for unit_id, members in groups.items():
        if unit_id in parent_rows:
            children = [m for m in members if m != unit_id]
            stats.parents += 1
            stats.merged_siblings += max(0, len(children) - 1)
            row = dict(parent_rows[unit_id])
            row["vector_rank"] = _min_rank(rows_by_id[m].get("vector_rank") for m in members)
            row["text_rank"] = _min_rank(rows_by_id[m].get("text_rank") for m in members)
            row["matched_children"] = sorted(
                ({"chunk_id": m, "score": round(scores[m], 6),
                  "vector_rank": rows_by_id[m].get("vector_rank"),
                  "text_rank": rows_by_id[m].get("text_rank"),
                  "content": (rows_by_id[m].get("content") or "")[:_TRACE_CHILD_CHARS]}
                 for m in children),
                key=lambda c: c["score"], reverse=True)
            out_rows[unit_id] = row
            out_scores[unit_id] = parent_score([scores[m] for m in members],
                                               sibling_weight=sibling_weight,
                                               sibling_cap=sibling_cap)
        else:                                    # standalone chunk, or an orphaned child
            (only,) = members
            out_rows[unit_id] = rows_by_id[only]
            out_scores[unit_id] = scores[only]
    stats.units = len(out_scores)
    return out_scores, out_rows, stats


def fuse_parent_rank(arms: list[tuple[str, list[dict], float]], parent_rows: dict[str, dict],
                     *, rrf_k: int) -> tuple[dict[str, float], dict[str, dict], CollapseStats]:
    """Reciprocal-rank fusion over PARENT units, ranked inside each arm first (the default).

    ``arms`` is ``[(name, rows_in_rank_order, weight), ...]`` with name "vector" or "text".

    Inside one arm, every hit is mapped to its unit (the parent for a child, itself otherwise)
    and units get DENSE ranks in order of their best hit: the parent of the #1 and #2 child
    hits is rank 1, the next distinct unit is rank 2, and so on. A unit's further hits in that
    arm add nothing to its score — they are recorded in ``matched_children`` only. Then the usual
    RRF sums weight / (rrf_k + unit_rank) across arms.

    Why: RRF is rank-based, so with raw child hits one parent's 5-8 children occupy 5-8 top
    ranks of an arm and push every other document down by as many places (the failure measured
    in child_rank fusion). Here a parent holds exactly one rank per arm — as a chunk did before
    R5.03 — but that rank is set by its most PRECISE child, which is the gain R5.03 is for.

    ``vector_rank`` / ``text_rank`` on the returned rows are these unit ranks; each matched
    child keeps its raw per-arm rank for the trace. A child whose parent is missing becomes its
    own unit and is counted in ``parents_missing`` (see collapse_to_parents).
    """
    stats = CollapseStats(granularity=GRANULARITY_PARENT_CHILD, fusion=FUSION_PARENT_RANK)
    scores: dict[str, float] = {}
    out_rows: dict[str, dict] = {}
    members: dict[str, dict[str, dict]] = {}     # unit -> child id -> trace entry
    children_seen: set[str] = set()
    missing_seen: set[str] = set()

    for arm, rows, weight in arms:
        ranked_units: set[str] = set()
        unit_rank = 0
        for raw_rank, row in enumerate(rows, start=1):
            stats.hits += 1
            hit_id = str(row["id"])
            is_child = row.get("chunk_role") == ROLE_CHILD and row.get("parent_id") is not None
            unit = hit_id
            if is_child:
                pid = str(row["parent_id"])
                if pid in parent_rows:
                    unit = pid
                    children_seen.add(hit_id)
                    entry = members.setdefault(unit, {}).setdefault(hit_id, {
                        "chunk_id": hit_id, "score": 0.0, "vector_rank": None,
                        "text_rank": None,
                        "content": (row.get("content") or "")[:_TRACE_CHILD_CHARS]})
                    entry[f"{arm}_rank"] = raw_rank
                    entry["score"] = round(entry["score"] + weight / (rrf_k + raw_rank), 6)
                else:
                    missing_seen.add(hit_id)
            if unit in ranked_units:
                stats.merged_siblings += 1       # a later hit of a unit already ranked here
                continue
            ranked_units.add(unit)
            unit_rank += 1
            scores[unit] = scores.get(unit, 0.0) + weight / (rrf_k + unit_rank)
            if unit not in out_rows:
                out_rows[unit] = dict(parent_rows[unit]) if unit in parent_rows else dict(row)
                if unit in missing_seen:
                    out_rows[unit]["parent_missing"] = True
            out_rows[unit][f"{arm}_rank"] = unit_rank

    for unit, kids in members.items():
        out_rows[unit]["matched_children"] = sorted(kids.values(), key=lambda c: c["score"],
                                                    reverse=True)
    stats.child_hits = len(children_seen) + len(missing_seen)
    stats.parents = len(members)
    stats.parents_missing = len(missing_seen)
    stats.units = len(scores)
    return scores, out_rows, stats


# ─────────────────────────────────────────────────────────────── parent fetch (DB)

PARENTS_SQL = sql_text("""
    SELECT c.id, c.document_id, c.content, c.metadata, c.chunk_role, c.parent_id,
           d.title, d.doc_type
    FROM knowledge_chunks c
    JOIN knowledge_documents d ON d.id = c.document_id
    WHERE c.tenant_id = CAST(:tenant_id AS uuid)
      AND c.id = ANY(CAST(:ids AS uuid[]))
""")


def parent_ids_of(rows) -> list[str]:
    """Distinct parent ids referenced by the child rows among the hits (stable order).
    Accepts a dict of rows (keyed by id) or any iterable of rows."""
    seen: dict[str, None] = {}
    for row in (rows.values() if isinstance(rows, dict) else rows):
        if row.get("chunk_role") == ROLE_CHILD and row.get("parent_id") is not None:
            seen[str(row["parent_id"])] = None
    return list(seen)


async def fetch_parents(db: AsyncSession, *, tenant_id: uuid.UUID,
                        ids: list[str]) -> tuple[dict[str, dict], float]:
    """Load parent rows by id in ONE round-trip (primary-key lookups, ~1-2 ms).

    Scoped by tenant as defence in depth: a parent_id can only ever point inside its own
    document, but a lookup that could cross tenants must not exist on the hot path.
    Returns (rows keyed by id, elapsed ms).
    """
    if not ids:
        return {}, 0.0
    started = time.perf_counter()
    rows = (await db.execute(PARENTS_SQL, {
        "tenant_id": str(tenant_id), "ids": [uuid.UUID(i) for i in ids],
    })).mappings().all()
    return {str(r["id"]): dict(r) for r in rows}, (time.perf_counter() - started) * 1000
