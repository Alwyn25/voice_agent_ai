"""Hybrid retrieval: pgvector semantic search + Postgres full-text, merged by
reciprocal rank fusion.

Why both: semantic search finds "setup fees" when the customer says "installation
charges"; keyword search reliably finds exact product names and numbers that
embeddings blur. RRF merges them without needing the two scores to be comparable.

Effective-date filtering is applied in SQL, so an expired promotion is not merely
ranked lower — it is unreachable. That is the guarantee the compliance design relies on.
"""
import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import date

import redis.asyncio as aioredis
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.engine.rag import reranker
from app.engine.rag.enrichment import QueryIntent, analyze_query, facts_of
from app.engine.rag import parent_child
from app.engine.rag.query_split import split_compound_query
from app.engine.rag.synonyms import build_alias_index, expand_terms, normalize_synonyms
from app.engine.rag.vector_index import apply_scan_config
from app.providers.embeddings.base import EmbeddingError
from app.providers.embeddings.router import get_embeddings

log = logging.getLogger(__name__)

# A bounded cache of QUERY embeddings, keyed by (model, text).
#
# An embedding is a pure function of (model, text), so the SAME query never needs to be
# embedded twice — the ~100–300 ms network call is the dominant cost of a retrieval. This
# is what makes the two retrieval TRIGGERS cheap to keep: when the knowledge_lookup tool
# retries a query the speculative path already ran this turn, or an FAQ recurs across calls,
# the vector is served from here instead of paying the round-trip again.
#
# Only single-query RETRIEVAL embeds pass through this; document ingestion embeds large
# one-off batches straight through the provider and must never pollute a small hot cache.
# Safe by construction: values are deterministic, so a concurrent double-miss just recomputes
# the identical vector — no lock needed on the event loop. Keyed on the model name because
# changing the embedding model invalidates every vector's meaning (and its dimensions), so a
# stale entry from a previous model must never be served. Bounded LRU: it cannot grow without
# limit, and the least-recently-used query falls out first.
_QVEC_CACHE: "OrderedDict[tuple[str, str], list[float]]" = OrderedDict()
_QVEC_CACHE_MAX = 1024

# P3.06: In-flight embed calls keyed by (model, text) — single-flight coalescing map.
# When the speculative prefetch and the real turn both embed the same query, the second
# awaiter joins the in-progress task instead of starting a redundant network round-trip.
_INFLIGHT_EMBEDS: "dict[tuple[str, str], asyncio.Future]" = {}

# R6.01: Regex that matches turns which carry zero retrievable intent — pure greetings,
# single-word acknowledgements, bare confirmations, and exit phrases. These turns never
# benefit from document retrieval, so skipping _embed_query() saves the ~300-500 ms
# OpenAI embedding round-trip. The pattern is intentionally narrow: short utterances only,
# anchored at both ends, case-insensitive, with optional punctuation. Any utterance that
# might carry a factual question falls through to full retrieval.
_SMALLTALK_RE = re.compile(
    r"^\s*("
    # greetings / call-openers
    r"(hi|hello|hey|hiya|howdy|namaste|helo|hii|hiii)"
    r"|(good\s+(morning|afternoon|evening|night|day))"
    r"|(is\s+this\s+(highland\s+greenz?|the\s+(sales\s+)?office|advora)(\s+.*)?)"
    # bare confirmations / acknowledgements
    r"|(yes|yeah|yep|yup|sure|ok|okay|alright|right|correct|absolutely|definitely|of\s+course)"
    r"|(no|nope|nah|not\s+really)"
    r"|(got\s+it|i\s+see|i\s+understand|understood|noted|sounds\s+good|perfect|great)"
    r"|(thanks?|thank\s+you|thanks?\s+a\s+lot|many\s+thanks|cheers)"
    r"|(you('re|\s+are)\s+welcome|no\s+problem|my\s+pleasure)"
    # call closers / exits
    r"|(bye|goodbye|good\s*bye|see\s+you|take\s+care|have\s+a\s+good\s+(day|one))"
    r"|(that('?s|\s+is)\s+(all|it|fine|good|enough|great|perfect|okay))"
    r"|(i('ll|\s+will)\s+(call|think|get)\s+back)"
    r"|(let\s+me\s+think(\s+about\s+it)?)"
    # short filler turns unlikely to reference documents
    r"|(okay\s+(calling|noted|sure|got\s+it)(\s+.*)?)"
    r"|(calling\s+(him|her|them|now)(\s+.*)?)"
    r"|(\d{1,2}\s+is\s+fine(\s+.*)?)"   # "Yes 11 is fine"
    r")\s*[.!?,]*\s*$",
    re.IGNORECASE,
)


async def _embed_network(query: str, key: "tuple[str, str]") -> list[float]:
    """Actual provider round-trip. Runs as an independent task in _INFLIGHT_EMBEDS so
    multiple awaiters share ONE call. Removes itself on completion so a later query
    re-embeds rather than awaiting a dead future."""
    try:
        embedded = await get_embeddings().embed([query])
        vec = embedded.vectors[0]
        _QVEC_CACHE[key] = vec
        while len(_QVEC_CACHE) > _QVEC_CACHE_MAX:
            _QVEC_CACHE.popitem(last=False)
        return vec
    finally:
        _INFLIGHT_EMBEDS.pop(key, None)


def cancel_inflight_embed(query: str) -> None:
    """Abort an in-flight embed for `query` (P3.06 supersede). Only called by the
    speculative path for its own stale key — never pulls a shared embed from under a
    real turn that has already coalesced onto it."""
    model = get_settings().embedding_model or ""
    task = _INFLIGHT_EMBEDS.pop((model, query), None)
    if task is not None and not task.done():
        task.cancel()


async def _embed_query(query: str,
                       session_cache: "dict[str, list[float]] | None" = None) -> list[float]:
    """The query's embedding. Checks session_cache first (fast, call-scoped), then the
    process-level LRU, then coalesces onto an in-flight embed for the same query, then
    starts a new network call. Writes back to both caches so subsequent callers pay no
    round-trip."""
    model = get_settings().embedding_model or ""
    # Session cache: per-call dict keyed only by text (model is fixed for a call).
    if session_cache is not None:
        hit = session_cache.get(query)
        if hit is not None:
            return hit
    # Process-level LRU: cross-call, cross-session hit for recurring FAQ queries.
    key = (model, query)
    hit = _QVEC_CACHE.get(key)
    if hit is not None:
        _QVEC_CACHE.move_to_end(key)
        if session_cache is not None:
            session_cache[query] = hit
        return hit
    # Single-flight: join an embed already running for this exact query rather than
    # starting a second network call. asyncio.shield keeps the shared task running
    # even if THIS awaiter is cancelled (e.g. the bridge cancelling a speculative embed).
    inflight = _INFLIGHT_EMBEDS.get(key)
    if inflight is None:
        inflight = asyncio.ensure_future(_embed_network(query, key))
        _INFLIGHT_EMBEDS[key] = inflight
    vec = await asyncio.shield(inflight)
    if session_cache is not None:
        session_cache[query] = vec
    return vec


async def warm_query_embedding(query: str,
                               session_cache: "dict[str, list[float]] | None" = None
                               ) -> bool:
    """Pre-compute and cache a query's embedding WITHOUT running full retrieval.
    P3.06 speculative prefetch: voice bridge calls this during the STT idle window so
    the real turn's embed call is a free session-cache hit. Returns True if a vector
    is now cached, False if the query normalised to nothing."""
    normalized = normalize_query(query)
    if not (normalized or "").strip():
        return False
    await _embed_query(normalized, session_cache=session_cache)
    return True


# Redis cache of whole RESULTS for popular questions (P3.05 Phase 4, ticket #34) --
# distinct from the query-embedding cache above. Cuts the full embed+SQL retrieval path
# (~80ms) for a question asked again, by anyone, for the same tenant within the window.
#
# Fail-open throughout, same philosophy as app.providers.llm.rate_limiter: a Redis hiccup
# must degrade to "no caching this time", never to a reason a turn fails.
#
# Invalidation is by VERSION, not by deleting individual query keys. Re-ingesting a
# document bumps kl:ver:{tenant_id}, which changes every subsequent cache key for that
# tenant -- old entries are simply never looked up again and expire on their own TTL.
# This avoids needing SCAN/pattern-delete (slow, and unsafe on a shared Redis in
# production) just to throw away a tenant's stale answers.
_KL_CACHE_TTL_SECONDS = 300  # 5 minutes, per the ticket

_kl_redis_client: aioredis.Redis | None = None


def _kl_redis() -> aioredis.Redis:
    global _kl_redis_client
    if _kl_redis_client is None:
        _kl_redis_client = aioredis.from_url(get_settings().redis_url, decode_responses=True)
    return _kl_redis_client


def _cache_signature(*, query: str, doc_type: str | None,
                     project_id: uuid.UUID | None, top_k: int,
                     knowledge_base_id: uuid.UUID | None = None,
                     granularity: str = parent_child.GRANULARITY_FLAT) -> str:
    # Hashed, not stored as plain text -- GDPR note in the ticket: no readable caller
    # text sits in Redis as (or inside) a key. Includes every parameter that changes
    # what SQL runs / what can be returned (doc_type, project_id, knowledge_base_id, top_k),
    # not just the query text -- a project- or KB-scoped lookup must never be served from a
    # differently scoped one's cached answer (that is exactly the cross-project leak the
    # project_id filter in retrieve() exists to prevent, one level up).
    payload = json.dumps({"q": query.strip().lower(), "doc_type": doc_type,
                          "project_id": str(project_id) if project_id else None,
                          "knowledge_base_id": str(knowledge_base_id) if knowledge_base_id else None,
                          "top_k": top_k,
                          # R5.03: flipping RETRIEVAL_GRANULARITY must never serve the other
                          # mode's cached answer for the rest of the TTL window.
                          "granularity": granularity}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


async def _cache_key(*, tenant_id: uuid.UUID, query: str, doc_type: str | None,
                     project_id: uuid.UUID | None, top_k: int, today: date,
                     knowledge_base_id: uuid.UUID | None = None,
                     granularity: str = parent_child.GRANULARITY_FLAT) -> str:
    try:
        version = await _kl_redis().get(f"kl:ver:{tenant_id}") or "0"
    except Exception:
        version = "0"
    sig = _cache_signature(query=query, doc_type=doc_type, project_id=project_id, top_k=top_k,
                           knowledge_base_id=knowledge_base_id,
                           granularity=granularity)
    return f"kl:{tenant_id}:{version}:{today.isoformat()}:{sig}"


async def _cache_get(key: str) -> dict | None:
    try:
        raw = await _kl_redis().get(key)
        return json.loads(raw) if raw else None
    except Exception:  # noqa: BLE001 -- a bad/missing Redis must never fail a lookup
        return None


async def _cache_set(key: str, payload: dict) -> None:
    try:
        await _kl_redis().set(key, json.dumps(payload), ex=_KL_CACHE_TTL_SECONDS)
    except Exception:  # noqa: BLE001 -- caching is a speed-up, never a requirement
        log.warning("knowledge cache: failed to write %s", key, exc_info=True)


async def invalidate_tenant_knowledge_cache(tenant_id: uuid.UUID) -> None:
    """Call after a document finishes (re-)ingesting for this tenant. Bumps the tenant's
    cache version so every previously cached answer for it stops being served -- cheap,
    and correct even if several documents finish ingesting around the same time."""
    try:
        await _kl_redis().incr(f"kl:ver:{tenant_id}")
    except Exception:  # noqa: BLE001 -- worst case, stale answers simply age out via TTL
        log.warning("knowledge cache: failed to invalidate tenant %s", tenant_id, exc_info=True)


# Enough to overturn the ~1% gap that put a 3 BHK cost sheet above the 2.5 BHK rows the
# customer had been asking about, without drowning a genuinely better match.
FOCUS_BOOST = 1.35
# Per-document window cap: now the RETRIEVAL_MAX_CHUNKS_PER_DOC setting (see the cap below).
POLICY_BOOST = 1.6      # enough to lift a policy chunk past a keyword-dense price list
PRICING_BOOST = 1.6     # lift the actual cost sheets past prose that merely mentions price
# A price question wants the document that HOLDS the numbers (doc_type 'pricing'), not the
# SOP prose that talks around them. Asked "what are the prices for a 3 BHK", hybrid ranking
# surfaced a "which prices we can quote" caveat chunk and an "our documents disagree" chunk
# above the 3 BHK cost sheet that actually states ₹11,499/sq ft — so the agent said "not
# published" while the rate sat one rank below the window. This is the pricing analogue of
# POLICY_BOOST: when the question is clearly about price, the priced documents earn the window,
# and the focus anchor ("3 BHK") then orders the right sheet to the front. Deliberately narrow —
# a bare "cost" inside an unrelated question should not trip it, so a subject word is required.
PRICING_MARKERS = ("price", "pricing", "cost sheet", "psf", "per square foot", "per sq ft",
                   "rate per", "how much", "budget", "emi", " lakh", " crore", "rupee",
                   "/sq", "per sqft", "cost of", "what does it cost", "what's the cost",
                   "price of", "priced")
# Questions about what the company offers overall, where a scope document is the right
# answer. Deliberately narrow: "what amenities" and "what is the price" are not these.
# Matched against the whole question, and paired: a phrase like "do you provide" only
# counts when the SUBJECT is the business itself, not a feature of it. "Do you provide
# rental services" is a scope question; "do you provide a swimming pool" is not, and an
# earlier version boosted both — which is how a question about amenities was answered
# with "we only sell apartments, no rentals".
SCOPE_SUBJECTS = ("propert", "flat", "apartment", "bhk", "configuration", "villa",
                  "rent", "rental", "lease", "resale", "commercial", "office space",
                  "plot", "penthouse")
SCOPE_ASKS = ("what kind", "what type", "what sort", "which", "do you sell",
              "do you have", "do you offer", "do you provide", "what do you sell",
              "what all", "available")
# A statement of interest carries the same scope intent as a question, and it is how
# most calls actually open: "I am looking for a flat" got no boost at all — it matches
# no SCOPE_ASKS phrase, because it is not phrased as a question — so plain hybrid
# ranking (which favours the longest, most keyword-dense document) answered from
# whichever price sheet ranked highest and the customer was told only 2 BHK existed.
SCOPE_STATEMENTS = ("looking for", "looking to buy", "looking to purchase",
                    "interested in", "want to see", "want to buy", "in the market for",
                    "shopping for", "want a", "need a")
RRF_K = 60  # rank-fusion damping; standard value
# R5.08: ceiling on the COMBINED price/area/bhk metadata lift for one chunk — parity with the
# single document-level PRICING_BOOST, so metadata can never outweigh the strongest existing
# membership signal. See apply_metadata_boosts().
META_BOOST_CAP = 1.6

# Stopwords that carry no retrieval signal in customer questions.
_QUERY_NOISE = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "will", "would", "should", "have", "has", "had", "i", "you", "we", "my", "your",
    "me", "it", "to", "of", "for", "on", "in", "at", "and", "or", "if", "so", "that",
    "this", "there", "then", "what", "how", "why", "when", "which", "who", "much",
    "many", "any", "some", "get", "got", "tell", "know", "want", "need", "please",
    "hi", "hello", "thanks", "up", "out", "not", "no", "yes", "be", "been", "am",
}


# "3bhk", "2.5 BHK", "3-bhk" — the ways a caller writes a configuration that a document spells
# "3 BHK". Left unnormalised, the keyword arm turns "3bhk" into the single token "3bhk", which
# matches NONE of the "3 BHK" chunks, so the config-specific cost sheet never scores and a real
# price answer ("₹11,499/sq ft") stays one rank below the window. Splitting the run into "3 bhk"
# lets the digit and "bhk" match the document's own two tokens. Idempotent on already-spaced
# input. Also written as a query-text normalisation (below) so the SEMANTIC arm sees it too.
_BHK_RUN = re.compile(r"(\d(?:\.\d+)?)\s*-?\s*bhk", re.IGNORECASE)


def normalize_query(query: str) -> str:
    """Caller-side spelling variants folded to the form the documents use. Currently: a
    configuration written without a space ("3bhk", "2.5-bhk") becomes "3 bhk" / "2.5 bhk"."""
    return _BHK_RUN.sub(lambda m: f"{m.group(1)} bhk", query or "")


def build_tsquery(query: str, alias_index: dict[str, set[str]] | None = None) -> str | None:
    """OR-join the significant words.

    ts_rank still ranks passages matching more terms higher, so OR gives recall without
    losing precision in the ordering — whereas AND (websearch/plainto default) means one
    absent word kills the whole match.

    ``alias_index`` (from :mod:`app.engine.rag.synonyms`) bridges vocabulary the keyword arm
    would otherwise miss: with it, a search for "generator" also ORs in "dg" and "power
    backup", so a chunk that only ever says "power backup" still matches. Expansion runs
    AFTER stopword filtering, so aliases are added to real query terms, never to noise.
    """
    words = re.findall(r"[a-zA-Z0-9]+", normalize_query(query).lower())
    terms = [w for w in words if len(w) > 2 and w not in _QUERY_NOISE]
    if not terms:
        terms = [w for w in words if len(w) > 2]
    if alias_index:
        terms = expand_terms(terms, alias_index)
    return " | ".join(dict.fromkeys(terms)) or None


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    title: str
    doc_type: str
    heading: str | None
    content: str
    score: float
    relevance: float = 0.0        # 0-1, relative to the best hit — for humans
    vector_rank: int | None = None
    text_rank: int | None = None
    # R5.01 debug fields (never sent to the model): position before re-ranking, the
    # cross-encoder's raw logit, and its position by that logit among the candidates.
    base_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None
    # R5.08: the chunk's stored facts (None = untagged) and which metadata lifts it received.
    # Debug only — deliberately NOT in as_context(), so the prompt does not grow.
    facts: dict | None = None
    meta_boosts: list[str] = field(default_factory=list)
    # R5.03: the child chunks whose match brought this parent into the window, best first
    # ({chunk_id, score, vector_rank, text_rank, content}). Empty for a standalone chunk and in
    # flat mode. Debug/trace only -- the model reads ``content`` (the parent), never these.
    matched_children: list[dict] = field(default_factory=list)

    def as_context(self) -> dict:
        source = self.title if not self.heading else f"{self.title} — {self.heading}"
        return {"chunk_id": self.chunk_id, "source": source, "doc_type": self.doc_type,
                "content": self.content, "score": round(self.relevance, 3)}


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    vector_hits: int = 0
    text_hits: int = 0
    excluded_expired: int = 0
    error: str | None = None
    latency_ms: int = 0
    # R5.08 diagnostics: mode, query intent, tagged/untagged counts, lifts applied, filter and
    # fallback. Surfaced in the planner's retrieval trace and /knowledge/search, and logged as
    # "rag.meta" — enough to answer "why did THIS chunk make the window?".
    metadata: dict = field(default_factory=dict)
    # R5.02 vector-arm diagnostics — surfaced in the planner's retrieval trace and the
    # /knowledge/search debug endpoint, so an empty/short answer can be traced to the index.
    vector_ms: float = 0.0              # DB time of the vector query (excl. embedding call)
    vector_pool: int = 0                # rows requested from the vector arm
    vector_scan: str = ""               # VectorScanConfig.describe() — settings actually used
    vector_eligible: int | None = None  # bounded eligible count; set only when underfilled
    vector_underfilled: bool = False    # True = matching rows existed that the scan did not return
    # R5.01: what the re-ranking step did this call (None = disabled). See rerank_diagnostics.
    rerank: dict | None = None
    skipped_smalltalk: bool = False     # R6.01: True = bail-out fired; embedding was not called
    # R5.03: what the child->parent step did (parent_child.CollapseStats.as_dict()); in flat
    # mode just {"granularity": "flat"}. Surfaced in the planner trace and /knowledge/search.
    parent_child: dict = field(default_factory=dict)

    def as_context(self) -> list[dict]:
        return [c.as_context() for c in self.chunks]

    def rerank_diagnostics(self) -> dict | None:
        return self.rerank

    def vector_diagnostics(self) -> dict:
        return {"vector_ms": round(self.vector_ms, 2), "pool": self.vector_pool,
                "returned": self.vector_hits, "scan": self.vector_scan,
                "eligible": self.vector_eligible, "underfilled": self.vector_underfilled}

    def parent_child_diagnostics(self) -> dict:
        return dict(self.parent_child)


VECTOR_SQL = sql_text("""
    SELECT c.id, c.document_id, c.content, c.metadata, c.chunk_role, c.parent_id,
           d.title, d.doc_type,
           1 - (c.embedding <=> CAST(:qvec AS vector)) AS similarity
    FROM knowledge_chunks c
    JOIN knowledge_documents d ON d.id = c.document_id
    WHERE c.tenant_id = CAST(:tenant_id AS uuid)
      AND c.chunk_role <> CAST(:exclude_role AS text)   -- R5.03: role scope, see parent_child.py
      AND d.status = 'ready'
      AND c.embedding IS NOT NULL
      AND (d.effective_from IS NULL OR d.effective_from <= CAST(:today AS date))
      AND (d.effective_to   IS NULL OR d.effective_to   >= CAST(:today AS date))
      AND (CAST(:doc_type AS text) IS NULL OR d.doc_type = CAST(:doc_type AS text))
      AND (CAST(:project_id AS uuid) IS NULL
           OR c.project_id = CAST(:project_id AS uuid)
           OR c.project_id IS NULL)
      -- R5.08 filter mode: only chunks whose stored facts carry the flag (e.g. has_price).
      -- An UNTAGGED chunk (ingested before R5.08, not yet backfilled) stays eligible, so a
      -- partially-tagged corpus can never lose knowledge to the filter. NULL flag = no filter.
      AND (CAST(:meta_flag AS text) IS NULL
           OR c.metadata->'facts' IS NULL
           OR c.metadata->'facts'->>CAST(:meta_flag AS text) = 'true')
      AND (CAST(:knowledge_base_id AS uuid) IS NULL
           OR c.knowledge_base_id = CAST(:knowledge_base_id AS uuid)
           OR c.knowledge_base_id IS NULL)
    ORDER BY c.embedding <=> CAST(:qvec AS vector)
    LIMIT CAST(:limit AS integer)
""")

# R5.02: the same query, shaped for an ITERATIVE HNSW scan (see vector_index.py).
#
# Filters, parameters and output columns are identical to VECTOR_SQL — only the shape differs:
#   * The inner ORDER BY distance ... LIMIT is what lets the planner drive the HNSW index; with
#     hnsw.iterative_scan on, the scan resumes until LIMIT rows SURVIVE the tenant/project/date
#     filters instead of stopping after ef_search candidates.
#   * relaxed_order may emit rows slightly out of distance order, so the MATERIALIZED CTE fixes
#     the result set first and the outer ORDER BY restores exact order (pgvector's documented
#     pattern). Downstream RRF depends on vector_rank, so order must be exact.
#   * The distance is computed once and similarity derived from it.
VECTOR_SQL_TUNED = sql_text("""
    WITH ann AS MATERIALIZED (
        SELECT c.id, c.document_id, c.content, c.metadata, c.chunk_role, c.parent_id,
               d.title, d.doc_type,
               c.embedding <=> CAST(:qvec AS vector) AS distance
        FROM knowledge_chunks c
        JOIN knowledge_documents d ON d.id = c.document_id
        WHERE c.tenant_id = CAST(:tenant_id AS uuid)
          AND c.chunk_role <> CAST(:exclude_role AS text)   -- R5.03: role scope, see parent_child.py
          AND d.status = 'ready'
          AND c.embedding IS NOT NULL
          AND (d.effective_from IS NULL OR d.effective_from <= CAST(:today AS date))
          AND (d.effective_to   IS NULL OR d.effective_to   >= CAST(:today AS date))
          AND (CAST(:doc_type AS text) IS NULL OR d.doc_type = CAST(:doc_type AS text))
          AND (CAST(:project_id AS uuid) IS NULL
               OR c.project_id = CAST(:project_id AS uuid)
               OR c.project_id IS NULL)
          AND (CAST(:knowledge_base_id AS uuid) IS NULL
               OR c.knowledge_base_id = CAST(:knowledge_base_id AS uuid)
               OR c.knowledge_base_id IS NULL)
        ORDER BY c.embedding <=> CAST(:qvec AS vector)
        LIMIT CAST(:limit AS integer)
    )
    SELECT id, document_id, content, metadata, chunk_role, parent_id, title, doc_type,
           1 - distance AS similarity
    FROM ann
    ORDER BY distance, id   -- R5.03: id breaks exact-distance ties deterministically
""")

# R5.02 debug signal: how many rows were ELIGIBLE for the vector arm, counted up to the pool
# size only (the inner LIMIT bounds the work to <= pool rows via the tenant index). Run only
# when the vector arm came back short, to tell "this tenant simply has few chunks" (fine) from
# "the index scan dropped rows that match" (the bug R5.02 guards against).
VECTOR_ELIGIBLE_SQL = sql_text("""
    SELECT count(*) FROM (
        SELECT 1
        FROM knowledge_chunks c
        JOIN knowledge_documents d ON d.id = c.document_id
        WHERE c.tenant_id = CAST(:tenant_id AS uuid)
          AND c.chunk_role <> CAST(:exclude_role AS text)   -- R5.03: role scope, see parent_child.py
          AND d.status = 'ready'
          AND c.embedding IS NOT NULL
          AND (d.effective_from IS NULL OR d.effective_from <= CAST(:today AS date))
          AND (d.effective_to   IS NULL OR d.effective_to   >= CAST(:today AS date))
          AND (CAST(:doc_type AS text) IS NULL OR d.doc_type = CAST(:doc_type AS text))
          AND (CAST(:project_id AS uuid) IS NULL
               OR c.project_id = CAST(:project_id AS uuid)
               OR c.project_id IS NULL)
          AND (CAST(:knowledge_base_id AS uuid) IS NULL
               OR c.knowledge_base_id = CAST(:knowledge_base_id AS uuid)
               OR c.knowledge_base_id IS NULL)
        LIMIT CAST(:limit AS integer)
    ) eligible
""")

TEXT_SQL = sql_text("""
    SELECT c.id, c.document_id, c.content, c.metadata, c.chunk_role, c.parent_id,
           d.title, d.doc_type,
           ts_rank(to_tsvector('english', c.content),
                   to_tsquery('english', CAST(:q AS text))) AS rank
    FROM knowledge_chunks c
    JOIN knowledge_documents d ON d.id = c.document_id
    WHERE c.tenant_id = CAST(:tenant_id AS uuid)
      AND c.chunk_role <> CAST(:exclude_role AS text)   -- R5.03: role scope, see parent_child.py
      AND d.status = 'ready'
      AND (d.effective_from IS NULL OR d.effective_from <= CAST(:today AS date))
      AND (d.effective_to   IS NULL OR d.effective_to   >= CAST(:today AS date))
      AND (CAST(:doc_type AS text) IS NULL OR d.doc_type = CAST(:doc_type AS text))
      AND (CAST(:project_id AS uuid) IS NULL
           OR c.project_id = CAST(:project_id AS uuid)
           OR c.project_id IS NULL)
      AND (CAST(:knowledge_base_id AS uuid) IS NULL
           OR c.knowledge_base_id = CAST(:knowledge_base_id AS uuid)
           OR c.knowledge_base_id IS NULL)
      AND to_tsvector('english', c.content) @@ to_tsquery('english', CAST(:q AS text))
    -- R5.03: ts_rank ties are common (same terms, same counts). Without a tie-break their order
    -- is the rows' physical order, which changes whenever rows are rewritten -- so the SAME
    -- query ranked differently after a re-ingest or backfill. c.id makes it reproducible.
    ORDER BY rank DESC, c.id
    LIMIT CAST(:limit AS integer)
""")

EXPIRED_SQL = sql_text("""
    SELECT count(*) FROM knowledge_chunks c
    JOIN knowledge_documents d ON d.id = c.document_id
    WHERE c.tenant_id = CAST(:tenant_id AS uuid)
      AND c.chunk_role <> 'child'      -- R5.03: count in pre-R5.03 chunk units
      AND (   (d.effective_from IS NOT NULL AND d.effective_from > CAST(:today AS date))
           OR (d.effective_to   IS NOT NULL AND d.effective_to   < CAST(:today AS date)))
""")


async def _tenant_settings(db: AsyncSession, tenant_id: uuid.UUID) -> dict:
    """The tenant's settings JSON — read for whatever a call site did not supply: the declared
    synonym groups (``retrieval_synonyms``) and the R5.03 per-tenant ``retrieval_granularity``.

    Kept here so any call site that omits them still gets what the planner path passes.
    Best-effort: a missing tenant or a non-dict settings blob degrades to {} (no expansion,
    global granularity) rather than raising on the hot path.
    """
    # Local import avoids a module-level dependency cycle (models import back through config).
    from app.db.models import Tenant
    from sqlalchemy import select as _select
    try:
        tenant = (await db.execute(
            _select(Tenant).where(Tenant.id == tenant_id))).scalar_one_or_none()
    except Exception:
        return {}
    return (tenant.settings if tenant and isinstance(tenant.settings, dict) else {}) or {}


def metadata_filter_flag(mode: str, intent: QueryIntent) -> str | None:
    """Which stored fact the vector arm is restricted to in ``filter`` mode, or None.

    Price wins over area when a question asks both ("price of the 1,239 sq ft flat"): the
    price chunks for that unit also state its area, whereas an area-only chunk cannot answer
    the price. BHK alone never filters — "tell me about the 3 BHK" wants amenities and specs
    too, which name no configuration.
    """
    if mode != "filter":
        return None
    if intent.wants_price:
        return "has_price"
    if intent.wants_area:
        return "has_area"
    return None


def apply_metadata_boosts(scores: dict[str, float], rows_by_id: dict[str, dict],
                          intent: QueryIntent, *, price_boost: float, area_boost: float,
                          bhk_boost: float, doc_price_boosted: set[str]) -> dict:
    """R5.08: lift chunks whose stored facts match what the query asks for. Mutates ``scores``
    in place and returns counts for the diagnostics. Pure — no I/O — so it is unit-tested
    directly (tests/test_chunk_enrichment.py).

    Applied BEFORE the base-score snapshot in retrieve(), so these are MEMBERSHIP signals, like
    the policy and pricing boosts: a chunk that states the price the caller asked for deserves
    a place in the window, not just a better seat once inside it. Each is a bounded multiply
    on the RRF score — it can overturn a near-tie, not a clearly better match.

      * price — a price question lifts chunks that state a price or rate. SKIPPED for chunks
        the document-level PRICING_BOOST already lifted, so a cost-sheet chunk is never
        boosted twice; the new signal is for priced chunks in documents NOT typed 'pricing'
        (a brochure, a policy) and for price questions PRICING_MARKERS does not recognise
        (Hinglish "charge kitna hai").
      * area  — a size question lifts chunks that state an area.
      * bhk   — a question naming a configuration lifts chunks that cover it. An exact set
        overlap on parsed values: 2.5 matches 2.5 only, never 2 or 3 (the substring focus
        boost cannot make that distinction reliably).

    Untagged chunks (no stored facts) are left exactly as they are — neutral, never penalised.
    """
    stats = {"tagged": 0, "untagged": 0, "price": 0, "area": 0, "bhk": 0}
    lifts: dict[str, list[str]] = {}
    for key, row in rows_by_id.items():
        if key not in scores:
            continue
        facts = facts_of(row.get("metadata"))
        if facts is None:
            stats["untagged"] += 1
            continue
        stats["tagged"] += 1
        applied: list[str] = []
        mult = 1.0
        if (intent.wants_price and facts.get("has_price") and key not in doc_price_boosted):
            mult *= price_boost
            applied.append("price")
        if intent.wants_area and facts.get("has_area"):
            mult *= area_boost
            applied.append("area")
        if intent.bhk:
            chunk_bhk = {float(b) for b in (facts.get("bhk") or [])
                         if isinstance(b, (int, float))}
            if chunk_bhk & intent.bhk:
                mult *= bhk_boost
                applied.append("bhk")
        # Signals compound (a 2.5 BHK cost-sheet row matches price AND area AND bhk), so the
        # combined lift is capped: metadata may break ties and admit a close runner-up, never
        # bury an unrelated but clearly better match.
        scores[key] *= min(mult, META_BOOST_CAP)
        for name in applied:
            stats[name] += 1
        if applied:
            lifts[key] = applied
    stats["lifts"] = lifts
    return stats
async def _vector_search(db: AsyncSession, s, result: RetrievalResult, params: dict, *,
                         pool: int, tenant_id: uuid.UUID) -> list:
    """Run the vector arm (R5.02) and record its diagnostics on ``result``.

    With tuning on: set the transaction-scoped HNSW scan settings, then run VECTOR_SQL_TUNED.
    With tuning off (RETRIEVAL_HNSW_TUNING_ENABLED=false): the exact pre-R5.02 VECTOR_SQL.
    Exceptions propagate to the caller's semantic-arm handler, unchanged from before.
    """
    started = time.perf_counter()
    scan = await apply_scan_config(db, s, pool=pool)
    sql = VECTOR_SQL_TUNED if scan.enabled else VECTOR_SQL
    rows = (await db.execute(sql, params)).mappings().all()
    result.vector_ms = (time.perf_counter() - started) * 1000
    result.vector_pool = pool
    result.vector_scan = scan.describe()

    # Short result: was the tenant's eligible set really this small, or did the scan drop rows?
    if len(rows) < pool and s.retrieval_vector_underfill_check:
        try:
            eligible = int((await db.execute(VECTOR_ELIGIBLE_SQL, params)).scalar() or 0)
            result.vector_eligible = eligible
            result.vector_underfilled = eligible > len(rows)
        except Exception as e:  # diagnostics must never cost the turn its answer
            log.warning("rag.vector eligible-count failed: %s: %s", e.__class__.__name__, e)

    log.info("rag.vector tenant=%s pool=%d returned=%d eligible=%s db_ms=%.1f scan=[%s]",
             tenant_id, pool, len(rows),
             "n/a" if result.vector_eligible is None else result.vector_eligible,
             result.vector_ms, result.vector_scan)
    if result.vector_underfilled:
        # The failure R5.02 exists to prevent: matching chunks existed, the index did not
        # return them. Usually means iterative scan is off/unsupported or max_scan_tuples is
        # too low for this tenant's share of the table. Investigate with
        # scripts/vector_index_report.py.
        log.warning("rag.vector.underfilled tenant=%s returned=%d of %d eligible (pool=%d) "
                    "scan=[%s]", tenant_id, len(rows), result.vector_eligible, pool,
                    result.vector_scan)

    if s.retrieval_vector_explain:
        try:
            plan = (await db.execute(sql_text("EXPLAIN (ANALYZE, BUFFERS) " + sql.text),
                                     params)).scalars().all()
            log.info("rag.vector EXPLAIN (scan=[%s]):\n%s", result.vector_scan, "\n".join(plan))
        except Exception as e:
            log.warning("rag.vector EXPLAIN failed: %s: %s", e.__class__.__name__, e)
    return rows


# tenant_id -> (monotonic time, glossary). The glossary is read on every reranked retrieval, so
# it is cached briefly instead of costing a DB round-trip per turn; an edit in tenant settings
# takes effect within _GLOSSARY_TTL_S.
_GLOSSARY_CACHE: dict[str, tuple[float, dict[str, str]]] = {}
_GLOSSARY_TTL_S = 300.0


async def _tenant_rerank_glossary(db: AsyncSession, tenant_id: uuid.UUID) -> dict[str, str]:
    """``tenant.settings["rerank_glossary"]``: the tenant's own Hinglish -> English domain
    vocabulary for the cross-encoder gloss (e.g. {"khel kood": "sports"}). Best-effort: any
    problem (missing tenant, malformed value, DB error) degrades to no tenant glossary."""
    key = str(tenant_id)
    hit = _GLOSSARY_CACHE.get(key)
    if hit and time.monotonic() - hit[0] < _GLOSSARY_TTL_S:
        return hit[1]
    from app.db.models import Tenant
    from sqlalchemy import select as _select
    glossary: dict[str, str] = {}
    try:
        tenant = (await db.execute(
            _select(Tenant).where(Tenant.id == tenant_id))).scalar_one_or_none()
        raw = (tenant.settings or {}).get("rerank_glossary") if tenant and isinstance(
            tenant.settings, dict) else None
        if isinstance(raw, dict):
            glossary = {str(k): str(v) for k, v in raw.items() if isinstance(k, str) and k.strip()}
    except Exception as e:  # noqa: BLE001 — a glossary must never cost the turn its answer
        log.warning("rag.rerank tenant glossary unavailable: %s: %s", e.__class__.__name__, e)
    _GLOSSARY_CACHE[key] = (time.monotonic(), glossary)
    return glossary


def _chunk_label(row: dict) -> str:
    """Short human label for logs/diagnostics: 'title > heading-leaf'."""
    heading = ((row.get("metadata") or {}).get("heading") or "").split(" > ")[-1]
    title = row.get("title") or "document"
    return f"{title} > {heading}" if heading and heading != title else title


def _rerank_passage(row: dict) -> str:
    """What the cross-encoder reads for one chunk. The document title goes first because the
    chunk body often does not repeat the subject the title carries — a cost-sheet row says
    '1,325 sq ft ... Total 1,55,31,250' while only the title says '2.5 BHK'."""
    return f"{row.get('title') or ''}\n{row.get('content') or ''}".strip()


async def _apply_rerank(query: str, base_scores: dict[str, float], rows_by_id: dict[str, dict],
                        s, result: RetrievalResult, top_k: int, *, db: AsyncSession,
                        tenant_id: uuid.UUID) -> dict[str, float] | None:
    """R5.01: re-score the top of the pool with the cross-encoder.

    Returns the NEW base relevance for every pooled chunk — fused scores for the candidates,
    base order preserved beneath them for the rest — which then replaces base_scores, so it
    decides window membership exactly as RRF did before; or None when the reranker did not
    produce usable scores — the caller then keeps the RRF ranking untouched. Always records
    what happened on ``result.rerank`` and in one ``rag.rerank`` log line.

    Everything downstream is unchanged: the per-document cap, the focus anchor (a tie-break on
    ORDER within the window) and the min-score cut all run on the returned scores. The policy /
    pricing boosts are already inside base_scores, so in "fuse" mode they keep their say via
    the base rank; "replace" mode deliberately drops them to measure the cross-encoder alone.
    """
    cfg = reranker.config_from_settings(s)
    base_order = [k for k, _ in sorted(base_scores.items(), key=lambda kv: kv[1], reverse=True)]
    for pos, key in enumerate(base_order, start=1):
        rows_by_id[key]["base_rank"] = pos
    candidates = base_order[:s.retrieval_rerank_candidates]

    # Language gate: only re-rank the languages the model is trusted on (see config). Uses the
    # responder's detector so "what language is this turn" has one answer across the system.
    # Imported here, not at module level: the planner package imports this module.
    from app.engine.planner.responder import detect_language
    language = detect_language(query)
    allowed = {x.strip().lower() for x in (s.retrieval_rerank_languages or "").split(",")}
    if "all" not in allowed and language not in allowed:
        result.rerank = {"status": reranker.SKIPPED_LANGUAGE, "language": language,
                         "model": cfg.label, "mode": s.retrieval_rerank_mode,
                         "candidates": 0, "scored": 0, "cache_hits": 0, "ms": 0.0, "error": None}
        log.info("rag.rerank status=%s language=%s fallback=rrf", reranker.SKIPPED_LANGUAGE,
                 language)
        return None

    # What the cross-encoder reads: an English gloss of a Hinglish question (the default model
    # is English-only). Settings-level glossary first, the tenant's own entries override it.
    ce_query = query
    if s.retrieval_rerank_translate:
        glossary = {**(s.retrieval_rerank_glossary or {}),
                    **(await _tenant_rerank_glossary(db, tenant_id))}
        ce_query = reranker.gloss_query(query, glossary)

    outcome = await reranker.score(
        cfg, ce_query, [(key, _rerank_passage(rows_by_id[key])) for key in candidates])
    diag = {"status": outcome.status, "language": language, "ce_query": ce_query,
            "model": outcome.model,
            "mode": s.retrieval_rerank_mode, "candidates": outcome.candidates,
            "scored": outcome.scored, "cache_hits": outcome.cache_hits,
            "ms": round(outcome.ms, 1), "error": outcome.error}
    result.rerank = diag

    if not outcome.ok:
        # skipped_few is routine (0-1 hits); everything else means ranking fell back to RRF.
        level = logging.INFO if outcome.status == reranker.SKIPPED_FEW else logging.WARNING
        log.log(level, "rag.rerank status=%s fallback=rrf ms=%.1f cand=%d model=%s error=%s",
                outcome.status, outcome.ms, outcome.candidates, outcome.model, outcome.error)
        return None

    fused = reranker.fuse_rankings(
        candidates, outcome.scores, mode=s.retrieval_rerank_mode, k=s.retrieval_rerank_rrf_k,
        base_weight=s.retrieval_rerank_base_weight, ce_weight=s.retrieval_rerank_weight)
    # Chunks below the candidate cut are NOT dropped: they keep their base order, strictly
    # beneath every re-scored candidate. The per-document cap relies on them — when two long
    # documents fill the top candidates, the cap skips their extra chunks and reaches further
    # down for breadth. Dropping them silently changed which chunks reached the window for
    # reasons unrelated to the cross-encoder (seen in the first eval run).
    floor = min(fused.values()) if fused else 0.0
    step = floor * 1e-4
    for pos, key in enumerate(base_order[len(candidates):], start=1):
        fused[key] = floor - step * pos

    # Per-chunk debug fields: CE logit and CE-only position (same tie-break as fuse_rankings).
    base_pos = {key: i for i, key in enumerate(candidates)}
    ce_order = sorted(candidates, key=lambda key: (-outcome.scores[key], base_pos[key]))
    for pos, key in enumerate(ce_order, start=1):
        rows_by_id[key]["rerank_rank"] = pos
        rows_by_id[key]["rerank_score"] = outcome.scores[key]

    # What the rerank CHANGED, in the terms that matter: which chunks entered / left the
    # top_k window (ignoring the per-doc cap, which applies identically either way), and
    # whether the chunk the model reads first changed.
    new_order = sorted(fused, key=fused.get, reverse=True)
    before, after = set(base_order[:top_k]), set(new_order[:top_k])
    diag.update({
        "entered": [_chunk_label(rows_by_id[k]) for k in new_order[:top_k] if k not in before],
        "left": [_chunk_label(rows_by_id[k]) for k in base_order[:top_k] if k not in after],
        "top1_before": _chunk_label(rows_by_id[base_order[0]]),
        "top1_after": _chunk_label(rows_by_id[new_order[0]]),
        "top1_changed": base_order[0] != new_order[0],
    })
    log.info("rag.rerank status=ok ms=%.1f cand=%d scored=%d cache_hits=%d mode=%s entered=%d "
             "top1_changed=%s top1=[%s] model=%s", outcome.ms, outcome.candidates,
             outcome.scored, outcome.cache_hits, s.retrieval_rerank_mode, len(diag["entered"]),
             diag["top1_changed"], diag["top1_after"], outcome.model)
    if s.retrieval_rerank_debug_log:
        lines = []
        for i, k in enumerate(new_order, start=1):
            row = rows_by_id[k]
            ce = (f"ce#{row['rerank_rank']:<2} ce={outcome.scores[k]:+7.3f}" if k in outcome.scores
                  else "(not a candidate)     ")
            lines.append(f"  new#{i:<2} base#{row['base_rank']:<2} {ce} fused={fused[k]:.5f}  "
                         f"{_chunk_label(row)}")
        log.info("rag.rerank table query=%r ce_query=%r\n%s", query, ce_query, "\n".join(lines))
    return fused
async def _expand_to_parents(db: AsyncSession, s, result: RetrievalResult,
                             scores: dict[str, float], rows_by_id: dict[str, dict], *,
                             tenant_id: uuid.UUID) -> tuple[dict[str, float], dict[str, dict]]:
    """R5.03: replace child hits by their parents (one batched fetch + the pure collapse).

    Fail-safe by construction: if the parent fetch fails, every child is served as its own
    unit (a precise ~200-char answer is far better than no answer) and the failure is
    recorded in the trace and logged -- the turn never fails because of this step.
    """
    parents, fetch_ms, fetch_error = await _fetch_parents_safe(
        db, tenant_id, parent_child.parent_ids_of(rows_by_id))
    new_scores, new_rows, stats = parent_child.collapse_to_parents(
        scores, rows_by_id, parents, sibling_weight=s.retrieval_parent_sibling_weight,
        sibling_cap=s.retrieval_parent_sibling_cap)
    _record_parent_stats(s, result, stats, fetch_ms, fetch_error, tenant_id)
    return new_scores, new_rows


def _fuse_chunks(s, vector_rows, text_rows) -> tuple[dict[str, float], dict[str, dict]]:
    """Chunk-level reciprocal rank fusion — the pre-R5.03 fusion, unchanged. Used in flat mode
    and by the child_rank fusion (which collapses to parents afterwards)."""
    scores: dict[str, float] = {}
    rows_by_id: dict[str, dict] = {}
    for rank, row in enumerate(vector_rows, start=1):
        key = str(row["id"])
        rows_by_id[key] = dict(row)
        rows_by_id[key]["vector_rank"] = rank
        scores[key] = scores.get(key, 0.0) + s.retrieval_vector_weight / (RRF_K + rank)
    for rank, row in enumerate(text_rows, start=1):
        key = str(row["id"])
        if key not in rows_by_id:
            rows_by_id[key] = dict(row)
        rows_by_id[key]["text_rank"] = rank
        scores[key] = scores.get(key, 0.0) + s.retrieval_text_weight / (RRF_K + rank)
    return scores, rows_by_id


async def _fetch_parents_safe(db: AsyncSession, tenant_id: uuid.UUID,
                              ids: list[str]) -> tuple[dict[str, dict], float, str | None]:
    """fetch_parents that never raises: on failure every child is served as its own unit (a
    precise ~200-char answer beats no answer) and the error is returned for the trace."""
    try:
        parents, fetch_ms = await parent_child.fetch_parents(db, tenant_id=tenant_id, ids=ids)
        return parents, fetch_ms, None
    except Exception as e:  # noqa: BLE001 -- degrade to child-level results, never fail
        err = f"{e.__class__.__name__}: {e}"
        log.warning("rag.parent_expand fetch failed tenant=%s parents=%d: %s",
                    tenant_id, len(ids), err)
        return {}, 0.0, err


def _record_parent_stats(s, result: RetrievalResult, stats, fetch_ms: float,
                         fetch_error: str | None, tenant_id: uuid.UUID) -> None:
    stats.fetch_ms, stats.fetch_error = fetch_ms, fetch_error
    stats.vector_scope, stats.text_scope = s.retrieval_pc_vector_scope, s.retrieval_pc_text_scope
    result.parent_child = stats.as_dict()
    log.info("rag.parent_expand tenant=%s fusion=%s scope=v:%s/t:%s hits=%d child_hits=%d "
             "parents=%d merged=%d missing=%d units=%d fetch_ms=%.1f", tenant_id, stats.fusion,
             stats.vector_scope, stats.text_scope, stats.hits, stats.child_hits, stats.parents,
             stats.merged_siblings, stats.parents_missing, stats.units, stats.fetch_ms)
    if stats.parents_missing and not fetch_error:
        # Should be impossible (FK cascade + atomic re-ingest). If it fires, a child points at a
        # parent that is gone -- inspect with scripts/backfill_parent_child.py --verify.
        log.warning("rag.parent_expand.missing tenant=%s children_without_parent=%d",
                    tenant_id, stats.parents_missing)


async def _fuse_by_parent(db: AsyncSession, s, result: RetrievalResult, vector_rows: list,
                          text_rows: list, *, tenant_id: uuid.UUID,
                          parent_vector_rows: list | None = None
                          ) -> tuple[dict[str, float], dict[str, dict]]:
    """R5.03 parent_rank fusion: one batched parent fetch, then rank parents inside each arm and
    fuse (parent_child.fuse_parent_rank). Same fail-safe as _expand_to_parents."""
    ids = parent_child.parent_ids_of(list(vector_rows) + list(text_rows))
    parents, fetch_ms, fetch_error = await _fetch_parents_safe(db, tenant_id, ids)
    scores, rows_by_id, stats = parent_child.fuse_parent_rank(
        [("vector", vector_rows, s.retrieval_vector_weight),
         ("text", text_rows, s.retrieval_text_weight)]
        + ([("vector_parent", parent_vector_rows,
             s.retrieval_vector_weight * s.retrieval_pc_parent_vector_weight)]
           if parent_vector_rows else []),
        parents, rrf_k=RRF_K)
    _record_parent_stats(s, result, stats, fetch_ms, fetch_error, tenant_id)
    return scores, rows_by_id


async def retrieve(db: AsyncSession, *, tenant_id: uuid.UUID, query: str,
                   top_k: int | None = None, doc_type: str | None = None,
                   on_date: date | None = None,
                   focus_terms: list[str] | None = None,
                   synonyms: list[list[str]] | None = None,
                   project_id: uuid.UUID | None = None,
                   knowledge_base_id: uuid.UUID | None = None,
                   session_embed_cache: "dict[str, list[float]] | None" = None,
                   rerank: bool | None = None,
                   granularity: str | None = None) -> RetrievalResult:
    """Retrieve for `query` — split into its sub-questions first when it asks more than one.

    A single question (the common case) goes straight to `_retrieve_one`, byte-for-byte the
    pre-split path. A compound one ("what's the rate, and how far is it from Wipro?") is split
    by query_split.split_compound_query; each part is retrieved on its own — earning only the
    boosts IT triggers — and the windows are merged by taking turns, so one half's boost can no
    longer evict the other half's evidence. See query_split.py for the bug this fixes.

    All parameters are as `_retrieve_one`; every call site (planner pre-retrieval, the
    knowledge_lookup tool, /knowledge/search, the eval scripts) gets the behaviour for free."""
    s = get_settings()
    # getattr: a settings stand-in without these fields (tests) gets the unchanged single path.
    parts = (split_compound_query(query, max_parts=getattr(s, "retrieval_split_max_parts", 3))
             if getattr(s, "retrieval_split_compound", False) else [query])
    kwargs = dict(tenant_id=tenant_id, top_k=top_k, doc_type=doc_type, on_date=on_date,
                  focus_terms=focus_terms, synonyms=synonyms, project_id=project_id,
                  knowledge_base_id=knowledge_base_id, session_embed_cache=session_embed_cache,
                  rerank=rerank, granularity=granularity)
    if len(parts) < 2:
        return await _retrieve_one(db, query=query, **kwargs)

    import time as _time
    started = _time.perf_counter()
    # The embeddings are network calls with no DB involvement: warm them all at once, so the
    # sequential per-part retrievals below (one AsyncSession cannot run queries concurrently)
    # hit the cache instead of paying one embedding round-trip each. A failure here is fine —
    # _retrieve_one embeds on a miss and degrades on its own.
    await asyncio.gather(*(_embed_query(normalize_query(p), session_cache=session_embed_cache)
                           for p in parts), return_exceptions=True)
    results = [await _retrieve_one(db, query=p, **kwargs) for p in parts]
    merged = _merge_part_results(query, parts, results, top_k=top_k or s.retrieval_top_k,
                                 max_per_doc=s.retrieval_max_chunks_per_doc)
    merged.latency_ms = int((_time.perf_counter() - started) * 1000)
    log.info("rag.split tenant=%s parts=%d chunks=%s q=%r", tenant_id, len(parts),
             [len(r.chunks) for r in results], query[:120])
    return merged


def _merge_part_results(query: str, parts: list[str], results: list[RetrievalResult], *,
                        top_k: int, max_per_doc: int) -> RetrievalResult:
    """One window from several sub-question windows: take turns down the ranks (part 1's best,
    part 2's best, part 1's second, …), skip duplicates, and apply the same per-document cap a
    single query gets — so each question the caller asked is represented, and no one document
    floods the merged window. Diagnostics from every part are kept for the trace."""
    merged = RetrievalResult(query=query)
    seen: set[str] = set()
    per_doc: dict[str, int] = {}
    held_back: list[RetrievedChunk] = []
    depth = max((len(r.chunks) for r in results), default=0)
    for rank in range(depth):
        for r in results:
            if rank >= len(r.chunks) or len(merged.chunks) >= top_k:
                continue
            c = r.chunks[rank]
            if c.chunk_id in seen:
                continue
            seen.add(c.chunk_id)
            if per_doc.get(c.document_id, 0) >= max_per_doc:
                held_back.append(c)
                continue
            per_doc[c.document_id] = per_doc.get(c.document_id, 0) + 1
            merged.chunks.append(c)
    # As for one query: if diversity left the window short, take the best of what was held.
    merged.chunks += held_back[:max(0, top_k - len(merged.chunks))]

    merged.vector_hits = sum(r.vector_hits for r in results)
    merged.text_hits = sum(r.text_hits for r in results)
    merged.excluded_expired = max((r.excluded_expired for r in results), default=0)
    errors = [r.error for r in results if r.error]
    merged.error = " | ".join(errors) if errors else None
    merged.parent_child = results[0].parent_child if results else {}
    merged.rerank = results[0].rerank if results else None
    merged.metadata = {"split": {"parts": parts,
                                 "per_part": [{"query": p, "chunks": len(r.chunks),
                                               "intent": (r.metadata or {}).get("intent")}
                                              for p, r in zip(parts, results)]}}
    return merged


async def _retrieve_one(db: AsyncSession, *, tenant_id: uuid.UUID, query: str,
                        top_k: int | None = None, doc_type: str | None = None,
                        on_date: date | None = None,
                        focus_terms: list[str] | None = None,
                        synonyms: list[list[str]] | None = None,
                        project_id: uuid.UUID | None = None,
                        knowledge_base_id: uuid.UUID | None = None,
                        session_embed_cache: "dict[str, list[float]] | None" = None,
                        rerank: bool | None = None,
                        granularity: str | None = None) -> RetrievalResult:
    """One query, one window — the retrieval path itself (see `retrieve` for splitting).

    ``project_id`` scopes retrieval to one project WITHIN a tenant. When given, only that
    project's chunks and tenant-wide (project_id IS NULL) chunks are eligible — this is what
    stops a Lemon Tree question being answered from a DSR cost sheet. When None (the default,
    and the single-project case) every chunk in the tenant is eligible, exactly as before.

    ``rerank`` (R5.01) forces the cross-encoder step on/off for this call — the debug endpoint
    and the eval use it to compare both orderings. None = RETRIEVAL_RERANK_ENABLED.

    ``knowledge_base_id`` (Vocal Backend Tasksheet P2) is the same kind of scope, one level
    up: when given, only that KB's chunks and ungrouped (knowledge_base_id IS NULL, pre-P2)
    chunks are eligible. POST /internal/retrieve calls this once per requested kb_id and
    merges — see app/api/knowledge_bases.py — rather than this function taking a list, so
    the heavily-tuned SQL strings above stay untouched beyond one added predicate each,
    mirroring project_id's exact shape.

    ``granularity`` (R5.03) — "flat" | "parent_child". None = the tenant's
    ``retrieval_granularity`` setting, else the global RETRIEVAL_GRANULARITY. The planner passes
    the value its turn context already resolved, so the hot path adds no tenant query."""
    import time as _time
    started = _time.perf_counter()
    s = get_settings()
    top_k = top_k or s.retrieval_top_k
    today = on_date or date.today()
    result = RetrievalResult(query=query)

    if not (query or "").strip():
        return result

    # Fold caller-side spelling variants ("3bhk" → "3 bhk") before either arm runs, so the
    # semantic embedding and the keyword tsquery both see the form the documents use. result
    # already holds the ORIGINAL query for the trace; this only steers the search.
    query = normalize_query(query)

    # R6.01: Early bail-out — skip the ~300-500 ms OpenAI embedding round-trip for turns
    # that carry zero retrievable intent (pure greetings, confirmations, call closers).
    # The regex check is ~0.05 ms; the length guard makes it unconditionally safe for longer
    # utterances that the pattern does not cover.
    if s.retrieval_smalltalk_skip_enabled and len(query) <= s.retrieval_smalltalk_max_chars:
        if _SMALLTALK_RE.match(query):
            log.debug("retrieval.smalltalk_skip query=%r", query)
            result.skipped_smalltalk = True
            return result

    # A popular question repeats across many different calls for the same tenant within a
    # short window — check the cache before paying for embedding + SQL at all. See the
    # cache helpers above `retrieve()` for the key scheme and invalidation.
    # R5.03: which chunk population each arm searches, and how hits become parents
    # (flat = pre-R5.03; see parent_child.py and the retrieval_pc_* settings).
    tenant_settings: dict | None = None
    if granularity is None:
        tenant_settings = await _tenant_settings(db, tenant_id)
        granularity = tenant_settings.get("retrieval_granularity")
    granularity = parent_child.resolve_granularity(granularity, s.retrieval_granularity)
    pc_mode = granularity == parent_child.GRANULARITY_PARENT_CHILD
    if pc_mode:
        vector_exclude = parent_child.scope_exclusion(s.retrieval_pc_vector_scope)
        text_exclude = parent_child.scope_exclusion(s.retrieval_pc_text_scope)
        mode_sig = (f"{granularity}:v={s.retrieval_pc_vector_scope}:t={s.retrieval_pc_text_scope}"
                    f":f={s.retrieval_pc_fusion}")
    else:
        vector_exclude = text_exclude = parent_child.excluded_role(granularity)
        mode_sig = granularity
    result.parent_child = {"granularity": granularity}

    cache_key = await _cache_key(tenant_id=tenant_id, query=query, doc_type=doc_type,
                                 project_id=project_id, top_k=top_k, today=today,
                                 knowledge_base_id=knowledge_base_id,
                                 granularity=mode_sig)
    cached = await _cache_get(cache_key)
    if cached is not None:
        result.chunks = [RetrievedChunk(**c) for c in cached.get("chunks", [])]
        result.vector_hits = cached.get("vector_hits", 0)
        result.text_hits = cached.get("text_hits", 0)
        result.excluded_expired = cached.get("excluded_expired", 0)
        result.parent_child = {**cached.get("parent_child", result.parent_child), "cached": True}
        result.latency_ms = int((_time.perf_counter() - started) * 1000)
        return result

    # Synonyms are a property of the TENANT'S DATA, not of the caller — so if a call site
    # did not supply them, load the tenant's declared groups here rather than silently
    # retrieving without the bridge. This is the single source of truth: it stops the two
    # retrieval call sites from ever diverging again.
    #
    # THE BUG THIS FIXES: the planner's speculative retrieval passed tc.retrieval_synonyms,
    # but the knowledge_lookup TOOL called retrieve() with none — so "is there a generator?"
    # bridged to "power backup" on the planner path and returned nothing on the tool path.
    # The model reaches for the tool when a product question arrives mid-booking, so the SAME
    # question got inconsistent answers depending on which path happened to serve it.
    #
    # None means "caller did not specify — use the tenant's"; an explicit list (including the
    # empty list the planner passes for a tenant with none declared) is respected as given,
    # so this never overrides a deliberate choice and never re-queries when one was made.
    if synonyms is None:
        if tenant_settings is None:
            tenant_settings = await _tenant_settings(db, tenant_id)
        synonyms = normalize_synonyms(tenant_settings.get("retrieval_synonyms"))

    pool = max(top_k * 4, 20)
    if pc_mode:
        # Several children collapse into one parent, so fetch more hits to still fill top_k.
        pool *= s.retrieval_parent_child_pool_factor
    # Each arm's rows, in rank order. Fused below — how depends on the mode (R5.03).
    vector_rows: list = []
    parent_vector_rows: list = []      # only with retrieval_pc_vector_scope="separate"
    text_rows: list = []

    # R5.08: what the question asks for, in the vocabulary of the chunks' stored facts. Mode
    # "off" skips the analysis entirely — ranking is then byte-for-byte pre-R5.08.
    meta_mode = s.retrieval_metadata_mode
    intent = analyze_query(query) if meta_mode != "off" else QueryIntent()
    meta_flag = metadata_filter_flag(meta_mode, intent)
    meta_diag: dict = {"mode": meta_mode, "intent": intent.describe(), "filter": meta_flag,
                       "filter_fallback": False}

    # --- semantic ---
    try:
        qvec = await _embed_query(query, session_cache=session_embed_cache)
        vec_literal = "[" + ",".join(f"{v:.6f}" for v in qvec) + "]"
        vparams = {"qvec": vec_literal, "tenant_id": str(tenant_id), "today": today,
                   "doc_type": doc_type, "project_id": str(project_id) if project_id else None,
                   "knowledge_base_id": str(knowledge_base_id) if knowledge_base_id else None,
                   "limit": pool, "meta_flag": meta_flag, "exclude_role": vector_exclude}
        t_vec = time.perf_counter()
        rows = (await db.execute(VECTOR_SQL, vparams)).mappings().all()
        if meta_flag and len(rows) < top_k:
            # The filter left too few chunks to fill the window — the tenant has little priced
            # content, or the extractor missed its format. Never answer from a starved pool:
            # re-run unfiltered, and record it so a recurring fallback is visible in the trace.
            meta_diag["filter_fallback"] = True
            meta_diag["filtered_rows"] = len(rows)
            rows = (await db.execute(VECTOR_SQL, {**vparams, "meta_flag": None})
                    ).mappings().all()
            vparams["meta_flag"] = None
        meta_diag["vector_ms"] = round((time.perf_counter() - t_vec) * 1000, 2)
        vector_rows = await _vector_search(db, s, result, vparams, pool=pool,
                                           tenant_id=tenant_id)
        result.vector_hits = len(vector_rows)
        if pc_mode and s.retrieval_pc_vector_scope == parent_child.SCOPE_SEPARATE:
            # R5.03 multi-granularity: the same query against PARENT vectors as its own list.
            # Its scan diagnostics go to a scratch result so the trace keeps the main arm's.
            scratch = RetrievalResult(query=query)
            parent_vector_rows = await _vector_search(
                db, s, scratch, {**vparams, "exclude_role": parent_child.ROLE_CHILD},
                pool=pool, tenant_id=tenant_id)
    except EmbeddingError as e:
        result.error = f"semantic search unavailable: {e}"
    except Exception as e:
        result.error = f"semantic search failed: {e.__class__.__name__}: {e}"

    # --- keyword ---
    # The alias index is built once and applied only to the lexical arm: the semantic arm
    # already generalises "generator" toward "power backup" through the embedding, so
    # widening it there would add noise for no recall it does not already have. It is the
    # exact-lexeme keyword arm that needs the bridge.
    alias_index = build_alias_index(synonyms) if synonyms else None
    tsquery = build_tsquery(query, alias_index)
    try:
        if tsquery is None:
            raise ValueError("no searchable terms")
        rows = (await db.execute(TEXT_SQL, {"q": tsquery, "tenant_id": str(tenant_id),
                                            "today": today, "doc_type": doc_type,
                                            "project_id": str(project_id) if project_id else None,
                                            "knowledge_base_id": (
                                                str(knowledge_base_id) if knowledge_base_id else None),
                                            "limit": pool, "exclude_role": text_exclude,
                                            })).mappings().all()
        text_rows = list(rows)
        result.text_hits = len(text_rows)
    except Exception as e:  # a malformed tsquery must not break the turn
        result.error = (result.error or "") + f" keyword search failed: {e.__class__.__name__}"

    # --- fusion ---
    # Everything below (boosts, per-document cap, min-score) runs on the fused UNITS. In flat
    # mode a unit is a chunk, exactly as before R5.03. In parent_child mode it is a parent (or a
    # standalone chunk), so the policy/pricing boosts see the parent's doc_type, the focus anchor
    # searches the parent's full text, and the per-document cap counts parents.
    if pc_mode and s.retrieval_pc_fusion == parent_child.FUSION_PARENT_RANK:
        scores, rows_by_id = await _fuse_by_parent(db, s, result, vector_rows, text_rows,
                                                   tenant_id=tenant_id,
                                                   parent_vector_rows=parent_vector_rows)
    else:
        scores, rows_by_id = _fuse_chunks(s, vector_rows, text_rows)
        if pc_mode and scores:
            scores, rows_by_id = await _expand_to_parents(db, s, result, scores, rows_by_id,
                                                          tenant_id=tenant_id)

    try:
        result.excluded_expired = int((await db.execute(
            EXPIRED_SQL, {"tenant_id": str(tenant_id), "today": today})).scalar() or 0)
    except Exception:
        result.excluded_expired = 0

    # Policy documents outrank the rest at equal relevance.
    #
    # Asked what configurations exist, the agent answered "two and three-bedroom
    # apartments" — because the 2 BHK and 3 BHK cost sheets are long, number-dense and
    # mention their configuration dozens of times, while the one document that lists ALL
    # of them says each once. Hybrid search ranked the cost sheets above it, only five
    # chunks reach the prompt, and the model answered accurately from a partial view.
    #
    # A document typed `policy` exists to state what is true in general — scope, what we
    # do and do not sell, which configurations exist. When it is relevant at all it should
    # be in front of the model, because the failure it prevents is the model generalising
    # from whichever price list happened to rank highest.
    # ...but only when the question is about SCOPE.
    #
    # An unconditional boost overcorrected. Asked "what amenities do you provide", the
    # policy document — which is about what we sell and do not sell — outranked the
    # amenities list, and the agent answered "we offer flats for ownership only, no
    # rentals" to a question about swimming pools. Right document, wrong question.
    #
    # Policy earns its lift when someone is asking what exists or what is on offer. For
    # anything more specific the ordinary ranking already knows better than a thumb on
    # the scale.
    low_q = query.lower()
    scope_question = (any(sub in low_q for sub in SCOPE_SUBJECTS)
                      and (any(a in low_q for a in SCOPE_ASKS)
                           or any(s in low_q for s in SCOPE_STATEMENTS)))

    if scope_question:
        for key, row in rows_by_id.items():
            if (row.get("doc_type") or "") == "policy" and key in scores:
                scores[key] *= POLICY_BOOST

    # A price question earns the priced documents their place in the window (membership, like
    # the policy boost above — see PRICING_BOOST). The focus anchor below then orders the right
    # variant's sheet to the front; this only ensures the sheets are IN the set to be ordered.
    pricing_question = any(m in low_q for m in PRICING_MARKERS)
    doc_price_boosted: set[str] = set()
    if pricing_question:
        for key, row in rows_by_id.items():
            if (row.get("doc_type") or "") == "pricing" and key in scores:
                scores[key] *= PRICING_BOOST
                doc_price_boosted.add(key)

    # R5.08: chunk-level metadata lifts (membership — before the base snapshot below). See
    # apply_metadata_boosts() for what each signal does and why it cannot double-count
    # with the document-level PRICING_BOOST just applied.
    lifts: dict[str, list[str]] = {}
    if meta_mode != "off" and intent.any:
        stats = apply_metadata_boosts(
            scores, rows_by_id, intent, price_boost=s.retrieval_meta_price_boost,
            area_boost=s.retrieval_meta_area_boost, bhk_boost=s.retrieval_meta_bhk_boost,
            doc_price_boosted=doc_price_boosted)
        lifts = stats.pop("lifts")
        meta_diag.update(stats)
    meta_diag["doc_pricing_boost"] = len(doc_price_boosted)

    # Snapshot relevance BEFORE the focus/preference anchor is applied. This is the score
    # that decides WHO IS IN the window; the anchor below may only decide the ORDER within
    # it. (The scope/policy boost is deliberately above this line: it is a membership signal
    # — a scope question genuinely wants the policy doc in the window — whereas the focus
    # anchor is a tie-break, not a gate. See below.)
    base_scores = dict(scores)

    # R5.01: let the cross-encoder re-score the top of the pool. On success the fused scores
    # REPLACE base relevance (membership) and seed the focus-ordered scores below, so every
    # rule after this point runs unchanged on the better ranking. On any reranker problem
    # (loading, busy, timeout, error) nothing is replaced and this call ranks exactly as
    # before R5.01. Chunks below the candidate cut keep their base order beneath every
    # candidate, so the per-document cap can still reach them for breadth.
    if s.retrieval_rerank_enabled if rerank is None else rerank:
        fused = await _apply_rerank(query, base_scores, rows_by_id, s, result, top_k,
                                    db=db, tenant_id=tenant_id)
        if fused is not None:
            base_scores = fused
            scores = dict(fused)

    # A chunk that names the thing the call is ABOUT is ORDERED ahead of one that does not —
    # but as a tie-break, never as a veto over what the query genuinely matched.
    #
    # Asked the cost of a 2.5 BHK, the search returned the 3 BHK cost sheet at 1.000 and the
    # 2.5 BHK rows at 0.992 — the right chunk was second by less than one percent, and this
    # boost is what lifts it to first. We know the subject: the planner extracted it and
    # working memory carried it, and the caller's standing unit preference is folded in too.
    #
    # THE TRAP a raw multiply fell into. The anchor is applied on EVERY knowledge turn for an
    # identified caller, including questions that have nothing to do with a unit — "is there a
    # daycare at the clubhouse?". A 1.35x boost on every "3 BHK" chunk then filled all eight
    # slots with cost sheets and pushed the creche chunk out of the window, so a booked 3 BHK
    # caller was told "no daycare details" while an unknown caller (no preference, no carried
    # focus) got the right answer from the same documents. The boost must reorder the window,
    # not decide who is in it — so membership is taken from base_scores above and the boost
    # only sorts the chosen set. Exact substring, so it cannot fire on a near-miss.
    boosted = dict(scores)
    for term in (focus_terms or []):
        needle = str(term).strip().lower()
        if len(needle) < 2:
            continue
        for key, row in rows_by_id.items():
            if key not in boosted:
                continue
            haystack = f"{row.get('title') or ''} {row.get('content') or ''}".lower()
            if needle in haystack:
                boosted[key] *= FOCUS_BOOST

    # Cap how much of the window one document may take — MEMBERSHIP by base relevance.
    #
    # A real query returned eight chunks: five from the two cost sheets, and the one chunk
    # that lists what is actually for sale never made it in. The cost sheets are long and
    # number-dense, so they win repeatedly on the same question — and the model then
    # answered "we have 2 BHK" because 2 BHK was what it had been shown.
    #
    # Breadth matters more than depth here. Three chunks from a price list say roughly what
    # two do; the fourth is worth less than a first chunk from a document nobody has seen
    # yet. Anything displaced is still ranked and simply falls below the cut.
    #
    # The cap is a setting (RETRIEVAL_MAX_CHUNKS_PER_DOC) because 2 can be too tight: one
    # document can hold several SIBLING facts a single question needs together. "Corner aur
    # premium mein farak?" needs the corner (Rs.50), premium (Rs.75) and corner+premium
    # (Rs.100) chunks — all in "Additional Charges" — and at 2 the Rs.75 one never reached the
    # model, which then said "premium is Rs.100" (failed 9/9 e2e runs).
    max_per_doc = s.retrieval_max_chunks_per_doc
    ranked_all = sorted(base_scores.items(), key=lambda kv: kv[1], reverse=True)
    # Score desc, then id: equal scores must not be ordered by dict insertion (= SQL row order).
    ranked_all = sorted(base_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    per_doc: dict[str, int] = {}
    selected, overflow = [], []
    for key, _ in ranked_all:
        doc_id = str(rows_by_id[key].get("document_id"))
        if per_doc.get(doc_id, 0) < max_per_doc:
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            selected.append(key)
        else:
            overflow.append(key)
        if len(selected) >= top_k:
            break
    # If diversity did not fill the window, take the best of what was held back.
    if len(selected) < top_k:
        selected += overflow[:top_k - len(selected)]
    # ORDER the chosen window by the focus-boosted score: the anchor settles which variant
    # leads (the right cost sheet first) within a membership base relevance already fixed.
    ranked = sorted(((key, boosted[key]) for key in selected),
                    key=lambda kv: (-kv[1], kv[0]))
    # The min-score cut measures a chunk's own BASE match to the query against the best base
    # match — never against a boosted score. Judging it on the boosted scale would let the
    # anchor re-evict through the threshold the very chunk membership-by-base kept, undoing
    # this fix; the anchor decides order, base relevance decides whether a chunk is worth
    # showing at all.
    best_base = max((base_scores[k] for k in selected), default=0.0)
    for key, score in ranked:
        row = rows_by_id[key]
        # normalise against the best hit so the threshold means "relative to what we found"
        relative = base_scores.get(key, 0.0) / best_base if best_base else 0.0
        if relative < s.retrieval_min_score:
            continue
        meta = row.get("metadata") or {}
        result.chunks.append(RetrievedChunk(
            chunk_id=key, document_id=str(row["document_id"]),
            title=row.get("title") or "document", doc_type=row.get("doc_type") or "",
            heading=meta.get("heading"), content=row["content"], score=score,
            relevance=round(relative, 3),
            vector_rank=row.get("vector_rank"), text_rank=row.get("text_rank"),
            base_rank=row.get("base_rank"), rerank_score=row.get("rerank_score"),
            rerank_rank=row.get("rerank_rank"),
            facts=facts_of(meta), meta_boosts=lifts.get(key, []),
            matched_children=row.get("matched_children") or []))

    # How many of the chunks that made the final window got a metadata lift — the number that
    # says whether R5.08 actually changed this answer, vs. merely re-scoring the pool.
    meta_diag["in_window_lifted"] = sum(1 for c in result.chunks if c.meta_boosts)
    result.metadata = meta_diag
    await _cache_set(cache_key, {"chunks": [asdict(c) for c in result.chunks],
                                 "vector_hits": result.vector_hits,
                                 "text_hits": result.text_hits,
                                 "excluded_expired": result.excluded_expired,
                                 "parent_child": result.parent_child})
    result.latency_ms = int((_time.perf_counter() - started) * 1000)
    # One line per retrieval that had something to match on; DEBUG otherwise, so a plain
    # amenities question does not spam the log. grep "rag.meta" to follow R5.08 decisions.
    log.log(logging.INFO if intent.any else logging.DEBUG,
            "rag.meta tenant=%s mode=%s intent=%s filter=%s fallback=%s tagged=%s untagged=%s "
            "lifted[price=%s area=%s bhk=%s] in_window_lifted=%d doc_pricing=%d q=%r",
            tenant_id, meta_mode, meta_diag["intent"], meta_flag, meta_diag["filter_fallback"],
            meta_diag.get("tagged", "-"), meta_diag.get("untagged", "-"),
            meta_diag.get("price", 0), meta_diag.get("area", 0), meta_diag.get("bhk", 0),
            meta_diag["in_window_lifted"], len(doc_price_boosted), query[:120])
    if meta_diag.get("untagged") and not meta_diag.get("tagged"):
        # Every candidate is untagged: enrichment has not run on this tenant's corpus, so R5.08
        # is silently doing nothing. Fix: python scripts/backfill_chunk_metadata.py --apply
        log.warning("rag.meta.untagged tenant=%s — no candidate chunk carries R5.08 facts; "
                    "run scripts/backfill_chunk_metadata.py", tenant_id)
    return result


def _selftest() -> int:
    """Pure checks on tsquery construction and synonym bridging — no DB, no embeddings.
    Run with ``python -m app.engine.rag.retrieval``.
    """
    failures = 0

    def check(name: str, cond: bool) -> None:
        nonlocal failures
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        if not cond:
            failures += 1

    # Baseline: stopwords out, significant words OR-joined, order/dedupe preserved.
    check("stopwords dropped", build_tsquery("what is the price of the flat") == "price | flat")
    check("empty query -> None", build_tsquery("   ") is None)

    # With a tenant alias index, the caller's word pulls its document-side synonyms into
    # the keyword query — the Feedback #5 case: "generator" must also match "power backup".
    idx = build_alias_index(normalize_synonyms(
        [["generator", "dg set", "power backup"]]))
    q = build_tsquery("is a generator available", idx)
    terms = set((q or "").split(" | "))
    check("original term present", "generator" in terms)
    check("alias words bridged in", {"power", "backup", "dg"} <= terms)
    # No alias index configured leaves the query exactly as it was — zero behaviour change
    # for a tenant that has declared no synonyms.
    check("no index is unchanged",
          build_tsquery("is a generator available") == "generator | available")

    # Configuration spelled without a space must tokenize the SAME as the spaced form, so the
    # keyword arm matches the "3 BHK" the documents use (the price-answer bug).
    check("normalize '3bhk' -> '3 bhk'", normalize_query("price of 3bhk") == "price of 3 bhk")
    check("normalize '2.5-bhk' -> '2.5 bhk'", normalize_query("a 2.5-bhk unit") == "a 2.5 bhk unit")
    check("normalize idempotent on spaced", normalize_query("3 bhk") == "3 bhk")
    check("tsquery '3bhk' matches spaced form",
          build_tsquery("price of 3bhk") == build_tsquery("price of 3 bhk"))
    check("tsquery '3bhk' yields bhk token",
          "bhk" in set((build_tsquery("3bhk") or "").split(" | ")))

    # Knowledge-lookup cache signature (ticket #34): must be deterministic, must change
    # whenever any parameter that changes what SQL can return changes, and must never embed
    # the raw query text itself (the GDPR note in the ticket — no readable caller text in
    # what gets stored as/inside a Redis key).
    base = _cache_signature(query="does it have a pool", doc_type=None,
                            project_id=None, top_k=4)
    check("cache signature is deterministic",
          base == _cache_signature(query="does it have a pool", doc_type=None,
                                   project_id=None, top_k=4))
    check("cache signature ignores case/whitespace (same question, same cache)",
          base == _cache_signature(query="  Does It Have A Pool  ", doc_type=None,
                                   project_id=None, top_k=4))
    check("cache signature changes with doc_type",
          base != _cache_signature(query="does it have a pool", doc_type="pricing",
                                   project_id=None, top_k=4))
    check("cache signature changes with project_id",
          base != _cache_signature(query="does it have a pool", doc_type=None,
                                   project_id=uuid.uuid4(), top_k=4))
    check("cache signature changes with top_k",
          base != _cache_signature(query="does it have a pool", doc_type=None,
                                   project_id=None, top_k=8))
    check("cache signature never contains the raw query text",
          "pool" not in base)

    print("RETRIEVAL SELFTEST:", "PASS" if not failures else f"{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())