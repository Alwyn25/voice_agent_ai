from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: str = "local"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://vap:vap@localhost:5432/vap"
    database_url_sync: str = "postgresql+psycopg2://vap:vap@localhost:5432/vap"
    redis_url: str = "redis://localhost:6379/0"

    # --- async DB connection pool (P2.05 support) ---
    # SQLAlchemy's async default pool is tiny (size 5 / overflow 10). That was fine while every
    # turn used the single call-scoped session, but parallel preconditions (P2.05) fan a
    # booking turn out to several side-sessions AT ONCE, so a turn briefly needs 1 (main) + N
    # (side) connections. With only a handful warm, the extra ones are built cold -- a fresh
    # TLS handshake to the remote Postgres that showed up live as a 145ms precondition spike.
    # A slightly larger, EXPLICIT pool (sized here, primed at startup by db_warm_connections)
    # keeps those connections warm so the fan-out reuses them instead of paying reconnect cost.
    #
    # Sizing note for production: total server connections ~= replicas x (db_pool_size +
    # db_max_overflow). Keep that under the Postgres max_connections budget (managed Postgres is
    # often ~100). Defaults below (10 + 20 = 30/replica) suit a handful of replicas; lower them
    # for many replicas or a small DB plan.
    db_pool_size: int = 10
    db_max_overflow: int = 20
    # Connections opened CONCURRENTLY at startup so the pool holds them ready before the first
    # booking (see _warm_up_io). Clamped to db_pool_size (a connection beyond pool_size is an
    # overflow one the pool discards on return, so warming it is wasted). Set to align with
    # parallel_preconditions_max_concurrency's typical real fan-out (~4 DB checks on
    # book_appointment); 0 disables multi-connection warming (single-connection warm-up stays).
    db_warm_connections: int = 4

    # LLM
    llm_provider: str = "openai"          # openai | groq | cerebras | mock
    openai_api_key: str = ""
    openai_model_main: str = "gpt-4o"        # reasoning + response
    openai_model_fast: str = "gpt-4o-mini"   # background jobs, the classifier call
    # Any OpenAI-compatible endpoint. Groq, Together, OpenRouter, Ollama and vLLM all
    # speak this protocol, so pointing the existing client at one is a config change
    # rather than a new provider:
    #     OPENAI_BASE_URL=https://api.groq.com/openai/v1
    #     OPENAI_API_KEY=gsk_...
    #     OPENAI_MODEL_MAIN=llama-3.3-70b-versatile
    # Embeddings are NOT affected — see embedding_base_url below. Groq has no embeddings
    # API at all, so retrieval keeps talking to whoever it was talking to.
    openai_base_url: str = ""
    # Groq — a first-class OpenAI-protocol provider, kept separate from the openai_* settings
    # so OpenRouter/OpenAI can stay configured side by side and you switch with one line
    # (LLM_PROVIDER=groq). Groq's free tier has NO per-prompt credit cap, which is the
    # OpenRouter-free-tier limitation that rejects this engine's ~8k-token planner prompt.
    #     LLM_PROVIDER=groq
    #     GROQ_API_KEY=gsk_...
    # The model ids below are Groq's own — the planner/responder request "main" or "fast"
    # tier abstractly (never a literal model id) and this provider maps that onto these, so
    # no call site needs to know which provider is active.
    # NOTE: the planner uses strict Structured Outputs (response_format json_schema). Pick a
    # Groq model that supports it; if a call fails with a response_format error, switch
    # groq_model_main to a structured-output-capable model.
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # llama-3.3-70b-versatile / llama-3.1-8b-instant (the previous defaults here) no longer
    # exist on Groq's real API -- confirmed live via GET /v1/models, which returned a genuine
    # 404 model_not_found for both. Replaced with real, currently-available models from that
    # same live call.
    groq_model_main: str = "openai/gpt-oss-120b"
    groq_model_fast: str = "openai/gpt-oss-20b"
    # Both gpt-oss models are REASONING models, confirmed live: a trivial two-word answer
    # spent 88-102 completion tokens on hidden reasoning (usage.completion_tokens_details.
    # reasoning_tokens) before writing the visible content -- the exact same failure mode
    # already found and fixed for Cerebras's gpt-oss-120b. Reusing that fix's floor value
    # here for the same reason: GroqProvider.complete_json/stream_text raise any caller's
    # max_tokens (400/300, sized for OpenAI) to at least this, so the reasoning step can't
    # silently starve the visible answer.
    groq_min_max_tokens: int = 1500
    # Cerebras -- confirmed via a live /v1/models call to actually offer gpt-oss-120b,
    # gemma-4-31b, qwen-3.8-27b (NOT llama-3.3-70b, which doesn't exist on this account).
    # Strict structured-output (response_format=json_schema) is confirmed working, live,
    # against this exact schema -- the earlier "unverified" note here was resolved.
    cerebras_api_key: str = ""
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    cerebras_model_main: str = "gpt-oss-120b"
    cerebras_model_fast: str = "gpt-oss-120b"
    # gpt-oss-120b is a REASONING model: it spends completion tokens on a hidden "thinking"
    # step before writing the actual JSON answer, on top of whatever the caller budgeted for
    # the visible response. The planner/responder call sites pass a fixed max_tokens (400/300)
    # sized for OpenAI, which has no such overhead -- confirmed live that this silently starves
    # Cerebras: one real call spent 458 of its 500-token budget on reasoning alone, truncating
    # the JSON mid-string; another spent the entire budget on reasoning and returned nothing.
    # CerebrasProvider.complete_json raises this floor under any caller's request rather than
    # touching the shared per-call values every provider reads, so OpenAI/Groq are unaffected.
    # 1500 covers the largest real reasoning-token count seen (458) several times over, on top
    # of a real full answer (a genuine, complete response measured at ~580 total completion
    # tokens end to end).
    cerebras_min_max_tokens: int = 1500
    # When on, LLM_PROVIDER is ignored in favour of LLM_FALLBACK_ORDER below: a plain,
    # comma-separated priority list (e.g. "cerebras,openai,groq") that this app walks in
    # order, sticking to the first entry until it fails with a real credit-exhaustion error
    # (HTTP 402) -- not re-probed per turn, since re-trying an already-dead provider on
    # every call would pay its full timeout/retry cost for no benefit; see
    # fallback_provider.py. Changing priority (e.g. moving Groq to the front once that
    # account is funded) is a one-line env edit here, never a code change -- that is the
    # entire point of this being config, not a hardcoded primary/fallback pair.
    llm_fallback_enabled: bool = False
    # Each name needs its own *_API_KEY set to actually be usable; an entry whose key is
    # missing is skipped (with a warning) rather than crashing startup, so this list can
    # include a provider you plan to enable later without breaking the app today.
    llm_fallback_order: str = "cerebras,groq,openai"
    # Retrieval keeps its own key and endpoint. Point the chat model at Groq and the
    # embeddings client would follow it there — where there is no embeddings API at all —
    # and every knowledge answer would fail with an auth error that looks nothing like
    # the cause. Leave these blank to use the OpenAI settings above.
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    # Ask the embedding endpoint to return exactly this many dimensions. Leave at 0 for
    # text-embedding-3-small's native 1536, which is what the column was built for.
    embedding_dimensions: int = 0
    llm_timeout_seconds: float = 12.0
    llm_max_retries: int = 1
    # Every LLM call (classifier, planner, responder, post-call summary) passes through
    # a per-provider, per-minute rate limiter before it reaches OpenAI/Groq -- see
    # app/providers/llm/rate_limiter.py. A request over the budget waits briefly rather
    # than firing and coming back as a 429. Protects against the PROVIDER's real limit,
    # not a spend/traffic cap -- set each *_rpm_limit to just under your account's actual
    # allowance.
    llm_rate_limit_enabled: bool = True
    # Placeholder -- OpenAI's real per-account limit depends on usage tier and can't be
    # known from this codebase. Check platform.openai.com/settings/organization/limits
    # and set OPENAI_RPM_LIMIT to match.
    openai_rpm_limit: int = 500
    # Groq's free-tier limit for llama-3.3-70b-versatile, confirmed directly against
    # their API. Raise this if/when the account moves to Groq's paid tier.
    groq_rpm_limit: int = 30
    # How long a call will wait for room under the budget before giving up and firing
    # anyway (fail-open -- the limiter must never be the reason a turn hangs or fails).
    llm_rate_limit_max_wait_seconds: float = 5.0
    # NOTE: the old `planner_mode` setting ("single_call" | "two_call") was removed — it was
    # read nowhere and its premise is obsolete. The "understanding" pass is NOT a serial extra
    # call: planner._understand_and_retrieve fires it CONCURRENTLY with speculative retrieval
    # (asyncio.gather) on the fast model, so it hides behind the ~300ms embedding+search that
    # runs regardless, and is skipped entirely on trivial turns and every retry/chain leg. There
    # is no separate call to "fold in", so the switch controlled nothing. (extra="ignore" above
    # means a stray PLANNER_MODE in an old .env is silently dropped, not an error.)
    llm_model_summary: str = "gpt-4o"      # memory quality compounds across every future
                                          # call; set to gpt-4o-mini to trade accuracy for cost
    # Tenant locale — override via TENANT_TIMEZONE env var for deployments outside India.
    # Used wherever the engine must name a time in local terms (TODAY/TIME in the planner
    # prompt, appointment read-backs). IANA tz database name, e.g. "America/New_York".
    tenant_timezone: str = "Asia/Kolkata"
    # Run the fast UNDERSTANDING classifier CONCURRENTLY WITH the strong PROPOSE call instead
    # of concurrently with retrieval (P4-1). The original design overlapped understand with a
    # speculative retrieval to "hide" it — but live measurement (replay_outliers.py) showed
    # retrieval is only ~15-30ms while understand is a full ~1.3s LLM round-trip, so the overlap
    # hides nothing and understand is effectively a serial ~1.3s call on EVERY non-trivial turn.
    # With this ON, a non-ambiguous, non-retry turn instead: (1) retrieves (~15ms, result always
    # kept), (2) runs understand() and propose() together via asyncio.gather, so understand hides
    # behind propose (the longer call) and the turn's wall drops by ~the understand time.
    # THE ONE BEHAVIOUR CHANGE: propose now always sees the retrieved knowledge, even on a
    # scheduling/status turn that understand would have routed away from documents. understand's
    # do_retrieve still gates what the RESPONDER (the only text the caller hears) is given, so
    # responder grounding is byte-for-byte unchanged; only the planner's INPUT gains context.
    # Off by default -- validate on the golden dataset before enabling (watch for a scheduling
    # turn that gets answered from a document). Project-ambiguous turns and retry/chain legs keep
    # the serial path (understand must resolve the project BEFORE propose's prompt is built).
    understanding_concurrent_with_propose: bool = False

    # Consent gate, semantic layer (app/engine/planner/consent.py). While an action is parked
    # awaiting the caller's yes, the fast understanding call also reads the reply against the
    # exact question asked and returns {decision, evidence} -- no extra model call, ~20-30
    # output tokens, and only on those turns. The deterministic layer always runs regardless;
    # turning this OFF only removes the model's read, so an unrecognised phrasing is re-asked
    # instead of understood. It never weakens the gate: a model "accept" is honoured only when
    # the quoted words are really in the transcript. Rollback switch: CONSENT_SEMANTIC_ENABLED.
    consent_semantic_enabled: bool = True
    # How many times a parked action is re-asked after a side question / unclear reply before
    # the engine stops appending the question (the action stays parked; a later "yes" then
    # gets a fresh read-back first, never a silent release).
    consent_max_reasks: int = 2

    # Hybrid planner routing: route the planner to a different model tier than the responder.
    # "main" (default) keeps the existing behaviour; "fast" routes plan() to CEREBRAS_MODEL_FAST
    # (e.g. qwen-3.8-27b) while the responder stays on the main tier (gpt-oss-120b).
    # The planner drives routing/tool selection, not customer-facing text, so a faster
    # non-reasoning model is acceptable there; set PLANNER_LLM_MODEL_TIER=fast to enable.
    planner_llm_model_tier: str = "main"

    # Custom Instructions (advisory) — author-time compile + per-tenant runtime caps.
    # The advisory block rides the responder's cacheable prefix every turn, so bound both
    # the row count and the total injected length (spec Open Q1 — tune against real usage).
    #
    # The compile step runs its OWN LLM, chosen independently of the voice-agent LLMs, so
    # you can point it at a cheaper/stronger model or a different vendor without touching
    # the live call path. OpenAI, OpenRouter and Groq all speak the OpenAI protocol, so one
    # configurable client covers them (and any OpenAI-compatible endpoint via ci_llm_base_url).
    # Leave ci_llm_provider EMPTY to reuse the voice agent's provider at its own main tier.
    ci_llm_provider: str = ""             # "" (reuse voice agent) | openai | openrouter | groq | mock
    ci_llm_api_key: str = ""              # API key for the CI provider (e.g. an OpenRouter key)
    ci_llm_base_url: str = ""             # override endpoint; blank = the provider's default
    ci_compile_model: str = ""            # model id for the compile step; "" -> openai_model_main
    # Completion budget for the compile call. The structured output carries the reasoning,
    # the (string-encoded) rule JSON, any proposed intents/targets to provision, a scope and
    # 2-3 test scenarios — and a REASONING model (e.g. Groq gpt-oss) also spends completion
    # tokens on its hidden reasoning. 900 was too tight once provisioning fields were added
    # (Groq returns 400 "max completion tokens reached before generating a valid document"),
    # so the floor is generous; the step is author-time and low-volume, so the cost is trivial.
    ci_compile_max_tokens: int = 3000
    ci_advisory_max_rows: int = 25        # max ACTIVE advisory instructions per tenant
    ci_advisory_max_chars: int = 4000     # max total chars of active advisory snippets
    # Dynamic Knowledge caps. Facts are cited VERBATIM in the responder's trusted PROJECT FACTS
    # tier, so bound how much trusted text one tenant can inject (prompt bloat + a break-out
    # surface). Enforced at author time (create/update), never on the hot turn path.
    dk_max_active_rows: int = 25          # max ACTIVE dynamic facts per tenant
    dk_max_fact_chars: int = 600          # max chars of a single fact's content
    dk_max_total_chars: int = 6000        # max total chars of active fact content
    # Routing instructions are clamped to a priority band at/above this floor, reserving the
    # lower band (0..floor-1) for hand-authored compliance rules (DNC, legal). A tenant rule
    # can therefore never be authored at a priority that would let it precede compliance
    # (spec §7 / Q4). Blocks are already un-overridable by the engine; this covers ordering.
    ci_routing_priority_min: int = 100

    # Post-call memory
    fact_confidence_threshold: float = 0.6   # below this a fact is stored for human review
    profile_promotion_threshold: float = 0.8 # confidence needed to overwrite a profile field
    max_facts_per_call: int = 12

    # Knowledge / RAG
    embedding_provider: str = "openai"           # openai | mock
    embedding_model: str = "text-embedding-3-small"
    chunk_target_chars: int = 900
    chunk_overlap_chars: int = 150
    # Semantic chunking (see app/engine/rag/chunkers.py). A long enumeration — an amenity
    # grid, a spec list — is split into groups of at most this many items so a single fact
    # ("dog park", "creche") gets a concentrated embedding instead of being 1/20th of an
    # averaged vector. Small enough to focus each item, large enough to keep neighbours for
    # a "list everything" question.
    chunk_list_group_size: int = 8
    chunk_min_chars: int = 20                     # a chunk shorter than this answers nothing
    # Normalisation of extracted text before chunking (app/engine/rag/normalize.py): page
    # headers/footers, hyphenated wraps, layout tables -> pipe tables, hard line wraps.
    #   auto   — every step except line-unwrapping always; unwrapping only when the text shows
    #            extraction signals. A no-op on clean markdown (verified on all seed documents).
    #   always — all steps (an ingestion service can also pass normalize_mode="always" per call).
    #   off    — text is chunked exactly as given.
    ingest_normalize_mode: Literal["auto", "always", "off"] = "auto"

    # R5.03: parent-child ("small-to-big") chunking — see app/engine/rag/parent_child.py.
    # At ingestion every chunk above becomes a PARENT and is also split into ~200-char CHILDREN.
    # Both are embedded, so which one retrieval searches is a pure query-time switch
    # (retrieval_granularity below) — no re-ingestion to turn it on, A/B it, or roll it back.
    # Off = ingestion writes exactly what it wrote before R5.03 (standalone chunks only).
    chunk_parent_child_enabled: bool = False
    chunk_child_target_chars: int = Field(default=200, ge=60, le=600)
    # Overlap (whole trailing sentences) between children cut from ONE long paragraph only;
    # never carried across paragraphs (one cost-sheet unit per paragraph must stay separate).
    # 100 measured on the DSR 2 BHK sheet (one paragraph, nine ~90-char unit records): at 60 one
    # record was cut across two children with no child holding it whole; at 100 all nine are
    # whole in at least one child, for two extra children.
    chunk_child_overlap_chars: int = Field(default=100, ge=0, le=300)
    chunk_child_list_group_size: int = Field(default=3, ge=1, le=20)
    # A child shorter than this is merged into its neighbour — never dropped, because in
    # parent-child mode text no child carries is unreachable.
    chunk_child_min_chars: int = Field(default=40, ge=1, le=300)
    # A chunk body at or below this is not split (it stays 'standalone'). Measured case: at 260
    # the 258-char "The project" chunk stayed whole, so its "900 flats" fact competed diluted
    # against focused ~200-char children of other documents and fell out of the window.
    chunk_child_split_min_chars: int = Field(default=200, ge=60, le=2000)
    # Prefix each child's breadcrumb with the document title (unless the heading contains it).
    # Off: measured on DSR it cost 4.5 pts Hit@1 — a shared prefix makes a document's children
    # look alike and blunts the precision they exist for.
    chunk_child_title_prefix: bool = False
    # Cut a long paragraph of number-led records ("875 sq ft, ... Total 1,03,93,750. 880 sq ft,
    # ...") at record boundaries instead of sentence packing (chunkers._records). Off: on DSR
    # it lost evidence on 2 questions with the default scopes; kept for record-heavy corpora.
    chunk_child_record_split: bool = False
    # 5 was too few once the knowledge base grew past a handful of documents. With 32
    # chunks, a question about configurations pulled three price-list chunks and left the
    # one document listing the full range outside the window — so the agent answered
    # "two and three-bedroom apartments" and denied the 1 BHK existed.
    #
    # 8 is affordable: measurement showed prompt size costs about 25 ms for four times the
    # tokens, so the extra chunks are nearly free on the clock and change what the model
    # can possibly know.
    retrieval_top_k: int = 8
    retrieval_min_score: float = 0.15            # below this a chunk is not worth showing
    retrieval_vector_weight: float = 1.0
    retrieval_text_weight: float = 0.8
    # Most chunks ONE document may place in the top_k window, so a long number-dense cost sheet
    # cannot crowd out breadth (retrieval.py, "Cap how much of the window"). 2 is the historical
    # value. The 2026-09-28 retrieval eval measured 3 at Complete@8 91.4% -> 94.3% (it lets
    # sibling facts from one document through together, e.g. corner/premium/both charges) with
    # no question losing evidence — pending an e2e confirmation before changing the default.
    # Required at 3 when retrieval_rerank_enabled is on (see there).
    retrieval_max_chunks_per_doc: int = Field(default=2, ge=1, le=20)

    # R5.08: chunk metadata enrichment (see app/engine/rag/enrichment.py). Every chunk is
    # tagged at ingestion with the prices, per-sq-ft rates, areas and BHK configurations it
    # states; retrieval matches those against what the QUERY asks for.
    #   off    — tags are ignored; retrieval ranks exactly as before R5.08 (kill switch).
    #   boost  — chunks holding the asked-for fact are lifted into the window (default).
    #   filter — boost, PLUS a price/area question searches only chunks that hold that fact
    #            in the vector arm. The keyword arm stays unfiltered as a safety net, and an
    #            under-filled filtered search re-runs unfiltered. Opt-in: a hard filter can hide
    #            a valid answer that states no number ("there is no 1 BHK cost sheet").
    retrieval_metadata_mode: Literal["off", "boost", "filter"] = "boost"
    # Membership lift for a chunk that states a price/rate when the question is about price.
    # Not applied on top of the document-level PRICING_BOOST — a chunk gets one or the other.
    retrieval_meta_price_boost: float = Field(default=1.3, ge=1.0, le=3.0)
    # Membership lift for a chunk that states an area when the question is about size.
    retrieval_meta_area_boost: float = Field(default=1.2, ge=1.0, le=3.0)
    # Lift for a chunk covering a BHK configuration the question names ("2.5 BHK price").
    retrieval_meta_bhk_boost: float = Field(default=1.2, ge=1.0, le=3.0)
    # Enrich chunks at ingestion. Off = chunks are stored untagged (retrieval treats them as
    # neutral). Kept separate from the mode so tagging can run before ranking is switched on.
    ingestion_enrich_metadata: bool = True
    # R5.03: what retrieval SEARCHES.
    #   flat          — standalone + parent chunks: byte-for-byte the pre-R5.03 behaviour.
    #   parent_child  — standalone + child chunks; each child hit is replaced by its parent's
    #                   text (deduplicated), so the model still reads ~900-char context.
    # Documents ingested before R5.03 have only standalone chunks and behave identically in
    # both modes, so this is safe to flip while a backfill is half done.
    retrieval_granularity: Literal["flat", "parent_child"] = "flat"
    # What each arm searches in parent_child mode: children | parents | both (standalone chunks
    # are always searched). Children give the vector arm precise ~200-char targets; parents give
    # a whole-section vector for broad questions ("how many flats in total") and keep the keyword
    # arm free of ts_rank's short-text bias (a 200-char child saying "BHK" five times outranks the
    # 900-char section that actually answers). See docs/r5.03-parent-child-chunking.md.
    # "separate" (vector arm only): search children AND parents as two ranked lists, so a section
    # gets one RRF vote for its best child (precision) and one for its whole-section vector
    # (broad questions). "both" instead mixes them in ONE list (a section keeps only its best).
    # Defaults = the only combination that passed the enable gate with ZERO regressions on the
    # DSR KB (2026-09-30, deterministic ordering): vector=separate, keyword=both -> Complete@8
    # 90.9 vs 88.6 flat, Hit@1 70.5 vs 65.9, MRR .801 vs .755; 6 improved / 0 worse (p=0.031).
    # Children alone win the top of the window but lose broad questions; the parent-vector list
    # keeps those. Cost: one extra vector query, ~+10 ms.
    retrieval_pc_vector_scope: Literal["children", "parents", "both", "separate"] = "separate"
    retrieval_pc_text_scope: Literal["children", "parents", "both"] = "both"
    # parent_rank: rank PARENTS inside each arm, then fuse (one rank per parent per arm — stops
    # one document's many children crowding every other document out). child_rank: fuse raw
    # child hits, collapse afterwards (first R5.03 version; kept for A/B). parent_child.py.
    retrieval_pc_fusion: Literal["parent_rank", "child_rank"] = "parent_rank"
    # "separate" vector scope: weight of the parent-vector list, as a multiple of
    # retrieval_vector_weight (the child-vector list keeps the full weight).
    retrieval_pc_parent_vector_weight: float = Field(default=1.0, ge=0.0, le=2.0)
    # child_rank fusion only: parent score = best child's RRF score plus this share of each
    # other matching child's score, capped at retrieval_parent_sibling_cap x the best. Default 0:
    # measured on the DSR KB (2026-09-30) 0.1 cost 2.3 pts Complete@8 and 9 pts Hit@1 vs 0.
    retrieval_parent_sibling_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval_parent_sibling_cap: float = Field(default=0.25, ge=0.0, le=1.0)
    # Children are ~4x as many rows as parents, and several collapse into one parent, so the
    # search pool is widened by this factor in parent_child mode to still fill top_k parents.
    retrieval_parent_child_pool_factor: int = Field(default=2, ge=1, le=5)

    # R5.02: HNSW scan tuning for the vector arm (see app/engine/rag/vector_index.py).
    # An HNSW scan collects ef_search nearest vectors across ALL tenants and only then applies
    # the tenant/project/date filters, so a filtered query can silently come back short.
    # Iterative scan (pgvector >= 0.8) keeps walking the graph until the LIMIT is filled.
    # All values are transaction-scoped (set_config is_local) — nothing leaks across requests.
    retrieval_hnsw_tuning_enabled: bool = True     # kill switch: false = pre-R5.02 query path
    # Candidate list size per scan step. Raised to at least the query's pool automatically.
    retrieval_hnsw_ef_search: int = Field(default=100, ge=1, le=1000)
    # relaxed_order (fast; rows re-sorted in SQL) | strict_order | off
    retrieval_hnsw_iterative_scan: Literal["relaxed_order", "strict_order", "off"] = "relaxed_order"
    # Upper bound on tuples one iterative scan may visit — the cost ceiling for a filter that
    # matches almost nothing. pgvector's own default.
    retrieval_hnsw_max_scan_tuples: int = Field(default=20000, ge=1)
    # Re-plan the vector query with the real tenant_id instead of a cached generic plan, so a
    # small tenant gets its btree index and a large one gets HNSW. ~0.3 ms planning per query.
    retrieval_vector_force_custom_plan: bool = True
    # When the vector arm returns fewer rows than asked, run a bounded count of eligible rows
    # (<= pool rows scanned) to tell "tenant simply has few chunks" from "index dropped rows".
    # Logs rag.vector.underfilled on the latter. Costs ~1 ms, and only on a short result.
    retrieval_vector_underfill_check: bool = True
    # DEBUG ONLY: log EXPLAIN (ANALYZE, BUFFERS) of every vector query — executes it twice.
    retrieval_vector_explain: bool = False

    # R5.01: cross-encoder re-ranking of the top of the hybrid pool (app/engine/rag/reranker.py).
    # false = the exact pre-R5.01 ranking path. Off by default pending an e2e confirmation.
    # Enable TOGETHER WITH retrieval_max_chunks_per_doc=3: in the 2026-09-28 retrieval eval
    # (70 queries) rerank + Hinglish gloss + cap 3 gave Complete@8 91.4% -> 94.3%, Hit@1
    # 65.7% -> 71.4%, MRR 0.775 -> 0.811 with no query losing evidence; rerank with cap 2
    # pushed one question's evidence (HG035, modular kitchen) out of the window.
    retrieval_rerank_enabled: bool = False
    # fuse    = rank-fusion of the existing order and the cross-encoder order (both get a vote)
    # replace = cross-encoder order alone within the candidate pool (for A/B only)
    retrieval_rerank_mode: Literal["fuse", "replace"] = "fuse"
    # Which query languages are re-ranked (comma-separated values of responder.detect_language:
    # english, hinglish, hindi; or "all"). The default model is trained on English MS MARCO
    # only; raw romanised Hinglish pushed evidence OUT of the window in the 2026-09-28 eval.
    # With retrieval_rerank_translate on (below) the model reads an English gloss instead and
    # every language gained, so "all" is the default. Set "english" if translation is off.
    retrieval_rerank_languages: str = "all"
    # Gloss romanised Hindi/Hinglish into English for the CROSS-ENCODER INPUT ONLY ("2bhk ka
    # price kya hai" -> "2bhk of price what is"). The search arms, the prompt and the logs keep
    # the caller's words. Deterministic word list, no model call, ~0 ms. See reranker.py.
    retrieval_rerank_translate: bool = True
    # Extra word/phrase glosses on top of the built-in grammar list, for every tenant
    # (env JSON, e.g. {"khel kood": "sports"}). A tenant adds its own domain vocabulary in
    # tenant.settings["rerank_glossary"] (same shape), which wins over this.
    retrieval_rerank_glossary: dict[str, str] = Field(default_factory=dict)
    # Model = HF repo id (or a local directory with tokenizer.json + the ONNX file). The revision
    # is PINNED so the model cannot change under a deployed build; the Dockerfile bakes exactly
    # this revision/file into the image. Change all three together.
    retrieval_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    retrieval_rerank_revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    # int8 export: ~2.5x faster than onnx/model.onnx (fp32) on CPU at near-identical scores.
    retrieval_rerank_onnx_file: str = "onnx/model_quint8_avx2.onnx"
    # HF cache root holding the model. Empty = env RETRIEVAL_RERANK_CACHE_DIR (set by the
    # Dockerfile to the baked copy), else HF's default (HF_HOME).
    retrieval_rerank_cache_dir: str = ""
    # Allow fetching the pinned files from HuggingFace when they are not on disk. Only ever
    # happens in the background (never on a caller's turn). Set false in production to
    # guarantee the baked copy is used.
    retrieval_rerank_allow_download: bool = True
    # How many of the top pre-rerank chunks are re-scored. Must exceed retrieval_top_k for the
    # reranker to be able to pull a chunk INTO the window. Cost is linear in this number.
    retrieval_rerank_candidates: int = Field(default=12, ge=2, le=50)
    # Token budget per (question + chunk) pair; a 900-char chunk is ~200-230 tokens. Cost is
    # roughly linear in this number.
    retrieval_rerank_max_tokens: int = Field(default=256, ge=32, le=512)
    # ORT threads per inference, and how many inferences may run at once in this process.
    # threads x max_concurrency = the most cores reranking can ever occupy — keep it well below
    # the box's core count so the voice pipeline always has CPU.
    retrieval_rerank_threads: int = Field(default=2, ge=1, le=16)
    retrieval_rerank_max_concurrency: int = Field(default=2, ge=1, le=8)
    # Hard deadline per rerank. On the planner path retrieval overlaps the understanding LLM
    # call, so this is hidden unless it exceeds that call; on the knowledge_lookup tool path it
    # is serial. Past it, the turn keeps the RRF order.
    retrieval_rerank_timeout_ms: int = Field(default=600, ge=10, le=10000)
    # Fusion weights / damping (mode=fuse). Equal weights = neither ordering dominates.
    retrieval_rerank_weight: float = Field(default=1.0, ge=0.0)
    retrieval_rerank_base_weight: float = Field(default=1.0, ge=0.0)
    retrieval_rerank_rrf_k: int = Field(default=60, ge=1)
    # DEBUG: log the full before/after candidate table for every retrieval at INFO level.
    retrieval_rerank_debug_log: bool = False
    # R6.01: Early retrieval bail-out — skip the OpenAI embedding round-trip (~300-500 ms)
    # for turns that are pure smalltalk, confirmations, or greetings and can never benefit
    # from document retrieval. The check runs in <0.1 ms (regex) before _embed_query().
    # Safe to keep on: patterns are anchored and require short utterances; any ambiguous turn
    # falls through to full retrieval as before.
    retrieval_smalltalk_skip_enabled: bool = True
    # Maximum character length below which the smalltalk check applies. Longer utterances are
    # almost certainly factual questions and bypass the check unconditionally.
    retrieval_smalltalk_max_chars: int = 120
    # Compound questions ("what's the rate for the 3 BHK, and how far is it from Wipro?") are
    # split into their sub-questions and each is retrieved on its own, then merged by taking
    # turns (app/engine/rag/query_split.py). Without it a boost earned by ONE half (a price
    # marker lifts every pricing document) fills the whole window and the other half's
    # document never reaches the model — which then invented a distance. A single question
    # takes the unchanged single-query path. false = always one query, exactly as before.
    retrieval_split_compound: bool = True
    # Most sub-questions retrieved for one turn (each is one extra keyword+vector query; the
    # embeddings run concurrently).
    retrieval_split_max_parts: int = Field(default=3, ge=2, le=5)

    # P2.02: commit d257134 already lets the responder stream its first sentence straight
    # to TTS mid-generation instead of waiting for the full reply -- but only on a turn
    # where a read-only tool (search_slots, etc.) just ran. A plain respond turn with no
    # tool at all (most factual Q&A: "does it have a pool?") gets none of that benefit
    # today and always waits for the complete reply. This extends the same, already-tested
    # mechanism to that case. Off by default pending its own real before/after test.
    stream_respond_no_tool_enabled: bool = False

    # P3.04 — deterministic response template cache. When True, write-tool turns whose reply
    # is fully determined by the tool output (book_appointment confirmed, reschedule confirmed)
    # skip the responder LLM entirely and return a pre-built phrase in the caller's language.
    # Off by default; flip RESPONDER_TEMPLATE_CACHE_ENABLED=true in .env to enable.
    # Watch app.engine.voice.templated_reply at DEBUG level to see hit/skip decisions.
    responder_template_cache_enabled: bool = False

    # Voice (app/api/voice_ws.py, app/engine/voice/). Deepgram is speech-in, ElevenLabs is
    # speech-out; the turn itself still runs through the same process_turn() a chat turn
    # does. Blank by default like the other provider keys above -- the socket refuses a
    # connection rather than starting a call it cannot actually run.
    deepgram_api_key: str = ""
    # Silence (ms) Deepgram waits before deciding an utterance is finished. Left at
    # Deepgram's own default (a few ms) this fires on any brief mid-sentence pause -- a
    # caller reading out a phone number in groups, or just thinking for a second, gets
    # split into two separate turns, each answered on its own. A caller saying "five nine
    # eight five three two six five five" as one number was heard as "5985326" (rejected
    # as an invalid phone number) followed by a stray "55". Raised from 500 to 700 after a
    # real call: a caller who paused mid-sentence to think ("kya aap mujhe teen bees... K
    # flat ke baare mein bata sakte hain? jaise ki uska...") still got split into two
    # separate turns at 500ms, each answered independently with no memory of the other.
    # 700ms survives a longer natural pause without making the agent feel slow to respond
    # once someone actually stops talking. This is a mitigation, not a full fix -- the
    # underlying cause is that a turn currently takes 5-8s to answer (several sequential
    # LLM calls), which is long enough that a caller often starts a second thought before
    # the first one is answered; no endpointing value fixes that outright.
    deepgram_endpointing_ms: int = 700
    # Repairs the specific failure the comment above describes, from the other side: even
    # at 700ms, a caller can still pause long enough between digit groups to get finalized
    # early. Rather than trying to prevent that (Deepgram's Finalize RPC fires unconditionally
    # the instant VAD goes quiet -- see bridge.py's hold-and-merge logic for why intercepting
    # it isn't the safer option), TurnBridge holds a transcript that looks like a bare,
    # still-incomplete number for a short grace window and merges it with whatever comes
    # next, rather than running the planner on half a phone number. See
    # app/engine/voice/entity_continuation.py.
    entity_continuation_enabled: bool = True
    # How long to hold a suspicious bare-number fragment waiting for a continuation. ~800ms
    # -- comparable to the endpointing pause above, since that's the pause length actually
    # observed to trigger this in the first place.
    entity_continuation_grace_ms: int = 800
    # Safety ceiling: total time a fragment can be held across repeated continuations before
    # giving up and running the planner on whatever's accumulated, so a persistent misread
    # (or a caller who trails off mid-number and never finishes) can't hang a turn forever.
    entity_continuation_max_hold_ms: int = 3000
    # Real production trace data showed VAD's silence wait (deepgram_endpointing_ms above)
    # as the single largest per-turn latency cost in one benchmark. Shortening it outright
    # risks the exact mid-sentence-pause bug fixed above. Smart Turn is a small, local ML
    # model (bundled with pipecat, no network call, ~12ms CPU inference) that judges from
    # the AUDIO ITSELF whether an utterance sounds complete, so a shorter timer can be
    # paired with a semantic check instead of relying on silence duration alone to catch
    # every real pause. Defaults to off: this only takes effect once explicitly enabled,
    # same as entity_continuation_enabled above -- existing behavior is unchanged until then.
    smart_turn_enabled: bool = False
    # Deliberately a conservative first cut, not competing head-on with what Smart Turn's
    # judgment alone might allow -- half the current 700ms, not e.g. 200ms, since Smart Turn
    # augments this timer rather than replacing it outright. Only used for VAD's stop_secs
    # when smart_turn_enabled is True; deepgram_endpointing_ms is used otherwise.
    smart_turn_endpointing_ms: int = 350
    # Same hold-and-merge shape as entity_continuation_grace_ms above: Smart Turn saying
    # "incomplete" doesn't cancel the already-fired Finalize RPC (can't -- same reason as
    # entity_continuation: see bridge.py), it holds the resulting transcript for this grace
    # window waiting for a continuation instead of running the planner on a sentence Smart
    # Turn itself flagged as unfinished.
    smart_turn_grace_ms: int = 900
    # Safety ceiling, same reasoning as entity_continuation_max_hold_ms: bounds how long a
    # turn can be held if Smart Turn keeps saying "incomplete" turn after turn, so a
    # persistent misread can't hang a turn forever.
    smart_turn_max_hold_ms: int = 3000
    # SileroVAD minimum audio volume before a frame is considered speech. Frames below
    # this RMS level are classified as silence regardless of the model's confidence,
    # which prevents keyboard clicks, mic handling noise, and low-level background hum
    # from triggering a false-start. Pipecat's default is 0.6; 0.6 is a safe baseline
    # that keeps the agent from reacting to near-silent audio — tune up (e.g. 0.7) in
    # noisy environments or down toward 0.4 for callers on very quiet lines.
    vad_min_volume: float = 0.6
    # --- Semantic endpointing (task P2.01) ------------------------------------------------
    # The single biggest TTFA lever in the latency benchmark: VAD's silence window is ~45%
    # of total time-to-first-audio. deepgram_endpointing_ms (700ms) and smart_turn_endpointing_ms
    # (350ms, a deliberately conservative first cut) both still spend most of that time simply
    # WAITING OUT SILENCE before a turn can start. Semantic endpointing collapses the silence
    # wait to ~120ms and instead trusts Smart Turn's judgement of the AUDIO to decide whether
    # the caller actually finished -- so a real pause is caught by MEANING, not by a long timer.
    #
    # This does NOT replace smart_turn_enabled: it is the aggressive end of the same mechanism.
    # When on, voice_ws.py builds the Smart Turn analyzer (whether or not smart_turn_enabled is
    # set, since the short window is only safe WITH the semantic check) and drives VAD stop_secs
    # from semantic_endpointing_ms below instead of the 350/700ms values. Off by default, exactly
    # like smart_turn_enabled and entity_continuation_enabled: flipping this in .env is the whole
    # rollout switch, and clearing it restores the previous behaviour with no code change. The
    # benchmark projects ~-580ms TTFA here; validate on real call recordings before enabling in
    # production (a too-short window can clip callers who pause mid-sentence -- the corroboration
    # and dictation-grace logic below are what keep that from happening).
    semantic_endpointing_enabled: bool = False
    # The shortened VAD silence window used for stop_secs when semantic_endpointing_enabled is on.
    # Defaulted to 200 -- the value real-call sweeps validated -- so semantic_endpointing_enabled=true
    # ALONE gives the production-grade config with no other keys required. The 120ms benchmark target
    # sits below the bundled Smart Turn model's reliable floor (it starves the model of trailing
    # silence -> false INCOMPLETE, and shatters digit dictation); 200ms captures ~93% of the latency
    # win reliably. Only lower it with fresh trace evidence and a better endpointing model.
    semantic_endpointing_ms: int = 200
    # Corroboration (task P2.01, semantic mode only). Real-call testing showed the bundled Smart
    # Turn model, starved of trailing silence at a 120ms window, misfires INCOMPLETE on turns that
    # are actually finished ("Yes.", "registered? Can you give that again?") -- each costing a full
    # continuation-grace hold for nothing. With this on, an INCOMPLETE verdict is only trusted
    # enough to HOLD when the transcript ITSELF also looks unfinished (a dangling connector, bare
    # digits, or a partial entity -- see entity_continuation.transcript_looks_unfinished). A
    # complete-looking clause the model merely misflagged -- including a bare "yes"/"ok" answer --
    # runs immediately. Smart Turn stays the primary judge; the transcript is a cheap backstop for
    # its noise at short windows. Turn off to trust the raw verdict unconditionally.
    semantic_require_incomplete_corroboration: bool = True
    # Continuation grace/ceiling used INSTEAD of smart_turn_grace_ms and entity_continuation_grace_ms
    # while semantic endpointing is on. Callers dictate a phone number or name in groups separated by
    # 0.9-2s pauses -- longer than the 800-900ms default graces, so fragments used to flush and run
    # as their own turns before the continuation arrived (the "shattered number" bug). 1500ms with a
    # 4000ms ceiling covers real dictation gaps; each merged fragment re-arms the window, and the
    # ceiling bounds a caller who trails off mid-number so a turn can never hang. Only a corroborated
    # hold (above) waits this long, so a misflagged complete turn is not delayed by it.
    semantic_continuation_grace_ms: int = 1500
    semantic_continuation_max_hold_ms: int = 4000
    # Debug trace for every end-of-turn decision (task P2.01 debuggability). When on, bridge.py
    # emits one structured log line per finalized transcript -- the Smart Turn verdict, word count,
    # which gate fired (smart_turn_hold / incomplete_overruled / entity_hold / ran), and the
    # resulting text -- so a specific call can be replayed to see exactly why a turn started when it
    # did. Kept behind a flag so it costs nothing in production; enable it while tuning
    # semantic_endpointing_ms or investigating a clipped/duplicated turn.
    endpointing_trace_enabled: bool = False
    # Barge-in carry-over (app/engine/voice/barge_in.py). The caller speaking while a turn is
    # still being planned cancels that turn (pipecat's interruption) and rolls it back -- which
    # used to throw the caller's words away with it: "book a site visit" vanished and the
    # follow-up "tomorrow morning" was planned on its own, with no idea what it was answering.
    # With this on, the cancelled turn's words are carried into the next transcript, so the
    # planner sees the whole request exactly as a chat turn would. The max age bounds how stale
    # a carried utterance may be before it is dropped instead of merged.
    voice_barge_in_carryover_enabled: bool = True
    voice_barge_in_carryover_max_age_s: float = 30.0
    # smart_format and punctuate were previously enabled to help with number formatting
    # ("eleven hundred" → 1100). Both options send the transcript through Deepgram's
    # post-processing pipeline which adds ~20-40 ms STT latency with no benefit for a
    # voice agent: the planner reads raw words fine, punctuation is irrelevant for
    # intent detection, and numerals (below) already handles the number-formatting case.
    # Disabled by default; re-enable via .env if you want formatted transcripts for
    # debugging or a chat channel that shows the raw STT output to users.
    deepgram_smart_format: bool = False
    deepgram_punctuate: bool = False
    deepgram_numerals: bool = True
    deepgram_model: str = "nova-3"
    # "multi" is Deepgram's code-switching mode -- it transcribes Hindi, English, and
    # Hindi-English mixed speech (Hinglish) within the same utterance without needing to
    # know in advance which one the caller will use. A caller here can freely move between
    # "what's the price" and "iska price kya hai" mid-call. A fixed single-language setting
    # (or leaving this unset, which quietly defaults to English) mangles real Hindi speech
    # into nonsense English-sounding text -- the responder prompt already knows how to
    # reply in Hindi/Hinglish (prompts.py), it just never gets a usable transcript to work
    # from otherwise.
    deepgram_language: str = "multi"
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""
    # REVERTED. eleven_multilingual_v2 was tried here to fix a real cross-language voice-
    # consistency complaint, but broke actual calls: ElevenLabs' websocket accepted the
    # configured elevenlabs_voice_id under this model, marked every TTS context "final"
    # immediately, and returned literally no audio -- silently, no error surfaced -- on
    # every single reply including the opening greeting ("no BotStartedSpeakingFrame within
    # 3.0s ... TTS context completed with no audio" in the logs). Most likely this specific
    # voice_id is not enabled for multilingual_v2 on this account/voice. Back on turbo_v2_5,
    # which is confirmed working, until that's resolved on the ElevenLabs side (a different
    # voice known to support multilingual_v2, or confirming account/voice compatibility)
    # rather than shipping a model change that silently kills every call.
    elevenlabs_model: str = "eleven_turbo_v2_5"
    # Left unset by ElevenLabsTTSService, these default to a high-stability, low-style
    # rendering that reviewers heard as "mechanical." Lower stability lets intonation
    # vary turn-to-turn instead of sounding identically flat every time; a non-zero
    # style adds expressiveness on top. Per ElevenLabs' own guidance this is the
    # standard starting point for a natural conversational voice, not an extreme setting.
    elevenlabs_stability: float = 0.45
    elevenlabs_similarity_boost: float = 0.8
    elevenlabs_style: float = 0.3
    elevenlabs_use_speaker_boost: bool = True

    # Filler-audio cache (P1.10 -- app/engine/voice/filler_audio.py). A small catalog of short,
    # CLOSED-CLASS lines the agent speaks verbatim: the tool-preamble fillers the engine emits
    # while a tool runs ("Let me check that for you...", "Give me just a moment...") AND the
    # common responder acknowledgements/closings ("Okay.", "Theek hai.", "Thank you!"). Each of
    # those turns pays the full ~280ms ElevenLabs first-audio latency even though the words never
    # change. Pre-synthesising the whole catalog ONCE at startup and replaying the cached PCM on
    # an exact match removes that latency on the turns they occur (fillers + common phrases are
    # ~25% of all turns per the Sept-2026 benchmark), at the cost of a few seconds of boot-time
    # synthesis and a few hundred KB of RAM. Off makes every line go through live TTS as before.
    filler_audio_cache_enabled: bool = True
    # Playback chunk size for the cached PCM. The pre-generated clip is one contiguous PCM
    # buffer; it is pushed downstream as a sequence of TTSAudioRawFrames of this duration so it
    # streams to the transport like real TTS output would (letting barge-in interrupt part-way
    # through, and keeping any single frame small). 40ms is a conservative, well-below-a-word
    # granularity; the exact value is not latency-sensitive since the whole clip is already in
    # memory. Must divide evenly into 16-bit samples -- odd byte counts are impossible here.
    filler_audio_chunk_ms: int = 40
    # Per-phrase ceiling for the one-off startup synthesis REST call to ElevenLabs. Generous
    # because it runs in the background at boot with no caller waiting; a phrase that times out
    # is simply skipped (that phrase falls back to live TTS) rather than delaying readiness.
    filler_audio_build_timeout_ms: int = 8000
    # TTFA (time-to-first-audio) instrumentation (P1.10 verification -- app/engine/voice/ttfa.py).
    # Logs one "voice TTFA" line per turn (ttfa_ms + source: filler_cache/filler_live/responder)
    # and feeds GET /health/ttfa's rolling per-source summary, so the filler's real effect on
    # how soon the caller hears audio can be measured on live calls rather than estimated. One
    # cheap INFO line per turn; off makes the tracker and probe a no-op.
    ttfa_logging_enabled: bool = True
    # Local, keyless engines for development -- no per-minute cost, noticeably lower quality
    # and higher latency than the cloud engines above, so production stays on deepgram/
    # elevenlabs. Switching either provider does not require the other's key to be set.
    stt_provider: str = "deepgram"        # deepgram | whisper
    tts_provider: str = "elevenlabs"      # elevenlabs | piper
    whisper_model: str = "base.en"        # tiny.en | base.en | small.en | medium.en
    piper_voice: str = "en_US-lessac-medium"

    # Confidence banding
    confidence_high: float = 0.80
    confidence_low: float = 0.55
    conf_stt_penalty: float = 0.85        # applied when stt_confidence < threshold
    conf_stt_threshold: float = 0.75
    # Below this STT confidence a WRITE action (anything that changes a record) is always read
    # back to the caller before it runs, however confident the planner is -- the engine-level
    # form of "don't act on words you may have misheard". Read-only lookups and plain answers
    # are never held back by it. Replaces per-tenant turn rules that blocked every tool or
    # forced an unanswerable question on any low-confidence turn (see planner.banded_action).
    # 0 disables.
    conf_stt_readback_below: float = 0.55
    conf_ungrounded_factor: float = 0.80  # factual answer with no retrieved knowledge
    conf_ambiguity_factor: float = 0.90   # model reported missing info
    conf_repeat_penalty: float = 0.15     # subtracted after repeated clarifications

    # Observability — P1.08 (profile the bridge, catch blocking calls on the async path)
    #
    # The loop monitor is a background sampler that flags anything stalling the asyncio
    # event loop (a synchronous DB/file call on the hot path is exactly what it catches).
    # It is cheap enough to leave on in production; leave it off and it costs nothing.
    #     LOOP_MONITOR_ENABLED=true
    loop_monitor_enabled: bool = False
    loop_monitor_interval_ms: int = 50      # how often to sample loop responsiveness
    loop_monitor_warn_ms: int = 100         # lag above this is logged as a stall
    # Deep mode turns on asyncio debug + slow-callback logging, which names the exact
    # blocking callsite ("Executing <Handle …> took N s"). Higher overhead — switch on
    # while investigating a stall, not as a steady-state setting.
    #     LOOP_MONITOR_DEEP=true
    loop_monitor_deep: bool = False
    loop_monitor_slow_callback_ms: int = 50

    # Break the per-turn "bridge" overhead (session load, message persist, trace write,
    # commit, …) out into named sub-stages on debug.timing, so the ~60 ms BRG figure is
    # attributable instead of an unexplained gap. Free to compute; on by default because
    # the numbers only appear in the already-present debug payload.
    bridge_timing_enabled: bool = True

    # Run a tool's DB-backed preconditions concurrently instead of one-after-another (task
    # P2.05). The executor's precondition loop today awaits each check in sequence, so
    # book_appointment pays centre_exists (12ms) + executive_exists (12ms) + slot_available
    # (35ms) + no_conflict (18ms) = ~77ms end to end; run concurrently the wall-time collapses
    # to ~max(35ms), saving ~40ms on booking/reschedule turns. Applies to inbound AND outbound
    # calls -- both run every tool through the one shared executor, so there is nothing
    # direction-specific to switch on.
    #
    # WHY A FLAG, DEFAULT OFF: true parallelism here cannot reuse ctx.db. A SQLAlchemy
    # AsyncSession is NOT safe for concurrent operations -- gathering several checks that each
    # await ctx.db.execute() on the SAME session raises InvalidRequestError. So each concurrent
    # DB check runs on its OWN short-lived read-only session (own pool connection). Off by
    # default -- exactly like semantic_endpointing_enabled and smart_turn_enabled -- so
    # flipping this in .env is the whole rollout switch and clearing it restores the byte-for-
    # byte sequential behaviour on the shared session with no code change.
    parallel_preconditions_enabled: bool = False
    # Ceiling on how many extra precondition connections may be checked out of the Postgres
    # pool AT ONCE, across all in-flight turns. The pool is small (SQLAlchemy default 5 + 10
    # overflow); without a cap, a burst of simultaneous booking turns each opening 4 side
    # sessions could starve it and stall unrelated queries. A semaphore bounds the blast
    # radius: over the cap, the extra checks simply queue (still no slower than sequential).
    parallel_preconditions_max_concurrency: int = 8
    # Debug trace for precondition parallelism (P2.05 debuggability). When on, the executor
    # emits one structured log line per tool turn -- which checks ran in parallel vs inline,
    # each check's wall-time, and the sequential-sum vs actual-wall saving -- so a specific
    # slow turn can be inspected. Per-check timings are ALSO folded into
    # outcome.validation["checks"] regardless of this flag, so they persist on the
    # tool_execution audit row. Off by default; enable while validating the rollout.
    parallel_preconditions_trace: bool = False

    # --- Speculative retrieval prefetch (P3.06) -------------------------------------------
    # Use the otherwise-idle STT endpointing window to warm the retrieval query embedding BEFORE
    # the final transcript lands. TurnBridge (bridge.py) embeds a stable confident interim in the
    # background; the real turn's embedding becomes a free in-memory cache hit. Off by default.
    voice_speculative_prefetch_enabled: bool = False
    voice_speculative_prefetch_min_words: int = 3
    voice_speculative_prefetch_min_confidence: float = 0.6
    voice_speculative_prefetch_trace_enabled: bool = False

    # Warm lazily-initialised I/O at startup (P1.08). The DEEP-mode loop trace showed the
    # FIRST live turn pays one-time costs — the asyncpg connection (TLS to remote Postgres),
    # SQLAlchemy's compiled-statement cache, and the OpenAI embeddings client (httpx pool) —
    # as ~200-400 ms event-loop stalls. Priming them at boot moves that cost off the first
    # caller. Best-effort and backgrounded, so it never delays or blocks startup.
    startup_warmup_enabled: bool = True

    # Auth
    jwt_secret: str = "change-me-in-.env"
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 30
    refresh_token_days: int = 7

    allowed_origins: str = "https://ai-voice-agent-ui-production.up.railway.app,http://localhost:5173,http://localhost:3000"

    # Document delivery (WhatsApp)
    # The send_document tool ships a shareable document (floor plan, brochure, price sheet,
    # …) to the caller over WhatsApp. It is provider-pluggable so a real integration is a
    # CONFIG change, never a code change:
    #
    #   whatsapp_provider=auto      # the default: use the cloud API when credentials are
    #                               # present, otherwise fall back to the stub. So the day
    #                               # the access token lands in .env, real sends begin — no
    #                               # deploy, no code edit.
    #   whatsapp_provider=cloud_api # force the real Meta WhatsApp Cloud API (fails loudly if
    #                               # credentials are missing, for a production box that must
    #                               # never silently stub).
    #   whatsapp_provider=stub      # force the stub even if credentials exist (staging/demo).
    #
    # To go live with Meta's WhatsApp Cloud API, set in .env:
    #   WHATSAPP_PROVIDER=auto            (or cloud_api)
    #   WHATSAPP_ACCESS_TOKEN=EAAG...     the permanent system-user token
    #   WHATSAPP_PHONE_NUMBER_ID=1234567890
    # base_url / api_version rarely change; document caption sender name is cosmetic.
    whatsapp_provider: str = "auto"          # auto | cloud_api | stub
    whatsapp_access_token: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_api_base_url: str = "https://graph.facebook.com"
    whatsapp_api_version: str = "v21.0"
    whatsapp_send_timeout_seconds: float = 8.0

    # ------------------------------------------------------------------ Telephony
    # The carrier that actually places outbound PSTN calls and streams the media. Behind a
    # provider abstraction (app/services/telephony/) exactly like WhatsApp above, so going
    # live with a real carrier is a CONFIG change, not a code change:
    #
    #   telephony_provider=stub      # the default: simulate the full call lifecycle
    #                                # (dial -> ring -> answered/busy/no_answer/voicemail) with
    #                                # no carrier account. Every state, retry and webhook path
    #                                # is exercised; no real audio is placed.
    #   telephony_provider=plivo     # (primary, India) / exotel / twilio -- real PSTN. Each
    #   telephony_provider=exotel    # reads its own credentials below and fails loudly if they
    #   telephony_provider=twilio    # are missing, so a production box never silently stubs.
    #   telephony_provider=auto      # a real provider when its credentials are present, else stub.
    telephony_provider: str = "stub"
    # The number outbound calls present as caller-ID. A tenant can override in its own
    # settings JSON (settings.caller_id); this is the platform-wide fallback.
    telephony_caller_id: str = "+910000000000"
    # Where a real carrier connects the answered call's media stream. The provider builds the
    # per-call stream URL from this; the stub ignores it.
    telephony_media_stream_base_url: str = ""
    # Full HTTPS URL Plivo POSTs call-lifecycle events to (answer, hangup, …).
    # Must be publicly reachable; use ngrok or a real domain in development.
    telephony_webhook_url: str = ""
    # Secret shared with the carrier's webhook config; when set, POST /webhooks/telephony/inbound
    # requires a matching X-Webhook-Secret header. Leave empty to skip (stub/dev only).
    telephony_webhook_secret: str = ""

    # Per-carrier credentials (blank until you go live; see telephony_provider above).
    plivo_auth_id: str = ""
    plivo_auth_token: str = ""
    exotel_api_key: str = ""
    exotel_api_token: str = ""
    exotel_subdomain: str = ""
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""

    # Dialer worker (app/services/telephony/dialer.py). It consumes queued Call rows and
    # originates them through the provider, respecting the concurrency cap and campaign pacing,
    # and re-checking the dial policy at dial time (a customer who opts out after a call was
    # queued is still protected).
    dialer_enabled: bool = True           # run the background dial loop in this process
    dialer_interval_seconds: float = 5.0  # how often the loop ticks
    dialer_max_concurrent_calls: int = 10  # per-tenant cap on simultaneous live calls
    dialer_batch_size: int = 25           # most calls to originate in one tick, per tenant

    # Retry policy for a dial that was not answered (busy / no answer / voicemail). The
    # calling-policy rules (consecutive-no-answer back-off, daily caps) still gate every retry
    # at dial time -- these just decide WHEN and HOW MANY times a retry is scheduled.
    dialer_max_attempts: int = 3          # including the first attempt
    dialer_retry_backoff_minutes: int = 15  # base wait before a retry; grows per attempt

    # Aggregator Worker (app/services/aggregator.py, Vocal Backend Tasksheet P7). Rolls
    # finished calls into agent_metrics_daily/daily_org_metrics on a fixed interval, the same
    # background-loop pattern as the dialer above.
    aggregator_enabled: bool = True
    aggregator_interval_seconds: float = 300.0  # 5 min — cheap query, no reason to run hot

    # Fernet key used by app/services/credentials.py to encrypt stored integration
    # credentials (Shopify/HubSpot/etc. API keys) at rest. "" in a fresh environment; must be
    # set to a real key before any integration credential is saved, or that write fails loudly
    # rather than silently storing a secret in plaintext.
    credentials_encryption_key: str = ""

    # Billing (app/services/billing/, Vocal Backend Tasksheet P8). "stub" is the default,
    # no-op provider (app/services/billing/stub.py) so billing/subscription endpoints work
    # in every environment without a real Stripe account configured.
    billing_provider: str = "stub"
    stripe_secret_key: str = ""

    # TEST-ONLY escape hatch for calling HOURS. When true, the outbound gate ignores the
    # time-of-day rules (legal_calling_hours, lead_preferred_hours, no_marketing_before_ten,
    # no_marketing_after_eight) so a dial can be exercised outside 09:00-21:00. Scoped to hours
    # ALONE — DNC, open-escalation, consent and frequency caps are still enforced. Default
    # false; leave it false in production.
    outbound_ignore_calling_hours: bool = False

    # TEST-ONLY escape hatch for the attempt/answer/lifetime COUNTERS (daily_attempt_cap,
    # marketing_daily_cap, stop_after_repeated_no_answer, back_off_after_*_no_answer,
    # failed_interaction_ceiling, lifetime_contact_ceiling, new_lead_one_attempt_a_day /
    # new_lead_cooling_period) — so testing against the same handful of seeded customers
    # doesn't get spent on "one call a day" instead of exercising the call. Scoped to those
    # counters ALONE — DNC, open-escalation, consent and calling hours are still enforced.
    # Default false; leave it false in production.
    outbound_ignore_frequency_caps: bool = False

    # Stub simulator controls (telephony_provider=stub only). The outcome is normally
    # "answered"; set one of the others to exercise the retry / voicemail paths, or use a
    # phone-suffix convention (a number ending in the two digits below forces that outcome)
    # so a demo can show both a connect and a no-answer deterministically.
    telephony_stub_outcome: str = "answered"   # answered|busy|no_answer|voicemail|failed
    telephony_stub_ring_seconds: float = 1.5   # simulated time from dial to outcome
    telephony_stub_talk_seconds: float = 8.0   # simulated answered-call duration
    telephony_stub_cost_per_minute: float = 0.6  # simulated per-minute price, stored on the Call

    # ------------------------------------------------------------------ GTM demo links
    # Private per-prospect demo pages for cold email (docs/gtm-demo-links.md). Everything
    # tunable lives here so the domain, limits, spend cap, lead email and leads sheet change
    # with a restart, never a deploy. Off by default: with DEMO_ENABLED=false nothing below
    # is read and the app behaves exactly as before. When it is true, app/demo/settings.py
    # refuses to start the app on a missing or contradictory combination (§13).
    demo_enabled: bool = False
    demo_tenant_slug_prefix: str = "demo-"   # each prospect's own tenant: demo-{short id}
    # Comma-separated tenant ids whose admins may use /admin/demo (our own staff tenants).
    # Roles are per tenant and "admin" is "*", so without this ANY customer's admin could
    # read every prospect and lead; this pins demo administration to our own organisation.
    demo_admin_tenant_ids: str = ""

    # Domain / URLs. Every link is {demo_public_base_url}/t/{token}, so moving to a new
    # domain is a config change; links already sent keep working while the old one resolves.
    demo_public_base_url: str = ""
    demo_allowed_origins: str = ""           # comma-separated; CORS + WebSocket Origin check
    dashboard_url: str = ""                  # "Open in admin" links in lead emails

    # Link lifetime
    demo_link_ttl_days: int = 7
    demo_default_timezone: str = "Asia/Kolkata"   # relative slots when a prospect has none

    # One link per company record, sent to every person there with their own recipient
    # code (?r=…), so we know who opened it without putting an email address in the URL.
    # The token is stored hashed (lookup) and encrypted with these Fernet keys (so the link
    # can be exported again for people added later). Comma-separated: the first encrypts,
    # all decrypt, so a key can be rotated by putting the new one first. Generate one with
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    demo_link_token_keys: str = ""
    # UTM tags on every exported URL, for the email tool's and any analytics' reports. The
    # campaign is the batch name (slugged), else demo_utm_campaign, else demo_<industry>.
    # The email step (e1…e4, li for a LinkedIn note) goes in utm_content.
    demo_utm_enabled: bool = True
    demo_utm_source: str = "email"
    demo_utm_medium: str = "cold_email"
    demo_utm_campaign: str = ""
    demo_export_steps: str = "e1,e2,e3,e4,li"   # one URL column per step in the export CSV
    # Ask a visitor who they are (optional, skippable) when we can't tell from their link:
    # off | unknown_only (no recipient code, or a forwarded one) | always
    demo_identify_prompt: Literal["off", "unknown_only", "always"] = "unknown_only"

    # Call limits (a link can override the first four for itself; see demo_links)
    demo_max_call_seconds: int = 300
    demo_max_calls_per_link_per_day: int = 10
    demo_max_concurrent_calls_per_link: int = 1
    demo_max_minutes_per_link: int = 60
    demo_max_concurrent_calls_global: int = 20

    # Spend. 0 = no cap. "Today" is measured in demo_budget_timezone.
    demo_daily_budget_usd: float = 25.0
    demo_budget_alert_pct: int = 80
    demo_budget_timezone: str = "Asia/Kolkata"

    # Abuse protection
    demo_rate_limit_per_min: int = 30
    demo_sessions_per_ip_per_hour: int = 5
    demo_turnstile_enabled: bool = False      # required outside local development...
    # ...unless this is false: an explicit, logged opt-out for running without the bot check
    # (only the per-IP, per-link, global and budget limits then stop a script).
    demo_turnstile_required: bool = True
    demo_turnstile_secret: str = ""
    demo_ip_hash_secret: str = "change-me"    # HMAC key for demo_sessions.ip_hash
    # Where the caller's IP comes from, for every per-IP limit. Behind proxies the socket
    # peer is the proxy, and the left part of X-Forwarded-For is whatever the client sent,
    # so the IP is the entry the outermost proxy WE run behind appended:
    #   x-forwarded-for + hops=N   the N-th entry from the right (Railway alone: 1;
    #                              Cloudflare → Railway: 2)
    #   cf-connecting-ip / x-real-ip   a single value set by the edge (only when the origin
    #                              can't be reached except through that edge)
    #   hops=0                     the socket peer (no proxy)
    demo_client_ip_header: str = "x-forwarded-for"
    demo_trusted_proxy_hops: int = 1

    # Uploads
    demo_upload_max_file_mb: int = 20
    demo_upload_max_files_per_prospect: int = 50

    # Lead email. smtp covers Google Workspace, Zoho, SES, Microsoft 365 or any mail server;
    # sendgrid/resend use demo_email_api_key; none = sheet only.
    demo_email_provider: Literal["smtp", "sendgrid", "resend", "none"] = "smtp"
    demo_lead_notify_emails: str = ""        # comma-separated recipients
    demo_alert_emails: str = ""              # budget / sync-failure alerts
    demo_lead_from_email: str = ""
    demo_lead_from_name: str = "Phonn Demo Leads"
    demo_lead_email_subject_prefix: str = "[Demo lead]"
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_starttls: bool = True
    demo_email_api_key: str = ""
    demo_notify_retry_minutes: int = 5
    demo_notify_max_attempts: int = 5

    # Leads Google Sheet (the sales team works leads there; demo_leads is the safety copy).
    # Empty sheet id = sheet off. The key is base64 of the service-account JSON, or a path.
    demo_leads_gsheet_id: str = ""
    google_service_account_json: str = ""
    demo_sheet_leads_tab: str = "Leads"
    demo_sheet_activity_tab: str = "Demo activity"   # empty = don't write the activity tab
    demo_sheet_sync_minutes: int = 5
    demo_sheet_auto_create_tabs: bool = True
    demo_lead_statuses: str = "New,Contacted,No answer,Follow up,Converted,Not interested"

    # Industries: launch definitions loaded on first start; admins manage them after that.
    demo_industries_seed_dir: str = "app/demo/industries/seed"
    # Workbook imports. A column the industry definition doesn't map is offered to the admin
    # as a new fact ("propose") or skipped ("ignore"); a definition's sheet.extra_columns
    # overrides this. Columns named here are bookkeeping and never become facts (comma-separated,
    # case-insensitive; a definition's sheet.ignore_columns adds to them).
    demo_import_new_columns: Literal["propose", "ignore"] = "propose"
    demo_import_ignore_columns: str = "#,Minimum met?"

    # Tracking / retention
    demo_webhook_url: str = ""               # optional: follow-up bucket changes POSTed here
    demo_transcript_retention_days: int = 90


@lru_cache
def get_settings() -> Settings:
    return Settings()