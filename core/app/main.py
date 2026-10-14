import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings

# Without this, the root logger has no handler and falls back to Python's built-in
# "last resort" one, which only prints WARNING and above -- every logger.info() and even
# logger.exception() in this app (voice call lifecycle, caught turn failures, tool
# errors...) was silently going nowhere, while LOG_LEVEL sat in .env doing nothing.
# pipecat's own loguru-based logging is separate from this and was never affected, which
# is why call setup/STT/TTS activity was always visible but nothing from our own code was.
# Set before the router imports below so nothing can log during import before this runs.
logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)

# uvicorn logs request lines in full; demo link tokens and voice ?token= must not land there.
from app.observability.log_redaction import install as install_log_redaction  # noqa: E402

install_log_redaction("uvicorn.access", "uvicorn.error")

from app.observability import LoopMonitor
from app.observability.loop_monitor import set_loop_monitor

from app.api.agents import router as agents_router
from app.api.analytics import router as analytics_router
from app.api.appointments import router as appointments_router
from app.api.auth import router as auth_router
from app.api.billing import router as billing_router
from app.api.calls import router as calls_router
from app.api.campaigns import router as campaigns_router
from app.api.compliance import router as compliance_router
from app.api.contacts import router as contacts_router
from app.api.conversations import router as conversations_router
from app.api.custom_instructions_admin import facts_router as dynamic_facts_router
from app.api.custom_instructions_admin import intents_router as intent_vocab_router
from app.api.custom_instructions_admin import router as custom_instructions_router
from app.api.custom_instructions_admin import targets_router as transfer_targets_router
from app.api.customers import router as customers_router
from app.api.dashboard import router as dashboard_router
from app.api.demo_admin import router as demo_admin_router
from app.api.demo_public import router as demo_public_router
from app.api.events_ws import router as events_router
from app.api.foundation import router as foundation_router
from app.api.health import router as health_router
from app.api.integrations import router as integrations_router
from app.api.integrations_tools import router as integrations_tools_router
from app.api.knowledge import router as knowledge_router
from app.api.knowledge_bases import internal_router as knowledge_internal_router
from app.api.knowledge_bases import router as knowledge_bases_router
from app.api.projects import router as projects_router
from app.api.reports import router as reports_router
from app.api.rules_admin import router as rules_router
from app.api.scheduling_admin import router as scheduling_router
from app.api.sessions import router as sessions_router
from app.api.support import router as support_router
from app.api.team import router as team_router
from app.api.telephony import router as telephony_router
from app.api.telephony_hooks import router as telephony_hooks_router
from app.api.templates import router as templates_router
from app.api.tool_metrics import router as tool_metrics_router
from app.api.tools_admin import router as tools_router
from app.api.users import router as users_router
from app.api.voice_ws import router as voice_ws_router
from app.api.plivo_ws import router as plivo_ws_router

logger = logging.getLogger(__name__)


async def _prewarm_voice_models() -> None:
    """Load whichever local STT/TTS models are configured once, up front, in the
    background. Without this the very first live call after a fresh start pays the
    full model-load cost (a one-time download can take a minute or more) inline, with
    a real caller waiting on the other end -- see app/api/voice_ws.py's _build_stt/
    _build_tts, which is what this calls to warm the same cache they'll read from.
    Best-effort: a prewarm failure must not stop the app from serving everything else.
    """
    settings = get_settings()
    if settings.stt_provider != "whisper" and settings.tts_provider != "piper":
        return
    from app.api.voice_ws import _build_stt, _build_tts
    try:
        if settings.stt_provider == "whisper":
            await _build_stt(settings)
        if settings.tts_provider == "piper":
            await _build_tts(settings)
        logger.info("voice model prewarm complete")
    except Exception:
        logger.exception("voice model prewarm failed -- first live call will load cold")


async def _prewarm_filler_audio() -> None:
    """Build the in-memory filler-audio cache at startup (P1.10).

    Synthesises a small set of deterministic filler lines ("Let me check that for you...",
    etc.) once via ElevenLabs and holds the PCM in memory, so the ~25% of turns that speak a
    filler skip the ~280ms live first-audio latency. No-op unless the feature is enabled and
    the TTS provider is ElevenLabs with credentials set -- see prewarm_filler_cache. Runs in
    the background like the other prewarms so it never delays readiness."""
    from app.engine.voice.filler_audio import prewarm_filler_cache
    await prewarm_filler_cache(get_settings())


async def _warm_up_io() -> None:
    """Prime lazily-initialised I/O so the FIRST live turn does not pay for it inline (P1.08).

    The DEEP-mode loop trace (docs/bridge-latency.md) showed the first turn after a fresh
    start stalling the event loop ~200-400 ms on one-time costs: the asyncpg connection
    (TLS handshake to the remote Postgres), SQLAlchemy compiling its statements the first
    time, and the OpenAI embeddings client building its httpx/TLS pool. Touching each here,
    at boot, moves that cost off a real caller.

    Runs in the background (like _prewarm_voice_models) so it never delays readiness, and is
    fully best-effort: any failure is logged and the app serves normally — a cold first turn
    is a latency blip, a failed startup is an outage.
    """
    if not get_settings().startup_warmup_enabled:
        return
    import time as _time
    t0 = _time.perf_counter()

    # DB: prime several pooled connections CONCURRENTLY (P2.05) so the pool holds them warm
    # before the first booking fans its preconditions out to parallel side-sessions -- otherwise
    # those side connections are built cold mid-call (the ~145ms precondition spike seen live).
    # warm_pool does the concurrent-hold trick; the extra ORM select warms SQLAlchemy's
    # statement-compile/loading cache that the hot queries reuse.
    try:
        from sqlalchemy import select

        from app.db.base import get_session_factory, warm_pool
        from app.db.models import Customer

        primed = await warm_pool(get_settings().db_warm_connections)
        async with get_session_factory()() as db:
            await db.execute(select(Customer).limit(1))   # warm ORM compile/loading cache
        logger.info("warm-up: DB pool primed (%d connection(s)) + ORM cache", primed)
    except Exception:
        logger.warning("warm-up: DB priming skipped (first turn will connect cold)",
                       exc_info=True)

    # Embeddings: build the client and do one tiny embed so the httpx/TLS pool to the
    # provider is open before the first retrieval. The mock provider returns instantly and
    # costs nothing; the real one is a single ~1-token request.
    try:
        from app.providers.embeddings.router import get_embeddings
        await get_embeddings().embed(["warm up"])
        logger.info("warm-up: embeddings client primed")
    except Exception:
        logger.warning("warm-up: embeddings priming skipped (first retrieval will init cold)",
                       exc_info=True)

    # R5.01 cross-encoder: load it now so the first knowledge turn is re-ranked instead of
    # served RRF order while the model loads. Only when enabled — a disabled reranker costs no
    # memory. warm_up never raises; a failure is logged and retrieval simply keeps RRF order.
    s = get_settings()
    if s.retrieval_rerank_enabled:
        from app.engine.rag import reranker
        cfg = reranker.config_from_settings(s)
        if await reranker.warm_up(cfg):
            logger.info("warm-up: reranker ready (%s)", cfg.label)
        else:
            logger.warning("warm-up: reranker NOT ready (%s) -- retrieval uses RRF order until "
                           "it loads; see GET /knowledge/reranker", cfg.label)

    logger.info("warm-up complete in %d ms", int((_time.perf_counter() - t0) * 1000))


async def _dialer_loop() -> None:
    """Run the outbound dialer on a fixed interval for the life of the process.

    Enabled by DIALER_ENABLED (default on). Each tick paces running campaigns, enqueues due
    callbacks and originates queued calls within the concurrency cap (app/services/telephony/
    dialer.py). A failing tick is logged and the loop continues — a transient DB blip must not
    stop outbound for the whole process. With multiple core replicas each would run its own
    loop; the concurrency cap is per-tenant per-process, so a shared lock is the scale-out
    upgrade and touches only this function.
    """
    from app.services.telephony.dialer import run_dialer_tick
    settings = get_settings()
    logger.info("dialer loop starting (interval=%ss, cap=%s/tenant)",
                settings.dialer_interval_seconds, settings.dialer_max_concurrent_calls)
    while True:
        try:
            await run_dialer_tick()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("dialer tick raised — continuing")
        await asyncio.sleep(max(1.0, settings.dialer_interval_seconds))


async def _aggregator_loop() -> None:
    """Run the Aggregator Worker on a fixed interval for the life of the process
    (Vocal Backend Tasksheet P7). Rolls finished Call-Runtime-driven calls into
    agent_metrics_daily/daily_org_metrics for any past day not yet rolled up
    (app/services/aggregator.py); a failing tick is logged and the loop continues,
    same discipline as _dialer_loop above."""
    from app.services.aggregator import run_aggregator_tick
    settings = get_settings()
    logger.info("aggregator loop starting (interval=%ss)", settings.aggregator_interval_seconds)
    while True:
        try:
            written = await run_aggregator_tick()
            if written:
                logger.info("aggregator tick wrote %d day rollup(s)", written)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("aggregator tick raised — continuing")
        await asyncio.sleep(max(1.0, settings.aggregator_interval_seconds))


async def _sync_demo_industries() -> None:
    """Create any launch industry (app/demo/industries/seed/*.yaml) missing from the
    database. Only ever creates; admin edits are never overwritten (registry.py). A failure
    is logged, not fatal: existing demos keep working, and the seeds are validated in CI."""
    from app.db.base import get_session_factory
    from app.demo.industries.registry import sync_seed_industries
    try:
        async with get_session_factory()() as db:
            await sync_seed_industries(db)
            await db.commit()
    except Exception:
        logger.exception("demo industry seed sync failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # GTM demo links (docs/gtm-demo-links.md §13): refuse to boot on a demo config that
    # would silently drop leads or break every demo call. A no-op while DEMO_ENABLED=false.
    from app.demo.settings import assert_demo_settings_valid
    assert_demo_settings_valid(get_settings())
    leads_task = None
    if get_settings().demo_enabled:
        asyncio.create_task(_sync_demo_industries())
        # Demo leads → Google Sheet + email, and the sheet's two-way sync (app/demo/leads).
        from app.demo.leads.service import worker_loop as demo_leads_loop
        leads_task = asyncio.create_task(demo_leads_loop())
    asyncio.create_task(_prewarm_voice_models())
    # Warm DB pool + embeddings client so the first live turn isn't cold (P1.08).
    asyncio.create_task(_warm_up_io())
    # Pre-generate the deterministic filler lines' TTS audio (P1.10) so filler turns skip
    # the ~280ms live-TTS latency. Background + best-effort: a failure just means fillers
    # fall back to live TTS, and it never delays readiness. See app/engine/voice/filler_audio.py.
    asyncio.create_task(_prewarm_filler_audio())
    dialer_task = None
    if get_settings().dialer_enabled:
        dialer_task = asyncio.create_task(_dialer_loop())
    aggregator_task = None
    if get_settings().aggregator_enabled:
        aggregator_task = asyncio.create_task(_aggregator_loop())

    # Event-loop stall monitor (P1.08), when enabled: a synchronous call stalling the
    # asyncio hot path names itself in the logs instead of just costing latency.
    monitor: LoopMonitor | None = None
    s = get_settings()
    if s.loop_monitor_enabled:
        monitor = LoopMonitor(
            interval_ms=s.loop_monitor_interval_ms,
            warn_threshold_ms=s.loop_monitor_warn_ms,
            deep=s.loop_monitor_deep,
            slow_callback_ms=s.loop_monitor_slow_callback_ms)
        monitor.start()
        set_loop_monitor(monitor)

    try:
        yield
    finally:
        if dialer_task is not None:
            dialer_task.cancel()
        if aggregator_task is not None:
            aggregator_task.cancel()
        if leads_task is not None:
            leads_task.cancel()
        if monitor is not None:
            await monitor.stop()
            set_loop_monitor(None)


app = FastAPI(title="Voice Agent Platform — Core", version="0.9.0", lifespan=lifespan)

# --- CORS Configuration ---
# Settings.allowed_origins (app/config.py) already reads the ALLOWED_ORIGINS env var via
# pydantic-settings, with a restrictive non-wildcard default — read it from there rather
# than re-reading the env var raw, so there is exactly one source of truth and no route to
# an unintended wide-open "*" default in an environment that forgot to set it.
origins = [o.strip() for o in get_settings().allowed_origins.split(",") if o.strip()]
# The demo page (GTM demo links) is served from its own origin(s) and calls /public/demo.
if get_settings().demo_enabled:
    origins += [o.strip() for o in get_settings().demo_allowed_origins.split(",")
                if o.strip() and o.strip() not in origins]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if "*" not in origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for _router in (health_router, auth_router, users_router, customers_router,
                scheduling_router, appointments_router, tools_router, tool_metrics_router,
                rules_router, compliance_router, knowledge_router, projects_router,
                custom_instructions_router, dynamic_facts_router,
                intent_vocab_router, transfer_targets_router, foundation_router,
                integrations_tools_router,
                conversations_router,
                calls_router, support_router, campaigns_router, analytics_router,
                events_router, sessions_router, voice_ws_router, plivo_ws_router,
                telephony_router, telephony_hooks_router,
                knowledge_bases_router, knowledge_internal_router, templates_router,
                agents_router, contacts_router, dashboard_router,
                billing_router, integrations_router, team_router, reports_router,
                demo_admin_router, demo_public_router):
    app.include_router(_router)
