from fastapi import APIRouter
from sqlalchemy import text

import redis.asyncio as aioredis

from app.config import get_settings
from app.db.base import get_engine
from app.observability.loop_monitor import get_loop_monitor

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/health/loop")
async def health_loop():
    """Event-loop health (P1.08). Reports the loop monitor's rolling stats so a stall on
    the async hot path is visible over HTTP, not only in the logs — ``max_lag_ms`` and
    ``spikes`` climbing means something is blocking the loop. ``enabled: false`` means the
    monitor was not switched on for this process (LOOP_MONITOR_ENABLED)."""
    monitor = get_loop_monitor()
    if monitor is None:
        return {"enabled": False}
    return {"enabled": True, **monitor.stats.as_dict()}


@router.get("/health/filler-audio")
async def health_filler_audio():
    """Filler-audio cache state (P1.10). Shows whether the pre-generated filler PCM built at
    startup, which phrases are cached, and the live hit/miss tally -- so a wording drift
    (a phrase that stops matching and silently falls back to live TTS) or a failed startup
    build is visible over HTTP, not only in the boot logs. Importing here, lazily, keeps the
    health router free of pipecat's heavy dependency tree unless this endpoint is hit."""
    from app.engine.voice.filler_audio import get_filler_cache
    return get_filler_cache().describe()


@router.get("/health/ttfa")
async def health_ttfa():
    """Time-to-first-audio summary (P1.10 verification). Rolling per-source stats
    (count/min/p50/avg/max ms) for how long after a transcript the caller first hears audio,
    grouped by what produced it: ``filler_cache`` / ``phrase_cache`` (the cache wins -- a
    tool-preamble filler resp. a common responder phrase served from memory), ``filler_live``
    (filler via live TTS), ``responder`` (live responder speech). Compare ``filler_cache`` vs
    ``filler_live`` for the cache saving, and the cached sources vs ``responder`` for the
    perceived-latency saving. Empty until
    voice turns have run. Lazy import keeps pipecat out of the health router until hit.

    P3.07: a source's nested ``out_buffer`` summary is how long that first audio then waited
    in the output transport before its first packet hit the socket (see ttfa.py's "wire"
    timing). ~0 ms confirms the 40 ms output packet adds no first-audio latency."""
    from app.engine.voice.ttfa import get_ttfa_stats
    return get_ttfa_stats().describe()


@router.get("/health/deep")
async def health_deep():
    result = {"status": "ok", "checks": {}}

    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
            ext = await conn.execute(text("SELECT extname FROM pg_extension WHERE extname='vector'"))
            result["checks"]["postgres"] = "ok"
            result["checks"]["pgvector"] = "ok" if ext.first() else "missing"
    except Exception as e:
        result["status"] = "degraded"
        result["checks"]["postgres"] = f"error: {e.__class__.__name__}"

    try:
        r = aioredis.from_url(get_settings().redis_url)
        await r.ping()
        await r.aclose()
        result["checks"]["redis"] = "ok"
    except Exception as e:
        result["status"] = "degraded"
        result["checks"]["redis"] = f"error: {e.__class__.__name__}"

    return result
