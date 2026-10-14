"""Rate limiting for outbound LLM calls — a request that would exceed the configured
requests-per-minute budget waits briefly rather than firing straight at the provider and
coming back as a 429.

Scoped per PROVIDER, globally (one shared budget for every OpenAI call, a separate one
for every Groq call) — not per-tenant. The budget is meant to sit just under what the
provider actually allows; this module protects against provider rate-limit errors, it is
not a spend/traffic governor.

Redis-backed (same client pattern as app.engine.memory.working) so the count stays
correct if this ever runs as more than one backend instance — an in-process counter would
let each replica think it has the full budget to itself. Every Redis operation is
fail-open, same philosophy as working.py: a Redis hiccup must degrade to "rate limiting
briefly does nothing", never to a new reason a turn fails.
"""
import asyncio
import logging
import time

import redis.asyncio as aioredis

from app.config import get_settings
from app.providers.llm.base import LLMError, LLMProvider, LLMResult

log = logging.getLogger(__name__)

# How long a single window's counter is kept before Redis expires it. A little over 60s
# so a key created right at the start of its minute never expires mid-window.
_WINDOW_TTL_SECONDS = 70
_POLL_INTERVAL_SECONDS = 0.25

_client: aioredis.Redis | None = None


def _redis() -> aioredis.Redis:
    global _client
    if _client is None:
        _client = aioredis.from_url(get_settings().redis_url, decode_responses=True)
    return _client


def _window_key(provider_name: str) -> str:
    # Fixed one-minute window. A burst straddling two windows could briefly exceed the
    # cap by a small margin -- accepted tradeoff: real traffic today sits nowhere near
    # these limits, so precision here matters far less than staying cheap and simple.
    minute = int(time.time() // 60)
    return f"llm_ratelimit:{provider_name}:{minute}"


async def acquire(provider_name: str, rpm_limit: int, max_wait_seconds: float) -> None:
    """Block until a slot is available under provider_name's per-minute budget, or until
    max_wait_seconds has passed -- whichever comes first. Never raises: any Redis failure
    or a wait timeout both fall through to letting the request proceed anyway."""
    deadline = time.perf_counter() + max_wait_seconds
    key = _window_key(provider_name)
    try:
        count = await _redis().incr(key)
        if count == 1:
            await _redis().expire(key, _WINDOW_TTL_SECONDS)
    except Exception:  # noqa: BLE001 -- fail open, never block a turn on Redis being down
        return

    while count > rpm_limit:
        if time.perf_counter() >= deadline:
            log.warning("llm rate limiter: gave up waiting for %s after %.1fs (limit=%d)",
                       provider_name, max_wait_seconds, rpm_limit)
            return
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        try:
            # The window may have rolled over while we slept -- re-derive the key each
            # poll rather than reusing the one captured above.
            key = _window_key(provider_name)
            count = int(await _redis().get(key) or 0)
        except Exception:  # noqa: BLE001
            return


class RateLimitedProvider:
    """Wraps a real LLMProvider so every complete_json() call passes through acquire()
    first. Transparent to every call site -- get_llm() returns this instead of the raw
    provider, and nothing else in the engine needs to know it exists.

    `rpm_limits` maps provider name -> its own configured budget (e.g. Groq's real limit is
    far lower than OpenAI/Cerebras's) -- looked up by the WRAPPED provider's live `.name` on
    every call, not fixed at construction, so a fallback chain that advances mid-process
    (FallbackLLMProvider) is rate-limited under whichever entry is actually active right now,
    never under the entry it started on. `default_rpm_limit` covers a name this dict doesn't
    know about, which should not normally happen but must never crash a turn."""
    def __init__(self, provider: LLMProvider, rpm_limits: dict[str, int],
                default_rpm_limit: int):
        self._provider = provider
        self._rpm_limits = rpm_limits
        self._default_rpm_limit = default_rpm_limit

    @property
    def name(self) -> str:
        # Read live rather than snapshotted at construction -- a wrapped provider's
        # identity can change after the fact (e.g. FallbackLLMProvider switching to its
        # fallback mid-process), and this must reflect whichever one is actually serving
        # calls right now.
        return self._provider.name

    def _rpm_limit_for_active_provider(self) -> int:
        return self._rpm_limits.get(self._provider.name, self._default_rpm_limit)

    async def complete_json(self, *, system: str, user: str, model: str | None = None,
                            temperature: float = 0.2, max_tokens: int = 900,
                            schema: dict | None = None,
                            schema_name: str = "response") -> LLMResult:
        await acquire(self._provider.name, self._rpm_limit_for_active_provider(),
                     get_settings().llm_rate_limit_max_wait_seconds)
        return await self._provider.complete_json(
            system=system, user=user, model=model, temperature=temperature,
            max_tokens=max_tokens, schema=schema, schema_name=schema_name)

    async def stream_text(self, *, system: str, user: str, model: str | None = None,
                          temperature: float = 0.2, max_tokens: int = 900,
                          schema: dict | None = None, schema_name: str = "response"):
        """Proxies to the wrapped provider's own stream_text, if it has one -- without
        this, wrapping a streaming-capable provider silently hid that capability behind
        `hasattr(provider, "stream_text")` returning False, and every streaming attempt
        fell back to the non-streaming path with no visible error. Rate-limited the same
        as complete_json: one real outbound request either way."""
        if not hasattr(self._provider, "stream_text"):
            raise LLMError(f"{self._provider.name} does not support streaming")
        await acquire(self._provider.name, self._rpm_limit_for_active_provider(),
                     get_settings().llm_rate_limit_max_wait_seconds)
        async for chunk in self._provider.stream_text(
                system=system, user=user, model=model, temperature=temperature,
                max_tokens=max_tokens, schema=schema, schema_name=schema_name):
            yield chunk
