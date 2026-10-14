"""Selects the active LLM provider.

Two modes, both driven entirely by config -- this module has no hardcoded notion of which
provider is "primary" or what a fallback order should be:

  - LLM_FALLBACK_ENABLED=false (default): use LLM_PROVIDER alone. Single provider, no
    chain -- exactly the original, pre-fallback behaviour.
  - LLM_FALLBACK_ENABLED=true: build an ORDERED CHAIN from LLM_FALLBACK_ORDER, a plain
    comma-separated env value (e.g. "cerebras,openai,groq"). Changing priority -- say,
    moving Groq to the front once that account is funded -- is a one-line env edit, never
    a code change. See fallback_provider.py for the sticky, credit-exhaustion-triggered
    switch logic between the entries in that chain.

A provider that fails to construct (its API key isn't set yet) is skipped with a warning
rather than crashing startup -- this lets LLM_FALLBACK_ORDER list a provider you plan to
enable later (e.g. Groq before its key exists) without breaking the app today.
"""
import logging
from functools import lru_cache

from app.config import get_settings
from app.providers.llm.base import LLMError, LLMProvider

logger = logging.getLogger(__name__)


def _build_named_provider(name: str, s) -> tuple[LLMProvider, int] | None:
    """Construct the named provider and its configured rpm limit, or None if it can't be
    built right now (unknown name, or a real LLMError such as a missing API key)."""
    try:
        if name == "cerebras":
            from app.providers.llm.cerebras_provider import CerebrasProvider
            return CerebrasProvider(), s.openai_rpm_limit
        if name == "openai":
            from app.providers.llm.openai_provider import OpenAIProvider
            return OpenAIProvider(), s.openai_rpm_limit
        if name == "groq":
            from app.providers.llm.groq_provider import GroqProvider
            return GroqProvider(), s.groq_rpm_limit
        if name == "mock":
            from app.providers.llm.mock_provider import MockProvider
            return MockProvider(), s.openai_rpm_limit
    except LLMError as e:
        logger.warning("llm-chain: skipping provider %r -- %s", name, e)
        return None
    logger.warning("llm-chain: unknown provider name %r, skipping", name)
    return None


@lru_cache
def get_llm() -> LLMProvider:
    s = get_settings()

    if not s.llm_fallback_enabled:
        built = _build_named_provider(s.llm_provider, s)
        if built is None:
            raise LLMError(f"LLM_PROVIDER={s.llm_provider!r} could not be configured")
        provider, rpm_limit = built
        rpm_limits = {provider.name: rpm_limit}
    else:
        names = [n.strip() for n in s.llm_fallback_order.split(",") if n.strip()]
        chain: list[tuple[str, LLMProvider]] = []
        rpm_limits = {}
        for name in names:
            built = _build_named_provider(name, s)
            if built is None:
                continue
            provider_obj, rpm_limit = built
            chain.append((name, provider_obj))
            rpm_limits[name] = rpm_limit
        if not chain:
            raise LLMError(f"LLM_FALLBACK_ORDER={s.llm_fallback_order!r} has no usable "
                           f"providers -- check each one's API key is set")
        if len(chain) == 1:
            # Only one entry actually constructed (the rest are unconfigured/unknown) --
            # nothing to fall back to, so skip the wrapper and use it directly.
            provider = chain[0][1]
        else:
            from app.providers.llm.fallback_provider import FallbackLLMProvider
            provider = FallbackLLMProvider(chain=chain)

    if not s.llm_rate_limit_enabled:
        return provider
    # Transparent to every call site -- understanding.py, planner.py, responder.py and
    # memory/pipeline.py all just call get_llm().complete_json(...) same as before.
    from app.providers.llm.rate_limiter import RateLimitedProvider
    return RateLimitedProvider(provider, rpm_limits=rpm_limits,
                               default_rpm_limit=s.openai_rpm_limit)
