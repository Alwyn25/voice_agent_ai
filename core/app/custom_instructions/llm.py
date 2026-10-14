"""LLM selection for the custom-instructions COMPILE step, independent of the voice-agent
LLMs.

Why its own selector: the compile step is author-time and low-volume, and an admin may want
a different vendor or a stronger/cheaper model for it than the one driving live calls —
without editing the LLM_PROVIDER (or LLM_FALLBACK_ORDER) that the planner and responder
depend on.

OpenAI, OpenRouter and Groq (and any OpenAI-compatible endpoint) all speak the same
protocol, so one configurable client covers them — differing only by base_url + key. Unlike
the shared GroqProvider, this passes the configured model id through UNCHANGED: it is not on
the planner path, so there is nothing to remap.

Config (app/config.py, all overridable in .env):
  ci_llm_provider  "" -> reuse the voice agent's provider (get_llm); else openai |
                   openrouter | groq | mock
  ci_llm_api_key   key for that provider
  ci_llm_base_url  endpoint override; blank -> the provider's default
  ci_compile_model model id; blank -> openai_model_main
"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.providers.llm.base import LLMError, LLMProvider
from app.providers.llm.openai_provider import OpenAIProvider

# Default endpoints for the OpenAI-compatible vendors we name. "" = the SDK default
# (api.openai.com). Any other compatible endpoint can be reached via ci_llm_base_url.
_DEFAULT_BASE_URLS = {
    "openai": "",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
}


class _ConfiguredOpenAICompatible(OpenAIProvider):
    """OpenAI-protocol client with an explicit key/base_url and NO model remapping."""

    def __init__(self, *, name: str, api_key: str, base_url: str) -> None:
        s = get_settings()
        if not api_key:
            raise LLMError(
                f"custom-instructions LLM: CI_LLM_API_KEY is not set for provider '{name}'")
        self.name = name
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=api_key, timeout=s.llm_timeout_seconds, max_retries=s.llm_max_retries,
            **({"base_url": base_url} if base_url else {}))


@lru_cache
def get_compiler_llm() -> LLMProvider:
    """The provider used for the compile step. Cached per process; mirrors get_llm()."""
    s = get_settings()
    provider = (s.ci_llm_provider or "").strip().lower()

    if not provider:
        # Not configured — reuse the voice agent's provider (current default behaviour).
        from app.providers.llm.router import get_llm
        return get_llm()

    if provider == "mock":
        from app.providers.llm.mock_provider import MockProvider
        return MockProvider()

    if provider not in _DEFAULT_BASE_URLS:
        raise LLMError(
            f"unknown ci_llm_provider '{provider}' "
            f"(expected empty, openai, openrouter, groq or mock)")

    base_url = _clean_base_url(s.ci_llm_base_url) or _DEFAULT_BASE_URLS[provider]
    return _ConfiguredOpenAICompatible(
        name=f"ci:{provider}", api_key=s.ci_llm_api_key, base_url=base_url)


def _clean_base_url(raw: str | None) -> str:
    """A defensive read of CI_LLM_BASE_URL. A common .env mistake is leaving an inline
    comment as the value ("# optional ...") or dropping the scheme — either produces an
    httpx UnsupportedProtocol crash deep in a request. Treat anything that is not a real
    http(s) URL as blank, so we fall back to the provider default instead."""
    val = (raw or "").split("#", 1)[0].strip()
    if val.startswith("http://") or val.startswith("https://"):
        return val
    return ""


def compiler_model() -> str:
    """The model id the compile step should request."""
    s = get_settings()
    return s.ci_compile_model or s.openai_model_main
