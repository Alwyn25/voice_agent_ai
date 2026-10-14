"""Groq adapter. Groq speaks the OpenAI protocol, so this reuses the OpenAI adapter's
request/response handling wholesale and only swaps in Groq's endpoint, key and model ids.

Why a first-class provider rather than just pointing OPENAI_BASE_URL at Groq: it keeps the
OpenAI/OpenRouter settings configured side by side, so switching is one line
(LLM_PROVIDER=groq) instead of rewriting three OPENAI_* vars and back. Groq's free tier has
no per-prompt credit cap — the OpenRouter-free-tier limit that rejects this engine's ~8k
planner prompt.

The planner/responder request "main" or "fast" tier abstractly (never a literal model id --
see openai_provider._resolve_model's docstring for the convention), so complete_json maps
that onto the configured Groq models here — no call site needs to know which provider is
active."""
from app.config import get_settings
from app.providers.llm.base import LLMError
from app.providers.llm.openai_provider import OpenAIProvider


class GroqProvider(OpenAIProvider):
    name = "groq"

    def __init__(self) -> None:
        s = get_settings()
        if not s.groq_api_key:
            raise LLMError("GROQ_API_KEY is not configured")
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=s.groq_api_key, timeout=s.llm_timeout_seconds,
            max_retries=s.llm_max_retries,
            base_url=s.groq_base_url or "https://api.groq.com/openai/v1")

    def _map_model(self, model: str | None) -> str:
        # `model` is an ABSTRACT tier request (None -> main, the literal string "fast" ->
        # fast), never a literal model id. Shared by complete_json and stream_text so
        # neither can drift from the other's mapping.
        s = get_settings()
        return s.groq_model_fast if model == "fast" else s.groq_model_main

    async def complete_json(self, *, system: str, user: str, model: str | None = None,
                            temperature: float = 0.2, max_tokens: int = 900,
                            schema: dict | None = None,
                            schema_name: str = "response"):
        s = get_settings()
        return await super().complete_json(
            system=system, user=user, model=self._map_model(model), temperature=temperature,
            # See groq_min_max_tokens's docstring: both gpt-oss models spend real completion
            # tokens on hidden reasoning before the visible answer, and every call site's
            # max_tokens (400/300) is sized for OpenAI, which has no such overhead.
            max_tokens=max(max_tokens, s.groq_min_max_tokens),
            schema=schema, schema_name=schema_name)

    async def stream_text(self, *, system: str, user: str, model: str | None = None,
                          temperature: float = 0.2, max_tokens: int = 900,
                          schema: dict | None = None,
                          schema_name: str = "response"):
        # Without this override, stream_text fell back to OpenAIProvider's version, which
        # defaults an absent model to OpenAI's own main-tier model and never remaps a
        # present one either — Groq rejects that id with a real 404, the exception is
        # swallowed by responder.py's try/except around _stream_first_sentence, and every
        # turn silently fell back to complete_json (the ~730ms non-streaming cost this
        # provider exists to avoid). Confirmed live before this fix: streaming with an
        # OpenAI model id passed straight through failed with "model gpt-4o-mini does not
        # exist" on Groq's real API.
        s = get_settings()
        async for token in super().stream_text(
                system=system, user=user, model=self._map_model(model),
                temperature=temperature, max_tokens=max(max_tokens, s.groq_min_max_tokens),
                schema=schema, schema_name=schema_name):
            yield token
