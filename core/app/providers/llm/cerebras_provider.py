"""Cerebras adapter. Cerebras speaks the OpenAI protocol, so this reuses the OpenAI
adapter's request/response handling wholesale and only swaps in Cerebras's endpoint, key
and model ids -- same pattern as GroqProvider.

Strict structured-output mode (response_format={"type": "json_schema", "strict": True,
...}) is confirmed working, live, against this project's real planner schema -- an earlier
concern here (raised before billing was enabled on the test account) turned out unfounded.

What IS real, confirmed live: gpt-oss-120b is a reasoning model that spends completion
tokens on a hidden "thinking" step before writing the visible JSON answer. The planner and
responder call sites budget a fixed max_tokens sized for OpenAI, which has no such overhead
-- see complete_json below for how this adapter compensates without touching those shared
call sites.

The floor (cerebras_min_max_tokens) only applies to confirmed reasoning models. Non-
reasoning models (qwen-3.8-27b, gemma-4-31b) write their answer directly with no hidden
thinking budget, so applying the floor to them wastes token budget and adds latency.

Also note: the model id assumed in early planning discussions (llama-3.3-70b) does not
exist on this account's real model list. The real, confirmed-available models (via a live
GET /v1/models call) are: gpt-oss-120b, gemma-4-31b, qwen-3.8-27b."""
from app.config import get_settings
from app.providers.llm.base import LLMError
from app.providers.llm.openai_provider import OpenAIProvider

# Confirmed reasoning models on Cerebras (spend hidden tokens before visible output).
# The min_max_tokens floor is ONLY applied when the resolved model is in this set.
_REASONING_MODELS: frozenset[str] = frozenset({"gpt-oss-120b", "gpt-oss-20b"})


class CerebrasProvider(OpenAIProvider):
    name = "cerebras"

    def __init__(self) -> None:
        s = get_settings()
        if not s.cerebras_api_key:
            raise LLMError("CEREBRAS_API_KEY is not configured")
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=s.cerebras_api_key, timeout=s.llm_timeout_seconds,
            max_retries=s.llm_max_retries,
            base_url=s.cerebras_base_url or "https://api.cerebras.ai/v1")

    def _map_model(self, model: str | None) -> str:
        # `model` is an ABSTRACT tier request (None -> main, the literal string "fast" ->
        # fast), never a literal model id -- see openai_provider._resolve_model's docstring
        # for the convention every real call site follows. Shared by complete_json and
        # stream_text so neither can drift from the other's mapping -- see stream_text's
        # docstring for what drifting here actually breaks.
        s = get_settings()
        return s.cerebras_model_fast if model == "fast" else s.cerebras_model_main

    def _effective_max_tokens(self, model_id: str, requested: int) -> int:
        s = get_settings()
        if model_id in _REASONING_MODELS:
            return max(requested, s.cerebras_min_max_tokens)
        return requested

    async def complete_json(self, *, system: str, user: str, model: str | None = None,
                            temperature: float = 0.2, max_tokens: int = 900,
                            schema: dict | None = None,
                            schema_name: str = "response"):
        resolved = self._map_model(model)
        return await super().complete_json(
            system=system, user=user, model=resolved, temperature=temperature,
            max_tokens=self._effective_max_tokens(resolved, max_tokens),
            schema=schema, schema_name=schema_name)

    async def stream_text(self, *, system: str, user: str, model: str | None = None,
                          temperature: float = 0.2, max_tokens: int = 900,
                          schema: dict | None = None,
                          schema_name: str = "response"):
        # Without this override, stream_text fell through to OpenAIProvider's version,
        # which never remaps the caller's model id -- Cerebras rejects the raw OpenAI id
        # (e.g. "gpt-4o") with a real 404 ("Model does not exist or you do not have access
        # to it"), the exception is swallowed by responder.py's try/except around
        # _stream_first_sentence, and every turn silently fell back to complete_json (the
        # non-streaming cost the whole responder-streaming feature exists to avoid).
        # Confirmed live before this fix: streaming with the caller's raw model id passed
        # straight through 404'd every single time while LLM_PROVIDER=cerebras. Same class
        # of bug already caught and fixed for GroqProvider -- Cerebras never got the
        # equivalent.
        resolved = self._map_model(model)
        async for token in super().stream_text(
                system=system, user=user, model=resolved,
                temperature=temperature,
                max_tokens=self._effective_max_tokens(resolved, max_tokens),
                schema=schema, schema_name=schema_name):
            yield token
