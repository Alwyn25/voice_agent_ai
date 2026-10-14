"""OpenAI adapter. Plain JSON mode guarantees parseable output, nothing more; a caller
that supplies a schema gets Structured Outputs, which guarantees the shape too — the
model cannot omit a required field or invent a value outside an enum, whereas JSON mode
alone let exactly that happen (a "confirm" action with no tool named, silently dropping
a field the prompt required)."""
import json
import time

from app.config import get_settings
from app.providers.llm.base import LLMError, LLMResult


def _resolve_model(model: str | None, *, main: str, fast: str) -> str:
    """`model` is an ABSTRACT tier request, never a literal id, from every real call site
    in this codebase: None means "main tier", the literal string "fast" means "fast tier".
    Any other value is a genuine explicit override (e.g. custom_instructions/llm.py's
    _ConfiguredOpenAICompatible, which intentionally passes a real model id straight
    through with no remapping) and is used as-is."""
    if model is None:
        return main
    if model == "fast":
        return fast
    return model


class OpenAIProvider:
    name = "openai"

    def __init__(self) -> None:
        s = get_settings()
        if not s.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not configured")
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=s.openai_api_key, timeout=s.llm_timeout_seconds,
                                   max_retries=s.llm_max_retries,
                                   **({"base_url": s.openai_base_url}
                                      if s.openai_base_url else {}))

    async def complete_json(self, *, system: str, user: str, model: str | None = None,
                            temperature: float = 0.2, max_tokens: int = 900,
                            schema: dict | None = None,
                            schema_name: str = "response") -> LLMResult:
        s = get_settings()
        model = _resolve_model(model, main=s.openai_model_main, fast=s.openai_model_fast)
        started = time.perf_counter()
        response_format = (
            {"type": "json_schema",
             "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
            if schema is not None else {"type": "json_object"})
        try:
            resp = await self._client.chat.completions.create(
                model=model,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
        except Exception as e:
            raise LLMError(f"{e.__class__.__name__}: {e}") from e

        text = (resp.choices[0].message.content or "").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"Model returned invalid JSON: {e}") from e

        usage = resp.usage
        # cached_tokens tells us how much of the prompt the provider served from its prefix
        # cache — the single number that PROVES whether the deliberate prompt-prefix ordering
        # (prompts.build_system_prompt) is actually paying off on the clock. Without it, a
        # caching regression (a stable block drifting below a varying one) is invisible until
        # latency creeps back up. Absent on providers/models that do not report it -> 0.
        cached = getattr(getattr(usage, "prompt_tokens_details", None),
                         "cached_tokens", 0) or 0
        return LLMResult(
            data=data, model=model, raw_text=text,
            latency_ms=int((time.perf_counter() - started) * 1000),
            tokens={"prompt": getattr(usage, "prompt_tokens", 0),
                    "completion": getattr(usage, "completion_tokens", 0),
                    "cached": cached,
                    "total": getattr(usage, "total_tokens", 0)} if usage else {})

    async def stream_text(self, *, system: str, user: str, model: str | None = None,
                          temperature: float = 0.2, max_tokens: int = 900,
                          schema: dict | None = None,
                          schema_name: str = "response"):
        """Async generator yielding raw text tokens from a streaming JSON completion.

        The caller must buffer the tokens and parse the final JSON themselves.
        Use complete_json() for the standard non-streaming path.
        """
        s = get_settings()
        model = _resolve_model(model, main=s.openai_model_main, fast=s.openai_model_fast)
        response_format = (
            {"type": "json_schema",
             "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
            if schema is not None else {"type": "json_object"})
        try:
            stream = await self._client.chat.completions.create(
                model=model, response_format=response_format,
                temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                stream=True)
        except Exception as e:
            raise LLMError(f"{e.__class__.__name__}: {e}") from e
        try:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            raise LLMError(f"{e.__class__.__name__}: {e}") from e
        finally:
            await stream.close()