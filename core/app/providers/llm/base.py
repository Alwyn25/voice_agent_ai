"""LLM provider interface. Engine code depends on this, never on a vendor SDK."""
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMResult:
    data: dict[str, Any]                    # parsed JSON response
    model: str
    tokens: dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0
    raw_text: str = ""


class LLMError(Exception):
    pass


class LLMProvider(Protocol):
    name: str

    async def complete_json(self, *, system: str, user: str, model: str | None = None,
                            temperature: float = 0.2, max_tokens: int = 900,
                            schema: dict | None = None,
                            schema_name: str = "response") -> LLMResult:
        """Return a parsed JSON object. Must raise LLMError on failure.

        schema — an optional strict JSON Schema (OpenAI Structured Outputs shape: every
        object needs "additionalProperties": false and every property listed in
        "required"). When given, the provider must guarantee the returned JSON conforms —
        not just that it parses — rather than relying on the prompt alone. Callers that
        pass no schema get today's best-effort JSON-mode behaviour."""
        ...
