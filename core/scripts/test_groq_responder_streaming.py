"""P1.03 integration test -- proves the responder path actually streams through Groq.

Without GroqProvider.stream_text, the responder's _stream_first_sentence() call
falls back silently to complete_json() (its caller catches LLMError and retries
non-streamed) -- adding the full non-streaming latency with no visible error.
This test fails loudly instead: it asserts stream_text is the method that ran,
and that it genuinely streamed more than one token rather than returning
everything in a single chunk.

Requires GROQ_API_KEY (real API call, no mocking -- this is what "integration"
means here: prove it against Groq's real endpoint, the same one production
calls).

Usage:
    LLM_PROVIDER=groq python scripts/test_groq_responder_streaming.py
"""
import asyncio
import os
import sys

os.environ.setdefault("LLM_PROVIDER", "groq")

from app.config import get_settings
get_settings.cache_clear()

from app.engine.planner import responder
from app.providers.llm.groq_provider import GroqProvider
from app.providers.llm.router import get_llm


async def main():
    s = get_settings()
    if s.llm_provider != "groq":
        print(f"FAIL: LLM_PROVIDER is {s.llm_provider!r}, expected 'groq' -- "
              f"set LLM_PROVIDER=groq to run this test")
        sys.exit(1)
    if not s.groq_api_key:
        print("FAIL: GROQ_API_KEY is not set -- this test calls Groq's real API")
        sys.exit(1)

    provider = get_llm()
    # get_llm() normally returns a RateLimitedProvider wrapping the real one (see
    # router.py) -- unwrap it the same way to find the actual GroqProvider instance.
    inner = getattr(provider, "_provider", provider)
    assert isinstance(inner, GroqProvider), (
        f"get_llm() resolved to {type(inner).__name__}, expected GroqProvider")

    stream_text_calls = []
    complete_json_calls = []
    orig_stream_text = GroqProvider.stream_text
    orig_complete_json = GroqProvider.complete_json

    async def spy_stream_text(self, **kwargs):
        stream_text_calls.append(kwargs)
        async for token in orig_stream_text(self, **kwargs):
            yield token

    async def spy_complete_json(self, **kwargs):
        complete_json_calls.append(kwargs)
        return await orig_complete_json(self, **kwargs)

    GroqProvider.stream_text = spy_stream_text
    GroqProvider.complete_json = spy_complete_json

    tokens_received = []

    def sentence_callback(sentence: str):
        tokens_received.append(sentence)

    try:
        raw_json, latency_ms = await responder._stream_first_sentence(
            system="You are a helpful assistant. Reply with a short, friendly greeting.",
            user="Say hello and ask how you can help today.",
            model=None,  # main tier -- GroqProvider maps this onto groq_model_main
            temperature=0.2,
            max_tokens=300,
            schema=responder.RESPONDER_SCHEMA,
            schema_name="responder_response",
            sentence_callback=sentence_callback,
        )
    finally:
        GroqProvider.stream_text = orig_stream_text
        GroqProvider.complete_json = orig_complete_json

    print(f"raw_json={raw_json!r}")
    print(f"latency_ms={latency_ms}")
    print(f"stream_text calls: {len(stream_text_calls)}")
    print(f"complete_json calls (should be 0 -- responder falls back to this "
          f"ONLY if streaming raised): {len(complete_json_calls)}")
    print(f"sentence_callback firings: {len(tokens_received)}")

    assert len(stream_text_calls) == 1, (
        f"expected exactly 1 stream_text call, got {len(stream_text_calls)} -- "
        f"the responder path is not routing through GroqProvider.stream_text")
    assert len(complete_json_calls) == 0, (
        f"complete_json was called {len(complete_json_calls)} time(s) -- streaming "
        f"failed and silently fell back to the slow path, exactly the bug this "
        f"ticket exists to catch")
    assert len(raw_json) > 0, "no content was streamed back at all"
    assert len(tokens_received) >= 1, "sentence_callback never fired -- no sentence boundary was found in the stream"

    print("\nPASS: the responder path calls GroqProvider.stream_text (not complete_json), "
          "and it genuinely streamed content back.")
    sys.exit(0)


asyncio.run(main())
