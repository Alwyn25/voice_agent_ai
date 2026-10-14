"""Unit tests for the sticky, ordered N-provider LLM fallback chain (FallbackLLMProvider)
and the config-driven chain-building in router.py.

What must hold, and is tested below:
  * first entry succeeds          -> no other entry is ever touched.
  * first entry fails, NOT a credit error (timeout, bad request, ...) -> raised through
    unchanged; the chain does NOT advance.
  * first entry fails WITH a credit error (402) -> this call is served by the next entry,
    and the chain advances.
  * once advanced -> every later call goes straight to the current entry; earlier entries
    are never called again (this is the whole point -- no per-call re-probe latency).
  * a 3-entry chain can advance twice (cerebras -> openai -> groq), in order, never
    skipping ahead or going back.
  * .name reflects whichever entry is actually serving calls right now.
  * same behaviour for stream_text as for complete_json.
  * router.py builds the chain from LLM_FALLBACK_ORDER and skips an entry whose provider
    fails to construct (e.g. a missing API key), rather than crashing.

Run: cd core && python -m pytest tests/test_llm_fallback.py -q
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers.llm.base import LLMError, LLMResult
from app.providers.llm.fallback_provider import FallbackLLMProvider


class _StatusError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class _FakeProvider:
    def __init__(self, name, *, fail_with: Exception | None = None):
        self.name = name
        self._fail_with = fail_with
        self.complete_json_calls = 0
        self.stream_text_calls = 0

    async def complete_json(self, **kwargs) -> LLMResult:
        self.complete_json_calls += 1
        if self._fail_with is not None:
            raise LLMError(f"{self._fail_with.__class__.__name__}: {self._fail_with}") \
                from self._fail_with
        return LLMResult(data={"ok": True}, model="fake-model")

    async def stream_text(self, **kwargs):
        self.stream_text_calls += 1
        if self._fail_with is not None:
            raise LLMError(f"{self._fail_with.__class__.__name__}: {self._fail_with}") \
                from self._fail_with
        for tok in ["hello", " world"]:
            yield tok


def _call_kwargs():
    return dict(system="sys", user="user text")


def _chain(*providers):
    return [(p.name, p) for p in providers]


def test_first_entry_success_never_touches_others():
    async def run():
        cerebras = _FakeProvider("cerebras")
        openai = _FakeProvider("openai")
        wrapper = FallbackLLMProvider(chain=_chain(cerebras, openai))

        result = await wrapper.complete_json(**_call_kwargs())

        assert result.data == {"ok": True}
        assert cerebras.complete_json_calls == 1
        assert openai.complete_json_calls == 0
        assert wrapper.name == "cerebras"
    asyncio.run(run())


def test_non_credit_error_raises_and_does_not_advance():
    async def run():
        cerebras = _FakeProvider("cerebras", fail_with=_StatusError(500, "server error"))
        openai = _FakeProvider("openai")
        wrapper = FallbackLLMProvider(chain=_chain(cerebras, openai))

        raised = False
        try:
            await wrapper.complete_json(**_call_kwargs())
        except LLMError:
            raised = True

        assert raised
        assert openai.complete_json_calls == 0
        assert wrapper.name == "cerebras"
    asyncio.run(run())


def test_credit_error_advances_to_next_entry():
    async def run():
        cerebras = _FakeProvider("cerebras", fail_with=_StatusError(402, "Payment required"))
        openai = _FakeProvider("openai")
        wrapper = FallbackLLMProvider(chain=_chain(cerebras, openai))

        result = await wrapper.complete_json(**_call_kwargs())

        assert result.data == {"ok": True}
        assert cerebras.complete_json_calls == 1
        assert openai.complete_json_calls == 1
        assert wrapper.name == "openai"
    asyncio.run(run())


def test_credit_error_detected_from_message_text_when_no_status_code():
    async def run():
        cerebras = _FakeProvider("cerebras", fail_with=RuntimeError(
            "Payment required to access this resource. Visit your billing tab."))
        openai = _FakeProvider("openai")
        wrapper = FallbackLLMProvider(chain=_chain(cerebras, openai))

        result = await wrapper.complete_json(**_call_kwargs())

        assert result.data == {"ok": True}
        assert wrapper.name == "openai"
    asyncio.run(run())


def test_once_advanced_earlier_entry_is_never_called_again():
    async def run():
        cerebras = _FakeProvider("cerebras", fail_with=_StatusError(402, "Payment required"))
        openai = _FakeProvider("openai")
        wrapper = FallbackLLMProvider(chain=_chain(cerebras, openai))

        await wrapper.complete_json(**_call_kwargs())
        assert cerebras.complete_json_calls == 1

        # Three more calls after advancing -- cerebras must stay at 1 call forever, since
        # re-probing it every turn is exactly the latency cost this feature avoids.
        for _ in range(3):
            await wrapper.complete_json(**_call_kwargs())
        assert cerebras.complete_json_calls == 1
        assert openai.complete_json_calls == 4
    asyncio.run(run())


def test_three_entry_chain_advances_in_order():
    async def run():
        cerebras = _FakeProvider("cerebras", fail_with=_StatusError(402, "Payment required"))
        openai = _FakeProvider("openai", fail_with=_StatusError(402, "Payment required"))
        groq = _FakeProvider("groq")
        wrapper = FallbackLLMProvider(chain=_chain(cerebras, openai, groq))

        result = await wrapper.complete_json(**_call_kwargs())

        assert result.data == {"ok": True}
        assert cerebras.complete_json_calls == 1
        assert openai.complete_json_calls == 1
        assert groq.complete_json_calls == 1
        assert wrapper.name == "groq"

        # A second call must go straight to groq -- neither earlier entry touched again.
        await wrapper.complete_json(**_call_kwargs())
        assert cerebras.complete_json_calls == 1
        assert openai.complete_json_calls == 1
        assert groq.complete_json_calls == 2
    asyncio.run(run())


def test_last_entry_credit_error_raises_through():
    async def run():
        cerebras = _FakeProvider("cerebras", fail_with=_StatusError(402, "Payment required"))
        openai = _FakeProvider("openai", fail_with=_StatusError(402, "Payment required"))
        wrapper = FallbackLLMProvider(chain=_chain(cerebras, openai))

        raised = False
        try:
            await wrapper.complete_json(**_call_kwargs())
        except LLMError:
            raised = True

        # Both entries are out of credit -- nothing left to advance to, so the error from
        # the last entry must propagate rather than looping or silently returning nothing.
        assert raised
        assert wrapper.name == "openai"
    asyncio.run(run())


def test_stream_text_success_never_touches_other_entries():
    async def run():
        cerebras = _FakeProvider("cerebras")
        openai = _FakeProvider("openai")
        wrapper = FallbackLLMProvider(chain=_chain(cerebras, openai))

        tokens = [tok async for tok in wrapper.stream_text(**_call_kwargs())]

        assert tokens == ["hello", " world"]
        assert cerebras.stream_text_calls == 1
        assert openai.stream_text_calls == 0
    asyncio.run(run())


def test_stream_text_credit_error_advances():
    async def run():
        cerebras = _FakeProvider("cerebras", fail_with=_StatusError(402, "Payment required"))
        openai = _FakeProvider("openai")
        wrapper = FallbackLLMProvider(chain=_chain(cerebras, openai))

        tokens = [tok async for tok in wrapper.stream_text(**_call_kwargs())]

        assert tokens == ["hello", " world"]
        assert wrapper.name == "openai"

        # Second call: cerebras must not be touched again.
        tokens2 = [tok async for tok in wrapper.stream_text(**_call_kwargs())]
        assert tokens2 == ["hello", " world"]
        assert cerebras.stream_text_calls == 1
        assert openai.stream_text_calls == 2
    asyncio.run(run())


def test_stream_text_non_credit_error_raises_and_does_not_advance():
    async def run():
        cerebras = _FakeProvider("cerebras", fail_with=_StatusError(500, "server error"))
        openai = _FakeProvider("openai")
        wrapper = FallbackLLMProvider(chain=_chain(cerebras, openai))

        raised = False
        try:
            async for _ in wrapper.stream_text(**_call_kwargs()):
                pass
        except LLMError:
            raised = True

        assert raised
        assert wrapper.name == "cerebras"
        assert openai.stream_text_calls == 0
    asyncio.run(run())


def test_rate_limit_error_skips_to_next_entry_for_this_call():
    async def run():
        cerebras = _FakeProvider("cerebras", fail_with=_StatusError(429, "too many requests"))
        openai = _FakeProvider("openai")
        wrapper = FallbackLLMProvider(chain=_chain(cerebras, openai))

        result = await wrapper.complete_json(**_call_kwargs())

        assert result.data == {"ok": True}
        assert cerebras.complete_json_calls == 1
        assert openai.complete_json_calls == 1
        # Unlike credit exhaustion, a 429 never changes the sticky baseline -- cerebras is
        # still the named primary, just skipped this one call while it cools down.
        assert wrapper.name == "cerebras"
    asyncio.run(run())


def test_rate_limit_cooldown_is_not_permanent():
    async def run():
        cerebras = _FakeProvider("cerebras", fail_with=_StatusError(429, "too many requests"))
        openai = _FakeProvider("openai")
        wrapper = FallbackLLMProvider(chain=_chain(cerebras, openai))

        await wrapper.complete_json(**_call_kwargs())
        assert cerebras.complete_json_calls == 1
        assert openai.complete_json_calls == 1

        # Still within the cooldown window -- cerebras is skipped again without being
        # called, same as the real deployment staying on its fallback for a bounded time.
        await wrapper.complete_json(**_call_kwargs())
        assert cerebras.complete_json_calls == 1
        assert openai.complete_json_calls == 2

        # Once the cooldown has passed, the primary is tried again automatically -- no
        # permanent demotion the way credit exhaustion causes.
        wrapper._rate_limited_until[0] = 0.0
        cerebras._fail_with = None
        result = await wrapper.complete_json(**_call_kwargs())
        assert result.data == {"ok": True}
        assert cerebras.complete_json_calls == 2
        assert openai.complete_json_calls == 2
        assert wrapper.name == "cerebras"
    asyncio.run(run())


def test_rate_limit_on_every_entry_raises_through_rather_than_looping():
    async def run():
        cerebras = _FakeProvider("cerebras", fail_with=_StatusError(429, "too many requests"))
        openai = _FakeProvider("openai", fail_with=_StatusError(429, "too many requests"))
        wrapper = FallbackLLMProvider(chain=_chain(cerebras, openai))

        raised = False
        try:
            await wrapper.complete_json(**_call_kwargs())
        except LLMError:
            raised = True

        assert raised
        # Each entry tried at most once this call -- no infinite loop when everyone is busy.
        assert cerebras.complete_json_calls == 1
        assert openai.complete_json_calls == 1
    asyncio.run(run())


def test_stream_text_rate_limit_skips_to_next_entry():
    async def run():
        cerebras = _FakeProvider("cerebras", fail_with=_StatusError(429, "too many requests"))
        openai = _FakeProvider("openai")
        wrapper = FallbackLLMProvider(chain=_chain(cerebras, openai))

        tokens = [tok async for tok in wrapper.stream_text(**_call_kwargs())]

        assert tokens == ["hello", " world"]
        assert wrapper.name == "cerebras"  # not a permanent switch, unlike credit exhaustion
        assert openai.stream_text_calls == 1
    asyncio.run(run())


def test_rate_limit_skip_then_credit_error_does_not_retry_the_dead_entry():
    # THE BUG: chain A -> B. A 429s (cooldown, not sticky), so this call serves B. B then
    # 402s. The old `_advance` moved `self._index` to `self._index + 1` (A's position + 1 =
    # B) instead of past the entry that actually failed (B) -- so B, confirmed out of
    # credit, stayed "current" and every later call retried it first.
    async def run():
        a = _FakeProvider("a", fail_with=_StatusError(429, "too many requests"))
        b = _FakeProvider("b", fail_with=_StatusError(402, "Payment required"))
        wrapper = FallbackLLMProvider(chain=_chain(a, b))

        raised = False
        try:
            await wrapper.complete_json(**_call_kwargs())
        except LLMError:
            raised = True
        # Only 2 entries exist, both tried once and both failed this call -- nothing left
        # to serve it, so the error must propagate.
        assert raised
        assert a.complete_json_calls == 1
        assert b.complete_json_calls == 1

        # A must NOT be retried again: it was only cooling down, but B's confirmed 402
        # moved the sticky position past both of them.
        try:
            await wrapper.complete_json(**_call_kwargs())
        except LLMError:
            pass
        assert a.complete_json_calls == 1
        assert b.complete_json_calls == 2
    asyncio.run(run())


def test_rate_limit_skip_then_credit_error_reaches_a_third_entry():
    # Same scenario as above, but with a third, healthy entry -- confirms the sticky
    # position lands PAST the entry that actually 402'd (b), not merely one past the old
    # `self._index` (which was still a, since a was only ever cooldown-skipped, not advanced).
    async def run():
        a = _FakeProvider("a", fail_with=_StatusError(429, "too many requests"))
        b = _FakeProvider("b", fail_with=_StatusError(402, "Payment required"))
        c = _FakeProvider("c")
        wrapper = FallbackLLMProvider(chain=_chain(a, b, c))

        result = await wrapper.complete_json(**_call_kwargs())
        assert result.data == {"ok": True}
        assert a.complete_json_calls == 1
        assert b.complete_json_calls == 1
        assert c.complete_json_calls == 1
        assert wrapper.name == "c"

        # A second call must go straight to c -- neither a nor b touched again.
        await wrapper.complete_json(**_call_kwargs())
        assert a.complete_json_calls == 1
        assert b.complete_json_calls == 1
        assert c.complete_json_calls == 2
    asyncio.run(run())


def test_rate_limit_never_wraps_back_to_an_earlier_entry():
    # THE BUG: chain A -> B. A 402s so the sticky index moves to B (1). B then 429s --
    # `_next_available_index` used to wrap with `% n`, landing back on A (confirmed dead)
    # instead of staying on B or raising. A must never be tried again.
    async def run():
        a = _FakeProvider("a", fail_with=_StatusError(402, "Payment required"))
        b = _FakeProvider("b", fail_with=_StatusError(429, "too many requests"))
        wrapper = FallbackLLMProvider(chain=_chain(a, b))

        raised = False
        try:
            await wrapper.complete_json(**_call_kwargs())
        except LLMError:
            raised = True

        assert raised
        assert a.complete_json_calls == 1       # the initial attempt that triggered _advance
        # b is tried twice within this one call: once after the advance, then again as the
        # "everyone is cooling down" fallback (_next_available_index has nowhere else to go
        # once it refuses to wrap back to a) -- the second attempt is what raises.
        assert b.complete_json_calls == 2
        assert wrapper.name == "b"

        # Next call: b is still cooling down, but there is nowhere else to go -- must try
        # b again (the only entry at or after the sticky index), never fall back to a.
        b._fail_with = None
        result = await wrapper.complete_json(**_call_kwargs())
        assert result.data == {"ok": True}
        assert a.complete_json_calls == 1
    asyncio.run(run())


def test_429_substring_in_an_unrelated_message_does_not_trigger_cooldown():
    # THE BUG: a plain "429" substring match fires on any error that happens to contain
    # those digits (a request id, an unrelated count) -- not just an actual rate limit.
    async def run():
        cerebras = _FakeProvider("cerebras", fail_with=RuntimeError(
            "request req_4291ab timed out"))
        openai = _FakeProvider("openai")
        wrapper = FallbackLLMProvider(chain=_chain(cerebras, openai))

        raised = False
        try:
            await wrapper.complete_json(**_call_kwargs())
        except LLMError:
            raised = True

        # Not a rate limit and not a credit error -- raised straight through, no failover.
        assert raised
        assert openai.complete_json_calls == 0
        assert wrapper.name == "cerebras"
        assert 0 not in wrapper._rate_limited_until
    asyncio.run(run())


def test_router_builds_chain_from_env_order_and_skips_unconfigured():
    """Config-driven: LLM_FALLBACK_ORDER lists 3 names; groq's key isn't set, so it must be
    skipped (with a warning) rather than crashing, leaving a working 2-entry chain."""
    import os
    os.environ["LLM_PROVIDER"] = "cerebras"
    os.environ["LLM_FALLBACK_ENABLED"] = "true"
    os.environ["LLM_FALLBACK_ORDER"] = "cerebras,openai,groq"
    os.environ["CEREBRAS_API_KEY"] = "csk-test-fake-key-for-construction-only"
    os.environ["OPENAI_API_KEY"] = "sk-test-fake-key-for-construction-only"
    os.environ.pop("GROQ_API_KEY", None)
    os.environ["LLM_RATE_LIMIT_ENABLED"] = "false"  # avoid needing a real Redis in this test

    from app.config import get_settings
    get_settings.cache_clear()
    from app.providers.llm.router import get_llm
    get_llm.cache_clear()

    llm = get_llm()
    assert llm.name == "cerebras"
    assert isinstance(llm, FallbackLLMProvider)
    names = [n for n, _ in llm._chain]
    assert names == ["cerebras", "openai"], (
        f"expected groq to be skipped (no GROQ_API_KEY), got chain {names!r}")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001 -- surface unexpected errors as failures too
            failures += 1
            print(f"ERROR {t.__name__}: {e.__class__.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
