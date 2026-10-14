"""Turn a raw usage count (tokens / characters / seconds) into a dollar figure.

WHY COMPUTED, NOT STORED. Provider pricing changes over time, and a dollar amount written
to a DecisionTrace row the day it was billed at one rate would silently go stale the day
the rate changes -- every historical row would then be quietly wrong with no record of
what changed or when. Instead, only the raw usage (tokens/characters/seconds -- facts that
never change after the turn happened) is persisted; cost is computed from CURRENT_RATES at
read time, in the analytics endpoints and the turn's own debug payload. Re-pricing history
if a plan changes is then just editing the table below, not a backfill migration.

RATES BELOW ARE APPROXIMATE PUBLIC LIST PRICES, not pulled from this project's actual
provider invoices/plan tier. Good enough for relative comparison between models/providers
(which is the actual ask -- "which one costs more"), not guaranteed to match a real bill to
the cent. Update PER_1M_TOKENS / TTS_PER_CHARACTER / STT_PER_MINUTE if the numbers drift or
the account's real plan pricing becomes known.
"""
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

# $ per 1,000,000 tokens, (input, output). Embeddings have no separate output cost, so
# their "output" side is 0 -- total tokens still price correctly since embeddings never
# populate a completion count anyway.
PER_1M_TOKENS: dict[str, tuple[Decimal, Decimal]] = {
    "gpt-4o": (Decimal("2.50"), Decimal("10.00")),
    "gpt-4o-mini": (Decimal("0.15"), Decimal("0.60")),
    "text-embedding-3-small": (Decimal("0.02"), Decimal("0")),
    # Groq's OpenAI-protocol models (see app/providers/llm/groq_provider.py) -- llama-3.3-70b
    # is the one actually configured for this project as of this session.
    "llama-3.3-70b-versatile": (Decimal("0.59"), Decimal("0.79")),
    # Groq's actually-configured main/fast tier (config.py's groq_model_main/fast) --
    # missing here means Groq-served traffic (the last fallback in the chain) prices at
    # $0.00. Rates verified directly from console.groq.com/docs, not guessed.
    "openai/gpt-oss-120b": (Decimal("0.15"), Decimal("0.60")),
    "openai/gpt-oss-20b": (Decimal("0.075"), Decimal("0.30")),
    # Cerebras (see app/providers/llm/cerebras_provider.py) -- gpt-oss-120b is this project's
    # configured CEREBRAS_MODEL_MAIN/FAST and its real primary LLM provider. Missing here
    # meant every Cerebras-served turn priced at exactly $0.00 (PER_1M_TOKENS.get() returns
    # None -> llm_cost_usd treats it as an unknown model, by design -- see the self-test
    # below), silently understating cost across 100% of live traffic whenever Cerebras is
    # up. Rate from cerebras.ai's own published pricing.
    "gpt-oss-120b": (Decimal("0.25"), Decimal("0.69")),
    # OpenRouter (see app/config.py's openai_base_url) prefixes model ids with the
    # upstream provider, e.g. "openai/gpt-4o-mini" -- this project ran on OpenRouter for
    # part of its history before switching to OpenAI directly. Same underlying model, same
    # rate (OpenRouter passes provider pricing through), aliased so a before/after provider
    # switch is comparable in /analytics/costs instead of the older rows silently pricing
    # at zero.
    "openai/gpt-4o": (Decimal("2.50"), Decimal("10.00")),
    "openai/gpt-4o-mini": (Decimal("0.15"), Decimal("0.60")),
}

# ElevenLabs bills per character of input text (confirmed via a real API error this
# session: "5 credits required" for an 11-character request). $ per credit varies by
# subscription tier; this uses a representative pay-as-you-go rate.
TTS_PER_CHARACTER = Decimal("0.00018")

# Deepgram bills per minute of audio processed (nova-3, pay-as-you-go).
STT_PER_MINUTE = Decimal("0.0043")

_CENTS_6DP = Decimal("0.000001")


def _round(value: Decimal) -> Decimal:
    return value.quantize(_CENTS_6DP, rounding=ROUND_HALF_UP)


def llm_cost_usd(model: str | None, tokens: dict | None) -> Decimal:
    """tokens is the {"prompt", "completion", "total"} shape LLMResult.tokens already uses
    (app/providers/llm/base.py). Unknown model or missing/zero tokens -> 0, never a guess."""
    if not model or not tokens:
        return Decimal("0")
    rate = PER_1M_TOKENS.get(model)
    if rate is None:
        return Decimal("0")
    input_rate, output_rate = rate
    prompt = Decimal(str(tokens.get("prompt") or 0))
    completion = Decimal(str(tokens.get("completion") or 0))
    return _round((prompt * input_rate + completion * output_rate) / Decimal("1000000"))


def tts_cost_usd(characters: int | None) -> Decimal:
    if not characters:
        return Decimal("0")
    return _round(Decimal(str(characters)) * TTS_PER_CHARACTER)


def stt_cost_usd(seconds: float | None) -> Decimal:
    if not seconds:
        return Decimal("0")
    minutes = Decimal(str(seconds)) / Decimal("60")
    return _round(minutes * STT_PER_MINUTE)


@dataclass
class TurnCost:
    """One turn's usage priced out -- the shape both the live debug payload
    (turn_runner.py) and the /analytics/costs endpoint build from."""
    llm_usd: Decimal
    tts_usd: Decimal
    stt_usd: Decimal

    @property
    def total_usd(self) -> Decimal:
        return self.llm_usd + self.tts_usd + self.stt_usd

    def as_dict(self) -> dict[str, str]:
        return {"llm": str(self.llm_usd), "tts": str(self.tts_usd),
                "stt": str(self.stt_usd), "total": str(self.total_usd)}


def turn_cost(*, tokens_by_stage: dict[str, dict] | None, model_used: str | None,
             tts_characters: int | None, stt_seconds: float | None) -> TurnCost:
    """The turn-level total: sum every LLM call this turn made (understanding + planner +
    responder, each may have run against a different model in principle, but today all
    share `model_used` -- see turn_runner.py's DecisionTrace construction) plus TTS/STT."""
    tokens_by_stage = tokens_by_stage or {}
    llm_total = sum((llm_cost_usd(model_used, stage_tokens)
                     for stage_tokens in tokens_by_stage.values()), Decimal("0"))
    return TurnCost(llm_usd=llm_total, tts_usd=tts_cost_usd(tts_characters),
                    stt_usd=stt_cost_usd(stt_seconds))


# --- self-check -------------------------------------------------------------------------
if __name__ == "__main__":
    failures = 0

    def check(label, cond):
        global failures
        print(("ok  " if cond else "FAIL") + "  " + label)
        if not cond:
            failures += 1

    check("gpt-4o-mini cost scales with tokens",
          llm_cost_usd("gpt-4o-mini", {"prompt": 1000, "completion": 500}) > 0)
    check("gpt-4o costs more than gpt-4o-mini for the same tokens",
          llm_cost_usd("gpt-4o", {"prompt": 1000, "completion": 500})
          > llm_cost_usd("gpt-4o-mini", {"prompt": 1000, "completion": 500}))
    check("unknown model prices as zero, not a guess",
          llm_cost_usd("some-future-model", {"prompt": 1000, "completion": 500}) == 0)
    check("no tokens prices as zero", llm_cost_usd("gpt-4o", None) == 0)
    check("no tokens prices as zero (empty dict)", llm_cost_usd("gpt-4o", {}) == 0)
    check("tts cost scales with characters", tts_cost_usd(1000) > tts_cost_usd(10))
    check("zero characters costs zero", tts_cost_usd(0) == 0)
    check("stt cost scales with duration", stt_cost_usd(120) > stt_cost_usd(10))
    check("zero seconds costs zero", stt_cost_usd(0) == 0)

    tc = turn_cost(
        tokens_by_stage={"understanding": {"prompt": 200, "completion": 50},
                         "planner": {"prompt": 3000, "completion": 150},
                         "responder": {"prompt": 1200, "completion": 80}},
        model_used="gpt-4o-mini", tts_characters=180, stt_seconds=4.2)
    check("turn_cost totals all three legs", tc.total_usd == tc.llm_usd + tc.tts_usd + tc.stt_usd)
    check("turn_cost.as_dict has the four expected keys",
          set(tc.as_dict().keys()) == {"llm", "tts", "stt", "total"})

    print("\nPRICING SELFTEST:", "PASS" if not failures else f"{failures} FAILURE(S)")
    import sys
    sys.exit(1 if failures else 0)
