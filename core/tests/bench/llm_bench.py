"""Live LLM latency measurements using the project's existing provider wrappers.

Only OpenAI and Groq are measured live (they're wired up in app.providers.llm).
All other providers in config.py (Anthropic, Google, Cerebras, Sarvam) use the
published / estimated figures from that file — their SDKs are not installed here.

Runs a realistic planner-sized prompt (kept under 2k tokens to avoid blowing
rate limits on the free Groq tier) and measures:
  - TTFT (time-to-first-token) via stream_text
  - Total generate time

This module does NOT import any core module except the providers. It sets up its
own minimal dotenv load so it can be run standalone without the full FastAPI app.
"""
from __future__ import annotations
import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── Minimal env bootstrap (no app.config import, no DB) ──────────────────────
_CORE_DIR = Path(__file__).resolve().parent.parent.parent  # core/
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(_CORE_DIR / ".env", override=False)
except ImportError:
    pass

# ── Measurement data classes ──────────────────────────────────────────────────

@dataclass
class LLMTimingResult:
    model_name: str
    provider: str
    model_id: str
    ttft_ms: float | None        # None = provider not measured live
    total_ms: float | None
    tokens_out: int
    tokens_per_sec: float | None
    cost_usd_per_call: float     # estimated from config matrix
    source: str                  # "measured" | "published" | "estimated"
    error: str | None = None


# ── Prompt used for timing ────────────────────────────────────────────────────
# Representative planner turn: ~900 input tokens, expect ~80 output tokens (JSON).

_SYSTEM = """You are a voice agent planner for an Indian real-estate company.
Given the customer's statement, decide which tool to call (if any) and return a JSON
object with the keys:
  tool_name (string or null),
  tool_args (object or null),
  intent_summary (string, ≤20 words).
Respond with ONLY valid JSON, no markdown."""

_USER = """Customer said: "Haan, mujhe kal dopahar 2 baje ki appointment chahiye Andheri centre mein,
2BHK flat ke baare mein baat karni hai. Plot number 42 ke baare mein details chahiye."
Session context: customer_id=cust_7821, offered_slots=[2025-01-15T14:00, 2025-01-15T15:00],
centre_id=ctr_andh_01, executive_id=exec_rahul_02.
Available tools: search_slots, book_appointment, inventory_query, knowledge_lookup."""

# ── Live measurement helpers ──────────────────────────────────────────────────

async def _measure_openai(model_id: str) -> tuple[float, float, int]:
    """Returns (ttft_ms, total_ms, tokens_out). Raises on auth/network error."""
    from app.providers.llm.openai_provider import OpenAIProvider

    provider = OpenAIProvider()
    tokens_out = 0
    ttft: float | None = None
    t0 = time.perf_counter()

    async for chunk in provider.stream_text(
        system=_SYSTEM, user=_USER, model=model_id,
        temperature=0.1, max_tokens=200,
    ):
        if ttft is None:
            ttft = (time.perf_counter() - t0) * 1000
        tokens_out += 1

    total_ms = (time.perf_counter() - t0) * 1000
    return ttft or total_ms, total_ms, tokens_out


async def _measure_groq(model_id: str) -> tuple[float, float, int]:
    """Returns (ttft_ms, total_ms, tokens_out). Raises on auth/network error."""
    from app.providers.llm.groq_provider import GroqProvider

    provider = GroqProvider()
    tokens_out = 0
    ttft: float | None = None
    t0 = time.perf_counter()

    async for chunk in provider.stream_text(
        system=_SYSTEM, user=_USER, model=model_id,
        temperature=0.1, max_tokens=200,
    ):
        if ttft is None:
            ttft = (time.perf_counter() - t0) * 1000
        tokens_out += 1

    total_ms = (time.perf_counter() - t0) * 1000
    return ttft or total_ms, total_ms, tokens_out


# ── Cost estimation from token counts ─────────────────────────────────────────

def _estimate_cost(model_in_per_1k: float, model_out_per_1k: float,
                   tokens_in: int = 200, tokens_out: int = 80) -> float:
    return (tokens_in / 1000 * model_in_per_1k) + (tokens_out / 1000 * model_out_per_1k)


# ── Run all measurements ───────────────────────────────────────────────────────

async def run_llm_bench(runs: int = 3) -> list[LLMTimingResult]:
    """Run live measurements for OpenAI and Groq; pull published for others."""
    from tests.bench.config import LLM_MODELS

    results: list[LLMTimingResult] = []

    for m in LLM_MODELS:
        cost = _estimate_cost(m.cost_in_per_1k, m.cost_out_per_1k)

        if m.provider == "openai" and os.getenv("OPENAI_API_KEY"):
            ttfts, totals, toks = [], [], []
            err = None
            for _ in range(runs):
                try:
                    tf, tot, tk = await _measure_openai(m.model_id)
                    ttfts.append(tf); totals.append(tot); toks.append(tk)
                    await asyncio.sleep(0.5)
                except Exception as e:
                    err = str(e)
                    break
            if ttfts:
                avg_ttft = sum(ttfts) / len(ttfts)
                avg_total = sum(totals) / len(totals)
                avg_toks = sum(toks) / len(toks)
                tps = avg_toks / (avg_total / 1000) if avg_total > 0 else 0
                results.append(LLMTimingResult(
                    model_name=m.name, provider=m.provider, model_id=m.model_id,
                    ttft_ms=round(avg_ttft, 1), total_ms=round(avg_total, 1),
                    tokens_out=round(avg_toks), tokens_per_sec=round(tps, 1),
                    cost_usd_per_call=cost, source="measured", error=err,
                ))
            else:
                results.append(_published_result(m, cost, err))

        elif m.provider == "groq" and os.getenv("GROQ_API_KEY"):
            ttfts, totals, toks = [], [], []
            err = None
            for _ in range(runs):
                try:
                    tf, tot, tk = await _measure_groq(m.model_id)
                    ttfts.append(tf); totals.append(tot); toks.append(tk)
                    await asyncio.sleep(1.5)  # respect 30 rpm free tier
                except Exception as e:
                    err = str(e)
                    break
            if ttfts:
                avg_ttft = sum(ttfts) / len(ttfts)
                avg_total = sum(totals) / len(totals)
                avg_toks = sum(toks) / len(toks)
                tps = avg_toks / (avg_total / 1000) if avg_total > 0 else 0
                results.append(LLMTimingResult(
                    model_name=m.name, provider=m.provider, model_id=m.model_id,
                    ttft_ms=round(avg_ttft, 1), total_ms=round(avg_total, 1),
                    tokens_out=round(avg_toks), tokens_per_sec=round(tps, 1),
                    cost_usd_per_call=cost, source="measured", error=err,
                ))
            else:
                results.append(_published_result(m, cost, "GROQ_API_KEY missing"))

        else:
            results.append(_published_result(m, cost))

    return results


def _published_result(m, cost: float, error: str | None = None) -> LLMTimingResult:
    return LLMTimingResult(
        model_name=m.name, provider=m.provider, model_id=m.model_id,
        ttft_ms=float(m.ttft_p50_ms), total_ms=None,
        tokens_out=80,
        tokens_per_sec=float(m.gen_tokens_per_sec),
        cost_usd_per_call=cost,
        source=m.source, error=error,
    )
