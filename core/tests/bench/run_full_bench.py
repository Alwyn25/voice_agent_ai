"""Full benchmark orchestrator — live tests + comprehensive PDF.

Usage (from core/ directory):
    python -m tests.bench.run_full_bench

Flags:
    --skip-llm     Skip LLM accuracy tests
    --skip-stt     Skip STT latency/WER tests
    --skip-tts     Skip TTS first-audio tests
    --out PATH     Override PDF output path

API keys read from ../.env (relative to core/):
    OPENAI_API_KEY      → LLM (GPT-4o, GPT-4o-mini) + Whisper STT
    DEEPGRAM_API_KEY    → Deepgram Nova-3 STT
    ELEVENLABS_API_KEY  → TTS latency test (turbo_v2_5 vs flash_v2_5)
    ELEVENLABS_VOICE_ID → required for EL tests
"""
from __future__ import annotations
import argparse, asyncio, sys, time
from pathlib import Path

_CORE = Path(__file__).resolve().parent.parent.parent
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))


def _banner(text: str) -> None:
    bar = "═" * (len(text) + 4)
    print(f"\n╔{bar}╗")
    print(f"║  {text}  ║")
    print(f"╚{bar}╝")


def _section(text: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {text}")
    print(f"{'─'*60}")


def _print_llm_summary(reports: list) -> None:
    _section("LLM Summary")
    print(f"  {'Model':<32} {'Src':<10} {'TTFT p50':>9} {'Tool%':>7} {'JSON%':>7} {'Cost/call':>11}")
    print(f"  {'-'*32} {'-'*10} {'-'*9} {'-'*7} {'-'*7} {'-'*11}")
    for r in reports:
        ttft = f"{r.ttft_p50:.0f}ms" if r.ttft_p50 else "n/a"
        tool = f"{r.tool_accuracy:.0%}" if r.results else "n/a"
        json_a = f"{r.json_accuracy:.0%}" if r.results else "n/a"
        cost = f"${r.cost_per_call:.5f}"
        src = "[LIVE]" if r.source == "measured" else "[EST]"
        print(f"  {r.model_name:<32} {src:<10} {ttft:>9} {tool:>7} {json_a:>7} {cost:>11}")


def _print_stt_summary(reports: list) -> None:
    _section("STT Summary")
    print(f"  {'Model':<30} {'Src':<8} {'Batch p50':>10} {'Avg WER':>9} {'$/min':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*9} {'-'*8}")
    for r in reports:
        lats = [x.latency_ms for x in r.results if not x.error]
        lat = f"{sorted(lats)[len(lats)//2]:.0f}ms" if lats else f"~{r.finalize_latency_ms}ms*"
        wers = [x.wer for x in r.results if not x.error]
        wer = f"{sum(wers)/len(wers):.1%}" if wers else f"~{r.hindi_wer_published:.0%}*"
        src = "[LIVE]" if r.source == "measured" else "[EST]"
        print(f"  {r.model_name:<30} {src:<8} {lat:>10} {wer:>9} ${r.cost_per_min:.4f}")


def _print_tts_summary(reports: list) -> None:
    _section("TTS Summary")
    print(f"  {'Model':<35} {'Src':<8} {'1st-Audio p50':>14} {'$/1k chars':>11}")
    print(f"  {'-'*35} {'-'*8} {'-'*14} {'-'*11}")
    for r in reports:
        fa = f"{r.p50_first_audio_ms:.0f}ms" if r.results else f"~{r.first_audio_ms_published}ms*"
        src = "[LIVE]" if r.source == "measured" else "[EST]"
        print(f"  {r.model_name:<35} {src:<8} {fa:>14} ${r.cost_per_1k_chars:.6f}")


async def _main(args: argparse.Namespace) -> None:
    t_start = time.perf_counter()
    _banner("Advora Voice Agent — Full Benchmark Suite")
    print(f"  Working directory: {_CORE}")

    llm_reports = []
    stt_reports = []
    tts_reports = []

    # ── 1. LLM accuracy test ──────────────────────────────────────────────────
    if not args.skip_llm:
        _section("Phase 1: LLM Accuracy & Latency (15 scenarios per model)")
        from tests.bench.live_llm_accuracy import run_llm_accuracy
        llm_reports = await run_llm_accuracy(delay_s=0.4)
        _print_llm_summary(llm_reports)
    else:
        _section("Phase 1: LLM [skipped] — loading estimated figures")
        from tests.bench.live_llm_accuracy import run_llm_accuracy
        llm_reports = await run_llm_accuracy.__wrapped__(delay_s=0) \
            if hasattr(run_llm_accuracy, "__wrapped__") else await run_llm_accuracy(delay_s=0)

    # ── 2. STT latency + WER ─────────────────────────────────────────────────
    if not args.skip_stt:
        _section("Phase 2: STT Latency & Accuracy (5 audio clips)")
        from tests.bench.live_stt_bench import run_stt_bench
        stt_reports = await run_stt_bench()
        _print_stt_summary(stt_reports)
    else:
        _section("Phase 2: STT [skipped] — using config estimates")
        from tests.bench.config import STT_MODELS
        from tests.bench.live_stt_bench import STTModelReport
        stt_reports = [
            STTModelReport(
                model_name=s.name, provider=s.provider, model_id=s.model_id,
                source="estimated", cost_per_min=s.cost_per_min,
                streaming_supported=s.streaming, finalize_latency_ms=s.finalize_latency_ms,
                hindi_wer_published=s.hindi_wer, notes=s.notes,
            )
            for s in STT_MODELS
        ]

    # ── 3. TTS first-audio latency ────────────────────────────────────────────
    if not args.skip_tts:
        _section("Phase 3: TTS First-Audio Latency (5 texts × 2 EL models)")
        from tests.bench.live_tts_bench import run_tts_bench
        tts_reports = await run_tts_bench()
        _print_tts_summary(tts_reports)
    else:
        _section("Phase 3: TTS [skipped] — using config estimates")
        from tests.bench.config import TTS_MODELS
        from tests.bench.live_tts_bench import TTSModelReport
        tts_reports = [
            TTSModelReport(
                model_name=t.name, provider=t.provider, model_id=t.model_id,
                source="estimated", first_audio_ms_published=t.first_audio_ms,
                cost_per_1k_chars=t.cost_per_1k_chars,
                hindi_support=t.hindi_support, notes=t.notes,
            )
            for t in TTS_MODELS
        ]

    # ── 4. Generate PDF ───────────────────────────────────────────────────────
    _section("Phase 4: Generating Comprehensive PDF Report")
    from tests.bench.full_report_v2 import generate_full_pdf
    out = Path(args.out) if args.out else None
    try:
        pdf_path = generate_full_pdf(llm_reports, stt_reports, tts_reports, out)
        elapsed = time.perf_counter() - t_start
        print(f"\n  ✓ PDF saved: {pdf_path}")
        print(f"  Total time: {elapsed:.1f}s")

        # Print quick recommendation
        live_llm = [r for r in llm_reports if r.source == "measured" and r.ttft_p50]
        if live_llm:
            best = min(live_llm, key=lambda r: r.ttft_p50)
            print(f"\n  Best live LLM: {best.model_name}  "
                  f"TTFT={best.ttft_p50:.0f}ms  "
                  f"tool-accuracy={best.tool_accuracy:.0%}")
        live_tts = [r for r in tts_reports if r.source == "measured"]
        if live_tts:
            best_tts = min(live_tts, key=lambda r: r.p50_first_audio_ms)
            print(f"  Best live TTS: {best_tts.model_name}  "
                  f"first-audio={best_tts.p50_first_audio_ms:.0f}ms")
    except Exception as e:
        print(f"\n  PDF generation error: {e}")
        import traceback; traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description="Full voice agent benchmark suite")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM accuracy tests")
    parser.add_argument("--skip-stt", action="store_true", help="Skip STT tests")
    parser.add_argument("--skip-tts", action="store_true", help="Skip TTS tests")
    parser.add_argument("--out", type=str, default=None, help="Override PDF output path")
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
