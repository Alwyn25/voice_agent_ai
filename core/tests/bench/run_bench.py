"""Benchmark orchestrator.

Usage (from the core/ directory):
    python -m tests.bench.run_bench

Flags:
    --no-live       Skip live LLM API calls (use published figures only)
    --runs N        How many live API call repetitions per model (default 3)
    --out PATH      Override PDF output path

The script:
  1. Runs live LLM measurements (OpenAI + Groq) if API keys are present
  2. Builds full-stack TTFA projections for all model combos
  3. Generates docs/Voice Agent/Voice_Agent_Latency_Report.pdf
  4. Prints a compact console summary
"""
from __future__ import annotations
import argparse
import asyncio
import sys
import time
from pathlib import Path

# Ensure core/ is on path so `from app.*` and `from tests.*` both resolve
_CORE = Path(__file__).resolve().parent.parent.parent
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))


def _print_section(title: str) -> None:
    bar = "─" * (len(title) + 4)
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def _print_tool_summary() -> None:
    _print_section("Tool Latency Summary (p50, all 17 tools)")
    from tests.bench.tool_steps import summary_table
    rows = summary_table()
    header = f"{'Tool':<26} {'Cat':<14} {'p50':>7} {'p95':>7} {'Timeout':>9} {'Headroom':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        flag = " ⚠" if r["headroom_ms"] < 200 else ""
        print(
            f"{r['tool']:<26} {r['category']:<14} "
            f"{r['p50_ms']:>6.0f}ms {r['p95_ms']:>6.0f}ms "
            f"{r['timeout_ms']:>8}ms {r['headroom_ms']:>9.0f}ms{flag}"
        )


def _print_llm_summary(llm_results: list) -> None:
    _print_section("LLM Model Comparison")
    header = f"{'Model':<30} {'Src':<11} {'TTFT p50':>10} {'Tok/s':>7} {'Cost/call':>12}"
    print(header)
    print("-" * len(header))
    for r in llm_results:
        ttft = f"{r.ttft_ms:.0f}ms" if r.ttft_ms else "—"
        tps = f"{r.tokens_per_sec:.0f}" if r.tokens_per_sec else "—"
        cost = f"${r.cost_usd_per_call:.6f}"
        err = f"  [{r.error[:40]}]" if r.error else ""
        print(f"{r.model_name:<30} {r.source:<11} {ttft:>10} {tps:>7} {cost:>12}{err}")


def _print_pipeline_summary(baseline_ms: float) -> None:
    _print_section("Top 10 Fastest Combos (no-tool turn, current VAD)")
    from tests.bench.pipeline_bench import run_pipeline_bench, top_n_fastest
    results = run_pipeline_bench(use_semantic_endpointing=False, include_tool_turns=False)
    top = top_n_fastest(results, n=10)
    header = f"{'STT':<26} {'LLM':<24} {'TTS':<26} {'TTFA':>8} {'vs Now':>9}"
    print(header)
    print("-" * len(header))
    for r in top:
        diff = r.total_ttfa_ms - baseline_ms
        sign = "-" if diff < 0 else "+"
        print(
            f"{r.stt_name.replace(' (current)',''):<26} "
            f"{r.llm_name:<24} "
            f"{r.tts_name.replace(' (current)',''):<26} "
            f"{r.total_ttfa_ms:>7.0f}ms {sign}{abs(diff):>7.0f}ms"
        )

    _print_section("Top 5 Fastest with Semantic Endpointing")
    sem = run_pipeline_bench(use_semantic_endpointing=True, include_tool_turns=False)
    sem_top = top_n_fastest(sem, n=5)
    for r in sem_top:
        diff = r.total_ttfa_ms - baseline_ms
        sign = "-" if diff < 0 else "+"
        print(
            f"{r.stt_name.replace(' (current)',''):<26} "
            f"{r.llm_name:<24} "
            f"{r.tts_name.replace(' (current)',''):<26} "
            f"{r.total_ttfa_ms:>7.0f}ms {sign}{abs(diff):>7.0f}ms"
        )


def _print_recommendation() -> None:
    _print_section("Recommendation")
    from tests.bench.pipeline_bench import recommendation
    rec = recommendation()
    print(f"  STT:           {rec['stt']}")
    print(f"  LLM Planner:   {rec['llm_planner']}")
    print(f"  LLM Responder: {rec['llm_responder']}")
    print(f"  TTS:           {rec['tts']}")
    print(f"  VAD:           {rec['vad']}")
    print(f"")
    print(f"  Projected TTFA (no-tool):   {rec['projected_ttfa_no_tool_ms']} ms")
    print(f"  Projected TTFA (with tool): {rec['projected_ttfa_with_tool_ms']} ms")
    print(f"  Current TTFA:               {rec['current_ttfa_ms']} ms")
    print(f"  Saving:                     −{rec['improvement_ms']} ms "
          f"({rec['improvement_ms'] / rec['current_ttfa_ms'] * 100:.0f}%)")
    print(f"")
    print(f"  {rec['rationale']}")


async def _main(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   Advora Voice Agent — Latency Benchmark Suite      ║")
    print("╚══════════════════════════════════════════════════════╝")

    # 1. Tool breakdown (always, no API keys needed)
    _print_tool_summary()

    # 2. LLM bench (live or published)
    if args.no_live:
        print("\n[skipping live LLM calls — using published figures]")
        from tests.bench.config import LLM_MODELS
        from tests.bench.llm_bench import _published_result, _estimate_cost
        llm_results = [_published_result(m, _estimate_cost(m.cost_in_per_1k, m.cost_out_per_1k))
                       for m in LLM_MODELS]
    else:
        print("\n[running live LLM measurements...]")
        from tests.bench.llm_bench import run_llm_bench
        llm_results = await run_llm_bench(runs=args.runs)
    _print_llm_summary(llm_results)

    # 3. Pipeline projections
    from tests.bench.pipeline_bench import current_baseline_breakdown
    baseline = current_baseline_breakdown()
    _print_pipeline_summary(baseline.total_ttfa_ms)

    # 4. Recommendation
    _print_recommendation()

    # 5. Generate PDF
    _print_section("Generating PDF Report")
    from tests.bench.pdf_report import generate_pdf
    out_path = Path(args.out) if args.out else None
    try:
        pdf_path = generate_pdf(llm_results=llm_results, output_path=out_path)
        elapsed = time.perf_counter() - t0
        print(f"\n  Report saved to: {pdf_path}")
        print(f"  Total bench time: {elapsed:.1f}s")
    except Exception as e:
        print(f"\n  PDF generation failed: {e}")
        import traceback; traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice agent latency benchmark")
    parser.add_argument("--no-live", action="store_true",
                        help="Skip live API calls; use published figures only")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of live API call repetitions per model (default 3)")
    parser.add_argument("--out", type=str, default=None,
                        help="Override PDF output path")
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
