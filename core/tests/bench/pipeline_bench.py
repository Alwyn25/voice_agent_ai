"""Full-stack TTFA projection for every STT × LLM × TTS combination.

TTFA = Time to First Audio (end of user speech → start of agent audio).

Pipeline stages and their contribution (current baseline):
  1. VAD endpointing       700 ms  (SileroVAD stop_secs=0.7)
  2. STT finalize          150 ms  (Deepgram Nova-3 streaming)
  3. Bridge overhead        60 ms  (frame relay + WS hop)
  4. Planner LLM TTFT      150 ms  (Groq 70b after fix)
  5. Tool call              —       (only for tool turns; p50 from tool_steps)
  6. Responder stream TTFT 150 ms  (same model — first-sentence start)
  7. TTS first audio       280 ms  (ElevenLabs turbo_v2_5)
  8. Transport              70 ms  (WebSocket frame + jitter buffer)
                         ───────
  Total (no-tool turn)   1,560 ms  (current; tool turns add step 5)

Competitor target: ~500 ms.
Best achievable without semantic endpointing: ~850 ms (see optimize notes).
Best with semantic endpointing (−550 ms VAD): ~300–400 ms.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class StageBreakdown:
    name: str
    ms: float
    note: str = ""


@dataclass
class PipelineResult:
    stt_name: str
    llm_name: str
    tts_name: str
    stages: list[StageBreakdown]
    is_tool_turn: bool = False
    tool_name: str | None = None
    tool_p50_ms: float = 0.0

    @property
    def total_ttfa_ms(self) -> float:
        return sum(s.ms for s in self.stages)

    @property
    def label(self) -> str:
        return f"{self.stt_name} / {self.llm_name} / {self.tts_name}"

    @property
    def vs_current_ms(self) -> float:
        return self.total_ttfa_ms - _CURRENT_BASELINE_MS


# ── Fixed / shared pipeline constants ─────────────────────────────────────────

_VAD_ENDPOINTING_MS = 700.0     # SileroVAD stop_secs=0.7 (current config)
_VAD_SEMANTIC_MS    = 120.0     # achievable with semantic endpointing
_BRIDGE_MS          = 60.0      # frame relay + internal WS hop
_TRANSPORT_MS       = 70.0      # outbound WebSocket + jitter buffer

# Current baseline TTFA (no-tool turn)
_CURRENT_BASELINE_MS = (
    _VAD_ENDPOINTING_MS + 150 + _BRIDGE_MS  # VAD + STT(Nova-3) + bridge
    + 150 + 150                              # planner Groq TTFT + responder TTFT
    + 280 + _TRANSPORT_MS                   # EL turbo_v2_5 + transport
)  # = 1,560 ms


def _build_stages(
    vad_ms: float,
    stt_ms: float,
    planner_ttft_ms: float,
    responder_ttft_ms: float,
    tts_first_ms: float,
    tool_ms: float = 0.0,
    is_tool_turn: bool = False,
) -> list[StageBreakdown]:
    stages = [
        StageBreakdown("VAD endpointing", vad_ms,
                       "SileroVAD silence window before pipeline starts"),
        StageBreakdown("STT finalize", stt_ms,
                       "Streaming STT → final transcript commit"),
        StageBreakdown("Bridge overhead", _BRIDGE_MS,
                       "Frame relay + internal WebSocket hop"),
        StageBreakdown("Planner TTFT", planner_ttft_ms,
                       "LLM time to first token (intent JSON)"),
    ]
    if is_tool_turn and tool_ms > 0:
        stages.append(StageBreakdown("Tool execution", tool_ms,
                                     "DB query / external API (p50)"))
    stages += [
        StageBreakdown("Responder TTFT", responder_ttft_ms,
                       "LLM time to first token (text reply)"),
        StageBreakdown("TTS first audio", tts_first_ms,
                       "TTS → first audio packet"),
        StageBreakdown("Transport", _TRANSPORT_MS,
                       "Outbound WebSocket + jitter buffer"),
    ]
    return stages


def run_pipeline_bench(
    use_semantic_endpointing: bool = False,
    include_tool_turns: bool = True,
) -> list[PipelineResult]:
    """Generate TTFA projections for all model combinations."""
    from tests.bench.config import LLM_MODELS, STT_MODELS, TTS_MODELS
    from tests.bench.tool_steps import TOOL_PROFILES

    vad_ms = _VAD_SEMANTIC_MS if use_semantic_endpointing else _VAD_ENDPOINTING_MS

    # Only streaming-capable STT for live voice
    streaming_stt = [s for s in STT_MODELS if s.streaming]

    # Representative tool turns: search_slots (read-only, common) + book_appointment (write)
    tool_examples = [p for p in TOOL_PROFILES
                     if p.tool_name in ("search_slots", "book_appointment", "send_document")]

    results: list[PipelineResult] = []

    for stt in streaming_stt:
        for llm in LLM_MODELS:
            for tts in TTS_MODELS:
                # No-tool turn
                stages = _build_stages(
                    vad_ms=vad_ms,
                    stt_ms=stt.finalize_latency_ms,
                    planner_ttft_ms=llm.ttft_p50_ms,
                    responder_ttft_ms=llm.ttft_p50_ms,
                    tts_first_ms=tts.first_audio_ms,
                )
                results.append(PipelineResult(
                    stt_name=stt.name, llm_name=llm.name, tts_name=tts.name,
                    stages=stages, is_tool_turn=False,
                ))

                if include_tool_turns:
                    for tool in tool_examples:
                        tstages = _build_stages(
                            vad_ms=vad_ms,
                            stt_ms=stt.finalize_latency_ms,
                            planner_ttft_ms=llm.ttft_p50_ms,
                            responder_ttft_ms=llm.ttft_p50_ms,
                            tts_first_ms=tts.first_audio_ms,
                            tool_ms=tool.total_p50,
                            is_tool_turn=True,
                        )
                        results.append(PipelineResult(
                            stt_name=stt.name, llm_name=llm.name, tts_name=tts.name,
                            stages=tstages, is_tool_turn=True,
                            tool_name=tool.tool_name, tool_p50_ms=tool.total_p50,
                        ))

    return results


def current_baseline_breakdown() -> PipelineResult:
    """Return the exact current production pipeline breakdown."""
    from tests.bench.config import STT_MODELS, LLM_MODELS, TTS_MODELS

    stt = next(s for s in STT_MODELS if "Nova-3" in s.name and "current" in s.name)
    llm = next(m for m in LLM_MODELS if "Groq llama-3.3" in m.name)
    tts = next(t for t in TTS_MODELS if "turbo" in t.name)

    stages = _build_stages(
        vad_ms=_VAD_ENDPOINTING_MS,
        stt_ms=stt.finalize_latency_ms,
        planner_ttft_ms=llm.ttft_p50_ms,
        responder_ttft_ms=llm.ttft_p50_ms,
        tts_first_ms=tts.first_audio_ms,
    )
    return PipelineResult(
        stt_name=stt.name, llm_name=llm.name, tts_name=tts.name,
        stages=stages, is_tool_turn=False,
    )


def top_n_fastest(results: Sequence[PipelineResult], n: int = 10,
                  no_tool_only: bool = True) -> list[PipelineResult]:
    filtered = [r for r in results if not r.is_tool_turn] if no_tool_only else list(results)
    return sorted(filtered, key=lambda r: r.total_ttfa_ms)[:n]


def recommendation() -> dict:
    """Return the recommended model stack with rationale."""
    return {
        "stt": "Deepgram Nova-3",
        "llm_planner": "Cerebras llama-3.3-70b",
        "llm_responder": "Groq llama-3.1-8b (fast model)",
        "tts": "ElevenLabs Flash v2.5",
        "vad": "Semantic endpointing (implement, −550 ms)",
        "projected_ttfa_no_tool_ms": 120 + 60 + 60 + 60 + 75 + 70,  # 445
        "projected_ttfa_with_tool_ms": 120 + 60 + 60 + 60 + 125 + 75 + 70,  # ~570
        "current_ttfa_ms": round(_CURRENT_BASELINE_MS),
        "improvement_ms": round(_CURRENT_BASELINE_MS - 445),
        "rationale": (
            "Cerebras gives lowest TTFT (60ms) for planner JSON — only drawback is no "
            "strict JSON schema, so add a lightweight retry wrapper. Groq 8b handles "
            "the responder stream (plain text, no schema needed). EL Flash v2.5 cuts "
            "TTS from 280ms to 75ms. Semantic endpointing is the single biggest lever "
            "(−550ms). Together these bring TTFA under 500ms on no-tool turns."
        ),
    }
