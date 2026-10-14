"""Model matrix and cost/latency reference data.

Data sources:
- LLM TTFT: Groq/OpenAI status pages + internal measurements
- STT latency: Deepgram, Sarvam, Google published specs + India deployment benchmarks
- TTS first-audio: ElevenLabs docs (flash vs turbo) + Azure/Sarvam benchmarks
- Costs: published pricing pages (Sep 2026)
- Models tagged [PDF] are documented in Advora_VoiceToText_India_TechRef_v1.pdf
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class LLMModel:
    name: str
    provider: str
    model_id: str
    ttft_p50_ms: int            # time to first token, p50
    ttft_p95_ms: int
    gen_tokens_per_sec: int     # streaming output speed
    cost_in_per_1k: float       # USD per 1k input tokens
    cost_out_per_1k: float
    structured_output: bool     # supports json_schema strict mode
    rpm_free: int               # free-tier rpm (0 = no free tier)
    notes: str
    source: str = "published"   # "measured" | "published" | "estimated"


@dataclass
class STTModel:
    name: str
    provider: str
    model_id: str
    finalize_latency_ms: int    # streaming: time from end of speech to final transcript
    hindi_wer: float            # word error rate on Hindi/Hinglish (lower = better)
    hinglish_support: str       # "native" | "good" | "partial" | "none"
    cost_per_min: float         # USD
    streaming: bool
    notes: str


@dataclass
class TTSModel:
    name: str
    provider: str
    model_id: str
    first_audio_ms: int         # time from text-in to first audio packet
    hindi_support: str          # "native" | "good" | "partial"
    cost_per_1k_chars: float
    notes: str


# ── LLM models ────────────────────────────────────────────────────────────────
# Planner prompt is ~8k tokens in, ~400 tokens out (JSON schema response)
# Responder prompt is ~3k tokens in, ~120 tokens out (streaming first sentence)

LLM_MODELS: list[LLMModel] = [
    LLMModel(
        name="GPT-4o",
        provider="openai",
        model_id="gpt-4o",
        ttft_p50_ms=420, ttft_p95_ms=900,
        gen_tokens_per_sec=60,
        cost_in_per_1k=0.0025, cost_out_per_1k=0.010,
        structured_output=True, rpm_free=0,
        notes="Current baseline; best reasoning; highest cost",
        source="measured",
    ),
    LLMModel(
        name="GPT-4o-mini",
        provider="openai",
        model_id="gpt-4o-mini",
        ttft_p50_ms=220, ttft_p95_ms=500,
        gen_tokens_per_sec=100,
        cost_in_per_1k=0.000150, cost_out_per_1k=0.000600,
        structured_output=True, rpm_free=0,
        notes="16× cheaper than GPT-4o; misses complex multi-step reasoning",
        source="measured",
    ),
    LLMModel(
        name="Groq llama-3.3-70b",
        provider="groq",
        model_id="llama-3.3-70b-versatile",
        ttft_p50_ms=150, ttft_p95_ms=350,
        gen_tokens_per_sec=280,
        cost_in_per_1k=0.000590, cost_out_per_1k=0.000790,
        structured_output=True, rpm_free=30,
        notes="Current after switch; 2.8× faster TTFT than GPT-4o; 30 rpm free",
        source="measured",
    ),
    LLMModel(
        name="Groq llama-3.1-8b",
        provider="groq",
        model_id="llama-3.1-8b-instant",
        ttft_p50_ms=80, ttft_p95_ms=180,
        gen_tokens_per_sec=750,
        cost_in_per_1k=0.000050, cost_out_per_1k=0.000080,
        structured_output=False, rpm_free=30,
        notes="Fastest available; no strict JSON schema; responder-only use",
        source="measured",
    ),
    LLMModel(
        name="Claude Haiku 4.5",
        provider="anthropic",
        model_id="claude-haiku-4-5-20251001",
        ttft_p50_ms=200, ttft_p95_ms=420,
        gen_tokens_per_sec=180,
        cost_in_per_1k=0.000250, cost_out_per_1k=0.001250,
        structured_output=True, rpm_free=0,
        notes="[PDF] Excellent for routing + short responses; strong tool-use",
        source="published",
    ),
    LLMModel(
        name="Claude Sonnet 5",
        provider="anthropic",
        model_id="claude-sonnet-5",
        ttft_p50_ms=300, ttft_p95_ms=650,
        gen_tokens_per_sec=120,
        cost_in_per_1k=0.003000, cost_out_per_1k=0.015000,
        structured_output=True, rpm_free=0,
        notes="[PDF] Best reasoning after GPT-4o; higher cost than Groq 70b",
        source="published",
    ),
    LLMModel(
        name="Gemini 1.5 Flash",
        provider="google",
        model_id="gemini-1.5-flash",
        ttft_p50_ms=180, ttft_p95_ms=400,
        gen_tokens_per_sec=200,
        cost_in_per_1k=0.000075, cost_out_per_1k=0.000300,
        structured_output=True, rpm_free=15,
        notes="[PDF] Very cheap; India data residency; 1M context",
        source="published",
    ),
    LLMModel(
        name="Cerebras llama-3.3-70b",
        provider="cerebras",
        model_id="llama-3.3-70b",
        ttft_p50_ms=60, ttft_p95_ms=120,
        gen_tokens_per_sec=2100,
        cost_in_per_1k=0.000600, cost_out_per_1k=0.000600,
        structured_output=False, rpm_free=0,
        notes="[PDF] Fastest TTFT of any provider; no strict JSON schema currently",
        source="published",
    ),
    LLMModel(
        name="Sarvam Saaras 2B",
        provider="sarvam",
        model_id="sarvam-2b-v0.5",
        ttft_p50_ms=120, ttft_p95_ms=280,
        gen_tokens_per_sec=400,
        cost_in_per_1k=0.000080, cost_out_per_1k=0.000080,
        structured_output=False, rpm_free=0,
        notes="[PDF] India-made; Hindi-native; limited English reasoning quality",
        source="published",
    ),
]

# ── STT models ────────────────────────────────────────────────────────────────

STT_MODELS: list[STTModel] = [
    STTModel(
        name="Deepgram Nova-3 (current)",
        provider="deepgram",
        model_id="nova-3",
        finalize_latency_ms=150,
        hindi_wer=0.08, hinglish_support="native",
        cost_per_min=0.0059,
        streaming=True,
        notes="Best Hinglish; code-switch 'multi' mode; model = nova-3",
    ),
    STTModel(
        name="Deepgram Nova-2",
        provider="deepgram",
        model_id="nova-2",
        finalize_latency_ms=150,
        hindi_wer=0.12, hinglish_support="good",
        cost_per_min=0.0043,
        streaming=True,
        notes="[PDF] Older gen; 27% cheaper; slightly higher Hindi WER",
    ),
    STTModel(
        name="Sarvam Bulbul-v2",
        provider="sarvam",
        model_id="sarvam-2b",
        finalize_latency_ms=200,
        hindi_wer=0.04, hinglish_support="native",
        cost_per_min=0.0020,
        streaming=True,
        notes="[PDF] Best pure Hindi WER; limited English; India-native",
    ),
    STTModel(
        name="Google Chirp v2",
        provider="google",
        model_id="chirp_2",
        finalize_latency_ms=250,
        hindi_wer=0.09, hinglish_support="good",
        cost_per_min=0.0160,
        streaming=True,
        notes="[PDF] Solid Hindi; most expensive; GCP infrastructure",
    ),
    STTModel(
        name="AssemblyAI Universal-2",
        provider="assemblyai",
        model_id="universal-2",
        finalize_latency_ms=220,
        hindi_wer=0.14, hinglish_support="partial",
        cost_per_min=0.0100,
        streaming=True,
        notes="[PDF] Strong English; Hindi support added late; partial Hinglish",
    ),
    STTModel(
        name="OpenAI Whisper large-v3",
        provider="openai",
        model_id="whisper-1",
        finalize_latency_ms=3000,
        hindi_wer=0.06, hinglish_support="good",
        cost_per_min=0.0060,
        streaming=False,
        notes="[PDF] Reference baseline; NOT streaming; batch/offline only",
    ),
]

# ── TTS models ─────────────────────────────────────────────────────────────────

TTS_MODELS: list[TTSModel] = [
    TTSModel(
        name="ElevenLabs turbo_v2_5 (current)",
        provider="elevenlabs",
        model_id="eleven_turbo_v2_5",
        first_audio_ms=280,
        hindi_support="partial",
        cost_per_1k_chars=0.000330,
        notes="Current; confirmed working; partial Hindi",
    ),
    TTSModel(
        name="ElevenLabs Flash v2.5",
        provider="elevenlabs",
        model_id="eleven_flash_v2_5",
        first_audio_ms=75,
        hindi_support="partial",
        cost_per_1k_chars=0.000330,
        notes="[PDF] 3.7× faster first audio; same price; same API",
    ),
    TTSModel(
        name="Sarvam Bulbul TTS",
        provider="sarvam",
        model_id="bulbul:v1",
        first_audio_ms=180,
        hindi_support="native",
        cost_per_1k_chars=0.000120,
        notes="[PDF] Best Hindi voice quality; 2.75× cheaper; India-hosted",
    ),
    TTSModel(
        name="Azure hi-IN-SwaraNeural",
        provider="azure",
        model_id="hi-IN-SwaraNeural",
        first_audio_ms=150,
        hindi_support="native",
        cost_per_1k_chars=0.000160,
        notes="[PDF] Native Hindi; Azure infra; good for pure Hindi calls",
    ),
    TTSModel(
        name="Google WaveNet hi-IN",
        provider="google",
        model_id="hi-IN-Wavenet-A",
        first_audio_ms=200,
        hindi_support="native",
        cost_per_1k_chars=0.000160,
        notes="[PDF] Good Hindi; GCP; slower than EL Flash",
    ),
]

# ── Tool timeout_ms from definitions (source of truth) ────────────────────────
TOOL_TIMEOUTS: dict[str, int] = {
    "search_slots": 4000,
    "book_appointment": 5000,
    "reschedule_appointment": 5000,
    "cancel_appointment": 4000,
    "locate_centre": 3000,
    "create_ticket": 4000,
    "create_customer": 5000,
    "update_customer_profile": 3000,
    "record_customer_fact": 3000,
    "schedule_callback": 3000,
    "record_do_not_contact": 3000,
    "verify_identity": 3000,
    "my_appointments": 3000,
    "my_tickets": 3000,
    "inventory_query": 4000,
    "knowledge_lookup": 10000,
    "send_document": 12000,
}
