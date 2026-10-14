"""Live STT latency + accuracy benchmark.

Flow:
  1. Generate 5 test audio clips via ElevenLabs TTS (English + Hinglish sentences
     with known text, so we can compute WER).
  2. Send each clip to Deepgram Nova-3 (batch REST API) — measure latency + WER.
  3. Send each clip to OpenAI Whisper-1 (batch REST API) — measure latency + WER.
  4. All other STT models use published/estimated figures.

Requires: DEEPGRAM_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, OPENAI_API_KEY
"""
from __future__ import annotations
import asyncio, io, os, sys, time
from dataclasses import dataclass, field
from pathlib import Path

_CORE = Path(__file__).resolve().parent.parent.parent
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from tests.bench.live_llm_accuracy import _load_env
_ENV = _load_env()

# ── Test sentences (known text → can compute WER) ─────────────────────────────
# Mix of English, Hinglish, Hindi-inflected English reflecting real estate calls

TEST_SENTENCES: list[tuple[str, str]] = [
    # (label, text_to_speak)
    ("en_booking",
     "Please book an appointment for tomorrow at two PM at the Andheri centre."),
    ("hi_slots",
     "Meri saari appointments dikhao, aur naye slots bhi batao."),
    ("hinglish_price",
     "Two BHK flat ka price kya hai? My budget is fifty lakhs."),
    ("en_cancel",
     "I need to cancel my appointment. Please cancel it as soon as possible."),
    ("hinglish_loan",
     "Home loan ke liye kya documents chahiye? Please explain the process."),
]


# ── WER (word error rate) ─────────────────────────────────────────────────────

def _wer(reference: str, hypothesis: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    n, m = len(ref), len(hyp)
    # Dynamic programming edit distance on word level
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            tmp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[m] / max(n, 1)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class STTCallResult:
    label: str
    reference_text: str
    transcript: str
    wer: float
    latency_ms: float
    audio_duration_ms: float
    error: str | None = None


@dataclass
class STTModelReport:
    model_name: str
    provider: str
    model_id: str
    source: str                 # "measured" | "estimated"
    results: list[STTCallResult] = field(default_factory=list)
    cost_per_min: float = 0.0
    streaming_supported: bool = True
    finalize_latency_ms: int = 0    # from config (estimated for streaming)
    hindi_wer_published: float = 0.0
    notes: str = ""

    @property
    def avg_latency_ms(self) -> float:
        vals = [r.latency_ms for r in self.results if not r.error]
        return sum(vals) / len(vals) if vals else 0

    @property
    def avg_wer(self) -> float:
        vals = [r.wer for r in self.results if not r.error]
        return sum(vals) / len(vals) if vals else 0

    @property
    def p95_latency_ms(self) -> float:
        vals = sorted(r.latency_ms for r in self.results if not r.error)
        if not vals: return 0
        return vals[min(int(len(vals) * 0.95), len(vals) - 1)]


# ── Audio generation via ElevenLabs TTS ───────────────────────────────────────

async def _generate_audio(text: str, voice_id: str, api_key: str) -> bytes | None:
    """Generate MP3 audio bytes for the given text using ElevenLabs."""
    import httpx
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
    }
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code == 200:
                return resp.content
            print(f"    ElevenLabs TTS error {resp.status_code}: {resp.text[:100]}")
    except Exception as e:
        print(f"    ElevenLabs TTS exception: {e}")
    return None


# ── Deepgram batch transcription ─────────────────────────────────────────────

async def _deepgram_transcribe(audio_bytes: bytes, dg_key: str,
                                model: str = "nova-3") -> tuple[str, float]:
    """Returns (transcript, latency_ms). Sends MP3 via Deepgram batch REST."""
    import httpx
    url = f"https://api.deepgram.com/v1/listen?model={model}&language=multi&smart_format=true"
    headers = {
        "Authorization": f"Token {dg_key}",
        "Content-Type": "audio/mpeg",
    }
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, content=audio_bytes, headers=headers)
        latency_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            transcript = (
                data.get("results", {})
                    .get("channels", [{}])[0]
                    .get("alternatives", [{}])[0]
                    .get("transcript", "")
            )
            return transcript, latency_ms
        return f"[HTTP {resp.status_code}]", latency_ms
    except Exception as e:
        return f"[error: {e}]", (time.perf_counter() - t0) * 1000


# ── OpenAI Whisper batch transcription ───────────────────────────────────────

async def _whisper_transcribe(audio_bytes: bytes, openai_key: str) -> tuple[str, float]:
    """Returns (transcript, latency_ms). Uses OpenAI Whisper-1 batch API."""
    import httpx
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {openai_key}"}
    files = {"file": ("audio.mp3", audio_bytes, "audio/mpeg"), "model": (None, "whisper-1")}
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, files=files)
        latency_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            return resp.json().get("text", ""), latency_ms
        return f"[HTTP {resp.status_code}]", latency_ms
    except Exception as e:
        return f"[error: {e}]", (time.perf_counter() - t0) * 1000


# ── Estimate audio duration (rough: ~150 words per minute) ───────────────────

def _estimate_duration_ms(text: str) -> float:
    words = len(text.split())
    return words / 150 * 60 * 1000


# ── Main entry ─────────────────────────────────────────────────────────────────

async def run_stt_bench() -> list[STTModelReport]:
    from tests.bench.config import STT_MODELS

    dg_key = _ENV.get("DEEPGRAM_API_KEY", "")
    openai_key = _ENV.get("OPENAI_API_KEY", "")
    el_key = _ENV.get("ELEVENLABS_API_KEY", "")
    voice_id = _ENV.get("ELEVENLABS_VOICE_ID", "")

    reports: list[STTModelReport] = []

    # ── Generate test audio clips ─────────────────────────────────────────────
    audio_clips: dict[str, bytes | None] = {}
    if el_key and voice_id:
        print("  Generating test audio clips via ElevenLabs TTS...")
        for label, text in TEST_SENTENCES:
            print(f"    [{label}] {text[:55]}...")
            audio = await _generate_audio(text, voice_id, el_key)
            audio_clips[label] = audio
            await asyncio.sleep(0.5)
    else:
        print("  ElevenLabs key/voice_id missing — skipping audio generation")
        audio_clips = {label: None for label, _ in TEST_SENTENCES}

    # ── Deepgram Nova-3 (live) ────────────────────────────────────────────────
    dg_report = STTModelReport(
        model_name="Deepgram Nova-3 (current)", provider="deepgram",
        model_id="nova-3", source="estimated",
        cost_per_min=0.0059, streaming_supported=True,
        finalize_latency_ms=150, hindi_wer_published=0.08,
        notes="Current production model; streaming with SileroVAD + Finalize RPC",
    )

    if dg_key and any(v for v in audio_clips.values()):
        print("\n  Testing Deepgram Nova-3 (batch REST, simulating streaming finalize)...")
        dg_report.source = "measured"
        for label, text in TEST_SENTENCES:
            audio = audio_clips.get(label)
            if audio:
                transcript, latency = await _deepgram_transcribe(audio, dg_key, "nova-3")
                w = _wer(text, transcript)
                dur = _estimate_duration_ms(text)
                dg_report.results.append(STTCallResult(
                    label=label, reference_text=text, transcript=transcript,
                    wer=w, latency_ms=latency, audio_duration_ms=dur,
                ))
                print(f"    [{label}] latency={latency:.0f}ms  WER={w:.1%}  "
                      f"transcript: {transcript[:55]}")
                await asyncio.sleep(0.3)
    else:
        print("  Deepgram: no key or no audio — using estimated figures")
        dg_report.results = []
    reports.append(dg_report)

    # ── Deepgram Nova-2 ───────────────────────────────────────────────────────
    dg2_report = STTModelReport(
        model_name="Deepgram Nova-2", provider="deepgram",
        model_id="nova-2", source="estimated",
        cost_per_min=0.0043, streaming_supported=True,
        finalize_latency_ms=150, hindi_wer_published=0.12,
        notes="Previous gen; 27% cheaper; slightly higher Hindi WER",
    )
    if dg_key and any(v for v in audio_clips.values()):
        print("\n  Testing Deepgram Nova-2 (batch)...")
        dg2_report.source = "measured"
        for label, text in TEST_SENTENCES:
            audio = audio_clips.get(label)
            if audio:
                transcript, latency = await _deepgram_transcribe(audio, dg_key, "nova-2")
                w = _wer(text, transcript)
                dur = _estimate_duration_ms(text)
                dg2_report.results.append(STTCallResult(
                    label=label, reference_text=text, transcript=transcript,
                    wer=w, latency_ms=latency, audio_duration_ms=dur,
                ))
                print(f"    [{label}] latency={latency:.0f}ms  WER={w:.1%}")
                await asyncio.sleep(0.3)
    reports.append(dg2_report)

    # ── OpenAI Whisper-1 ─────────────────────────────────────────────────────
    wh_report = STTModelReport(
        model_name="OpenAI Whisper large-v3", provider="openai",
        model_id="whisper-1", source="estimated",
        cost_per_min=0.0060, streaming_supported=False,
        finalize_latency_ms=3000, hindi_wer_published=0.06,
        notes="Batch only — NOT usable for real-time streaming voice agent; reference baseline",
    )
    if openai_key and any(v for v in audio_clips.values()):
        print("\n  Testing OpenAI Whisper-1 (batch)...")
        wh_report.source = "measured"
        for label, text in TEST_SENTENCES:
            audio = audio_clips.get(label)
            if audio:
                transcript, latency = await _whisper_transcribe(audio, openai_key)
                w = _wer(text, transcript)
                dur = _estimate_duration_ms(text)
                wh_report.results.append(STTCallResult(
                    label=label, reference_text=text, transcript=transcript,
                    wer=w, latency_ms=latency, audio_duration_ms=dur,
                ))
                print(f"    [{label}] latency={latency:.0f}ms  WER={w:.1%}  "
                      f"transcript: {transcript[:55]}")
                await asyncio.sleep(0.5)
    reports.append(wh_report)

    # ── Remaining models — estimated only ─────────────────────────────────────
    from tests.bench.config import STT_MODELS as _all_stt
    estimated_ids = {"nova-3", "nova-2", "whisper-1"}
    for s in _all_stt:
        if s.model_id in estimated_ids:
            continue
        reports.append(STTModelReport(
            model_name=s.name, provider=s.provider, model_id=s.model_id,
            source="estimated", cost_per_min=s.cost_per_min,
            streaming_supported=s.streaming, finalize_latency_ms=s.finalize_latency_ms,
            hindi_wer_published=s.hindi_wer, notes=s.notes, results=[],
        ))

    return reports
