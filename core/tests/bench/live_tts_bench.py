"""Live TTS first-audio latency benchmark.

Tests ElevenLabs eleven_turbo_v2_5 (current) vs eleven_flash_v2_5 (recommended)
by streaming the HTTP response and timing the first non-empty audio chunk.

Metrics:
  - connection_ms   : time to first HTTP response byte (TCP + TLS + TTFB)
  - first_audio_ms  : time to first audio data chunk (what the listener hears)
  - total_ms        : time to full audio generation complete
  - bytes_per_sec   : throughput

Other TTS providers (Sarvam, Azure, Google) use published/estimated figures.
"""
from __future__ import annotations
import asyncio, os, sys, time
from dataclasses import dataclass, field
from pathlib import Path

_CORE = Path(__file__).resolve().parent.parent.parent
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from tests.bench.live_llm_accuracy import _load_env
_ENV = _load_env()

# ── Test texts (realistic agent responses — short to match actual agent output) ─

TEST_TEXTS: list[tuple[str, str]] = [
    ("short_confirm",
     "Great, I have booked your appointment for tomorrow at two PM at Andheri centre."),
    ("medium_info",
     "Sure, let me check the available slots for this week. "
     "We have openings on Wednesday at eleven AM and Friday at three PM. "
     "Which one works better for you?"),
    ("long_inventory",
     "We currently have three two-BHK units available. "
     "The first is on the seventh floor at sixty-two lakhs. "
     "The second is a corner unit on the twelfth floor at sixty-eight lakhs. "
     "The third is a ready-to-move unit at fifty-eight lakhs. "
     "All units include covered parking and are RERA registered."),
    ("hinglish_cancel",
     "Aapki appointment successfully cancel ho gayi hai. "
     "Kya aap doosri date ke liye reschedule karna chahenge?"),
    ("short_greeting",
     "Namaste! Main Advora AI ka voice assistant hun. Aap kaise help kar sakta hun?"),
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TTSCallResult:
    label: str
    text_len_chars: int
    connection_ms: float
    first_audio_ms: float
    total_ms: float
    total_bytes: int
    bytes_per_sec: float
    error: str | None = None


@dataclass
class TTSModelReport:
    model_name: str
    provider: str
    model_id: str
    source: str                   # "measured" | "estimated"
    first_audio_ms_published: int
    cost_per_1k_chars: float
    hindi_support: str
    results: list[TTSCallResult] = field(default_factory=list)
    notes: str = ""

    @property
    def avg_first_audio_ms(self) -> float:
        vals = [r.first_audio_ms for r in self.results if not r.error]
        return sum(vals) / len(vals) if vals else float(self.first_audio_ms_published)

    @property
    def p50_first_audio_ms(self) -> float:
        vals = sorted(r.first_audio_ms for r in self.results if not r.error)
        if not vals: return float(self.first_audio_ms_published)
        return vals[len(vals) // 2]

    @property
    def p95_first_audio_ms(self) -> float:
        vals = sorted(r.first_audio_ms for r in self.results if not r.error)
        if not vals: return float(self.first_audio_ms_published) * 2
        return vals[min(int(len(vals) * 0.95), len(vals) - 1)]


# ── ElevenLabs streaming test ─────────────────────────────────────────────────

async def _test_elevenlabs_model(
    model_id: str,
    voice_id: str,
    api_key: str,
    delay_s: float = 0.8,
) -> list[TTSCallResult]:
    import httpx
    results: list[TTSCallResult] = []

    for label, text in TEST_TEXTS:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.45,
                "similarity_boost": 0.8,
                "style": 0.3,
                "use_speaker_boost": True,
            },
        }
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        connection_ms: float = 0
        first_audio_ms: float = 0
        total_bytes = 0
        err = None

        try:
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=30) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    connection_ms = (time.perf_counter() - t0) * 1000
                    if resp.status_code != 200:
                        err = f"HTTP {resp.status_code}"
                        results.append(TTSCallResult(
                            label=label, text_len_chars=len(text),
                            connection_ms=connection_ms, first_audio_ms=0,
                            total_ms=0, total_bytes=0, bytes_per_sec=0, error=err,
                        ))
                        continue
                    async for chunk in resp.aiter_bytes(chunk_size=1024):
                        if chunk:
                            if first_audio_ms == 0:
                                first_audio_ms = (time.perf_counter() - t0) * 1000
                            total_bytes += len(chunk)
            total_ms = (time.perf_counter() - t0) * 1000
            bps = total_bytes / (total_ms / 1000) if total_ms > 0 else 0

            results.append(TTSCallResult(
                label=label, text_len_chars=len(text),
                connection_ms=connection_ms,
                first_audio_ms=first_audio_ms,
                total_ms=total_ms, total_bytes=total_bytes,
                bytes_per_sec=bps,
            ))
            print(f"    [{label}] connect={connection_ms:.0f}ms  "
                  f"first_audio={first_audio_ms:.0f}ms  "
                  f"total={total_ms:.0f}ms  {total_bytes:,}B")

        except Exception as e:
            err = str(e)[:100]
            results.append(TTSCallResult(
                label=label, text_len_chars=len(text),
                connection_ms=0, first_audio_ms=0,
                total_ms=0, total_bytes=0, bytes_per_sec=0, error=err,
            ))
            print(f"    [{label}] ERROR: {err}")

        await asyncio.sleep(delay_s)

    return results


# ── Main entry ─────────────────────────────────────────────────────────────────

async def run_tts_bench() -> list[TTSModelReport]:
    from tests.bench.config import TTS_MODELS

    el_key = _ENV.get("ELEVENLABS_API_KEY", "")
    voice_id = _ENV.get("ELEVENLABS_VOICE_ID", "")

    reports: list[TTSModelReport] = []

    for t in TTS_MODELS:
        report = TTSModelReport(
            model_name=t.name, provider=t.provider, model_id=t.model_id,
            source="estimated", first_audio_ms_published=t.first_audio_ms,
            cost_per_1k_chars=t.cost_per_1k_chars,
            hindi_support=t.hindi_support, notes=t.notes,
        )

        if t.provider == "elevenlabs" and el_key and voice_id:
            print(f"\n  Testing {t.name} ({t.model_id}) — live streaming...")
            try:
                report.results = await _test_elevenlabs_model(
                    t.model_id, voice_id, el_key
                )
                report.source = "measured"
            except Exception as e:
                print(f"  Error testing {t.model_id}: {e}")
        else:
            print(f"  {t.name}: estimated (no key or non-EL provider)")

        reports.append(report)

    return reports
