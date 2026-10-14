#!/usr/bin/env python3
"""Verify the filler-audio cache (P1.10) — matching contract, catalog integrity, chunking,
and (optionally) a real end-to-end ElevenLabs build.

    docker compose exec core python scripts/check_filler_audio.py
    docker compose exec core python scripts/check_filler_audio.py --build   # real synthesis

Two layers.

OFFLINE (default, no network, no credentials) is the regression gate for the part most
likely to silently break: the exact-match contract between the words the engine SPEAKS and
the words pre-generated at startup. If those drift, every cache hit quietly becomes a miss
and the latency win evaporates with no error anywhere. This layer proves:

  * every filler turn_runner.py streams -- each intent (read-only / generic) in each
    language (English / Hindi / Hinglish) -- is in the cached catalog and normalise-matches,
  * no two catalog phrases collide under normalisation (which would serve one line's audio
    for another),
  * cached PCM chunks reassemble exactly and never split a 16-bit sample,
  * build() populates the cache, get() hits every cached phrase and misses a normal reply,
    and the skip-conditions (disabled / wrong provider / missing creds) leave it empty.

The offline build test injects synthetic PCM in place of the ElevenLabs call, so it needs
no key and makes no request.

--build runs the REAL startup synthesis against ElevenLabs using the process settings and
prints the resulting cache description — the same thing GET /health/filler-audio returns —
so a real voice/key/model problem is caught here rather than on the first live call.

Stdlib + the app package only, like the other check_*.py scripts.
"""
import argparse
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

# The app package is importable as `app.*` when run from core/ (as the other scripts are).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.engine.voice import filler_audio as fa


def _fake_settings(**overrides):
    """A minimal settings stand-in with the attributes build()/synthesis read."""
    base = dict(
        filler_audio_cache_enabled=True,
        tts_provider="elevenlabs",
        elevenlabs_api_key="test-key",
        elevenlabs_voice_id="test-voice",
        elevenlabs_model="eleven_turbo_v2_5",
        elevenlabs_stability=0.45,
        elevenlabs_similarity_boost=0.8,
        elevenlabs_style=0.3,
        elevenlabs_use_speaker_boost=True,
        filler_audio_build_timeout_ms=8000,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _Check:
    """Tiny pass/fail accumulator so one failing assertion doesn't hide the rest."""

    def __init__(self):
        self.failed = 0
        self.passed = 0

    def ok(self, cond: bool, label: str):
        if cond:
            self.passed += 1
            print(f"  PASS  {label}")
        else:
            self.failed += 1
            print(f"  FAIL  {label}")


async def _offline(chk: _Check) -> None:
    print("catalog integrity")
    catalog = fa.CACHED_FILLER_PHRASES       # tool-preamble fillers (also the de-dup set)
    all_catalog = fa.ALL_CACHED_PHRASES      # everything pre-generated: fillers + common
    read_en = fa.filler_phrase(fa.FILLER_INTENT_READ_ONLY, fa.LANG_ENGLISH)
    gen_en = fa.filler_phrase(fa.FILLER_INTENT_GENERIC, fa.LANG_ENGLISH)
    chk.ok(read_en in catalog, "read-only English filler is in the cached catalog")
    chk.ok(gen_en in catalog, "generic English filler is in the cached catalog")
    # Every filler_phrase() result -- every intent x language -- must be pre-generated.
    intents = (fa.FILLER_INTENT_READ_ONLY, fa.FILLER_INTENT_GENERIC)
    langs = (fa.LANG_ENGLISH, fa.LANG_HINDI, fa.LANG_HINGLISH)
    chk.ok(all(fa.filler_phrase(i, l) in catalog for i in intents for l in langs),
           "every intent x language filler is in the cached catalog")
    chk.ok(all(p in all_catalog for p in catalog), "fillers are part of the full catalog")
    # Every phrase pre-generated -- filler AND common -- must normalise to a unique key, or one
    # phrase's audio would be served for another.
    keys = [fa._normalize(p) for p in all_catalog]
    chk.ok(len(set(keys)) == len(keys),
           "no two phrases across the full catalog collide under normalisation")
    chk.ok(all(keys), "no catalog phrase normalises to empty")

    print("common phrases (P1.10 top-N coverage) are cached but NOT treated as fillers")
    common = tuple(p for variants in fa._COMMON_PHRASES.values() for p in variants)
    chk.ok(len(common) > 0, "there is at least one common phrase in the catalog")
    chk.ok(all(p in all_catalog for p in common),
           "every common phrase is pre-generated in the full catalog")
    # THE de-dup contract: a common phrase is the responder's OWN reply segment, so it must NOT
    # read as a filler -- otherwise the bridge drops it from _prepushed and re-speaks the first
    # sentence (double-speak). This guards that fillers and common phrases stay disjoint.
    chk.ok(not any(fa.is_filler_phrase(p) for p in common),
           "no common phrase is classified as a filler (protects bridge de-dup)")
    chk.ok(all(fa.is_filler_phrase(p) for p in catalog),
           "every filler IS classified as a filler")
    # Hindi common phrases must actually be Devanagari (no silent English fallback slipping in).
    hi_common = fa._COMMON_PHRASES.get(fa.LANG_HINDI, ())
    chk.ok(all(any("ऀ" <= c <= "ॿ" for c in p) for p in hi_common),
           "every Hindi common phrase is in Devanagari script")

    print("language selection returns a distinct, language-appropriate filler")
    # Each language must yield a DIFFERENT string (no silent fallback to English for Hindi/
    # Hinglish), and Devanagari must actually be Devanagari.
    for intent in intents:
        variants = {l: fa.filler_phrase(intent, l) for l in langs}
        chk.ok(len(set(variants.values())) == 3,
               f"{intent}: english/hindi/hinglish fillers are all distinct")
        chk.ok(any("ऀ" <= c <= "ॿ" for c in variants[fa.LANG_HINDI]),
               f"{intent}: hindi filler is in Devanagari script")
    # An unknown language falls back to English rather than erroring.
    chk.ok(fa.filler_phrase(fa.FILLER_INTENT_READ_ONLY, "klingon") == read_en,
           "unknown language falls back to the English filler")

    print("matching tolerates incidental variation")
    # The words the engine emits verbatim must match the catalog entry regardless of trailing
    # ellipsis, case and stray whitespace — the differences that actually occur in practice.
    for raw, expected_phrase in [
        ("Let me check that for you...", read_en),
        ("  LET ME   check that for you  ", read_en),
        ("Give me just a moment", gen_en),
    ]:
        chk.ok(fa._normalize(raw) == fa._normalize(expected_phrase),
               f"{raw!r} matches {expected_phrase!r}")

    print("PCM chunking is lossless and sample-aligned")
    pcm = bytes(range(256)) * 40  # 10240 bytes of arbitrary PCM
    chunks = list(fa._iter_pcm_chunks(pcm, 1280))
    chk.ok(b"".join(chunks) == pcm, "chunks reassemble to the original PCM exactly")
    chk.ok(all(len(c) % 2 == 0 for c in chunks), "no chunk splits a 16-bit sample")
    chk.ok(all(len(c) == 1280 for c in chunks[:-1]), "every non-final chunk is full-size")

    print("build() populates the cache and get() hits/misses correctly")
    # Patch out the network: return deterministic synthetic PCM so this stays offline.
    orig_synth = fa._synthesize_pcm

    async def _fake_synth(settings, text, sample_rate, timeout_s):
        return (b"\x01\x00") * (sample_rate // 100)  # ~10ms of non-silent PCM

    fa._synthesize_pcm = _fake_synth
    try:
        cache = fa.FillerAudioCache()
        await cache.build(_fake_settings())
        chk.ok(cache.ready, "cache reports ready after a successful build")
        chk.ok(cache.describe()["phrase_count"] == len(all_catalog),
               "every phrase in the full catalog (fillers + common) was cached")
        # A hit for every intent x language filler, in its own language...
        chk.ok(all(cache.get(fa.filler_phrase(i, l)) is not None
                   for i in intents for l in langs),
               "every intent x language filler is a cache hit")
        chk.ok(cache.get("Give me just a moment") is not None,
               "generic English filler hits without its trailing ellipsis")
        # ...and a hit for every common phrase (this is the new P1.10 coverage)...
        chk.ok(all(cache.get(p) is not None for p in common),
               "every common phrase is a cache hit")
        chk.ok(cache.get("Okay") is not None and cache.get("  theek   hai  ") is not None,
               "common phrases hit despite stripped punctuation / case / whitespace")
        # ...and a miss for an ordinary reply, which must fall through to live TTS.
        chk.ok(cache.get("Your appointment is confirmed for 4 PM on Saturday.") is None,
               "a normal reply is a cache miss (served live)")
        hits_before = cache.hits
        cache.get(read_en)
        chk.ok(cache.hits == hits_before + 1, "hit counter increments on a hit")

        print("skip-conditions leave the cache empty (safe no-op)")
        for label, settings in [
            ("disabled by config", _fake_settings(filler_audio_cache_enabled=False)),
            ("non-elevenlabs provider", _fake_settings(tts_provider="piper")),
            ("missing voice id", _fake_settings(elevenlabs_voice_id="")),
        ]:
            c = fa.FillerAudioCache()
            await c.build(settings)
            chk.ok(not c.ready and c.get(read_en) is None,
                   f"{label}: cache stays empty and returns None")
    finally:
        fa._synthesize_pcm = orig_synth


async def _real_build(chk: _Check) -> None:
    """Run the real startup synthesis against ElevenLabs and print the cache description."""
    from app.config import get_settings
    print("real ElevenLabs build (using process settings)")
    cache = fa.FillerAudioCache()
    await cache.build(get_settings())
    desc = cache.describe()
    chk.ok(desc["ready"], "real build produced at least one cached phrase")
    print(f"  cached {desc['phrase_count']} phrase(s), sample_rate={desc['sample_rate']}:")
    for p in desc["phrases"]:
        print(f"    - {p['text']!r}  ({p['bytes']} bytes)")


async def _main(do_build: bool) -> int:
    chk = _Check()
    await _offline(chk)
    if do_build:
        await _real_build(chk)
    print(f"\n{chk.passed} passed, {chk.failed} failed")
    return 1 if chk.failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true",
                    help="also run a real ElevenLabs synthesis and print the cache description")
    args = ap.parse_args()
    sys.exit(asyncio.run(_main(args.build)))
