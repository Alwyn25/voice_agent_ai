# Filler-audio cache & TTFA instrumentation (P1.10)

**Goal:** cut how long a caller waits after they finish speaking before they hear *any* audio back
(time-to-first-audio, TTFA), on the turns where the engine speaks a deterministic filler.

---

## 1. What was built

### 1.1 Filler-audio cache — `core/app/engine/voice/filler_audio.py`

A handful of the agent's spoken lines are **fillers**: short, non-committal acknowledgements the
engine speaks over the same TTS path as the real reply, purely to cover the 1–2 s tool round-trip
with a voice instead of silence. They are emitted in two shapes:


| Intent      | When                                                            | English line                 |
| ----------- | --------------------------------------------------------------- | ---------------------------- |
| `read_only` | about to look something up (slots, price, inventory, knowledge) | "Let me check that for you…" |
| `generic`   | about to write/do something (book, ticket, callback)            | "Give me just a moment…"     |


Each phrase exists in **English / Hindi (Devanagari) / Hinglish (Roman)**. Because these strings are
chosen by *our code*, not the LLM, they are identical every call, yet each filler turn still paid the
full live ElevenLabs first-audio latency to synthesise words that never change.

**Plus a "top-N" catalog of common responder phrases (closes the P1.10 gap).** The task also calls
for pre-generating the common short responses the agent speaks — `'Theek hai'`, `'Okay'`, etc. These
are **not** fillers: they are the responder's own reply segments. They are cached and served the same
way, but kept strictly separate from the filler set (see *Safety* below). The catalog:

| Kind | Phrases (EN / HI / Hinglish) |
|---|---|
| Tool-preamble fillers | "Let me check that for you…" · "Give me just a moment…" (× 3 langs = 6) |
| Common acknowledgements/closings | "Okay." · "Sure." · "Got it." · "Thank you!" + HI/Hinglish equivalents (12) |

**18 clips total**, synthesised once at startup. They are served opportunistically: whenever any
spoken segment — filler or responder — exactly matches a cached phrase, the cached PCM is served and
the live TTS call is skipped. The responder is never *forced* to emit these; the cache simply catches
them on the turns it naturally does.

**The change:** at startup (`prewarm_filler_cache`, backgrounded from `main.py`'s lifespan) every
phrase is synthesised **once** via ElevenLabs' REST API — same voice/model/voice-settings as the
live pipeline, so a cached clip is indistinguishable — and the raw PCM is held in memory. At call
time, `FillerAudioProcessor` sits just upstream of the TTS service: a `TTSSpeakFrame` whose text
matches a cached phrase is replayed from memory as the exact frame sequence TTS would emit
(`TTSStarted → TTSAudioRaw… → TTSStopped`) and the speak frame is swallowed so ElevenLabs is never
called. Every other line passes straight through to live TTS unchanged.

**Safety properties:**

- Matching is **exact, normalised equality** against a small curated set (`_FILLER_TABLE`, the single
source of truth) — never fuzzy/semantic — so a cached clip can never be substituted for a real reply.
- `turn_runner.py` builds the filler it speaks from the *same* table via `filler_phrase(intent, language)`,
so the spoken words and the cached words cannot drift apart.
- A disabled feature, an empty/failed cache, or a miss simply means the line is spoken live, exactly as
before. Correctness never depends on a cache hit.
- Filler language is chosen with the same `responder.detect_language(user_input)` the responder uses, so
the filler never precedes an English reply with a Hindi "one moment" or vice-versa.
- **Common phrases are decoupled from fillers.** `is_filler_phrase()` (the bridge's de-dup exclusion set)
is built from the *filler* catalog only — common phrases return `False`. This is load-bearing: a common
phrase is a real reply segment, so it must stay in the bridge's `_prepushed` de-dup; if it were treated
as a filler, the bridge would replay the first sentence and the caller would hear it twice. The
verification script asserts the two sets are disjoint.
- **Common phrases are curated closed-class only.** `_normalize` strips trailing punctuation, so `"Okay?"`
folds to `"Okay."` — acceptable for acknowledgements, but the catalog must never include a word whose
question intonation must sound different from its statement form.



### 1.2 Bridge de-dup fix — `core/app/engine/voice/bridge.py`

Fillers flow through the bridge's `_sentence_callback` like the responder's streamed sentences, but they
are **not** response segments. `is_filler_phrase()` now keeps them out of `_prepushed`, so the post-turn
"skip the sentence already streamed" de-dup compares the right entry and the caller never hears the real
first sentence twice.

### 1.3 TTFA instrumentation — `core/app/engine/voice/ttfa.py`

Built to *measure* the feature, because the engine's own `debug.timing.total_ms` measures reply-compute
time, which the filler deliberately does not change. For each turn, `ttfa_ms` = time from the finalized
transcript reaching the engine (`t0`) until the first `TTSAudioRawFrame` leaves the TTS stage, tagged with
the `source` that produced it:

- `filler_cache` — pre-generated tool-preamble filler served from memory (**a win**)
- `phrase_cache` — a common responder phrase served from memory (**a win** — the new top-N coverage)
- `filler_live` — a filler, but via live TTS (cache miss/disabled)
- `responder` — no cache hit this turn; first audio is the responder's own live speech

`TurnBridge` stamps `t0`, `FillerAudioProcessor` attributes the source, and a `TTFAProbe` placed *after*
the TTS stage records the first audio frame uniformly for both cached and live audio.

> **Scope of** `ttfa_ms`**:** it covers the **engine + TTS** portion only — exactly what the filler affects.
> It excludes the upstream VAD silence window (~700 ms) + STT finalize (they happen before the bridge
> sees the transcript) and the final network hop to the browser. Full mic-to-ear ≈ `ttfa_ms` + ~850 ms.



### 1.4 Observability & config

- `GET /health/filler-audio` — cache ready-state, which phrases are cached, live hit/miss tally.
- `GET /health/ttfa` — rolling per-source count/min/p50/avg/max.
- `check_filler_audio.py` — offline regression gate for the match contract + `--build` for a real synth.
- Config: `filler_audio_cache_enabled` (default on), `filler_audio_chunk_ms` (40), `filler_audio_build_timeout_ms` (8000), `ttfa_logging_enabled` (default on).

---



## 2. Results — live calls, 2026-09-10

Two runs were compared: cache **ON** (filler turns served from memory) vs cache **OFF** (same fillers
still spoken, but via live TTS). All figures are `ttfa_ms` (engine + TTS portion).

### 2.1 Cache ON run


| source         | n   | min  | p50      | avg  | max  |
| -------------- | --- | ---- | -------- | ---- | ---- |
| `filler_cache` | 4   | 2127 | **2538** | 2543 | 3143 |
| `responder`    | 9   | 1284 | 4200     | 4052 | 5466 |




### 2.2 Cache OFF run


| source        | n   | min  | p50      | avg  | max  |
| ------------- | --- | ---- | -------- | ---- | ---- |
| `filler_live` | 3   | 3220 | **3671** | 3572 | 3825 |
| `responder`   | 5   | 3035 | 3773     | 3879 | 5374 |




### 2.3 Headline numbers

**Win #1 — the cache saving (**`filler_cache` **vs** `filler_live`**):**


|     | cached  | live    | saving        |
| --- | ------- | ------- | ------------- |
| p50 | 2538 ms | 3671 ms | **−1,133 ms** |
| avg | 2543 ms | 3572 ms | **−1,029 ms** |
| min | 2127 ms | 3220 ms | **−1,093 ms** |


Serving the filler from memory removes **~1.0–1.1 s** of first-audio latency on filler turns — the live
ElevenLabs first-audio round-trip, plus connection setup, over an India→US network path.

**Win #2 — the perceived-latency win (filler vs waiting for the responder):**
On a tool turn, without any filler the caller hears nothing until the whole tool + responder + TTS chain
finishes (the `responder` rows, ~4.0–4.2 s p50). A cached filler puts a voice in their ear at ~2.5 s —
**~1.5 s sooner** — and the real reply streams in behind it.

### 2.4 Is this good? — yes, with honest caveats

- **The feature works and the effect is real.** Cache ON reliably lands first audio ~1 s earlier than
cache OFF on filler turns, and the `filler_cache` source is being attributed correctly.
- **The saving is ~4× larger than the benchmark estimate (~280 ms).** The benchmark used ElevenLabs'
*published* first-audio spec; the live India-RTT cost it actually removes is closer to ~1 s. The cached
path is essentially instant (PCM already in memory), so it skips the entire live-TTS setup, not just the
model's synthesis time.
- **Absolute TTFA is still high** (~2.5 s cached filler, ~4 s responder). That is **not** a P1.10 miss —
`ttfa_ms` is dominated by the planner/responder LLM (GPT-4o TTFT p50 ~1.1 s, p95 ~5.1 s) and, mic-to-ear,
by the excluded VAD (~700 ms) + STT. P1.10 is orthogonal to those; they are addressed by the roadmap's
planner swap (GPT-4o-mini → Cerebras) and semantic VAD.
- **Sample is small** (4 cached / 3 live turns). The direction and magnitude are clear and consistent, but
these are indicative numbers from a manual session, not a statistically sampled benchmark. Recommend
letting `/health/ttfa` accumulate over a longer call session to confirm the p50s hold.

> **Note:** the numbers above predate the common-phrase catalog (the `phrase_cache` source). They
> measure tool-preamble fillers only. The common-phrase coverage uses the identical cache/serve path,
> so the per-hit saving is the same ~1 s; its *overall* impact depends on how often the responder emits
> a catalog phrase as a standalone segment, which `/health/filler-audio` (hits/misses) and the
> `phrase_cache` bucket on `/health/ttfa` will show over a real session. Re-measure before quoting a
> turn-coverage number.

### 2.5 Scope vs. the task (P1.10)

| Task text | Built | Status |
|---|---|---|
| "Pre-generate common filler TTS audio at startup" | filler + common catalog, synth once at boot | ✅ |
| "Top-10 filler responses (`'Theek hai'`, `'Let me check'`, …)" | 18 clips (6 fillers + 12 common, 3 langs) | ✅ covered |
| "When responder selects a matching phrase, serve cached bytes" | opportunistic exact-match serve in `FillerAudioProcessor` | ✅ |
| "…as **MP3** bytes" | raw **PCM** (`pcm_16000`) | ⚠️ deliberate — matches transport, no decode step |

The one intentional deviation from the task wording is **PCM instead of MP3**: the voice transport plays
16-bit mono PCM, so caching PCM lets a hit be replayed as byte-identical `TTSAudioRawFrame`s with zero
decode; MP3 would add a decode hop for no benefit.

---



## 3. How to reproduce / monitor

```bash
# Per-turn lines during a live call
docker compose logs -f core | grep "voice TTFA"

# Rolling per-source summary
curl -s http://localhost:8000/health/ttfa

# Cache state + hit/miss tally
curl -s http://localhost:8000/health/filler-audio

# Offline match-contract regression gate (no key/network)
docker compose exec core python scripts/check_filler_audio.py
# ...plus a real ElevenLabs build
docker compose exec core python scripts/check_filler_audio.py --build
```

To A/B it again: toggle `filler_audio_cache_enabled` (ON → `filler_cache`, OFF → `filler_live`) and
compare the two source buckets on `/health/ttfa`.

---



## 4. Files touched


| File                                     | Change                                                           |
| ---------------------------------------- | ---------------------------------------------------------------- |
| `core/app/engine/voice/filler_audio.py`  | **new** — cache, processor, filler + common-phrase catalog, match contract |
| `core/app/engine/voice/ttfa.py`          | **new** — `TurnTiming`, `TTFAProbe`, `TTFAStats`                 |
| `core/scripts/check_filler_audio.py`     | **new** — offline + `--build` verification                       |
| `core/app/engine/planner/turn_runner.py` | build filler from the shared catalog, language-matched           |
| `core/app/engine/voice/bridge.py`        | stamp `t0`; keep fillers out of the de-dup `_prepushed`          |
| `core/app/api/voice_ws.py`               | wire `FillerAudioProcessor` + `TTFAProbe` into the pipeline      |
| `core/app/main.py`                       | background `prewarm_filler_audio()` at startup                   |
| `core/app/api/health.py`                 | `GET /health/filler-audio`, `GET /health/ttfa`                   |
| `core/app/config.py`                     | 4 settings (cache enable, chunk ms, build timeout, TTFA logging) |


