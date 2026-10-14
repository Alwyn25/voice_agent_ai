# Voice Tool-Calling Bug — Root Cause & Fix Log

**Area:** Voice pipeline · planner · rule engine  |  **Owner:** Nikhil  |  **Branch:** `dev` (uncommitted working copy)
**Status:** fixed in code; DSR rule set v2 live locally  |  **Updated:** 13 Oct 2026

The report: on a **voice** call the agent did not call the tool the caller asked for, or called an
unrelated one — while the same conversation **typed in chat** behaved correctly. This log records how
the investigation proceeded, the six root causes it found (none of them in tool definitions, tool
execution or the database lookups themselves), what was changed for each, and how it was verified.
Written so the next engineer can see *why* each fix has the shape it has.

---



## 00 · The puzzle

Voice and chat run the **same** engine. `TurnBridge` (`app/engine/voice/bridge.py`) hands every
finalized transcript to `process_turn()` — the exact function `/sessions/{id}/turns` calls for a chat
turn. Same planner, same rules, same tools, same validation.

So a tool-calling difference between the two channels cannot come from the planner "being different
on voice". It has to come from **what reaches** `process_turn()` **differently on voice, or from
anything inside it that branches on voice-only inputs.** That framed the whole investigation.

What a voice turn passes that a chat turn does not:


| Voice-only input                                          | Where it acts                                                                |
| --------------------------------------------------------- | ---------------------------------------------------------------------------- |
| The transcript itself (Deepgram output, not typed text)   | everything downstream                                                        |
| How transcripts become turns (pipecat frames, barge-in)   | which words a turn sees                                                      |
| `stt_confidence`                                          | confidence banding, rule context, prompt                                     |
| `channel="voice"`                                         | rule context (`session.channel`), prompt                                     |
| A long-lived DB session shared across the whole call      | persistence, close-out                                                       |
| `sentence_callback`, `session_embed_cache`, `stt_seconds` | TTS streaming, embedding cache, cost — **checked, no effect on tool choice** |




## 01 · How we investigated

1. **Code path audit** — bridge → `process_turn` → `plan()` → rules → tool executor, listing every
  branch on the inputs above.
2. **Real calls** — `decision_traces` and `messages` for every recorded voice call (Ajmera demo tenant
  and DSR), comparing what the caller said, what Deepgram transcribed, what the planner proposed, and
   which tools ran.
3. **pipecat 1.7.0 source** (read inside the `core` container) — to confirm how interruptions cancel
  work and how transcripts are emitted.
4. **Live test calls** after each fix, reading the container logs turn by turn.
5. **A text-driven voice harness** (`scripts/voice_harness.py`, §06) — typed text → local TTS →
  real Deepgram → `process_turn()`, to reproduce voice turns without a browser.
6. **Direct rule-engine evaluation** — DSR's live rules evaluated for identical turns as chat and as
  voice at different STT confidences.

---



## 02 · Root causes



### RC1 — A barge-in cancelled the turn being planned and threw the caller's words away

**Severity: high · all tenants · voice only**

A voice turn runs inline on `TurnBridge`'s pipecat frame-processing task. pipecat cancels that task
whenever it decides a new user turn has started:

`MinWordsUserTurnStartStrategy` → `UserTurnProcessor._on_user_turn_started` →
`broadcast_interruption()` → `FrameProcessor._start_interruption()` → cancel the processor's current
task (a `TranscriptionFrame` is not an `UninterruptibleFrame`).

The start strategy is configured with `min_words=3` **while the bot is speaking**, but drops to
**1 word — counting interim transcripts — while the bot is silent**. The bot is silent for the whole
time it is planning a turn: **4–27 s** in the recorded calls. So any word in that window — a
follow-up ("…tomorrow morning"), a check-in ("hello?"), a "hmm" — cancelled the turn mid-plan.

On cancellation the bridge rolls back the DB (correct for consistency) — which also erased the
customer `Message`. Working memory was never saved. **The caller's request existed nowhere.** The
interrupting fragment then ran as a new turn with no context:

> "I want to book a site visit" *(cancelled)* → "tomorrow morning" planned **alone** → no booking
> intent in context → greeting, status check, unrelated lookup, or no tool.

A chat turn can never lose a message this way — which is exactly the reported symptom.

### RC2 — Turn persistence could be split by a cancellation

**Severity: medium · all tenants · voice (and chat on client disconnect)**

The end of `process_turn()` saved working memory (Redis) and **then** committed Postgres as two
separate awaits. A cancellation landing between them left Redis advanced (turn index, pending action,
offered slots) over a rolled-back DB — the next turn planned against state the two stores disagreed
on. It also meant the bridge could not tell whether a cancelled turn had actually been saved.

### RC3 — Speech-to-text misheard the domain's own words, and the planner acted on them

**Severity: high · all tenants · voice only**

Every recorded voice call misheard the same kind of word:


| Caller said  | Deepgram transcript                               | Consequence                                                                              |
| ------------ | ------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| RERA         | "rare", "herrera", "raidra", "रहेगा", "a c trara" | RERA question not recognised                                                             |
| possession   | "position"                                        | **routed to a booking-status lookup** ("may I have the name the visit is booked under?") |
| 3 BHK        | "3 bsk"                                           | configuration missed                                                                     |
| book a visit | "bouquet visite", "mokai visit", "bouquet visser" | booking intent missed                                                                    |


Why Deepgram had no help:

- `keyterms.py` deliberately boosted **only proper nouns**, and its length filter dropped short words
like "BHK" — exactly the acronyms that were misheard.
- A GTM demo tenant keeps its project in `tenant.settings.project_facts`, not in `projects` rows, so
**its own project name was never boosted**.
- The Plivo telephony path (`plivo_ws.py`) passed **no keyterms at all**.
- Nothing told the planner its input was a speech transcript in which a similar-sounding word may
stand for a domain term.



### RC4 — Low audio confidence broke the lookup chain; a slot was booked without being read out

**Severity: high · all tenants · voice only**

`score_confidence()` multiplies the planner's self-confidence by **0.85** when
`stt_confidence < 0.75`; chat sends no STT confidence and never pays it. A normal 0.9 plan becomes
**0.765 → band "medium"**, and banding turned **every** medium `invoke_tool` into `confirm` —
including read-only lookups.

A booking is a chain: `locate_centre` → `search_slots` → offer → confirm → `book_appointment`. The
chain re-plans after each lookup, re-scores with the same STT penalty, gets "confirm" for the next
*lookup*, and `turn_runner` breaks the chain. Live trace:


| Turn | Caller (STT conf)                               | Tool steps          | Agent said                                              |
| ---- | ----------------------------------------------- | ------------------- | ------------------------------------------------------- |
| 7    | "i want to mokai visit tomorrow morning" (0.72) | 1 (`locate_centre`) | "Let me check the available slots…"                     |
| 8    | "okay" (0.57)                                   | 1 (`search_slots`)  | "Right — let me check the available slots…"             |
| 9    | "okay" (0.68)                                   | `book_appointment`  | **"Your visit is confirmed for Wednesday at 11:00 AM"** |


The 11:00 slot was found on turn 8 but **never spoken**; the caller's next "okay" booked it.

### RC5 — Calls were never closed after a barge-in

**Severity: medium · all tenants · voice only**

The bridge's rollback after a cancelled turn **expires every ORM object** on the call's shared
session, but only `Conversation` was refreshed. The end-of-call block in `voice_ws.py` then read
`call.answered_at` → synchronous lazy-load outside an awaited context → `MissingGreenlet`, which
aborted the whole close-out: conversation left `active`, call left `answered`, no duration, working
memory never cleared, post-call memory task never run. Reproduced against the real DB.

The Plivo handler had **no close-out at all** — and no `on_client_disconnected` handler, so a hangup
did not even end the pipeline.

### RC6 — DSR tenant rules made voice turns skip or block tools

**Severity: high · DSR · voice only (by construction)**

DSR's `dsr_turn_policy` carried three rules that only ever fire on voice:


| Rule                         | Condition                                | What it actually did                                                                                                                                                                                  |
| ---------------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `low_speech_confidence`      | `turn.stt_confidence < 0.55`             | `require utterance_confirmed` → derived slot `utterance`, which nothing can ever capture → the turn became **"Before I set that up, may I ask what your utterance is?"** with no tool, twice per call |
| `very_low_speech_no_actions` | `turn.stt_confidence < 0.35`             | `block tool:*` → **every tool blocked**, lookups and the tool behind a "yes" included                                                                                                                 |
| `recording_disclosure_first` | `channel == voice` and `turn_index == 0` | **never fired** (the caller's first turn is index 1), its `recording_disclosed` flag can't be set by anything, and the disclosure was **never spoken**                                                |


Deepgram's multi-language model scores short, *correctly heard* answers low as a matter of course —
"11" at **0.32**, "okay" at **0.57**, "bocale c tomorrow" at **0.43** — so the confidence rules fired
on ordinary speech. Rule-engine evaluation of the live rules, before the fix:


| Turn             | Result                                     |
| ---------------- | ------------------------------------------ |
| chat             | nothing                                    |
| voice, conf 0.95 | nothing                                    |
| voice, conf 0.45 | forced requirement `low_speech_confidence` |
| voice, conf 0.30 | **all tools blocked** + forced requirement |


---



## 03 · Fixes



### Fix 1 (RC1) — Barge-in carry-over

`app/engine/voice/barge_in.py` (new, pure) · `app/engine/voice/bridge.py`

When a barge-in cancels a turn **before it was saved**, the bridge keeps that turn's text and merges
it into the next transcript, so the planner sees the caller's whole request — exactly what a chat
turn would have received.

- `"I want to book a site visit"` + `"tomorrow morning"` → one turn:
`"I want to book a site visit tomorrow morning"`.
- **Repeat detection** (whole-word): if the caller simply re-says the cancelled words, they are not
duplicated.
- **Max age** (`VOICE_BARGE_IN_CARRYOVER_MAX_AGE_S`, default 30 s), measured from the *first*
cancellation in a chain, so stale words are dropped rather than merged.
- Merged STT confidence = the **lower** of the two halves, so low-confidence handling still sees it.
- Every cancellation is now logged (`voice turn cancelled by barge-in`, `voice barge-in carry-over merged|repeated|expired`) — previously a cancelled turn left no trace at all.
- Kill switch: `VOICE_BARGE_IN_CARRYOVER_ENABLED` (default `true`).



### Fix 2 (RC2) — Cancellation-safe persistence

`app/engine/planner/turn_runner.py` (`_persist_turn`, `TurnCancelledAfterCommit`)

Redis save + Postgres commit now run as one unit in their own task, shielded. If a cancellation
arrives mid-persist, `_persist_turn` **finishes the write in place** (so the caller, which holds the
per-call DB lock, never touches the session while a commit is in flight) and then raises
`TurnCancelledAfterCommit` — a subclass of `CancelledError`, so pipecat's cancellation completes
normally. The bridge reads it as "this turn is saved": no rollback, no carry-over. A write that itself
fails during cancellation surfaces as a plain cancellation, so the bridge rolls back and carries over.

### Fix 3 (RC3) — A per-call lexicon, used twice

`app/engine/voice/lexicon.py` (new) · `app/engine/voice/asr_hints.py` (new) · `keyterms.py` ·
`packs/base.py` · `packs/real_estate_india/pack.py` · `voice_ws.py` · `plivo_ws.py` ·
`turn_runner.py` · `planner.py` · `prompts.py` · `understanding.py`

Domain-agnostic by design — nothing industry-specific is hardcoded in the engine; everything comes
from data the engine already uses to understand a tenant.

**3a. The call lexicon** — built once per call from:

1. the tenant's proper nouns (name, projects + aliases, `project_facts` names, locations) — now
  including demo tenants' `project_facts`;
2. the industry pack's vocabulary — new `Vocabulary.stt_keyterms` (curated, most-misheard first;
  real-estate: RERA, BHK, possession, site visit, carpet area…), plus `technical_terms` and nouns;
3. **action phrases derived from the pack's commitment noun** — "book a site visit", "book a visit",
  "reschedule my visit"… (healthcare: "book a consultation"; SaaS: "book a demo call");
4. the tenant's custom intents (`intent_vocabulary` names + aliases), e.g. "loan enquiry".

**3b. Use 1 — Deepgram keyterm boosting**, on **both** the browser path and Plivo (which previously
had none). Proper nouns keep their 40-term budget; domain terms get their own 40 (under Deepgram's
100), and are exempt from the short-word filter.

**3c. Use 2 — misheard-word hints.** Each voice transcript is compared with the lexicon by a small,
dependency-free phonetic matcher:

- a consonant-skeleton key (vowels dropped, voicing pairs and common spellings merged) plus spelling
similarity; spelled acronyms (BHK) matched letter-by-letter;
- one word for one word must have an **identical** sound key (rejects "person"→possession,
"session"→possession); phrases use a graded score (so "mokai visit" → "book a visit");
- a word already heard correctly is never hinted; spans covered by a correct term are skipped.

Hints are **never rewrites** — "what position is the flat facing" is genuinely "position". They are
shown to the planner and the fast classifier (*"SPEECH-TO-TEXT CHECK: "position" → probably
"possession"…"*), folded into the retrieval query as extra terms, logged (`voice asr hints`) and
returned in the turn debug envelope (`asr_hints`). Every voice turn's prompt also carries a one-line,
domain-agnostic note that the input is a speech transcript.

Tuning, on real transcripts: every recorded mishearing above the threshold is caught (scores
0.78–0.96); **zero** false positives across 23 ordinary sentences (including the real "who is the
person coding" and "where is the site office"). Healthcare ("consultashun"), SaaS ("onboarting") and
tenant-intent ("lone enquiry") cases are caught too.

### Fix 4 (RC4) — Read-only lookups are never demoted

`app/engine/planner/planner.py` (`banded_action`) · `app/engine/tools/types.py` (`READ_ONLY_TOOLS`)

Confidence banding is now a small pure function. A **read-only lookup is never demoted to
"confirm"** — it changes nothing, so there is nothing to confirm, and demoting it only broke the
chain. Writes still band exactly as before, and a "low" band still escalates any action. The list of
read-only tools now has one definition (`READ_ONLY_TOOLS`), shared by banding and by the TTS
pipelining in `turn_runner`, so the two cannot drift.

Result: a voice booking now reaches *"Shall I book your site visit for tomorrow at 11:00 AM?"* in one
turn (`locate_centre` + `search_slots` in the same turn), as in chat.

### Fix 5 (RC5) — Reliable call close-out

`app/api/voice_ws.py` · `app/api/plivo_ws.py`

- **Browser voice:** `conv` and `call` are refreshed (awaited) before any attribute is read; the
conversation id is captured as a plain value at setup for logging; the commit runs in a
`try/except` with a guarded rollback; working-memory clear and the post-call task **always** run.
- **Plivo:** added `on_client_disconnected` (a hangup now ends the pipeline) and a close-out that
completes the **conversation** only. The **Call row is deliberately left alone**: its lifecycle
belongs to `apply_call_event` (`services/telephony/events.py`) via Plivo's hangup webhook, which
carries the authoritative duration, cost and hangup cause and drives retries/campaign progress — and
ignores events for an already-terminal call, so closing it from the websocket would drop that data.



### Fix 6 (RC6) — DSR's voice-only rules removed; their intents moved where they work

`scripts/dsr_rules/dsr_turn_policy.yaml` · `planner.banded_action` · `app/api/sessions.py`
(`with_recording_disclosure`) · `seed_dsr_settings.py` (both copies) · `config.py`


| Removed rule                 | Legitimate intent                       | Now enforced by                                                                                                                                                                                                                                           |
| ---------------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `low_speech_confidence`      | don't act on possibly-misheard words    | **Engine, all tenants:** below `CONF_STT_READBACK_BELOW` (0.55) any **write** is read back before it runs, whatever the band. Lookups and answers are never held back. `0` disables.                                                                      |
| `very_low_speech_no_actions` | same                                    | same — blocking lookups on confidence alone was wrong (correct "11" at 0.32)                                                                                                                                                                              |
| `recording_disclosure_first` | disclose recording before anything else | **New tenant setting** `recording_disclosure`, spoken in the voice opening line, before the caller's first word. Placed as its own sentence before the greeting's closing question. Used by browser voice, Plivo, console voice sessions and the harness. |


DSR's greeting is now: *"Thanks for calling DSR Infra Projects. This call may be recorded for quality
and training purposes. How can I help you today?"*

Applied locally: `dsr_turn_policy` **v1 → v2** (activated through the rules API, v1 retained for
rollback) and `settings.recording_disclosure` set on the DSR tenant (merge; no other key touched).
After the change, the same rule-engine evaluation returns **nothing** for chat and for voice at 0.95,
0.45 and 0.30, and on the first voice turn — voice now matches chat.

---



## 04 · New settings


| Setting                                | Default  | Meaning                                                              |
| -------------------------------------- | -------- | -------------------------------------------------------------------- |
| `VOICE_BARGE_IN_CARRYOVER_ENABLED`     | `true`   | merge a barge-in-cancelled turn's words into the next transcript     |
| `VOICE_BARGE_IN_CARRYOVER_MAX_AGE_S`   | `30`     | oldest carried words that may still be merged                        |
| `CONF_STT_READBACK_BELOW`              | `0.55`   | below this STT confidence, a write is always read back; `0` disables |
| `tenant.settings.recording_disclosure` | unset    | sentence spoken in every voice opening line for that tenant          |
| `Vocabulary.stt_keyterms` (pack)       | per pack | curated domain words boosted in STT and matched for hints            |


---



## 05 · Verification

**Unit tests** (new): `test_barge_in.py`, `test_persist_turn.py`, `test_asr_hints.py`,
`test_confidence_banding.py`, `test_voice_rule_parity.py`; `test_keyterms.py` extended. Full container
suite: **678 passed** — the single failure, `test_demo_settings::test_defaults_are_off`, is
environmental (the container sets `DEMO_ENABLED=true`) and unrelated.

**Bridge with real pipecat** (scratch check in the container): a cancelled turn rolls back and keeps
its words; the next turn receives the merged text; a turn cancelled after commit does neither.

**Live calls (Ajmera demo tenant), after Fixes 1–3:**

- Carry-over fired three times; `"i want to mokai visit"` + `"tomorrow morning"` was planned as one
request and went straight to the slot search.
- "is it rera approved", "when is the possession", "what's the price of a 3 bhk" were all transcribed
correctly (conf 0.97–0.99) — previously misheard on every call.

**Voice harness, after Fix 4:**


| Typed                                        | Deepgram heard | Outcome                                                                   |
| -------------------------------------------- | -------------- | ------------------------------------------------------------------------- |
| when is the position                         | "position"     | hint → possession → "June 2027" (previously a booking-name ask)           |
| is it rare approved                          | **"rera"**     | keyterm boost corrected it inside Deepgram                                |
| i want to book a site visit tomorrow morning | exact          | `locate_centre` + `search_slots` in one turn → "Shall I book … 11:00 AM?" |


**Close-out:** the `MissingGreenlet` was reproduced against the real DB with the old read and is gone
with the refresh.

**DSR rules:** evaluated against the live rule engine before/after (§02 RC6, §03 Fix 6).

---



## 06 · Testing voice without a browser — `scripts/voice_harness.py`

Typed text → **Piper** (local, free TTS) → **real Deepgram** (the call's model, language, formatting
flags and call-lexicon keyterms) → `process_turn()` on a `channel="voice"` conversation with the call
lexicon in working memory. Prints, per turn: what Deepgram heard (flagged when it differs from what
was typed), ASR hints, intent, action, confidence band, tools run, pending confirmation, reply and
timing. Demo tenants start the way a demo-link call does.

```bash
docker compose exec -it core python scripts/voice_harness.py --tenant "DSR Infra Projects"
docker compose exec core python scripts/voice_harness.py --tenant "Ajmera Realty & Infra" \
    "is it rera approved" "when is the possession" "book a site visit tomorrow morning"
```

Options: `--file` (one query per line), `--text-only` (skip TTS/STT to separate STT problems from
engine problems), `--voice` (another Piper voice, e.g. Hindi), `--debug` (full debug envelope).
Not exercised by the harness: live-stream endpointing and barge-in. It writes what a real test call
writes, and closes the conversation at the end.

---



## 07 · Known limitations & follow-ups

- **Short interruptions still cost a re-plan.** Carry-over keeps the words, but a "hmm" while the agent
is thinking still cancels the in-flight turn and re-runs it merged (extra latency and LLM spend). A
follow-up could raise the start strategy's word threshold while a turn is in flight.
- **Deepgram** `is_final` **segmentation** (pipecat emits a turn per final segment) was not observed
splitting turns in the recorded calls, but long continuous utterances were not specifically tested.
- **Hints miss some forms:** "raidra"/"trara" score just under the threshold (RERA is now boosted at the
STT level instead); split acronyms ("3 beat hk") and Devanagari transcripts are not matched.
- **Motorworks demo tenant** (`scripts/demo_rules/motorworks_turn_policy.yaml`) carries the same three
voice-only rules — not changed here.
- **Other unsatisfiable** `require` **rules in DSR** (`quote_only_published_prices`,
`knowledge_retrieved`, `sales_executive_confirmed`, …) have never fired in DSR's history, but if one
did it would hijack the turn the same way RC6 did. Worth converting to advisory guidance, and adding
a rule-activation check that rejects a `require` nothing can satisfy.

---



## 08 · Files

**New:** `app/engine/voice/barge_in.py`, `app/engine/voice/lexicon.py`, `app/engine/voice/asr_hints.py`,
`scripts/voice_harness.py`, `tests/test_barge_in.py`, `tests/test_persist_turn.py`,
`tests/test_asr_hints.py`, `tests/test_confidence_banding.py`, `tests/test_voice_rule_parity.py`

**Modified:** `app/engine/voice/bridge.py`, `app/engine/voice/keyterms.py`,
`app/engine/planner/turn_runner.py`, `app/engine/planner/planner.py`, `app/engine/planner/prompts.py`,
`app/engine/planner/understanding.py`, `app/engine/tools/types.py`, `app/api/voice_ws.py`,
`app/api/plivo_ws.py`, `app/api/sessions.py`, `app/packs/base.py`,
`app/packs/real_estate_india/pack.py`, `app/config.py`, `scripts/dsr_rules/dsr_turn_policy.yaml`,
`scripts/seed_dsr_settings.py`, `app/packs/real_estate_india/seeds/seed_dsr_settings.py`,
`tests/test_keyterms.py`

**Rollback:** DSR rule set — re-activate `dsr_turn_policy` v1 in the Rule Engine screen; carry-over —
`VOICE_BARGE_IN_CARRYOVER_ENABLED=false`; write read-back — `CONF_STT_READBACK_BELOW=0`; disclosure —
remove `recording_disclosure` from the tenant's settings.