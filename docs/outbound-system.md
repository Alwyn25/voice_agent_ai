# Outbound Calling System

_Last updated: 2026-08-27_

This document describes the outbound calling system end to end: how a call is decided,
placed, and conducted, what is built and working today, and exactly what remains per the PRD
(_The AI Voice Agent.md_, §7 Campaigns, §9 Calls & Telephony, §Voice Pipeline "Life of a Call").

The guiding principle: **the engine is channel- and direction-agnostic.** Every rule, tool,
retrieval, booking and postcondition runs identically whether a turn arrives typed or spoken,
inbound or outbound. Outbound adds three things around that unchanged core — a compliance
**gate**, a **telephony** layer that places calls, and an outbound **conversation opening**
(identity confirmation + purpose-shaped dialog).

---

## 1. Architecture at a glance

```
                 ┌───────────────────────── triggers ─────────────────────────┐
   manual dial ──┤  POST /calls/outbound   campaigns   callbacks   retries     │
                 └───────────────┬─────────────────────────────────────────────┘
                                 ▼
                   place_outbound_call()  ── the SINGLE gate every dial passes
                   app/services/outbound.py   (DNC · hours · retry · escalation)
                                 │  writes Call(status="queued", purpose, campaign_id, attempt)
                                 ▼
                   Dialer worker  (app/services/telephony/dialer.py)
                     · concurrency cap + campaign pacing (token bucket)
                     · RE-CHECKS the gate at dial time
                     · provider.originate(DialRequest)
                                 │
                                 ▼
                   TelephonyProvider  (stub | plivo | exotel | twilio)
                     · originate() → carrier places the PSTN call
                     · parse_webhook()/verify_signature() → CallEvent
                                 │  carrier events  →  POST /webhooks/telephony/{provider}
                                 ▼
                   apply_call_event()  ── the SINGLE writer of Call lifecycle state
                   app/services/telephony/events.py   (retries · campaign stats · cost · events)
                                 │  on "answered": media stream ↔ pipecat
                                 ▼
                   process_turn()  ── the SAME brain inbound uses
                     · outbound identity gate  (app/engine/planner/outbound_flow.py)
                     · purpose-shaped prompt block  (app/engine/planner/prompts.py)
                     · rules / tools / retrieval / booking  (unchanged)
```

Three layers, three "single-funnel" invariants that make the system hard to get subtly wrong:

| Invariant | Where | Why it matters |
|---|---|---|
| Every dial passes **one** gate | `place_outbound_call()` | No feature can bypass DNC / calling hours |
| Every carrier event has **one** writer | `apply_call_event()` | Retries, cost, stats stay consistent |
| Every turn runs **one** brain | `process_turn()` | Spoken == typed; inbound == outbound |

---

## 2. The compliance gate

`app/services/outbound.py` — `check_can_call()` (dry-run) and `place_outbound_call()` (create
the queued `Call`). The gate evaluates the tenant's `outbound_call` rule set plus a hard
open-escalation check ("a human owns this customer — no automated contact until resolved").

The DSR calling policy (`core/scripts/dsr_rules/dsr_calling_policy.yaml`) is the reference rule
set: `dnc_absolute` (priority 0, terminal), calling-hours windows, per-purpose caps
(`marketing_daily_cap` = 1/day, `daily_attempt_cap` = 2/day), journey-stage rules
(`visit_scheduled_no_promotions`, `negotiation_is_human_work`), and segment rules
(channel-partner, out-of-city, VIP).

**Purpose sensitivity.** Many rules fire only for `session.purpose == marketing`. The same
customer can be **blocked for marketing but allowed for service** — e.g. a lead with a booked
visit. The purpose is chosen per call and re-checked at dial time.

Triggers, all through the one gate:
- **Manual** — `POST /calls/outbound` ([app/api/calls.py](../core/app/api/calls.py))
- **Campaigns** — create/preview/start/pause/stop/targets ([app/api/campaigns.py](../core/app/api/campaigns.py))
- **Callbacks** — "call me later" due ([dialer `_enqueue_due_callbacks`](../core/app/services/telephony/dialer.py))
- **Retries** — scheduled by `apply_call_event` after busy/no-answer/voicemail

---

## 3. Telephony layer (provider-agnostic, stub-first)

`app/services/telephony/` — mirrors the WhatsApp messaging abstraction. A carrier does exactly
two things: place a call, and translate its callbacks into one canonical event.

| File | Responsibility |
|---|---|
| `base.py` | `TelephonyProvider` ABC, `DialRequest`, `OriginateResult`, `CallEvent`, the `STATUS_*` vocabulary, `TERMINAL_/RETRYABLE_/ACTIVE_STATUSES` |
| `stub.py` | `StubTelephonyProvider` — simulates the full lifecycle (ring → answer → completed, or busy/no-answer/voicemail/failed) with synthetic duration + cost, no carrier account |
| `provider.py` | Selection: `stub` (default) \| `plivo`/`exotel`/`twilio` \| `auto`. Cached per credentials. Real adapters currently raise `NotImplementedError` |
| `events.py` | `apply_call_event()` — the single writer: mutates the `Call`, schedules retries, advances campaign targets, records cost + analytics, publishes dashboard events. Idempotent on terminal state |
| `dialer.py` | The worker tick: enqueue campaign targets (paced) + due callbacks, then originate queued calls within the per-tenant concurrency cap, re-checking the gate at dial time |

**API surface:**
- `POST /telephony/tick` — run one dialer pass now (demo/testing) · `GET /telephony/status` — live dialer state ([app/api/telephony.py](../core/app/api/telephony.py))
- `POST /webhooks/telephony/{provider}` — carrier callbacks, **secured by provider signature** (not JWT), verify → `parse_webhook` → `apply_call_event` ([app/api/telephony_hooks.py](../core/app/api/telephony_hooks.py))

**Background loop:** `app/main.py` runs `run_dialer_tick()` every `dialer_interval_seconds`
when `dialer_enabled` is true.

**Stub demo trick:** the callee's last two digits force an outcome — `…00` no-answer,
`…11` busy, `…22` voicemail, `…99` failed; anything else answers.

**Retry policy** (`events._schedule_retry`): busy/no-answer/voicemail queue a fresh `Call`
(`attempt+1`, `scheduled_for = now + backoff·attempt`) up to `dialer_max_attempts`. Retries are
still gated at dial time, so the calling policy's back-off has the final say.

---

## 4. Outbound conversation system

The behavioural half — what makes an outbound call feel right and stay compliant. Lives in
`app/engine/planner/outbound_flow.py` (pure, self-tested via
`python -m app.engine.planner.outbound_flow`) and is wired into `process_turn`.

### 4.1 Identity gate (deterministic)

We placed the call, so per Voice Pipeline §3.5 the agent confirms it reached the right person
**before any business**. The opener (both channels) asks _"Am I speaking with {name}?"_. The
next turn is classified and the planner's draft is **overridden** so a pitch never reaches the
wrong person. State lives on the session (`outbound_stage`, `call_purpose`,
`outbound_identity_tries`); the gate is a final guard in
[turn_runner.py](../core/app/engine/planner/turn_runner.py), placed before the visit-status gate.

| Caller's answer | Branch | Behaviour |
|---|---|---|
| "Yes / speaking / that's me" | `confirmed` | Open the purpose in the right register; hand to the normal engine |
| "No / wrong number / nobody by that name" | `wrong_person` | Apologise, **end call**, tag record `wrong_number` (flag-for-review, **not** auto-suppressed) |
| "He's not here / not available" | `unavailable` | Offer a callback |
| "I'm busy / call me later" | `not_now` | Offer a callback |
| "Who is this? / what's this about?" | `gatekeeper` | Disclose purpose briefly, re-ask identity |
| "Stop calling me" | `opt_out` | **Not intercepted** — falls to the normal DNC path (`record_do_not_contact`), one audit trail for every opt-out |
| unclear | `unclear` | Re-ask identity (twice, then let the call proceed) |

Operator-confirmed policy decisions: wrong number → **flag for review** (a one-off wrong answer
must not permanently block a lead); unavailable/busy → **offer a callback** (matches the PRD
callback queue).

**Identity carries into the visit-status gate.** The inbound visit-status gate asks a returning
caller for their name before reading a booking back, because inbound caller-ID can match the
wrong record. On outbound that check is already satisfied: we dialled *this* record's number and
the caller confirmed "yes, this is {name}" — strictly stronger than caller-ID. So once
`outbound_stage == confirmed`, a later "when is my visit?" is answered straight from the dialled
record (the same deterministic `confirmed_reply` text) **without re-asking the name**. The carry
is keyed on `outbound` + `STAGE_CONFIRMED` only (never `callback`/`closed`, where identity was
not confirmed), so the inbound always-ask path is byte-for-byte unchanged. See the split Turn-A
branch in [turn_runner.py](../core/app/engine/planner/turn_runner.py).

### 4.2 Purpose-shaped dialog

Once identity is confirmed, an `OUTBOUND CALL` block is added to the system prompt
([prompts.py](../core/app/engine/planner/prompts.py)) setting objective + tone by purpose. The
purpose is read from the linked `Call` in
[turn_context.py](../core/app/engine/planner/turn_context.py) (`TurnContext.call_purpose`).

| Purpose | Objective / tone |
|---|---|
| `marketing` | Warm, brief; spark interest, offer a site visit; accept "not interested" gracefully |
| `service` | Resolve the specific matter (confirm/reschedule/follow-up); don't turn into a pitch |
| `feedback_survey` | Post-visit; ask briefly how it went; don't sell |
| `retention` / `follow_up` | Re-engage around what changed; low pressure |
| `upsell` | Offer options that build on prior interest |

**Lead with the enquiry's project (outbound only).** A call we placed is about the lead's
*stated interest*, so it should open with that project — not a generic "new options" followed by
"which of our projects?". `resolve_project` (project_resolution.py) has an outbound-only signal
`enquiry_interest`: the lead's most recent `CustomerProductInterest` at level `interested` that
names a project. It sits after the existing signals (mention · pin · dialled DID · visit history)
and before tenant-default/ambiguous, so a stronger signal still wins and it pins on success
(≤ one query per call). The query is gated on `direction == "outbound"`, so **inbound resolution
runs the exact same queries and cost it did before**. The confirmed-identity opener then names
the resolved project (`purpose_opener(..., project=…)`); disambiguation remains the true fallback
for a lead with no enquiry on file.

**Purpose questions during disambiguation.** A multi-project tenant forces a "which project?"
clarification when the turn needs a project but none is resolved. If the caller answers that with
a *meta* question — "what is this about?", "who is this?" — the fast classifier tends to mislabel
it a product question, and the engine used to re-pin the identical project menu, deaf to the
question and looping every turn. On an outbound call the engine now discloses the call's purpose
(`outbound_flow.purpose_disclosure`, purpose-aware, no identity re-ask) and *then* steers to the
choice. Guarded on `outbound`, so inbound disambiguation is unchanged.

### 4.3 Console UI (test harness)

The Test Console ([voice-agent-ui/src/pages/console/ConsolePage.jsx](../voice-agent-ui/src/pages/console/ConsolePage.jsx))
drives `process_turn` directly. Outbound has **Direction**, a **Purpose** selector, and both
**Start conversation** (chat) and **Start voice call** (voice) — the same two-button UX inbound
has. The compliance gate runs server-side either way; a blocked dial disables Start ("Refused
by policy"). Voice plays the customer via the browser mic — no carrier needed to demo the full
conversation.

---

## 5. Data model & config

**Migration `0011_call_scheduling`** adds to `calls`: `scheduled_for`, `attempt`, `purpose`,
`campaign_id`, and the `ix_calls_dispatch` index. Apply with
`docker compose exec core alembic upgrade head`.

**Config** (`app/config.py`): `telephony_provider` (stub default), `telephony_caller_id`,
`telephony_media_stream_base_url`, per-carrier credentials, `telephony_stub_*` (outcome, ring,
talk, cost), and `dialer_*` (`enabled`, `interval_seconds`, `max_concurrent_calls`,
`batch_size`, `max_attempts`, `retry_backoff_minutes`).

**Test/reset script** `core/scripts/unblock_outbound.py` — resolves open escalations (and
optionally clears DNC / resets attempt counters) so a persona is callable for testing, then
re-runs the real gate and prints the verdict. E.g.
`python scripts/unblock_outbound.py --name "Rohan Mehta" --reset-attempts`.

---

## 6. How to test (stub, no carrier)

1. `docker compose exec core alembic upgrade head`
2. `docker compose exec core python scripts/unblock_outbound.py --name "Priya Nair" --reset-attempts`
3. **Conversation** — Console → Outbound → pick a Purpose → pick the customer → Start
   conversation / Start voice call. The agent opens with the identity question; exercise each
   branch (see §4.1).
4. **Dialer lifecycle** — queue a call, then `POST /telephony/tick` (or wait for the loop) and
   watch the `Call` move `queued → ringing → answered → completed`. Force outcomes with the
   number-suffix trick (§3).

---

## 7. PRD compliance

### ✅ Done

| PRD requirement | Where |
|---|---|
| Pre-call gate: DNC, calling hours, retry limits, consent, escalation (§3, §DNC) | `services/outbound.py` + rule sets |
| DNC `priority:0, terminal` — no dialer path skips it | `dsr_calling_policy.yaml` + gate |
| Manual `POST /calls/outbound` through the gate (§9) | `api/calls.py` |
| Campaigns: create / preview / start / pause / stop / targets (§7) | `api/campaigns.py` |
| Per-dial re-check at dial time (§7) | `dialer._originate_one` |
| Escalation `resolve` unlocks customer (§8) | `api/support.py` |
| Outbound in the console — chat + voice, both directions | `ConsolePage.jsx`, `voice_ws.py` |
| **Identity confirmation** "Am I speaking with…?" + all branches (§3.5) | `outbound_flow.py`, `turn_runner.py` |
| **Purpose-driven conversation style** | `outbound_flow.purpose_guidance`, `prompts.py` |
| Provider abstraction (Plivo/Exotel/Twilio), swappable (Voice §2) | `telephony/base.py`, `provider.py` |
| Stub simulates full lifecycle | `telephony/stub.py` |
| `POST /webhooks/telephony/{provider}` + signature verification (§9) | `api/telephony_hooks.py` |
| Dialer worker: concurrency cap + campaign pacing (§7) | `telephony/dialer.py` |
| Retry on busy/no-answer/voicemail + backoff + max attempts (§3.4) | `events._schedule_retry` |
| "Call me later" callback due → dialed (§9) | `dialer._enqueue_due_callbacks` |
| Cost per call recorded (§Cost) | `events.apply_call_event` (stub value) |
| Canonical events → single Call-state writer | `events.apply_call_event` |
| Dashboard events: `call.started`/`call.ended`/`campaign.progress` (§11) | `services/events.publish` |
| Manual dial tick + live dialer status API | `api/telephony.py` |

### ⚠️ Partial / stubbed

| Requirement | Gap |
|---|---|
| Cost & recording from a real carrier | Stub emits synthetic cost + `stub://` recording URL |
| Signature verification / event parsing | Framework in place; real `verify_signature`/`parse_webhook` land with each carrier adapter |

### ❌ Pending (next increments)

| Requirement | Note |
|---|---|
| **Real carrier adapters** (Plivo/Exotel/Twilio `originate`/`parse_webhook`) | Selector raises `NotImplementedError` by design (stub-first). India-primary: **Plivo** |
| **PSTN media bridge** — a real answered call → pipecat via `TwilioFrameSerializer`/equivalent | Only browser-mic `/ws/voice` exists; no `/ws/telephony` accepting carrier media. **The main gap for real voice.** |
| **Answering-machine detection + voicemail message** (§3.4) | Stub marks `voicemail`; no real AMD or leave-message |
| **Recording upload to object storage + signed URL** (§9) | `/calls/{id}/recording` still returns a placeholder |
| **Recorded-call disclosure audio played first** (§3 step 4) | `disclosure_played` column exists, not wired into the voice flow |
| **Per-tenant daily budget: 80% alert, 100% pause campaigns** (§Cost) | Not implemented |
| **Live human transfer / call-leg bridging on escalation** (§Escalation immediate) | Separate feature, not built |

---

## 8. Suggested next steps

1. **Plivo adapter** — implement `originate`, `parse_webhook`, `verify_signature` against the
   existing ABC; wire into `provider._real_adapter`. No other change needed.
2. **`/ws/telephony` media bridge** — accept the carrier's media-stream WebSocket, build the
   same pipecat pipeline + `TurnBridge` as `/ws/voice` with the carrier serializer (µ-law 8k),
   linked to the `Call`. This unlocks real outbound (and inbound) voice.
3. **Recording + disclosure** — pre-generated disclosure audio played first (`disclosure_played`),
   recording upload to object storage, signed URLs on `/calls/{id}/recording`.
4. **Budget guard** — per-tenant daily spend; 80% alert, 100% pauses outbound campaigns
   (inbound/service continue).

---

## 9. File index

| Area | Files |
|---|---|
| Gate | `app/services/outbound.py`, `scripts/dsr_rules/dsr_calling_policy.yaml` |
| Telephony | `app/services/telephony/{base,stub,provider,events,dialer}.py`, `app/api/{telephony,telephony_hooks}.py` |
| Conversation | `app/engine/planner/outbound_flow.py`, `turn_runner.py`, `prompts.py`, `turn_context.py` |
| Channels | `app/api/voice_ws.py`, `app/api/sessions.py` |
| Data | `migrations/versions/0011_call_scheduling.py`, `app/db/models/conversations.py` (Call) |
| Config | `app/config.py` (`telephony_*`, `dialer_*`) |
| Tooling | `scripts/unblock_outbound.py` |
| UI | `voice-agent-ui/src/pages/console/ConsolePage.jsx`, `.../useVoiceCall.js` |
