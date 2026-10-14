# Handoff

## Current Handoff
**From:** SDE  
**To:** QA Engineer + Security Engineer  
**Feature:** stage-10-plivo  
**Phase:** 4 (Implementation) → 5 (Verification)

### What was built
See `.10x/decisions/sde/stage-10-plivo.md` for full details.

Summary:
- `core/app/services/telephony/plivo.py` — `PlivoTelephonyProvider` (originate, hangup, verify_signature, parse_webhook)
- `core/app/config.py` — added `telephony_webhook_url` field
- `core/app/services/telephony/provider.py` — wired plivo into `_real_adapter`
- `core/app/api/telephony_hooks.py` — fixed form-body parsing bug (was `request.json()`, now content-type-aware)
- `core/tests/telephony/test_plivo.py` — 16 unit tests, all passing

### What was added in this continuation
- `telephony_hooks.py` now returns Plivo `<Stream>` XML for answered events (so Plivo
  knows to open an audio WebSocket to `/ws/plivo/{call_uuid}`)
- `core/app/api/plivo_ws.py` — new file; the pipecat pipeline for Plivo calls
  with `PlivoFrameSerializer` (mulaw 8kHz JSON ↔ PCM 16kHz AudioRawFrame)
- `core/app/main.py` — `plivo_ws_router` registered

### What to test (QA)
1. Run `pytest core/tests/telephony/test_plivo.py -v` — must be 16/16 green
2. Set these in `.env`:
   - `TELEPHONY_PROVIDER=plivo`
   - `PLIVO_AUTH_ID=...`, `PLIVO_AUTH_TOKEN=...`
   - `TELEPHONY_WEBHOOK_URL=https://unbutton-rekindle-ellipse.ngrok-free.dev/webhooks/telephony/plivo`
   - `TELEPHONY_MEDIA_STREAM_BASE_URL=wss://unbutton-rekindle-ellipse.ngrok-free.dev/ws/plivo`
   - `TELEPHONY_CALLER_ID=+1XXXXXXXXXX`
3. Run: `ngrok http 8000 --url https://unbutton-rekindle-ellipse.ngrok-free.dev`
4. Start the app; `POST /telephony/tick` → should dial a queued call via Plivo API
5. When Plivo answers:
   - Logs show `apply_call_event` (status=answered) + `plivo call started`
   - The webhook returns XML (visible in Plivo dashboard's "call flow" trace)
   - Plivo opens WebSocket to `/ws/plivo/{call_uuid}`
   - The opening greeting TTS audio plays on the phone
6. Full round-trip: caller speaks → STT → TurnBridge → planner → TTS → audio on phone

### Security checkpoints
1. `verify_signature()` — confirm `hmac.compare_digest` prevents timing attacks
2. `telephony_webhook_url` must be HTTPS in production (Plivo rejects HTTP for V2 signatures)
3. Credentials never logged — confirm `plivo_auth_id`/`plivo_auth_token` don't appear in logs
4. `/ws/plivo/{call_uuid}` has no auth — acceptable because call_uuid is a random Plivo UUID;
   consider adding a short-lived nonce or IP allowlist for Plivo's WebSocket servers in prod

---

## Handoff History

### SDE → QA (initial state bootstrap)
Context captured during Discovery Protocol scan of Stages 1–9. `.10x/` bootstrapped from codebase.
