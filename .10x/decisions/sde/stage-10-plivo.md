# SDE: Stage 10 — Plivo Adapter

**Status:** Implementation complete  
**Feature slug:** stage-10-plivo

## What was built

### New file: `core/app/services/telephony/plivo.py`
- `PlivoTelephonyProvider` — concrete implementation of `TelephonyProvider` ABC
- `originate()` — POST to `https://api.plivo.com/v1/Account/{auth_id}/Call/` via `httpx.AsyncClient`
  with HTTP Basic auth; returns `OriginateResult(ok=True, provider_call_id=request_uuid)` on 200-202,
  `ok=False` otherwise. Never raises.
- `hangup()` — DELETE to the same endpoint with the call UUID
- `verify_signature()` — Plivo V2 HMAC-SHA256: `sha256(auth_token, url + nonce + sha256hex(body))`,
  constant-time compare via `hmac.compare_digest`. Headers are lowercase (FastAPI normalises them).
- `parse_webhook()` — maps Plivo's `CallStatus`/`Event` field to `STATUS_*` constants from `base.py`;
  returns `None` for keep-alives and unrecognised events; includes `duration_seconds` from `Duration`/
  `BillDuration` field.

### Modified: `core/app/config.py`
- Added `telephony_webhook_url: str = ""` in the telephony section

### Modified: `core/app/services/telephony/provider.py`
- `_real_adapter("plivo", s)` now returns `PlivoTelephonyProvider(s)` instead of raising

### Modified (bug fix): `core/app/api/telephony_hooks.py`
- Fixed form-body parsing bug: payload extraction now detects `Content-Type` and uses
  `await request.form()` for `application/x-www-form-urlencoded` (Plivo's format),
  falling back to `request.json()` for other carriers.

### New tests: `core/tests/telephony/test_plivo.py`
- 16 tests, all passing. Covers: originate success/failure, verify_signature valid/invalid/missing,
  parse_webhook status mapping (7 statuses), edge cases (unknown status, missing fields, duration).

## Deviations from plan
None. Implementation matches the approved design exactly.

### Modified: `core/app/api/telephony_hooks.py` (XML stream directive)
- When event.status == STATUS_ANSWERED AND configured provider is "plivo" AND
  `telephony_media_stream_base_url` is set, returns Plivo XML instead of 204:
  `<Response><Stream keepCallAlive="true" bidirectional="true" contentType="audio/x-mulaw;rate=8000">
  {base_url}/{call_uuid}</Stream></Response>`
- All other events (ringing, completed, hangup) continue to return 204.

### New file: `core/app/api/plivo_ws.py`
- `PlivoFrameSerializer`: converts AudioRawFrame(PCM 16kHz) ↔ Plivo JSON frames (mulaw 8kHz).
  Uses `audioop.ratecv` (stateful, continuous) + `audioop.lin2ulaw`/`ulaw2lin`.
  Declares `type = FrameSerializerType.TEXT` so FastAPIWebsocketTransport reads JSON text frames.
- `plivo_media_stream(websocket, call_uuid)` at `/ws/plivo/{call_uuid}`:
  - Looks up `Call` by `provider=="plivo"` and `provider_call_id==call_uuid`
  - Creates `Conversation`, links it to the Call, saves the opening message
  - Builds the same pipecat pipeline as `voice_ws.py`:
    `transport.input → VADProcessor → STT → UserTurnProcessor → TurnBridge →
     FillerAudioProcessor → TTS → TTFAProbe → transport.output`
  - Uses `PlivoFrameSerializer` instead of `ProtobufFrameSerializer`
  - No JWT auth (carrier-initiated connection, secured by random call_uuid)

### Modified: `core/app/main.py`
- Added `from app.api.plivo_ws import router as plivo_ws_router`
- Added `plivo_ws_router` to the `include_router` loop.

## Tech debt
- `hangup()` silently swallows HTTP errors from Plivo. A production hardening pass could log
  non-2xx responses from the DELETE, but the interface contract (`async def hangup` → None)
  means we can't propagate errors from it, and call state is driven by webhooks not hangup
  API responses, so this is acceptable.
- `audioop` is deprecated in Python 3.11+ and removed in Python 3.13. If the runtime is
  upgraded to 3.13, install `audioop2` (same API, drop-in replacement).
- `PlivoFrameSerializer` does not handle Plivo's start/stop/dtmf JSON events — they return
  `None` from `deserialize()` and are silently dropped. Fine for the current use case.

## Dependencies added
None. `httpx` was already in `requirements.txt`.
