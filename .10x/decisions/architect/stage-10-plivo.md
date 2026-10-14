# Architect: Stage 10 — Plivo Adapter

**Feature:** stage-10-plivo  
**Date:** 2026-09-30

## Component Design

### New file: `core/app/services/telephony/plivo.py`
One class: `PlivoTelephonyProvider(TelephonyProvider)`.

```
PlivoTelephonyProvider
  name = "plivo"
  
  __init__(settings: Settings)
    → stores auth_id, auth_token, async httpx.AsyncClient
  
  async originate(req: DialRequest) → OriginateResult
    → POST https://api.plivo.com/v1/Account/{auth_id}/Call/
    → body: from_, to, answer_url (media stream or IVR), time_limit
    → on 201: return OriginateResult(ok=True, provider_call_id=response["request_uuid"])
    → on 4xx/5xx: return OriginateResult(ok=False, error=...)
  
  async hangup(provider_call_id: str) → None
    → DELETE https://api.plivo.com/v1/Account/{auth_id}/Call/{provider_call_id}/
    → best-effort, swallow non-fatal errors
  
  verify_signature(headers, body) → bool
    → Plivo uses X-Plivo-Signature-V2 + X-Plivo-Signature-V2-Nonce
    → HMAC-SHA256(auth_token, full_url + nonce + body) → compare to header
    → See: https://www.plivo.com/docs/sms/api/overview/signing-plivo-requests
  
  parse_webhook(payload: dict) → CallEvent | None
    → Map Plivo event params to canonical CallEvent
    → Key params: CallUUID, CallStatus (ringing/answered/completed/busy/failed/no-answer),
      HangupCause, RecordUrl, Duration, TotalCost, BillDuration
```

### Modified file: `core/app/services/telephony/provider.py`
Replace the `NotImplementedError` in `_real_adapter()` for the "plivo" case.

### No other files change.
- `dialer.py`: unchanged — calls `provider.originate(req)`, provider-agnostic
- `events.py`: unchanged — `apply_call_event` is canonical
- `telephony_hooks.py`: unchanged — routes to the configured provider's `verify_signature` + `parse_webhook`
- `config.py`: Plivo credentials already defined (`plivo_auth_id`, `plivo_auth_token`)

## Data Flow

```
dialer.py
  → _originate_one(call, provider)
  → PlivoTelephonyProvider.originate(DialRequest)
  → POST api.plivo.com → 201 → OriginateResult(ok=True, provider_call_id=uuid)
  → call.provider = "plivo", call.provider_call_id = uuid, call.status = "ringing"

[Plivo dials the number]

Plivo POSTs to /webhooks/telephony/plivo
  → telephony_hooks.py: verify_signature → parse_webhook → CallEvent
  → apply_call_event(db, call, event) → status transitions, retries, analytics
```

## Plivo-specific Mapping

| Plivo `CallStatus` | Canonical status |
|---|---|
| `ringing` / `in-progress` | `ringing` |
| `answered` | `answered` |
| `completed` | `completed` |
| `busy` | `busy` |
| `failed` | `failed` |
| `no-answer` | `no_answer` |
| `machine` | `voicemail` |

## Failure Modes
- Plivo credentials missing: `TelephonyError` raised at provider construction (already in `_real_adapter`)
- Plivo 4xx on originate: `OriginateResult(ok=False)` — dialer records and moves on
- Plivo 5xx / timeout: same — never raises, always returns
- Webhook with wrong signature: 403 (already in `telephony_hooks.py`)
- Recording URL in webhook: stored on `call.recording_url` via `apply_call_event` (already wired)
