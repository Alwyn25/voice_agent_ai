# Engineering Manager: Stage 10 — Plivo Adapter Task Breakdown

**Feature:** stage-10-plivo  
**Date:** 2026-09-30  
**Estimate:** 3-4 hours total

## Task List (ordered)

### Task 1: `PlivoTelephonyProvider.originate()` — ~45 min
- Create `core/app/services/telephony/plivo.py`
- Implement `__init__` with `httpx.AsyncClient(auth=(auth_id, auth_token), base_url=...)`
- Implement `originate()`: POST `/Call/`, map response to `OriginateResult`
- Handle all HTTP errors as `ok=False` (never raise)

### Task 2: `PlivoTelephonyProvider.hangup()` — ~15 min
- DELETE `/Call/{provider_call_id}/`
- Best-effort: swallow all exceptions, log warning

### Task 3: `PlivoTelephonyProvider.verify_signature()` — ~30 min
- Read Plivo's V2 signing spec
- Implement HMAC-SHA256: `HMAC(auth_token, url + nonce + body_hash)`
- Compare to `X-Plivo-Signature-V2` header (constant-time compare)

### Task 4: `PlivoTelephonyProvider.parse_webhook()` — ~30 min
- Map Plivo webhook params to `CallEvent`
- Status mapping table (ringing → ringing, completed → completed, etc.)
- Extract `CallUUID`, `HangupCause`, `RecordUrl`, `Duration`, `TotalCost`

### Task 5: Wire into `provider._real_adapter()` — ~10 min
- Add `from app.services.telephony.plivo import PlivoTelephonyProvider`
- Replace `raise NotImplementedError` for "plivo" case

### Task 6: Unit tests — ~60 min
- `tests/telephony/test_plivo.py`
- Test `originate()` success + error cases (mock httpx)
- Test `verify_signature()` with known HMAC vectors
- Test `parse_webhook()` for all status mappings

### Task 7: End-to-end test — ~20 min
- Extend `run_conversation_tests.py` or add a new smoke test
- Verify dialer → Plivo adapter → simulated webhook → apply_call_event

## Dependencies
- Task 5 depends on Tasks 1-4
- Task 6 can run in parallel with Task 5
- Task 7 depends on Task 5+6

## Risk
- Plivo V2 signature spec: the nonce and URL normalization are fiddly. Allocate extra time if tests fail.
- `httpx.AsyncClient` lifecycle: must close the client properly (use `__aenter__`/`__aexit__` or a module-level client closed at app shutdown).
