# Security: Stage 10 — Plivo Adapter

**Feature:** stage-10-plivo  
**Date:** 2026-09-30

## Threat Model

### Webhook Spoofing (HIGH)
Plivo webhook at `/webhooks/telephony/plivo` mutates Call state. Without signature verification,
any attacker who knows the URL can forge call events (marking calls as answered, completed, etc.).

**Mitigation:** `verify_signature()` must be implemented correctly using Plivo V2 HMAC-SHA256.
Must use constant-time compare (`hmac.compare_digest`) to prevent timing attacks.
The existing `telephony_hooks.py` already returns 403 on verification failure.

### Credential Exposure (HIGH)
`plivo_auth_id` and `plivo_auth_token` are carrier credentials.

**Mitigation:** Already in `Settings` via pydantic-settings / `.env`. Never logged. ✓

### Recording URL Access (MEDIUM)
Plivo webhook includes a `RecordUrl` pointing to a recording file. This URL is stored on
the `call.recording_url` column and accessed via `GET /calls/{id}/recording` (audited).

**Mitigation:** Already implemented in Stage 9 — recording access is audited. ✓
Plivo recording URLs are pre-signed and time-limited. No additional action needed.

### SSRF via RecordUrl (LOW)
If the system ever proxies recording content by fetching the `RecordUrl`, that URL is
attacker-controlled if webhook spoofing succeeds. Currently the URL is only stored, not fetched.

**Mitigation:** Do not proxy recording content. Store and return the URL only. ✓

## Auth Review
- `originate()` uses HTTP Basic auth (auth_id + auth_token) — standard Plivo pattern ✓
- `verify_signature()` uses HMAC-SHA256 with `hmac.compare_digest` — constant-time ✓
- Webhook route has no JWT requirement (correct — carriers cannot present JWT) ✓
- The 403 path in `telephony_hooks.py` is already in place ✓

## No New Permissions Needed
Stage 10 adds no new API routes. The existing `calls:write` / `calls:read` permissions cover it.
