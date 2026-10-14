# Senior Engineer: Stage 10 — Plivo Adapter Approach

**Feature:** stage-10-plivo  
**Date:** 2026-09-30

## Implementation Guidance

### Reference: `stub.py` + `base.py`
These are the contract. `PlivoTelephonyProvider` must satisfy `TelephonyProvider`'s ABC
exactly as `StubTelephonyProvider` does. Read both before writing a line.

### httpx.AsyncClient lifecycle
The existing providers (e.g. `app/providers/llm/openai_provider.py`) create an `httpx.AsyncClient`
at construction time and hold it. Follow the same pattern. The provider is cached in `_cache` in
`provider.py`, so the client lives for the process lifetime — **do not** create a new client per call.

Register cleanup in `app/main.py` lifespan if the client needs explicit close (check whether
the LLM providers do this; if they do, add Plivo to the same cleanup list).

### Signature verification — Plivo V2
Plivo's V2 signing (recommended since 2022):
1. Compute HMAC-SHA256 of `full_callback_url + nonce + sha256_hex(body)`
   where `nonce` = value of `X-Plivo-Signature-V2-Nonce` header
2. Base64-encode the HMAC digest
3. Compare (constant-time) against `X-Plivo-Signature-V2` header

The `full_callback_url` is this server's own URL for the webhook endpoint. It must match
exactly what Plivo has configured. Add `telephony_webhook_base_url: str = ""` to Settings
for production; fall back to building it from request if the operator doesn't set it.

Actually — looking at `telephony_hooks.py`, it doesn't pass the URL to `verify_signature`.
The signature contract is `verify_signature(headers, body) -> bool`. To get the URL, we need
access to the request. Two options:
  a. Add `url: str = ""` to `verify_signature` signature (breaking change to ABC)  
  b. Store the expected URL on the provider at construction time from config
Go with (b): add `telephony_webhook_url: str = ""` to `Settings` and have Plivo read it
in `__init__`. The stub's `verify_signature` accepts everything so no ABC break is needed.

### parse_webhook — Plivo event params
Plivo sends form-encoded POST, not JSON. FastAPI's webhook handler reads `.json()` but
Plivo sends `application/x-www-form-urlencoded`. **Important**: check what
`telephony_hooks.py` does with the body. It calls `await request.json()`. For Plivo
this will fail — need to use `await request.form()` when the provider is Plivo.

**Actually** — re-reading `telephony_hooks.py`:
```python
try:
    payload = await request.json()
except Exception:
    payload = {}
```
It already handles the JSON failure with an empty dict fallback. But then `payload` would be `{}`.
This means `parse_webhook(payload={})` would receive nothing useful.

**Fix needed in `telephony_hooks.py`**: parse body as form if content-type is `application/x-www-form-urlencoded`.

Alternative: in `parse_webhook`, accept raw `body: bytes` and parse it ourselves. But the ABC says `payload: dict`. The cleanest fix is to update `telephony_hooks.py` to detect content type and provide the correct dict to `parse_webhook`. This is a bug in the existing code that Stage 10 surfaces — it was designed for JSON providers first.

### Error handling pattern (from dialer.py)
```python
result = await provider.originate(req)
if not result.ok:
    call.status = STATUS_FAILED
    # ...
```
The provider NEVER raises for an expected failure. Only `TelephonyError` for genuine faults.
Follow this contract strictly in PlivoTelephonyProvider.
