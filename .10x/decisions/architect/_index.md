# Architect Decisions Index

## Active Features
| Slug | Description | Status |
|------|-------------|--------|
| stage-10-plivo | Plivo telephony adapter | Design complete |

## Cross-cutting Architecture [DISCOVERED]
- Provider abstraction: every external service = `base.py` ABC + `stub.py` + `provider.py`
- All call lifecycle state changes flow through `apply_call_event` — single writer pattern
- No new endpoints needed for Stage 10; webhook router already handles `POST /webhooks/telephony/{provider}`
- Signature verification is the provider's responsibility; the webhook route delegates to it
- All providers are carrier-neutral from the dialer's perspective
