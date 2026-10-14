# Security Decisions Index

## Active Features
| Slug | Description | Status |
|------|-------------|--------|
| stage-10-plivo | Webhook HMAC verification, credential handling | Review complete |

## Cross-cutting Security Notes [DISCOVERED]
- All credentials in `.env` via pydantic-settings — never hardcoded
- JWT for API auth (30m access / 7d refresh)
- API keys (SHA256 hashed, scoped)
- Principal dependency (`require(permission)`) on all protected routes
- Recording access is audited
- Webhook routes are not behind JWT (carriers cannot present tokens) — secured by provider signature
