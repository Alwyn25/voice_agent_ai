# Senior Engineer Decisions Index

## Active Features
| Slug | Description | Status |
|------|-------------|--------|
| stage-10-plivo | Plivo adapter | Approach complete |

## Cross-cutting Implementation Notes [DISCOVERED]
- httpx.AsyncClient held at provider construction time (cached in `_cache`)
- All HTTP calls are async; never use `requests` or `urllib` in service layer
- `hmac.compare_digest` for all HMAC comparisons (constant-time, security requirement)
- Form-encoded webhook bodies: `telephony_hooks.py` needs a content-type check for Plivo
