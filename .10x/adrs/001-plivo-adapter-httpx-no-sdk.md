# ADR-001: Use httpx directly for Plivo API — no Plivo Python SDK

**Status:** Accepted  
**Date:** 2026-09-30  
**Feature:** stage-10-plivo  
**Author:** 10x-Team (Architect + CTO)

## Context
Stage 10 requires implementing the Plivo carrier adapter. Plivo provides an official
Python SDK (`plivo`). The project already has `httpx==0.28.1` in requirements.txt, used
for OpenAI and other provider calls. The Plivo REST API surface needed here is three endpoints:
POST /Call/, DELETE /Call/{uuid}/, and HMAC-SHA256 signature verification (no HTTP call).

## Decision
Call Plivo's REST API directly using `httpx.AsyncClient` with HTTP Basic auth
(`plivo_auth_id` + `plivo_auth_token`). Do not add the `plivo` SDK package.

## Alternatives Considered

| Alternative | Pros | Cons | Why Not |
|-------------|------|------|---------|
| `plivo` Python SDK | Official, auto-updated with API changes | Adds a dependency; the SDK is synchronous (wraps `requests`), requiring `asyncio.run_in_executor` or a thread — not consistent with the rest of the codebase | Adds friction, is not async-native |
| `httpx` directly (chosen) | Already in project, async-native, consistent with all other provider calls | Must handle Plivo API details manually | The API surface is tiny; 3 endpoints, well-documented |
| `aiohttp` | Async | Not in project; adds a dep | Redundant with httpx |

## Consequences

### Positive
- Zero new dependencies
- Async-native, consistent with all other HTTP calls in the codebase
- Easy to test: mock `httpx.AsyncClient.post` exactly as done elsewhere

### Negative
- If Plivo changes its API significantly, we update our code rather than upgrading the SDK
- HMAC signature logic must be written manually (trivial: ~5 lines of Python)

### Risks
- Signature verification correctness: mitigated by unit tests against known Plivo test vectors

## Dependencies
- Requires `plivo_auth_id` and `plivo_auth_token` in Settings (already defined in `config.py`)
- Plivo API base URL: `https://api.plivo.com/v1/Account/{auth_id}/`
