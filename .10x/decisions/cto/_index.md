# CTO Decisions Index

## Active Features
| Slug | Description | Status |
|------|-------------|--------|
| stage-10-plivo | Plivo telephony adapter | In progress |

## Cross-cutting Tech Decisions [DISCOVERED]
- Python 3.11 (pinned in pyproject/Dockerfile — PRD says 3.12 but actual code uses 3.11)
- FastAPI async throughout
- Provider abstraction pattern: every external service has a `base.py` ABC, a `stub.py`, and a `provider.py` that reads config. Real adapters added as new files.
- `httpx` is the HTTP client for all external APIs (OpenAI, embeddings, etc.)
- No SDK dependencies where the REST API is simple enough to call directly
