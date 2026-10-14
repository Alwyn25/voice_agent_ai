# Staff Engineer Decisions Index [DISCOVERED]

## Codebase Patterns
- **Provider pattern:** `base.py` (ABC) + `stub.py` + `provider.py` (factory) + adapter files. Used for LLM, embeddings, telephony, WhatsApp.
- **Error handling:** Providers return result objects (never raise for expected failures). `TelephonyError` / `MessagingError` are reserved for programming faults.
- **Config:** `pydantic-settings` from `.env`. All credentials blank by default; adapters fail loudly if credentials are missing when required.
- **Async:** All DB and HTTP calls are async. `httpx.AsyncClient` for HTTP. SQLAlchemy async sessions.
- **Single writer:** Call lifecycle state changes → `apply_call_event()` only. Never mutate `call.status` elsewhere.
- **Tests:** `scripts/` contains integration tests; unit tests presumably in `tests/` (not yet created for telephony).

## Active Features
| Slug | Description | Status |
|------|-------------|--------|
| stage-10-plivo | Plivo adapter | In progress |
