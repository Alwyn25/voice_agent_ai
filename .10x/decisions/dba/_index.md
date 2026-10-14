# DBA Decisions Index [DISCOVERED]

## Schema State
- 34 tables, 4 migrations (0001-0004), SQLAlchemy 2.0 async
- UUID PKs everywhere, all timestamps TZ-aware, `tenant_id` on every business table
- `calls` table: has `provider`, `provider_call_id`, `recording_url`, `cost`, `duration_seconds` — all used by Stage 10
- pgvector (HNSW + GIN index on embeddings)

## Stage 10: No DB Changes
The `calls` table already has every column Stage 10 needs. No migration required.

## Active Features
| Slug | Description | Status |
|------|-------------|--------|
| stage-10-plivo | No schema changes | N/A |
