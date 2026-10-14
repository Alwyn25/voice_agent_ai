# AI Voice Sales & Customer Lifecycle Agent Platform

Monorepo:

- `core/` — Core Platform (FastAPI): engine + business services + APIs
- `voice/` — Voice Session Service (from Stage 10)
- `workers/` — Background workers (from Stage 7)
- `packages/shared/` — Shared schemas (the frozen process_turn contract)
- `dashboard/` — React admin (separate track)

## Quick start (Docker)

```
cp .env.example .env
docker compose up --build
curl http://localhost:8000/health/deep
```



## Quick start (local, without Docker for core)

```
docker compose up -d postgres redis
cd core && python -m venv .venv && .venv\Scripts\activate   (Windows)
pip install -r requirements.txt -e ../packages/shared
alembic upgrade head
uvicorn app.main:app --reload
```

