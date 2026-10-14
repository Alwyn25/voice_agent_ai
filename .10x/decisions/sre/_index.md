# SRE Decisions Index [DISCOVERED]

## Observability [DISCOVERED]
- Structured logging throughout (standard Python `logging`)
- `app/observability/` — `pricing.py`, `tool_metrics.py`, `queries.py`
- Profile scripts: `profile_turns.py`, `profile_bridge.py`
- Dashboard WebSocket for live events

## Stage 10 SRE Notes
- Plivo call failures appear in `calls_by_status` via `GET /telephony/status`
- `apply_call_event` publishes `call.ended` with `connected` flag — feeds `GET /analytics/overview`
- Failed originates logged at WARNING level (already in `dialer.py`)
- No new SLOs required for Stage 10 — existing call analytics cover it

## Active Features
| Slug | Description | Status |
|------|-------------|--------|
| stage-10-plivo | No new monitoring; existing analytics cover it | N/A |
