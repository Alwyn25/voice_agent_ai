# DevOps Decisions Index [DISCOVERED]

## Infrastructure
- Docker Compose: pgvector/pg16, redis:7, core service
- Deployment: Azure Ubuntu VM
- CI/CD: not yet configured in this repo

## Stage 10 Deployment Notes
- Add `PLIVO_AUTH_ID` + `PLIVO_AUTH_TOKEN` + `TELEPHONY_PROVIDER=plivo` to production .env
- Add `TELEPHONY_WEBHOOK_URL=https://<your-domain>/webhooks/telephony/plivo` to production .env
- Configure Plivo dashboard to POST call events to that webhook URL

## Active Features
| Slug | Description | Status |
|------|-------------|--------|
| stage-10-plivo | No infra changes; env vars only | Pending |
