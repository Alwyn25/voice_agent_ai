# Product Manager Decisions Index

## Cross-cutting [DISCOVERED]
- **Product:** AI Voice Sales & Customer Lifecycle Agent for real estate (DSR, Lemontree tenants)
- **Users:** Sales supervisors (campaigns), agents (calls), analysts (reports), admins
- **Current capability gap:** System can simulate calls end-to-end with the stub but cannot place real PSTN calls. Stage 10 closes this gap.
- **Success criteria for Stage 10:** `telephony_provider=plivo` in .env with real credentials → the dialer originates an actual PSTN call, the webhook updates call state, recordings are stored.

## Active Features
| Slug | Description | Status |
|------|-------------|--------|
| stage-10-plivo | First real carrier adapter — Plivo | In progress |
