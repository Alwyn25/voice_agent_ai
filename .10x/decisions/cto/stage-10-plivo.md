# CTO: Stage 10 — Plivo Adapter

**Feature:** stage-10-plivo  
**Date:** 2026-09-30  
**Status:** [DISCOVERED → CONFIRMED]

## Decision
Build the Plivo carrier adapter. All infrastructure is in place (abstraction, dialer, webhooks,
stub, events, retries, policy gate). Only the concrete Plivo HTTP calls are missing.

## Build vs Buy
Not applicable — this is a first-party adapter to an already-chosen third-party carrier.
The carrier selection (Plivo as primary for India) is documented in the PRD.

## Strategic Context
- Plivo is the primary carrier for India (pricing, DID availability).
- Exotel and Twilio adapters are named in the PRD but not required for Stage 10.
- Stage 10 is the "first stage where a real phone rings" — the commercial viability gate.

## Scope Decision
Narrow: PlivoTelephonyProvider only. Do not implement Exotel or Twilio in this stage.
The abstraction means adding them later is a new file + a branch in `_real_adapter`, nothing else.

## Tech Direction
Use `httpx` (already in requirements.txt) for Plivo's REST API.
Do NOT add the `plivo` Python SDK — it wraps httpx anyway, adds a dependency, and the
Plivo REST API is small enough to call directly. The SDK adds no value here.
