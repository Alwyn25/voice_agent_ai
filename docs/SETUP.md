# Running this locally

Everything runs in Docker. You need Docker Desktop and an OpenAI key with credit on it.

---

## 1. Configure

```bash
cp .env.example .env
```

Fill in the two lines marked REQUIRED:

- `JWT_SECRET` — any long random string. `python -c "import secrets; print(secrets.token_urlsafe(48))"`
- `OPENAI_API_KEY` — **must have credit**. An empty balance returns 429 on every turn and
  the agent will look broken rather than out of money.

Leave the rest alone. The database hostnames are compose service names, not `localhost` —
inside the core container, `localhost` is the container itself.

## 2. Start

```bash
docker compose up -d --build
docker compose logs core
```

Wait for `Uvicorn running on http://0.0.0.0:8000`. Migrations run first, so the API is not
listening for a few seconds after the container starts — **commands chained straight onto
`up` with `&&` will lose that race** and fail with "connection refused".

## 3. Seed

Four commands, in this order. Each depends on the one before it.

```bash
# Tenant, admin login, tool contracts, state machine, permissions.
docker compose exec core python scripts/seed.py \
  --tenant "DSR Infra Projects" --email you@example.com --password changeme \
  --reseed-tools
```

Expect **15 tool definitions**. Fewer means a contract failed to insert — the script names
which one and why.

```bash
# The real-estate demo: rule packs, centres, working hours, eight leads across the journey.
docker compose exec core python scripts/seed_dsr.py

# The knowledge base: 16 documents from the DSR brochure and cost sheets.
docker compose exec core python scripts/seed_knowledge_dsr.py

# Puts the demo leads into known states and clears anything left from earlier runs.
docker compose exec core python scripts/reset_visits.py
```

Expect **zero** documents flagged as not ingested. Anything reported there is invisible to
the agent, and a pricing question against a missing cost sheet will look like a reasoning
failure when it is a data one. Usually a timeout — re-run it, it is idempotent.

## 4. The frontend

Separate repository. It expects the API on `127.0.0.1:8000`, which is where compose puts
it.

```bash
npm install
npm run dev
```

Sign in with the email and password you passed to `seed.py`.

---

## Demo data

Eight leads, one per journey stage, so every rule path has someone to exercise it:

| | State | Why they exist |
|---|---|---|
| Vikram Singh | negotiation | VIP, has complained — escalation paths |
| Rohan Mehta | lead | the ordinary happy path |
| **Kavya Menon** | **do_not_contact** | **every outbound attempt must be refused** |
| Priya Nair | site_visit_done | post-visit follow-up |
| Sanjay Gupta | qualified | ready to book |
| Meera Iyer | dormant | re-engagement |
| Aditya Verma | site_visit_scheduled | has a booking — reschedule and cancel |
| Nisha Pillai | contacted | evening calls only |

Kavya is the important one. Any outbound attempt to her should be refused with the rule
named. If it is not, compliance is broken and nothing else matters.

## Re-seeding after a change

| You changed | Run |
|---|---|
| a tool contract, `allowed_states`, preconditions | `seed.py … --reseed-tools` |
| a rule YAML under `scripts/dsr_rules/` | `seed_dsr.py --only rules` |
| a knowledge document | `seed_knowledge_dsr.py` |
| Python only | `docker compose up -d --build core` |

Contracts and rules are **database rows**. Editing the file and rebuilding is not enough —
the change compiles, the behaviour does not move, and it looks like the fix did not work.

## Known limitations

- **Telephony is not built.** Conversations run over the chat console; there is no STT,
  TTS or Plivo integration.
- **Turns take 2 to 9 seconds.** Each is two to four model calls at roughly 1.3s of fixed
  overhead from India. Acceptable for chat, too slow for voice without streaming.
- **English only.** The architecture supports other languages; the guards are English
  regexes and Hindi is untested.
- The brochure and cost sheets are transcribed into `seed_knowledge_dsr.py` by hand. There
  is no PDF ingestion.
