# Bridge latency & event-loop blocking (P1.08)

**Task:** Profile the bridge (BRG ≈ 60 ms) stage and remove blocking calls from the
asyncio path.

## What "the bridge" is here

The latency benchmark report models a turn as a waterfall:

```
VAD → STT → BRIDGE → Planner → Tool → Responder → TTS → Transport
                ▲
        "frame relay + WS hop", ~60 ms
```

The Excel task list points this task at `core/app/engine/memory/pipeline.py`, but that
file is the **post-call** memory pipeline — it runs *after* a call ends and is not on the
live turn path. The bridge overhead actually lives inside **`process_turn`**
(`core/app/engine/planner/turn_runner.py`): the per-turn orchestration that is *not*
planner/tool/responder/retrieval work —

1. `session_load` — read working memory from Redis
2. `input_persist` — write the customer's message + flush
3. `customer_load` — load the customer row
4. `trace_persist` — write the decision trace + flush
5. `session_save` — write working memory back to Redis
6. `commit` — the transaction commit

`scripts/profile_turns.py` already saw this as an unexplained "other / gap" bucket. P1.08
breaks it open and adds a detector for the specific failure the task names — a synchronous
call stalling the event loop.

## What was added

| Piece | File | Purpose |
| --- | --- | --- |
| `StageTimer` | `app/observability/stage_timer.py` | Times the six bridge sub-stages; adds `bridge_*_ms`, a `bridge_ms` (their direct sum) and `unattributed_ms` (the honest "not instrumented" leftover) to `debug.timing`. |
| `LoopMonitor` | `app/observability/loop_monitor.py` | Background sampler that flags anything blocking the event loop and names the in-flight task. |
| Instrumentation | `turn_runner.py` (`process_turn`) | Six `async with _bridge.stage(...)` wrappers; `_with_bridge_timing()` assembles the payload. |
| Startup wiring | `app/main.py` (`lifespan`) | Starts the monitor at boot (when enabled) and stops it on shutdown, restoring loop settings. |
| Startup warm-up | `app/main.py` (`_warm_up_io`) | Primes the DB pool, ORM statement cache, and embeddings client at boot so the first live turn isn't cold (`STARTUP_WARMUP_ENABLED`). |
| Health endpoint | `GET /health/loop` | Exposes loop stats (max lag, spike count) over HTTP. |
| Profiler | `scripts/profile_bridge.py` | Bridge breakdown over HTTP + in-process cProfile of `process_turn`. |
| Verifier | `scripts/verify_bridge_observability.py` | Self-contained proof both tools work (CI-safe). |

Everything is **additive and config-gated** — with the flags off, behaviour and the
response shape are byte-for-byte unchanged.

## Configuration (`.env`)

```bash
# Break the bridge out into sub-stages on debug.timing (cheap; on by default)
BRIDGE_TIMING_ENABLED=true

# Warm the DB pool + embeddings client at boot so the first live turn isn't cold (on by default)
STARTUP_WARMUP_ENABLED=true

# Event-loop blocking detector (cheap; safe to leave on in production)
LOOP_MONITOR_ENABLED=true
LOOP_MONITOR_INTERVAL_MS=50      # sampling rate
LOOP_MONITOR_WARN_MS=100         # lag above this logs a stall

# Deep mode: asyncio debug + slow-callback logging names the EXACT blocking callsite.
# Higher overhead — turn on only while investigating.
LOOP_MONITOR_DEEP=true
LOOP_MONITOR_SLOW_CALLBACK_MS=50
```

## How to use it

### 1. See where the 60 ms goes

```bash
docker compose exec core python scripts/profile_bridge.py --runs 8
```

Prints, per turn shape, the `total`, `bridge`, and each sub-stage (median + p95), then
ranks the sub-stages. On a cloud-Postgres stack most of the bridge is **network RTT**
(Redis `GET`/`SET` + two Postgres flushes + the commit), ~15 ms base each — that is
latency, not a blocked loop, and is addressed by connection pooling / co-locating the DB,
not by `to_thread`.

Every turn also carries the numbers in its API response under `debug.timing`:

```json
"timing": {
  "planner_ms": 150, "tool_ms": 0, "responder_ms": 150, "total_ms": 512,
  "bridge_ms": 61,          // direct SUM of the six bridge_*_ms below (real DB/Redis cost)
  "unattributed_ms": 151,   // wall-clock not in any timed stage — usually an extra model call
  "bridge_session_load_ms": 9, "bridge_input_persist_ms": 12,
  "bridge_customer_load_ms": 8, "bridge_trace_persist_ms": 14,
  "bridge_session_save_ms": 7, "bridge_commit_ms": 11
}
```

> **Reading `bridge_ms` vs `unattributed_ms`.** `bridge_ms` is a *measurement* — the sum of
> the six sub-stages — so it is always the true orchestration overhead (~10–15 ms on this
> stack) and can never be inflated by time we didn't wrap. `unattributed_ms` is the residual
> (`total − planner − tool − responder − retrieval − bridge`): a big value there means a turn
> spent real time in a stage that isn't instrumented yet (e.g. a turn that makes an auxiliary
> LLM call not tagged planner/responder). That is a signal to *wrap that stage*, not to hunt
> for DB latency in the bridge. Earlier iterations folded this leftover into `bridge_ms`,
> which made a chit-chat turn read ~13 ms but a multi-call turn read thousands — the split
> keeps the bridge number honest while never letting the leftover hide.

### 2. Catch a real blocking call

Run the app with `LOOP_MONITOR_ENABLED=true`. If a synchronous call ever stalls the loop,
you get a log line naming the in-flight task:

```
WARNING app.observability.loop: event-loop stalled 214 ms (threshold 100 ms).
        In-flight: Task-42@retrieve:193
```

and `GET /health/loop` shows the running totals:

```json
{"enabled": true, "samples": 128400, "spikes": 3, "max_lag_ms": 214.0, ...}
```

To pin the **exact line**, restart with `LOOP_MONITOR_DEEP=true` and read asyncio's own
message: `Executing <Handle ...> took 0.214 seconds`.

### 3. Find sync hotspots by function (cProfile)

```bash
docker compose exec core python scripts/profile_bridge.py --cprofile 20
```

Profiles `process_turn` in-process with the **LLM mocked** (so model network time doesn't
dominate) and prints the top functions by `tottime` (where a synchronous CPU blocker
ranks) and `cumtime` (the call tree, I/O waits included).

> ⚠️ cProfile mode runs real turns and **writes to the DB** (messages, traces). Point it at
> a demo/throwaway conversation, not production data.

## The post-call pipeline (`memory/pipeline.py`)

The task list names `core/app/engine/memory/pipeline.py`. That file is **not** the TTFA
bridge — it is the **post-call** pipeline (summary + fact extraction), run via
`background.add_task(run_post_call_task, …)` from the `/end` endpoints. But it runs as a
FastAPI **BackgroundTask on the same event loop as live turns**, so a stall there taxes
concurrent calls. It was reviewed and:

- **No blocking call to remove** — every DB and LLM call is already awaited; there is no
  sync file/DB/HTTP or `time.sleep`. The `LoopMonitor` covers it automatically because it
  watches the whole loop.
- **Fixed a real latency pattern:** the fact merge called `await db.flush()` *once per
  fact* (up to `max_facts_per_call` DB round-trips). It now collects the supersede pairs
  and flushes **once**, then wires the `superseded_by` ids — provably equivalent (verified
  by an equivalence simulation), N round-trips → 1.
- **Instrumented** with the same `StageTimer` (`load`, `analyze_llm`, `facts_merge`,
  `audit`, `total_ms`); `run_post_call_task` logs the breakdown at DEBUG and now logs
  exceptions that were previously swallowed silently.

## Reading the result: is it a *block* or just *latency*?

- **High in `tottime`, not obviously async I/O** (a regex loop, `jsonschema` validation, a
  rules evaluation, JSON serialization) → a genuine blocker. Fix by moving it off the loop
  with `asyncio.to_thread(...)`, or by caching the work.
- **High only in `cumtime` / shows as an awaited round-trip** (Redis, Postgres) → latency,
  not a block. Fix with pooling, fewer round-trips, or DB locality — not threading.

The per-turn hot path was already fully async (providers use `AsyncOpenAI`, working memory
uses `redis.asyncio`, DB uses `asyncpg`, event publish is in-memory `put_nowait`), which is
why cProfile shows no steady-state app blocker. The one synchronous blocker that DID exist —
`bcrypt` on the auth path, which stalls *all* in-flight turns whenever a login runs — was
found by the loop monitor and moved off the loop (see the DEEP-mode result above). The
lasting deliverable is the **instrumentation that caught it and will catch the next one**.
Re-run the two scripts after any hot-path change.

## Findings from the first profiling run

Measured on the containerised stack (real Redis + Railway Postgres, real LLM for HTTP mode,
LLM mocked for cProfile):

- **The bridge is ~11–15 ms, not 60 ms.** Sub-stage medians: `commit` 6, `trace_persist` 5,
  `session_save` 4, `input_persist` 2, `customer_load` 1, `session_load` 0. Every one is an
  already-awaited DB/Redis round-trip. The report's 60 ms was a top-down estimate; bottom-up
  it is ~4× smaller.
- **No synchronous blocker on the hot path.** cProfile (LLM mocked) shows the `tottime`
  top-20 is entirely async machinery (`epoll.poll`, socket send/recv, greenlet switch,
  asyncio `_run_once`) and SQLAlchemy statement *compilation* — no app-level regex/JSON/
  validation/`sleep` hotspot. `process_turn` is 0.021 s of `tottime` across 1,278 calls. A
  source scan of the hot path (retrieval, rules, turn_context) found no `time.sleep`,
  `requests`, sync file/DB, or CPU reranker either.
- **So P1.08's premise does not hold on this stack**: there is no 60 ms bridge and no sync
  call to move off the loop. Proving that — and leaving a detector for the next regression —
  is the deliverable. The −30 ms the task budgeted is not available because the time it
  targeted is not being spent.
- **Two real leads the run surfaced:**
  1. **Live-stack loop stalls** — the monitor logged spikes up to **~470 ms**. cProfile
     (mocked) shows no *steady-state app* blocker, so DEEP mode was used to name them (below).
  2. **`unattributed_ms` on multi-call turns** — the `own-record lookup` shape showed
     ~2.7 s outside every timed stage (that turn makes an auxiliary model call not tagged
     planner/responder). It is correctly reported as `unattributed_ms`, not bridge; wrapping
     that call in its own `StageTimer` stage would attribute it.

### DEEP-mode result — what the stalls actually were

Running with `LOOP_MONITOR_DEEP=true` named the callsites. Read the `wait_for=` field in
each `Executing <Task …>` line — it says what the stalled task was *suspended on*, which is
the key to attribution:

- **Dominant stalls are DB/client COLD-START, one-time per worker boot.** The big spikes were
  suspended on `asyncpg/connection.py:432` (establishing a DB connection — TLS handshake to
  the remote Railway Postgres), `asyncpg/prepared_stmt.py:256` (first statement prepare), and
  `_retrieve` at `planner.py:334` right before the *first* OpenAI embeddings call (lazy client
  init + first pgvector query compile). Evidence they are cold-start: they cluster in the
  first ~2 s after a (re)start and do not recur on warm turns. In production a worker boots
  rarely and before it serves live calls, so these are low-impact — but they are now
  **mitigated**: `_warm_up_io()` in `app/main.py` (gated by `STARTUP_WARMUP_ENABLED`, on by
  default) opens a pooled DB connection + one ORM select and does one tiny embed at startup,
  in the background, so the first live turn finds the pool, statement cache, and embeddings
  client already warm. Best-effort: a warm-up failure is logged and the app serves normally.
- **`bcrypt` on the auth path — a genuine synchronous blocker, FIXED.** Separately from the
  cold-start noise, `app/security/auth.py` called `bcrypt.checkpw`/`hashpw` **directly inside
  async handlers**. bcrypt burns ~250–400 ms of CPU *by design*, and on the loop that freezes
  every in-flight voice turn — and unlike cold-start it recurs on **every** login / user
  create / update. Fixed by adding `verify_password_async` / `hash_password_async` (bcrypt via
  `asyncio.to_thread`; its C code drops the GIL, so the loop stays free) and switching the
  async handlers in `api/auth.py` and `api/users.py` to them; the sync versions remain for
  sync callers (`scripts/seed.py`, tests). This is the recurring blocker P1.08 targets — a
  DB-connection future in a `wait_for=` is DB latency, but bcrypt on the loop is a true block.

> **Note on reading `docker compose logs`.** It is cumulative (everything since container
> start), and the dev server runs `uvicorn --reload`. To attribute cleanly: restart `core`,
> run the profiler **twice** (first run warms the pool + embeddings client), and inspect only
> the second, warm run (`docker compose logs --since 2m core | grep -E "Executing <|stalled"`).
> A ~400 ms stall that survives into the warm run is a real lead; a first-boot one is not.

### Pinning the live-stack stalls (DEEP mode)

The monitor runs inside the **server** process and reads its config at boot, so the flags
must be set on the `core` service and the server restarted — setting them on the `exec`'d
profiler (a separate process) does nothing. Put them in the `.env` the app loads:

```bash
# .env (loaded by app.config)
LOOP_MONITOR_ENABLED=true
LOOP_MONITOR_DEEP=true
LOOP_MONITOR_SLOW_CALLBACK_MS=50
```

```bash
docker compose up -d core                                  # restart so the flags take effect
docker compose exec core python scripts/profile_bridge.py --runs 2   # generate load
docker compose logs core | grep "Executing <"              # the exact blocking callsite(s)
```

`LOOP_MONITOR_DEEP` enables asyncio debug + slow-callback logging, so any callback or task
step that overruns `LOOP_MONITOR_SLOW_CALLBACK_MS` names itself. Match on `Executing <` (not
`<Handle` alone) — asyncio logs a slow **task step** as `Executing <Task …>` and a slow bare
**callback** as `Executing <Handle …>`; the `app.observability.loop` "event-loop stalled …"
lines carry the aggregate lag alongside. **Turn DEEP back off afterwards** — asyncio debug
mode adds overhead and is not a steady-state setting.

## Verify the tooling itself

```bash
python core/scripts/verify_bridge_observability.py
```

Self-contained (no server/DB): confirms `StageTimer` attributes sub-stages and
`LoopMonitor` catches a deliberate 300 ms block. Exit code gates CI.
