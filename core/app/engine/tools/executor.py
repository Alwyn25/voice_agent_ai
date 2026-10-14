"""Tool executor — the gate every action passes through.

Order (planner steps 8-10):
  1. resolve tool + permission
  1b. rule gate (block / require / escalate) — nothing executes past a block
  2. validate params against JSON Schema
  3. idempotency check (same key already succeeded -> return prior result)
  4. allowed-state gate, then preconditions (live data)
  5. execute implementation (with timeout)
  6. postconditions (proof of effect)
  7. state transition effect
  8. persist tool_execution row
Nothing can reach step 5 without passing 1-4.
"""
import asyncio
import logging
import time
import uuid
from dataclasses import replace
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from sqlalchemy import event, select

from app.config import get_settings
from app.db.base import get_session_factory
from app.db.models import ToolExecution
from app.engine.rules import engine as rule_engine
from app.engine.rules.context import build_context
from app.engine.state_machine.machine import apply_trigger
from app.engine.tools import contracts, registry
from app.engine.tools.types import CheckResult, ToolContext, ToolOutcome
from app.observability import tool_metrics

DEFAULT_TIMEOUT_MS = 5000

log = logging.getLogger(__name__)


# ===== P2.05: parallel preconditions =======================================================
# A tool's DB-backed preconditions (centre_exists, executive_exists, slot_available,
# no_conflict, …) are independent read-only SELECTs, yet the executor runs them one after
# another. Running the DB ones concurrently collapses their wall-time from a SUM to a MAX —
# roughly 77ms -> 35ms on a book_appointment turn, ~40ms saved. Applies to inbound AND
# outbound calls: both drive every tool through this one executor. Enabled by
# parallel_preconditions_enabled (off by default — see config.py for the full rationale).
#
# THE HAZARD this module handles carefully: a SQLAlchemy AsyncSession is NOT safe for
# concurrent use. Gathering several checks that each `await ctx.db.execute()` on the SAME
# session raises InvalidRequestError. So each concurrent DB check runs on its OWN short-lived,
# read-only session (its own pool connection); the in-memory checks (no DB) run inline on ctx.


# One semaphore per event loop, bounding how many precondition side-connections may be open at
# once across ALL in-flight turns — a burst of simultaneous booking turns must not exhaust the
# small Postgres pool. Keyed by loop id so tests that spin up fresh loops each get their own.
_precond_sems: dict[int, tuple[asyncio.Semaphore, int]] = {}


def _precondition_semaphore(size: int) -> asyncio.Semaphore:
    loop_id = id(asyncio.get_running_loop())
    sem, cached = _precond_sems.get(loop_id, (None, 0))
    if sem is None or cached != size:
        sem = asyncio.Semaphore(max(1, size))
        _precond_sems[loop_id] = (sem, size)
    return sem


def _has_pending_tool_write(db: Any) -> bool:
    """True when a tool has already flushed an UNCOMMITTED write on this session this turn.

    Parallel preconditions read on separate connections that see only COMMITTED data, so while
    such a write is pending we must stay on the shared session — otherwise a second tool in a
    chain could miss a row the first tool just wrote (e.g. a freshly-booked appointment that a
    following no_conflict check should see). The DB session is call-scoped and commits once per
    turn, so this is cleared at each commit and the frequent single-tool turn is never blocked
    (its preconditions run before the tool writes)."""
    return bool(db.info.get("_pending_tool_write"))


def _mark_pending_tool_write(db: Any) -> None:
    """Record (from _record, after a tool has flushed its rows) that an uncommitted tool write
    is now pending. Registers a one-time listener that clears the flag when the turn's
    transaction ends, so the next turn on this long-lived session starts clean."""
    if not db.info.get("_pending_write_listener"):
        def _clear(sync_session: Any) -> None:
            sync_session.info["_pending_tool_write"] = False
        # AsyncSession.info proxies to sync_session.info, so clearing the sync session's dict
        # clears the same one _has_pending_tool_write reads. Commit is the normal turn end;
        # rollback covers an aborted turn.
        event.listen(db.sync_session, "after_commit", _clear)
        event.listen(db.sync_session, "after_rollback", _clear)
        db.info["_pending_write_listener"] = True
    db.info["_pending_tool_write"] = True


def _check_entry(name: str, res: CheckResult, started: float) -> dict:
    """One precondition's trace row. `ms` (added for P2.05) rides along into the tool_execution
    audit row via outcome.validation["checks"], so a slow check is attributable after the
    fact, parallel or not."""
    return {"check": name, "ok": res.ok, "message": res.message, "data": res.data,
            "ms": round((time.perf_counter() - started) * 1000, 1)}


async def _preconditions_sequential(ctx: ToolContext,
                                    resolved: list[tuple[str, dict]]) -> list[dict]:
    """Original path: each check awaited on the shared session, in order, stopping at the first
    failure. Runs with the flag off, and as the fallback when the parallel path hits trouble —
    so it stays behaviourally identical to the pre-P2.05 loop (short-circuit included)."""
    checks_run: list[dict] = []
    for name, args in resolved:
        t0 = time.perf_counter()
        res = await contracts.run_check(name, ctx, args)
        checks_run.append(_check_entry(name, res, t0))
        if not res.ok:
            break
    return checks_run


async def _preconditions_parallel(ctx: ToolContext,
                                  resolved: list[tuple[str, dict]],
                                  settings: Any) -> list[dict]:
    """In-memory checks inline, DB checks concurrent (each on its own read-only session), trace
    returned in DECLARED order so the caller still blocks on the first not-ok check exactly as
    the sequential loop would.

    Unlike the sequential path this does NOT short-circuit — once dispatched, every DB check
    runs even if an earlier one fails. That is safe because preconditions are read-only and
    side-effect-free, so the reported outcome (first not-ok in declared order) is identical; the
    only cost is a few wasted SELECTs when an early check fails."""
    factory = get_session_factory()
    sem = _precondition_semaphore(settings.parallel_preconditions_max_concurrency)

    async def run_db_check(name: str, args: dict) -> dict:
        t0 = time.perf_counter()
        async with sem:                       # bound concurrent pool checkouts
            async with factory() as side:     # own connection; read-only, never committed
                # replace() shares customer/extra/tenant_id/project_id by reference — the check
                # sees the same context, only a different db handle.
                res = await contracts.run_check(name, replace(ctx, db=side), args)
        return _check_entry(name, res, t0)

    entries: dict[int, dict] = {}
    tasks: dict[int, asyncio.Task] = {}
    for i, (name, args) in enumerate(resolved):
        if contracts.is_in_memory_check(name):
            t0 = time.perf_counter()
            entries[i] = _check_entry(name, await contracts.run_check(name, ctx, args), t0)
        else:
            tasks[i] = asyncio.ensure_future(run_db_check(name, args))

    if tasks:
        gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for i, result in zip(tasks.keys(), gathered):
            if isinstance(result, Exception):
                # run_check swallows check-level errors, so an exception here is infrastructure
                # (pool timeout, connection drop). Re-raise so the caller falls back to a clean
                # sequential run rather than blocking a turn on a half-finished parallel set.
                raise result
            entries[i] = result

    return [entries[i] for i in range(len(resolved))]


async def _run_preconditions(ctx: ToolContext, tool: Any, params: dict) -> list[dict]:
    """Evaluate tool.preconditions and return their trace rows in declared order.

    Sequential on the shared session by default; concurrent on isolated sessions when
    parallel_preconditions_enabled AND it is safe to do so (no uncommitted intra-turn tool
    write a check might need to read). Any failure in the parallel path degrades to the proven
    sequential path, so this can never fail a turn the original loop would have passed."""
    specs = tool.preconditions or []
    if not specs:
        return []

    # Resolve $-references once (pure, no IO) so both paths work from identical inputs.
    resolved = [(spec.get("check"), contracts.resolve_refs(spec.get("args", {}), params, ctx))
                for spec in specs]

    settings = get_settings()
    if not settings.parallel_preconditions_enabled or _has_pending_tool_write(ctx.db):
        return await _preconditions_sequential(ctx, resolved)

    t0 = time.perf_counter()
    try:
        checks_run = await _preconditions_parallel(ctx, resolved, settings)
    except Exception:
        log.warning("parallel preconditions failed; using sequential", exc_info=True)
        return await _preconditions_sequential(ctx, resolved)

    if settings.parallel_preconditions_trace:
        wall = (time.perf_counter() - t0) * 1000
        seq_sum = sum(c.get("ms", 0.0) for c in checks_run)
        log.info("preconditions[parallel] tool=%s checks=%d wall=%.1fms seq_sum=%.1fms "
                 "saved=%.1fms detail=%s", getattr(tool, "name", "?"), len(checks_run), wall,
                 seq_sum, max(0.0, seq_sum - wall),
                 {c["check"]: c.get("ms") for c in checks_run})
    return checks_run


def _render_idempotency_key(template: str | None, ctx: ToolContext, params: dict) -> str | None:
    if not template:
        return None
    values = {"customer_id": str(ctx.customer.id) if ctx.customer else "none",
              "conversation_id": str(ctx.conversation_id) if ctx.conversation_id else "none",
              **{k: str(v) for k, v in params.items()}}
    try:
        return template.format(**values)[:300]
    except KeyError:
        return None


def validate_params(schema: dict, params: dict) -> list[str]:
    """Format-aware validation. Without a FormatChecker, "format": "uuid" is only an
    annotation and a centre name would sail through as a centre_id."""
    if not schema:
        return []
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return [f"{'.'.join(str(p) for p in e.path) or '(root)'}: {e.message}"
            for e in validator.iter_errors(params)]


async def execute(ctx: ToolContext, tool_name: str, params: dict, *,
                  principal_kind: str = "agent", dry_run: bool = False,
                  check_rules: bool = True) -> ToolOutcome:
    """Public entry point. Wraps _execute so every outcome — including blocked and
    skipped_duplicate ones that never reach _record()/ToolExecution below — is captured
    for performance tracking, without every call site needing to know that."""
    started = time.perf_counter()
    outcome = await _execute(ctx, tool_name, params, principal_kind=principal_kind,
                             dry_run=dry_run, check_rules=check_rules)
    if not outcome.duration_ms:
        outcome.duration_ms = int((time.perf_counter() - started) * 1000)
    await tool_metrics.record(ctx, outcome)
    return outcome


async def _execute(ctx: ToolContext, tool_name: str, params: dict, *,
                   principal_kind: str = "agent", dry_run: bool = False,
                   check_rules: bool = True) -> ToolOutcome:
    started = time.perf_counter()
    tool = await registry.get_tool(ctx.db, ctx.tenant_id, tool_name)
    if tool is None or not tool.is_enabled:
        return ToolOutcome(status="blocked", tool_name=tool_name,
                           error=f"Tool '{tool_name}' is not available",
                           failed_check="tool_available")

    # --- permission ---
    if tool.permission_level in ("supervisor_only", "human_only") and principal_kind == "agent":
        return ToolOutcome(status="blocked", tool_name=tool_name,
                           error=f"Tool '{tool_name}' requires {tool.permission_level}",
                           failed_check="permission_level")

    # --- schema validation ---
    errors = validate_params(tool.input_schema or {}, params)
    if errors:
        return ToolOutcome(status="blocked", tool_name=tool_name,
                           error="Parameter validation failed: " + "; ".join(errors),
                           failed_check="input_schema", validation={"errors": errors})

    # --- idempotency ---
    idem_key = _render_idempotency_key(tool.idempotency_key, ctx, params)
    if idem_key:
        prior = (await ctx.db.execute(select(ToolExecution).where(
            ToolExecution.tenant_id == ctx.tenant_id,
            ToolExecution.idempotency_key == idem_key,
            ToolExecution.status == "success"))).scalars().first()
        if prior is not None:
            return ToolOutcome(status="skipped_duplicate", tool_name=tool_name,
                               output=prior.output or {}, execution_id=prior.id,
                               validation={"idempotency_key": idem_key})

    # --- allowed state (after idempotency: a completed action
    #     must report its prior result, not a state block) ---
    current_state = ctx.customer.current_state if ctx.customer else None
    if tool.allowed_states and current_state and current_state not in tool.allowed_states:
        return ToolOutcome(status="blocked", tool_name=tool_name,
                           error=f"Tool not allowed in state '{current_state}'",
                           failed_check="allowed_states")

    # --- rule gate (block always wins; requirements must be satisfied first) ---
    rules_trace: list[dict] = []
    if check_rules:
        rule_ctx = await build_context(
            ctx.db, ctx.tenant_id, evaluation_point=f"tool:{tool_name}",
            customer=ctx.customer, proposal={"tool": tool_name, "params": params},
            session={"identity_verified": ctx.identity_verified,
                     "channel": ctx.extra.get("channel"),
                     "direction": ctx.extra.get("direction"),
                     # Captured conversation entities + the per-condition ask counter, so a
                     # slot-based `require` (e.g. "collect budget before offering a visit") can
                     # be satisfied by real captured data and stand aside after a couple of asks.
                     "__slots__": ctx.extra.get("slots") or {},
                     "__require_asks__": ctx.extra.get("require_asks") or {}},
            turn=ctx.extra.get("turn"))
        decision = await rule_engine.evaluate_point(
            ctx.db, ctx.tenant_id, evaluation_point=f"tool:{tool_name}", ctx=rule_ctx)
        rules_trace = decision.evaluated
        if not decision.allowed:
            v = decision.blocks[0]
            return ToolOutcome(status="blocked", tool_name=tool_name,
                               error=v.reason, failed_check=f"rule:{v.rule_id}",
                               failure_action={"action": "respond", "rule": v.rule_id},
                               validation={"rules_evaluated": rules_trace})
        unmet = rule_engine.unmet_requirements(decision, rule_ctx)
        if unmet:
            r = unmet[0]
            return ToolOutcome(status="blocked", tool_name=tool_name,
                               error=r.body.get("reason", "Requirement not met"),
                               failed_check=f"requirement:{r.body.get('condition')}",
                               # slot/ask/condition let the turn layer surface the rule's own
                               # question deterministically (no extra model call) and count the
                               # asks for the anti-trap — carried as data on the rule, so a
                               # deleted custom instruction leaves nothing behind.
                               failure_action={"action": "clarify",
                                               "prompt_hint": r.body.get("prompt_hint"),
                                               "ask": r.body.get("ask"),
                                               "slot": rule_engine.require_slot(r.body),
                                               "condition": r.body.get("condition"),
                                               "rule": r.rule_id},
                               validation={"rules_evaluated": rules_trace})
        if decision.escalations:
            e = decision.escalations[0]
            return ToolOutcome(status="blocked", tool_name=tool_name, error=e.reason,
                               failed_check=f"escalation:{e.rule_id}",
                               failure_action={"action": "escalate",
                                               "type": e.body.get("type", "immediate"),
                                               "rule": e.rule_id},
                               validation={"rules_evaluated": rules_trace})

    # --- preconditions ---
    # Sequential on the shared session by default; concurrent on isolated read-only sessions
    # when parallel_preconditions_enabled (P2.05). Either way the trace comes back in declared
    # order, so blocking on the first not-ok check is identical to the original inline loop.
    checks_run = await _run_preconditions(ctx, tool, params)
    for entry in checks_run:
        if not entry["ok"]:
            name = entry["check"]
            policy = (tool.failure_policy or {}).get("on_precondition_fail", {})
            action = policy.get(name) or policy.get("default") or {"action": "respond"}
            return ToolOutcome(status="blocked", tool_name=tool_name, error=entry["message"],
                               failed_check=name, failure_action=action,
                               output=entry["data"],
                               validation={"checks": checks_run, "rules_evaluated": rules_trace})

    if dry_run:
        return ToolOutcome(status="success", tool_name=tool_name,
                           output={"dry_run": True, "would_execute": tool_name},
                           validation={"checks": checks_run, "idempotency_key": idem_key,
                                      "rules_evaluated": rules_trace},
                           duration_ms=int((time.perf_counter() - started) * 1000))

    # --- execute ---
    timeout_ms = int((tool.failure_policy or {}).get("timeout_ms", DEFAULT_TIMEOUT_MS))
    output: dict[str, Any] = {}
    error: str | None = None
    try:
        impl = registry.load_implementation(tool.implementation)
        # Run inside a SAVEPOINT: if the tool raises (bad type, constraint violation),
        # the failure rolls back to this point and the session stays usable. Without
        # this a single bad tool call takes down the whole turn.
        async with ctx.db.begin_nested():
            output = await asyncio.wait_for(impl(ctx, params), timeout=timeout_ms / 1000)
    except asyncio.TimeoutError:
        error = f"Tool timed out after {timeout_ms}ms"
    except Exception as e:
        error = f"{e.__class__.__name__}: {e}"

    if error:
        outcome = ToolOutcome(status="failed", tool_name=tool_name, error=error,
                              failure_action=(tool.failure_policy or {}).get(
                                  "on_execution_error", {"action": "escalate"}),
                              validation={"checks": checks_run},
                              duration_ms=int((time.perf_counter() - started) * 1000))
        await _record(ctx, outcome, params, idem_key)
        return outcome

    # --- postconditions ---
    for spec in (tool.postconditions or []):
        name = spec.get("check")
        args = contracts.resolve_refs(spec.get("args", {}), params, ctx, result=output)
        res = await contracts.run_check(name, ctx, args)
        checks_run.append({"check": name, "ok": res.ok, "message": res.message, "post": True})
        if not res.ok:
            outcome = ToolOutcome(
                status="failed", tool_name=tool_name,
                error=f"Postcondition '{name}' failed: {res.message}",
                failed_check=name, output=output,
                failure_action={"action": "escalate", "reason": "postcondition_failed"},
                validation={"checks": checks_run},
                duration_ms=int((time.perf_counter() - started) * 1000))
            await _record(ctx, outcome, params, idem_key)
            return outcome

    # --- effects: state transition ---
    state_changed_to = None
    effect_state = (tool.effects or {}).get("state_transition")
    if effect_state and ctx.customer is not None:
        result = await apply_trigger(ctx.db, ctx.customer, f"tool:{tool_name}",
                                     actor=ctx.actor, reason=f"tool {tool_name} succeeded")
        if result:
            state_changed_to = result.to_state

    outcome = ToolOutcome(status="success", tool_name=tool_name, output=output,
                          state_changed_to=state_changed_to,
                          validation={"checks": checks_run, "idempotency_key": idem_key,
                                      "rules_evaluated": rules_trace},
                          duration_ms=int((time.perf_counter() - started) * 1000))
    await _record(ctx, outcome, params, idem_key)
    return outcome


async def _record(ctx: ToolContext, outcome: ToolOutcome, params: dict, idem_key: str | None) -> None:
    row = ToolExecution(
        tenant_id=ctx.tenant_id, decision_trace_id=ctx.decision_trace_id,
        conversation_id=ctx.conversation_id,
        customer_id=ctx.customer.id if ctx.customer else None,
        tool_name=outcome.tool_name, input=_jsonable(params), output=_jsonable(outcome.output),
        status=outcome.status, error=outcome.error, duration_ms=outcome.duration_ms,
        idempotency_key=idem_key)
    ctx.db.add(row)
    await ctx.db.flush()
    outcome.execution_id = row.id
    # A tool has now flushed (uncommitted) rows on this session — this audit row, and for a
    # mutation tool its business rows too. Any LATER tool in the same turn must evaluate its
    # preconditions sequentially on this shared session so it can see them; a parallel side
    # connection would not (see _has_pending_tool_write). Cleared on the turn's commit.
    _mark_pending_tool_write(ctx.db)


def _jsonable(d: Any) -> Any:
    if isinstance(d, dict):
        return {_clean(k): _jsonable(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_jsonable(v) for v in d]
    if isinstance(d, uuid.UUID):
        return str(d)
    if hasattr(d, "isoformat"):
        return d.isoformat()
    if isinstance(d, str):
        return _clean(d)
    return d


# Postgres cannot store a NUL byte in a text or JSONB column, and the model does emit them:
# asked for a centre named "DSR Highland Greenz — Site Office" it sent the em-dash back as
# \u0000. The tool itself ran perfectly — 64 slots found — and then writing the audit row
# raised UntranslatableCharacterError, which surfaced to the caller as a 500 on a call that
# had actually succeeded. Audit must never be able to fail a request that worked, so
# anything unstorable is stripped on the way in.
_UNSTORABLE = dict.fromkeys(range(0x20), None)      # NUL and the other C0 controls
_UNSTORABLE.pop(0x09), _UNSTORABLE.pop(0x0A), _UNSTORABLE.pop(0x0D)   # keep tab, newline, CR


def _clean(value: str) -> str:
    return value.translate(_UNSTORABLE) if isinstance(value, str) else value
