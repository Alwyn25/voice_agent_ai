#!/usr/bin/env python3
"""Prove the P1.08 observability actually works — no server, no database needed.

Two things must be true for the bridge work to be trustworthy:

  1. StageTimer records sub-stage durations, namespaces them, and derives bridge_ms as a
     residual so nothing unwrapped can hide.
  2. LoopMonitor actually notices a synchronous call blocking the event loop and reports
     it — a detector that cannot catch a deliberate block is worthless.

This runs both as self-contained checks with fake work, so it is safe to run anywhere and
in CI:

    docker compose exec core python scripts/verify_bridge_observability.py
    python core/scripts/verify_bridge_observability.py     # also fine, stdlib + app only

Exit code is 0 on success, 1 on failure, so it can gate a pipeline.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.observability.stage_timer import StageTimer          # noqa: E402
from app.observability.loop_monitor import LoopMonitor        # noqa: E402

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures += 1


async def timer_checks() -> None:
    print("StageTimer")
    t = StageTimer()
    async with t.stage("session_load"):
        await asyncio.sleep(0.02)          # ~20 ms of (simulated) Redis I/O
    with t.stage("commit"):
        await asyncio.sleep(0.01)          # ~10 ms of (simulated) DB commit
    # ~30 ms of work we deliberately DO NOT wrap — it stands in for an auxiliary model call
    # a turn can make without tagging it planner/responder. The whole point of the residual
    # split is that this time must land in unattributed_ms, and must NOT inflate bridge_ms.
    await asyncio.sleep(0.03)

    # Heavy stages are timed elsewhere on a real turn; here they are just labels/values.
    timing = {"planner_ms": 5, "tool_ms": 0, "responder_ms": 3, "retrieval_ms": 0}
    timing["total_ms"] = t.total_ms()
    t.merge_into(timing, prefix="bridge_")
    timing["bridge_ms"] = t.sum_ms()               # direct sum of the wrapped sub-stages
    timing["unattributed_ms"] = t.residual_ms(timing)   # the honest "not instrumented" bucket

    check("sub-stages recorded and namespaced",
          "bridge_session_load_ms" in timing and "bridge_commit_ms" in timing,
          str({k: v for k, v in timing.items() if k.startswith("bridge_")}))
    check("session_load timed (~20 ms)", timing["bridge_session_load_ms"] >= 15,
          f"{timing['bridge_session_load_ms']} ms")
    # bridge_ms is now the DIRECT sum of the two wrapped stages (~30 ms), NOT the whole turn
    # (~60 ms with the untimed sleep). If it ever equals total it means the residual leaked
    # back in — exactly the bug this split fixes.
    check("bridge_ms is the direct sub-stage sum (~30 ms), well under total",
          25 <= timing["bridge_ms"] <= 45 and timing["bridge_ms"] < timing["total_ms"],
          f"bridge={timing['bridge_ms']} of total={timing['total_ms']}")
    check("untimed work surfaces as unattributed_ms, not as fake bridge I/O",
          timing["unattributed_ms"] > 0,
          f"unattributed={timing['unattributed_ms']} ms (the ~30 ms left unwrapped)")


async def monitor_checks() -> None:
    print("LoopMonitor")
    m = LoopMonitor(interval_ms=20, warn_threshold_ms=80, snapshot_tasks=5)
    m.start()
    await asyncio.sleep(0.1)               # healthy loop — should produce no spike
    healthy_spikes = m.stats.spikes
    time.sleep(0.3)                        # THE BLOCK: 300 ms of synchronous work
    await asyncio.sleep(0.1)               # let the sampler resume and record it
    await m.stop()

    check("no spurious spike on a healthy loop", healthy_spikes == 0,
          f"{healthy_spikes} spike(s) before the block")
    check("caught the 300 ms synchronous block", m.stats.spikes >= 1,
          f"{m.stats.spikes} spike(s)")
    check("measured the stall size (~300 ms)", m.stats.max_lag_ms >= 200,
          f"max_lag={m.stats.max_lag_ms:.0f} ms")
    check("restored loop settings after stop",
          not asyncio.get_event_loop().get_debug())


async def offload_checks() -> None:
    """Prove the REMEDIATION, not just the detection: the same CPU-heavy work that stalls
    the loop when called directly (monitor_checks above) must NOT stall it when pushed to a
    worker thread with ``asyncio.to_thread`` — this is exactly the pattern applied to bcrypt
    on the /auth/login path (app/security/auth.py, P1.08). ``time.sleep`` stands in for the
    blocking C routine; like bcrypt it holds no Python work and lets the thread run free.
    """
    print("Offload (asyncio.to_thread) keeps the loop free")
    m = LoopMonitor(interval_ms=20, warn_threshold_ms=80, snapshot_tasks=5)
    m.start()
    await asyncio.sleep(0.1)                      # settle
    await asyncio.to_thread(time.sleep, 0.3)      # 300 ms of work, OFF the loop
    await asyncio.sleep(0.1)                      # give the sampler room to have run
    await m.stop()

    check("offloaded work did NOT stall the loop", m.stats.spikes == 0,
          f"{m.stats.spikes} spike(s), max_lag={m.stats.max_lag_ms:.0f} ms "
          "(should be ~0 — the block ran in a thread)")


async def main() -> int:
    print("\nVerifying P1.08 bridge observability\n" + "-" * 40)
    await timer_checks()
    await monitor_checks()
    await offload_checks()
    print("-" * 40)
    print("All checks passed.\n" if not failures else f"{failures} check(s) FAILED.\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
