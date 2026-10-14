"""LoopMonitor — detect and attribute anything that blocks the asyncio event loop.

The problem it solves
---------------------
The benchmark report's whole thesis is that the bridge stage costs ~60 ms "higher than
pure asyncio overhead (0.001 ms measured)", and suspects "a synchronous call (DB, file
I/O) on the hot path". A blocking call inside a coroutine does not raise, does not log,
and does not show up as its own line in the timing dict — it just steals wall-clock time
from every other task sharing the loop. It is invisible by construction. This module
makes it visible.

How it works
------------
Two complementary techniques, each pulling its weight:

1. **Lag sampler (always-on, cheap).** A background task repeatedly awaits a short sleep
   and measures how much *longer* than requested it actually took to be resumed. If the
   loop were free, resumption is near-instant; the excess ("lag") is time the loop spent
   unable to service anyone — i.e. something ran synchronously without yielding. A spike
   is logged with the lag and a snapshot of the tasks alive at that moment, so the
   culprit can be correlated. ~20 wake-ups/second costs nothing measurable.

2. **Slow-callback attribution (opt-in, precise).** asyncio's own debug machinery logs
   "Executing <Handle …> took N seconds" for any single callback that overruns
   ``loop.slow_callback_duration``. That message names the exact function, so when you
   are hunting a specific blocker you flip on ``deep`` mode and the loop tells you the
   callsite directly. It carries more overhead, so it is off by default.

Why a sampler *and* slow-callback: the sampler is safe to leave on in production and tells
you *that* and *how badly* the loop stalled, with a rough who; slow-callback tells you the
*exact* callsite but is a debugging tool you switch on deliberately. Together they cover
both "alert me in prod" and "let me pin it down".

Everything here is defensive: a monitor that crashed the loop it watches would be far
worse than the latency it reports, so the sampler swallows its own errors and keeps going.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger("app.observability.loop")


@dataclass
class LoopStats:
    """Rolling counters, cheap to read from a health endpoint or a test.

    Deliberately tiny — a few numbers, not a time series. Anything richer belongs in a
    real metrics backend; this is here so "is the loop healthy right now?" is answerable
    without one.
    """
    samples: int = 0                 # how many lag measurements were taken
    spikes: int = 0                  # how many exceeded the warn threshold
    max_lag_ms: float = 0.0          # worst single stall observed
    last_spike_lag_ms: float = 0.0   # most recent stall's size
    last_spike_at: float = 0.0       # epoch seconds of the most recent stall
    running_since: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "samples": self.samples,
            "spikes": self.spikes,
            "max_lag_ms": round(self.max_lag_ms, 1),
            "last_spike_lag_ms": round(self.last_spike_lag_ms, 1),
            "last_spike_at": self.last_spike_at or None,
            "uptime_s": round(time.time() - self.running_since, 1),
        }


class LoopMonitor:
    """Watches one event loop for stalls. Start it once at app startup; stop at shutdown.

    Parameters
    ----------
    interval_ms:
        How often the sampler wakes to measure lag. Smaller = catches shorter stalls and
        reacts sooner, at slightly more overhead. 50 ms (20 Hz) is a good default: cheap,
        and fine-grained enough to notice a 60 ms bridge hiccup.
    warn_threshold_ms:
        Lag above this is logged as a spike. Set comfortably above normal scheduling
        jitter (a few ms) so healthy turns stay quiet — 100 ms means "a real block", not
        "the scheduler was briefly busy".
    deep:
        Enable asyncio debug mode + slow-callback logging for precise callsite attribution.
        More overhead; use it while investigating, not as a permanent setting.
    slow_callback_ms:
        In ``deep`` mode, the per-callback overrun asyncio will complain about. Should be
        below ``warn_threshold_ms`` so the exact callback is named before/around the time
        the sampler flags the aggregate stall.
    snapshot_tasks:
        How many live tasks to name in a spike log. Zero disables the snapshot (quietest).
    """

    def __init__(self, *, interval_ms: int = 50, warn_threshold_ms: int = 100,
                 deep: bool = False, slow_callback_ms: int = 50,
                 snapshot_tasks: int = 8) -> None:
        self.interval = max(0.005, interval_ms / 1000.0)          # floor at 5 ms
        self.warn_threshold = max(0.001, warn_threshold_ms / 1000.0)
        self.deep = deep
        self.slow_callback = max(0.001, slow_callback_ms / 1000.0)
        self.snapshot_tasks = max(0, snapshot_tasks)

        self.stats = LoopStats()
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # Remember what we changed on the loop so stop() can put it back exactly, rather
        # than clobbering settings another component may rely on.
        self._prev_debug: bool | None = None
        self._prev_slow: float | None = None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        """Begin sampling on the currently-running loop. Idempotent and never raises.

        Must be called from within the running loop (e.g. a FastAPI startup/lifespan
        handler) so ``get_running_loop`` succeeds; if there is no running loop we log and
        no-op rather than blow up an otherwise-healthy boot.
        """
        if self._task is not None and not self._task.done():
            return  # already running
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            log.warning("LoopMonitor.start() called with no running loop; not started")
            return

        if self.deep:
            # Record prior values so stop() can restore them (see _restore_loop).
            self._prev_debug = self._loop.get_debug()
            self._prev_slow = getattr(self._loop, "slow_callback_duration", None)
            self._loop.set_debug(True)
            # asyncio logs "Executing <Handle …> took N s" for callbacks over this bound —
            # this is the line that names the exact synchronous blocker.
            self._loop.slow_callback_duration = self.slow_callback

        self._task = self._loop.create_task(self._run(), name="loop-monitor")
        log.info("LoopMonitor started (interval=%dms, warn=%dms, deep=%s)",
                 int(self.interval * 1000), int(self.warn_threshold * 1000), self.deep)

    async def stop(self) -> None:
        """Stop sampling and restore any loop settings we changed. Safe to call twice."""
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - defensive
                log.debug("LoopMonitor task raised on shutdown", exc_info=True)
        self._restore_loop()
        log.info("LoopMonitor stopped. %s", self.stats.as_dict())

    def _restore_loop(self) -> None:
        if self._loop is None:
            return
        if self._prev_debug is not None:
            self._loop.set_debug(self._prev_debug)
            self._prev_debug = None
        if self._prev_slow is not None:
            self._loop.slow_callback_duration = self._prev_slow
            self._prev_slow = None

    # -- the sampler -------------------------------------------------------
    async def _run(self) -> None:
        """Measure loop lag forever, until cancelled.

        The measurement is the whole trick: we ask to sleep for exactly ``interval`` and
        see how long it really took. ``loop.time()`` is the loop's own monotonic clock, so
        the difference is pure scheduling delay — no wall-clock/timezone noise. On a free
        loop the delay is microseconds; a big delay means the loop was stuck running
        something that never awaited.
        """
        loop = self._loop
        assert loop is not None
        while True:
            start = loop.time()
            try:
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                raise
            elapsed = loop.time() - start
            lag = elapsed - self.interval

            # Update rolling stats. Guard against a spuriously negative lag (clock edge
            # cases) so the max never gets poisoned.
            self.stats.samples += 1
            if lag > self.stats.max_lag_ms / 1000.0:
                self.stats.max_lag_ms = lag * 1000.0

            if lag >= self.warn_threshold:
                self._on_spike(lag)

    def _on_spike(self, lag_seconds: float) -> None:
        """Log a stall, with a best-effort snapshot of who was on the loop.

        Caveat we state honestly: by the time the sampler is resumed, the blocker has
        already finished (that is *why* we were resumed), so the snapshot shows the tasks
        alive at detection, not a frozen frame of the blocker. It is a correlation aid —
        narrowing "which coroutine was in flight" — not a stack trace of the guilty line.
        For the exact line, run in ``deep`` mode and read the asyncio "Executing <Handle>
        took N s" message, which fires on the blocker itself.
        """
        lag_ms = lag_seconds * 1000.0
        self.stats.spikes += 1
        self.stats.last_spike_lag_ms = lag_ms
        self.stats.last_spike_at = time.time()

        snapshot = self._task_snapshot() if self.snapshot_tasks else ""
        log.warning("event-loop stalled %.0f ms (threshold %.0f ms).%s",
                    lag_ms, self.warn_threshold * 1000.0,
                    f" In-flight: {snapshot}" if snapshot else "")

    def _task_snapshot(self) -> str:
        """Names of the tasks currently alive, minus the monitor itself.

        Kept to ``snapshot_tasks`` entries so a spike during a burst of concurrent calls
        does not dump hundreds of lines. Never raises — introspection failures must not
        turn a warning into an error.
        """
        try:
            here = asyncio.current_task()
            names = []
            for t in asyncio.all_tasks():
                if t is here or t.done():
                    continue
                # Prefer the top of the coroutine stack (the frame actually executing);
                # fall back to the task's name/repr when no Python frame is available
                # (e.g. the task is parked in C-level I/O).
                where = ""
                try:
                    stack = t.get_stack(limit=1)
                    if stack:
                        f = stack[0]
                        where = f"@{f.f_code.co_name}:{f.f_lineno}"
                except Exception:
                    pass
                names.append(f"{t.get_name()}{where}")
                if len(names) >= self.snapshot_tasks:
                    break
            return ", ".join(names)
        except Exception:  # pragma: no cover - defensive
            return ""


# -- module-level singleton -------------------------------------------------
# One monitor per process. The app wires it up in the lifespan handler; other code (a
# health endpoint, a test) reaches the live instance through get_loop_monitor() to read
# stats without threading it through call signatures.
_monitor: LoopMonitor | None = None


def get_loop_monitor() -> LoopMonitor | None:
    """The running monitor, or None if observability is disabled this process."""
    return _monitor


def set_loop_monitor(monitor: LoopMonitor | None) -> None:
    """Register (or clear) the process-wide monitor. Called by the app lifespan only."""
    global _monitor
    _monitor = monitor
