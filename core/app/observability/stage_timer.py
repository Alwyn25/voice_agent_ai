"""StageTimer — attribute a turn's wall-clock time to named sub-stages.

Why this exists
---------------
``process_turn`` already times the three expensive stages (planner, tool, responder)
plus retrieval. What it does NOT time is everything else: loading working memory from
Redis, writing the customer's message, loading the customer row, assembling and writing
the decision trace, saving the session, and the final commit. The benchmark report calls
this leftover the "bridge" (BRG) stage — "frame relay + WS hop", ~60 ms — and
``scripts/profile_turns.py`` only ever saw it as an unexplained "gap between the stage
sum and the wall clock". You cannot shrink what you cannot see, so this makes it visible.

Design goals
------------
- **Cheap.** One ``time.perf_counter()`` pair per stage. Perf counter is monotonic and
  the highest-resolution clock available, so nested/adjacent stages never go negative.
- **Non-throwing.** Instrumentation must never change behaviour. A timer that raised on
  a hot path would turn an observability aid into an outage, so every public method is
  defensive and the context manager records the elapsed time even when the wrapped block
  raises (the stage still cost time; hiding that would be a lie).
- **Ergonomic.** ``async with timer.stage("session_load"): ...`` reads at the call site,
  and ``timer.merge_into(timing)`` drops the numbers straight into the existing
  ``debug.timing`` dict that the API already returns — no new response shape to plumb.

This class is deliberately synchronous-friendly: the context managers work for both
``with`` and ``async with`` because timing itself does no awaiting. The ``async``
flavour exists only so it can wrap ``await`` expressions without an extra indent.
"""
from __future__ import annotations

import time
from types import TracebackType


class _StageScope:
    """A single ``with`` / ``async with`` block whose duration is recorded on exit.

    Supports both protocols so the same object can wrap plain code and ``await``
    expressions. The elapsed time is written on ``__exit__`` regardless of whether the
    body raised — an exception does not make the time spent unreal, and swallowing the
    measurement would hide exactly the slow, failing stage you most want to see.
    """

    __slots__ = ("_timer", "_name", "_start")

    def __init__(self, timer: "StageTimer", name: str) -> None:
        self._timer = timer
        self._name = name
        self._start = 0.0

    # -- sync protocol ------------------------------------------------------
    def __enter__(self) -> "_StageScope":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, tb: TracebackType | None) -> bool:
        # Record even on error; return False so the exception propagates untouched.
        self._timer._record(self._name, time.perf_counter() - self._start)
        return False

    # -- async protocol (identical timing; no awaiting of our own) -----------
    async def __aenter__(self) -> "_StageScope":
        self._start = time.perf_counter()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None,
                        exc: BaseException | None, tb: TracebackType | None) -> bool:
        self._timer._record(self._name, time.perf_counter() - self._start)
        return False


class StageTimer:
    """Accumulates named sub-stage durations for one turn.

    Millisecond integers are used throughout to match the existing ``timing`` dict on the
    turn's ``debug`` payload (``planner_ms``, ``tool_ms``, …), so the bridge numbers sit
    alongside the stage numbers in the same units and the same place.

    Repeated stages accumulate: two Redis round-trips both tagged ``"session_io"`` sum
    into one figure, which is what you want when a stage happens more than once in a turn.
    """

    __slots__ = ("_ms", "_started")

    def __init__(self) -> None:
        # stage name -> accumulated milliseconds (float internally, rounded on export)
        self._ms: dict[str, float] = {}
        # Wall-clock origin, so total() reflects the whole turn even across stages that
        # were not individually wrapped.
        self._started = time.perf_counter()

    def stage(self, name: str) -> _StageScope:
        """Time a block of the turn.

            with timer.stage("customer_load"):
                customer = await db.execute(...)

        or, wrapping an await without an extra indent level::

            async with timer.stage("commit"):
                await db.commit()
        """
        return _StageScope(self, name)

    def _record(self, name: str, seconds: float) -> None:
        # Guard against a clock hiccup producing a negative interval — perf_counter is
        # monotonic so this should never fire, but instrumentation code has no business
        # ever emitting a negative latency into a dashboard.
        ms = max(0.0, seconds * 1000.0)
        self._ms[name] = self._ms.get(name, 0.0) + ms

    def add_ms(self, name: str, ms: float) -> None:
        """Fold in a duration measured elsewhere (e.g. a stage another function timed)."""
        self._record(name, max(0.0, ms) / 1000.0)

    def total_ms(self) -> int:
        """Whole-turn wall clock since this timer was created."""
        return int((time.perf_counter() - self._started) * 1000)

    def as_dict(self) -> dict[str, int]:
        """Per-stage milliseconds, rounded to ints for the response/trace."""
        return {name: int(round(ms)) for name, ms in self._ms.items()}

    def merge_into(self, timing: dict[str, int], *, prefix: str = "") -> dict[str, int]:
        """Copy the recorded stages into an existing ``timing`` dict, in place.

        Keys become ``{prefix}{stage}_ms`` (e.g. ``bridge_session_load_ms``): the ``_ms``
        suffix matches the existing convention (``planner_ms``, ``tool_ms``) since a timer
        is always in milliseconds, and ``prefix`` namespaces the stages so they never
        collide with the planner/tool/responder keys already present. The same dict is
        returned for convenient chaining.
        """
        for name, ms in self.as_dict().items():
            timing[f"{prefix}{name}_ms"] = ms
        return timing

    def sum_ms(self) -> int:
        """Total of the wrapped sub-stages — the DIRECT cost of everything this timer timed.

        This is the honest "bridge" figure: the six orchestration steps we actually wrapped
        (session load, message persist, customer load, trace write, session save, commit),
        summed. Unlike a whole-turn residual it cannot silently absorb time we never
        measured — an untimed auxiliary LLM call lands in ``residual_ms`` below, not here,
        so ``bridge_ms`` stays a number you can defend as "this is the DB/Redis overhead".
        """
        return int(round(sum(self._ms.values())))

    def residual_ms(self, timing: dict[str, int], *,
                    heavy_keys: tuple[str, ...] = (
                        "planner_ms", "tool_ms", "responder_ms", "retrieval_ms")) -> int:
        """Wall-clock left over after EVERY stage we measured is subtracted from the total.

        ``residual = total − heavy_stages − wrapped_sub_stages``. This is the honest
        "we did not instrument this" bucket, kept as its own line (``unattributed_ms``)
        rather than folded into the bridge. A large value is a signal that a turn spent
        real time somewhere untimed — e.g. an auxiliary model call that is not tagged
        planner/responder — and the fix is to wrap THAT stage, not to hunt for DB latency
        in the bridge. Because the leftover always surfaces here, nothing can hide; it just
        no longer gets mislabelled as bridge I/O.

        ``total`` is read from the caller's authoritative ``timing['total_ms']`` when
        present (the whole-turn clock started at the top of ``process_turn``), falling back
        to this timer's own origin otherwise.
        """
        total = int(timing.get("total_ms", self.total_ms()))
        measured = self.sum_ms() + sum(int(timing.get(k, 0)) for k in heavy_keys)
        return max(0, total - measured)
