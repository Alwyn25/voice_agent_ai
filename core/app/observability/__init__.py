"""Observability primitives for the voice agent's hot path.

Two independent tools live here, both built for P1.08 (profile the BRG/bridge
stage and remove blocking calls from the asyncio path):

- ``stage_timer``  — attributes per-turn "bridge" overhead into named sub-stages
                     (session load, message persist, trace write, commit, …) so the
                     otherwise-invisible ~60 ms is measurable and regression-checkable.
- ``loop_monitor`` — a low-overhead background sampler that detects when *anything*
                     stalls the asyncio event loop and logs the offending task's stack,
                     so a synchronous call on the hot path names itself instead of just
                     showing up as latency.

Both are inert unless enabled in config, so importing this package costs nothing at
request time.
"""

from app.observability.loop_monitor import LoopMonitor, get_loop_monitor
from app.observability.stage_timer import StageTimer

__all__ = ["StageTimer", "LoopMonitor", "get_loop_monitor"]
