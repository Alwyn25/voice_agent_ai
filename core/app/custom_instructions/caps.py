"""Per-tenant caps on advisory instructions.

The advisory block rides in the responder's cacheable prefix on every turn, so an
unbounded number (or length) of rows would bloat the prompt and blow the latency/cost
budget. Caps are enforced at AUTHOR time (activation), not on the hot turn path — fail the
admin fast, keep the turn clean. Defaults live in settings so they are tunable per
deployment without a code change (spec Open Q1: tune against real tenant usage).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings


class CapExceeded(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


@dataclass
class Caps:
    max_active_rows: int
    max_total_chars: int


def current_caps() -> Caps:
    s = get_settings()
    return Caps(max_active_rows=s.ci_advisory_max_rows,
                max_total_chars=s.ci_advisory_max_chars)


def check_activation(*, active_snippets: list[str], new_snippet: str) -> None:
    """Raise CapExceeded if activating ``new_snippet`` alongside ``active_snippets`` would
    breach a cap. ``active_snippets`` must EXCLUDE the row being activated (callers pass the
    other currently-active advisory snippets for the same tenant)."""
    caps = current_caps()
    if len(active_snippets) + 1 > caps.max_active_rows:
        raise CapExceeded(
            f"Activating this instruction would exceed the per-tenant limit of "
            f"{caps.max_active_rows} active advisory instructions. Deactivate one first.")
    total = sum(len(s or "") for s in active_snippets) + len(new_snippet or "")
    if total > caps.max_total_chars:
        raise CapExceeded(
            f"Activating this instruction would push the total advisory text to {total} "
            f"characters, over the {caps.max_total_chars}-character budget. Shorten or "
            f"deactivate an instruction.")
