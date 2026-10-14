"""CallState (Vocal Backend Tasksheet P3) — the state every compiled call-flow
graph carries through a call. Every node factory in nodes.py reads a subset of
this and returns a partial dict; LangGraph merges it into the running state.
"""
from typing import Any, TypedDict


class Turn(TypedDict):
    role: str  # customer|agent
    text: str


class CallState(TypedDict):
    call_id: str
    transcript: list[Turn]
    captures: dict[str, Any]
    tool_results: dict[str, Any]
    retrieved_chunks: list[dict]
    intent: str | None
    sentiment: str | None
    handoff: bool


def new_call_state(call_id: str) -> CallState:
    return CallState(call_id=call_id, transcript=[], captures={}, tool_results={},
                     retrieved_chunks=[], intent=None, sentiment=None, handoff=False)
