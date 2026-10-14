"""The frozen process_turn contract (v1). Do not change without a version bump."""
from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class TurnAction(str, Enum):
    SPOKE = "spoke"
    TOOL_EXECUTED = "tool_executed"
    CLARIFY = "clarify"
    CONFIRM = "confirm"
    ESCALATE = "escalate"
    END_CALL = "end_call"
    TRANSFER = "transfer"


class TurnMetadata(BaseModel):
    stt_confidence: float | None = None
    interrupted: bool = False
    audio_offset_ms: int | None = None


class ProcessTurnRequest(BaseModel):
    conversation_id: UUID
    tenant_id: UUID
    customer_id: UUID | None = None
    channel: str = Field(pattern="^(voice|chat)$")
    user_input: str
    input_metadata: TurnMetadata = TurnMetadata()


class ToolResult(BaseModel):
    tool_name: str
    status: str  # success | failed | blocked
    output: dict[str, Any] = {}
    error: str | None = None


class EscalationDirective(BaseModel):
    escalation_type: str  # immediate | scheduled | ticket
    reason: str
    context_summary: str


class AgentResponse(BaseModel):
    response_text: str
    response_segments: list[str] = []
    action: TurnAction
    executed_tools: list[ToolResult] = []
    state_changed_to: str | None = None
    escalation: EscalationDirective | None = None
    end_call: bool = False
    trace_id: UUID | None = None
