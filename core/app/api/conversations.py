"""Conversations + the planner 'think' endpoint (execution arrives in the next stage)."""
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import Conversation, Customer, Message
from app.engine.memory import working
from app.engine.memory.pipeline import run_post_call, run_post_call_task
from app.engine.planner.planner import plan
from app.engine.planner.turn_runner import process_turn
from app.security.deps import Principal, require

router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationCreate(BaseModel):
    customer_id: uuid.UUID | None = None
    phone: str | None = None
    channel: str = "chat"          # chat | voice
    direction: str = "inbound"     # inbound | outbound


@router.post("", status_code=201)
async def create_conversation(body: ConversationCreate,
                              p: Principal = Depends(require("conversations:read")),
                              db: AsyncSession = Depends(get_db)):
    customer_id = body.customer_id
    if customer_id is None and body.phone:
        from app.services.phone import normalize_phone
        c = (await db.execute(select(Customer).where(
            Customer.tenant_id == p.tenant_id,
            Customer.phone_normalized == normalize_phone(body.phone)))).scalar_one_or_none()
        customer_id = c.id if c else None
    conv = Conversation(tenant_id=p.tenant_id, customer_id=customer_id, channel=body.channel,
                        direction=body.direction, status="active",
                        started_at=datetime.now(timezone.utc))
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return {"id": str(conv.id), "customer_id": str(conv.customer_id) if conv.customer_id else None,
            "channel": conv.channel, "direction": conv.direction, "status": conv.status}


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: uuid.UUID,
                           p: Principal = Depends(require("conversations:read")),
                           db: AsyncSession = Depends(get_db)):
    conv = await _conv(db, p.tenant_id, conversation_id)
    msgs = (await db.execute(select(Message).where(
        Message.conversation_id == conv.id).order_by(Message.created_at))).scalars().all()
    return {"id": str(conv.id), "customer_id": str(conv.customer_id) if conv.customer_id else None,
            "channel": conv.channel, "direction": conv.direction, "status": conv.status,
            "summary": conv.summary, "outcome": conv.outcome,
            "messages": [{"role": m.role, "content": m.content, "at": m.created_at}
                         for m in msgs]}


@router.get("")
async def list_conversations(p: Principal = Depends(require("conversations:read")),
                             db: AsyncSession = Depends(get_db),
                             customer_id: uuid.UUID | None = None,
                             page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=100)):
    q = select(Conversation).where(Conversation.tenant_id == p.tenant_id)
    if customer_id:
        q = q.where(Conversation.customer_id == customer_id)
    rows = (await db.execute(q.order_by(Conversation.created_at.desc())
                             .offset((page - 1) * page_size).limit(page_size))).scalars().all()
    return {"items": [{"id": str(c.id), "customer_id": str(c.customer_id) if c.customer_id else None,
                       "channel": c.channel, "direction": c.direction, "status": c.status,
                       "outcome": c.outcome, "started_at": c.started_at} for c in rows],
            "page": page, "page_size": page_size}


class PlanRequest(BaseModel):
    user_input: str
    stt_confidence: float | None = None
    identity_verified: bool = False
    pending_action: dict | None = None
    record_message: bool = True


@router.post("/{conversation_id}/plan")
async def plan_turn(conversation_id: uuid.UUID, body: PlanRequest,
                    p: Principal = Depends(require("conversations:read")),
                    db: AsyncSession = Depends(get_db)):
    """Runs planner steps 1-7 and returns the proposal WITHOUT executing anything.

    This is the engine's thinking exposed for inspection: which tools were offered,
    which rules fired, what the model proposed, and how confident we are in it.
    """
    conv = await _conv(db, p.tenant_id, conversation_id)
    if body.record_message:
        db.add(Message(tenant_id=p.tenant_id, conversation_id=conv.id,
                       role="customer", content=body.user_input))
        await db.flush()

    proposal = await plan(db, tenant_id=p.tenant_id, conversation=conv,
                          user_input=body.user_input, stt_confidence=body.stt_confidence,
                          identity_verified=body.identity_verified,
                          pending_action=body.pending_action)
    await db.commit()

    c = proposal.confidence
    return {
        "conversation_id": str(conv.id),
        "understood": proposal.understood,
        "action": proposal.action,
        "tool": proposal.tool,
        "params": proposal.params,
        "response_draft": proposal.response_draft,
        "reasoning": proposal.reasoning,
        "confidence": {"score": c.score if c else 0, "band": c.band if c else "low",
                       "self_reported": c.self_confidence if c else 0,
                       "adjustments": c.adjustments if c else []},
        "candidates_offered": proposal.candidates,
        "rules_fired": [r for r in proposal.rules_evaluated if r.get("matched")],
        "forced_escalation": proposal.forced_escalation,
        "escalate_after_tool": proposal.escalate_after_tool,
        "memory_used": proposal.memory_used,
        "knowledge_used": proposal.knowledge_used,
        "retrieval": proposal.retrieval,
        "missing_info": proposal.missing_info,
        "model": proposal.model, "tokens": proposal.tokens,
        "latency_ms": proposal.latency_ms, "error": proposal.error,
        "note": "Planner output only — no action was executed (that is the next stage).",
    }


async def _conv(db: AsyncSession, tenant_id: uuid.UUID, cid: uuid.UUID) -> Conversation:
    conv = (await db.execute(select(Conversation).where(
        Conversation.tenant_id == tenant_id, Conversation.id == cid))).scalar_one_or_none()
    if conv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    return conv


class TurnRequest(BaseModel):
    user_input: str
    stt_confidence: float | None = None
    channel: str | None = None


@router.post("/{conversation_id}/turns")
async def take_turn(conversation_id: uuid.UUID, body: TurnRequest,
                    p: Principal = Depends(require("conversations:read")),
                    db: AsyncSession = Depends(get_db)):
    """The real process_turn: understand, decide, validate, execute, respond, record.

    This is the single entry point every channel uses (voice, chat, and later WhatsApp).
    """
    conv = await _conv(db, p.tenant_id, conversation_id)
    if conv.status != "active":
        raise HTTPException(status.HTTP_409_CONFLICT, f"Conversation is {conv.status}")

    result = await process_turn(db, tenant_id=p.tenant_id, conversation=conv,
                                user_input=body.user_input,
                                stt_confidence=body.stt_confidence, actor=p.actor)
    return {
        "response_text": result.response_text,
        "response_segments": result.response_segments,
        "action": result.action,
        "executed_tools": result.executed_tools,
        "state_changed_to": result.state_changed_to,
        "escalation": result.escalation,
        "end_call": result.end_call,
        "trace_id": str(result.trace_id) if result.trace_id else None,
        "confidence": result.confidence,
        "pending_confirmation": result.pending_confirmation,
        "debug": result.debug,
        "tool_failure": (result.executed_tools[0] if result.executed_tools
                         and result.executed_tools[0].get("status") not in ("success", "skipped_duplicate")
                         else None),
    }


class EndRequest(BaseModel):
    reason: str | None = None
    outcome: str | None = None
    analyse: bool = True


@router.post("/{conversation_id}/end")
async def end_conversation(conversation_id: uuid.UUID, body: EndRequest,
                           background: BackgroundTasks,
                           p: Principal = Depends(require("conversations:read")),
                           db: AsyncSession = Depends(get_db)):
    """Close the conversation and queue post-call analysis.

    The analysis (summary, fact extraction, profile promotion) runs in the background so
    hanging up is never delayed by it. Set analyse=false to skip, or call
    POST /conversations/{id}/analyse to run it synchronously."""
    conv = await _conv(db, p.tenant_id, conversation_id)
    conv.status = "completed"
    conv.ended_at = datetime.now(timezone.utc)
    if body.outcome:
        conv.outcome = body.outcome
    await db.commit()
    await working.clear(conv.id)
    if body.analyse:
        background.add_task(run_post_call_task, p.tenant_id, conv.id)
    return {"id": str(conv.id), "status": conv.status, "outcome": conv.outcome,
            "analysis_queued": body.analyse}


@router.post("/{conversation_id}/analyse")
async def analyse_conversation(conversation_id: uuid.UUID,
                               p: Principal = Depends(require("conversations:read")),
                               db: AsyncSession = Depends(get_db)):
    """Run the post-call memory pipeline synchronously and return what it learned.

    Safe to re-run: facts with unchanged values are left alone, changed values supersede
    the previous row rather than overwriting it."""
    await _conv(db, p.tenant_id, conversation_id)
    report = await run_post_call(db, tenant_id=p.tenant_id, conversation_id=conversation_id)
    await db.commit()
    return {
        "conversation_id": str(report.conversation_id),
        "skipped": report.skipped,
        "summary": report.summary, "outcome": report.outcome, "sentiment": report.sentiment,
        "facts_added": report.facts_added,
        "facts_superseded": report.facts_superseded,
        "facts_unchanged": report.facts_unchanged,
        "candidates_for_review": report.candidates,
        "profile_updated": report.profile_updated,
        "product_interests": report.product_interests,
        "model": report.model, "tokens": report.tokens,
    }


@router.get("/{conversation_id}/traces")
async def conversation_traces(conversation_id: uuid.UUID,
                              p: Principal = Depends(require("audit:read")),
                              db: AsyncSession = Depends(get_db)):
    """Every decision made in this conversation, in order — the flight recorder."""
    from app.db.models import DecisionTrace
    await _conv(db, p.tenant_id, conversation_id)
    rows = (await db.execute(select(DecisionTrace).where(
        DecisionTrace.tenant_id == p.tenant_id,
        DecisionTrace.conversation_id == conversation_id)
        .order_by(DecisionTrace.turn_index))).scalars().all()
    return [{"trace_id": str(t.id), "turn": t.turn_index, "input": t.input_text,
             "reasoning": t.planner_reasoning, "proposed_tool": t.proposed_tool,
             "params": t.tool_params, "confidence": float(t.confidence) if t.confidence else None,
             "band": t.confidence_band, "action_taken": t.action_taken,
             "rules_evaluated": t.rules_evaluated,
             "rules_matched": [r.get("rule") for r in (t.rules_evaluated or []) if r.get("matched")],
             "memory_used": t.retrieved_memory, "knowledge_used": t.retrieved_chunks,
             "validation": t.validation_result,
             "blocked_reason": (t.validation_result or {}).get("error"),
             "failed_check": (t.validation_result or {}).get("failed_check"), "latency_ms": t.latency_ms,
             "model": t.model_used, "tokens": t.tokens, "created_at": t.created_at} for t in rows]
