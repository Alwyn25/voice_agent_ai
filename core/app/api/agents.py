"""Vocal Backend Tasksheet P4 (Agents): a deployable agent binds a published
call-flow template version (see P3), an optional phone number, and its own
knowledge-base/tool attachments. agent_metrics_daily is backfilled zero until
P6 (Calls Runtime) and P7 (Overview) exist to write real numbers — every
roster/summary/metrics query below already reads from it, so nothing here
needs to change once those land.
"""
import uuid
from datetime import date, timedelta

import app.integrations  # noqa: F401 — registers every mock adapter (see app/integrations/adapters/)
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import (Agent, AgentGuardrail, AgentKnowledgeBase, AgentMetricsDaily, AgentTool,
                           Campaign, KnowledgeBase, PhoneNumber, TemplateVersion, ToolConnection)
from app.integrations.base import get_adapter
from app.security.deps import Principal, require
from app.services.activity import append_activity_event

router = APIRouter(prefix="/agents", tags=["agents"])

STATUSES = ("draft", "live", "paused")
DIRECTIONS = ("inbound", "outbound")


async def _agent(db: AsyncSession, tenant_id: uuid.UUID, agent_id: uuid.UUID) -> Agent:
    row = (await db.execute(select(Agent).where(
        Agent.tenant_id == tenant_id, Agent.id == agent_id,
        Agent.is_deleted == False))).scalar_one_or_none()  # noqa: E712
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Agent not found")
    return row


async def _metrics_rollup(db: AsyncSession, tenant_id: uuid.UUID, agent_ids: list[uuid.UUID],
                          since: date) -> dict[uuid.UUID, dict]:
    """Summed rollup per agent since `since` — one query for the whole roster,
    not one per agent (see P3/P4's own N+1 lesson from the Wave 4 review)."""
    if not agent_ids:
        return {}
    rows = (await db.execute(
        select(AgentMetricsDaily.agent_id, func.sum(AgentMetricsDaily.calls),
              func.sum(AgentMetricsDaily.resolved), func.avg(AgentMetricsDaily.avg_duration_seconds),
              func.sum(AgentMetricsDaily.actions_taken))
        .where(AgentMetricsDaily.tenant_id == tenant_id, AgentMetricsDaily.agent_id.in_(agent_ids),
              AgentMetricsDaily.date >= since)
        .group_by(AgentMetricsDaily.agent_id))).all()
    out = {}
    for agent_id, calls, resolved, avg_dur, actions in rows:
        calls = int(calls or 0)
        out[agent_id] = {
            "calls": calls,
            "resolved_pct": round(100 * int(resolved or 0) / calls, 1) if calls else None,
            "avg_duration_seconds": round(float(avg_dur), 1) if avg_dur is not None else None,
            "actions_taken": int(actions or 0),
        }
    return out


async def _campaign_assignments(db: AsyncSession, tenant_id: uuid.UUID,
                                agent_ids: list[uuid.UUID]) -> dict[uuid.UUID, uuid.UUID]:
    """Vocal Backend Tasksheet P5 gave Campaign an agent_id (migration 0024) — this is now a
    real join, not the honestly-null placeholder P4 shipped with. An agent can in principle be
    referenced by more than one campaign; the most recently created one wins, since that is the
    one the roster card cares about."""
    if not agent_ids:
        return {}
    rows = (await db.execute(select(Campaign.agent_id, Campaign.id)
        .where(Campaign.tenant_id == tenant_id, Campaign.agent_id.in_(agent_ids))
        .order_by(Campaign.created_at.desc()))).all()
    out: dict[uuid.UUID, uuid.UUID] = {}
    for agent_id, campaign_id in rows:
        out.setdefault(agent_id, campaign_id)
    return out


async def _phone_map(db: AsyncSession, tenant_id: uuid.UUID, phone_ids: list[uuid.UUID]) -> dict[uuid.UUID, str]:
    if not phone_ids:
        return {}
    rows = (await db.execute(select(PhoneNumber).where(
        PhoneNumber.tenant_id == tenant_id, PhoneNumber.id.in_(phone_ids)))).scalars().all()
    return {p.id: p.e164 for p in rows}


def _agent_out(a: Agent, metrics: dict | None, campaign_id: uuid.UUID | None = None, phone: str | None = None) -> dict:
    m = metrics or {"calls": 0, "resolved_pct": None, "avg_duration_seconds": None, "actions_taken": 0}
    return {"id": str(a.id), "name": a.name, "purpose_label": a.purpose_label,
           "direction": a.direction, "status": a.status, "voice": a.voice,
           "template_version_id": str(a.template_version_id) if a.template_version_id else None,
           "phone_number_id": str(a.phone_number_id) if a.phone_number_id else None,
           "phone": phone,
           "campaign_id": str(campaign_id) if campaign_id else None,
           "calls_7d": m["calls"], "resolved_pct_7d": m["resolved_pct"],
           "avg_duration_seconds_7d": m["avg_duration_seconds"], "actions_taken_7d": m["actions_taken"],
           "created_at": a.created_at, "updated_at": a.updated_at,
           # Behaviour tab
           "description": a.description, "language": a.language,
           "system_instructions": a.system_instructions, "personality": a.personality or {},
           "greeting_message": a.greeting_message, "closing_message": a.closing_message,
           # Voice tab
           "stt_model": a.stt_model, "tts_model": a.tts_model, "voice_accent": a.voice_accent,
           "speaking_speed": a.speaking_speed, "response_latency_ms": a.response_latency_ms,
           "interruption_sensitivity": a.interruption_sensitivity,
           "silence_timeout_seconds": a.silence_timeout_seconds,
           "background_noise": a.background_noise, "call_quality": a.call_quality}


@router.get("")
async def list_agents(p: Principal = Depends(require("agents:read")), db: AsyncSession = Depends(get_db)):
    """Roster: joins the latest metrics rollup and phone assignment in one pass
    (Definition of Done: roster fields satisfiable from one query per concern,
    not N+1)."""
    agents = (await db.execute(select(Agent).where(
        Agent.tenant_id == p.tenant_id, Agent.is_deleted == False)  # noqa: E712
        .order_by(Agent.created_at.desc()))).scalars().all()
    agent_ids = [a.id for a in agents]
    metrics = await _metrics_rollup(db, p.tenant_id, agent_ids, date.today() - timedelta(days=7))
    campaigns = await _campaign_assignments(db, p.tenant_id, agent_ids)
    phone_ids = [a.phone_number_id for a in agents if a.phone_number_id]
    phone_map = await _phone_map(db, p.tenant_id, phone_ids)
    return [_agent_out(a, metrics.get(a.id), campaigns.get(a.id), phone_map.get(a.phone_number_id)) for a in agents]


@router.get("/summary")
async def agents_summary(p: Principal = Depends(require("agents:read")), db: AsyncSession = Depends(get_db)):
    """Scalar header stats for the roster page (UI shows AGENTS DEFINED / LIVE
    NOW / CALLS TODAY)."""
    agents_defined = int((await db.execute(select(func.count()).select_from(Agent).where(
        Agent.tenant_id == p.tenant_id, Agent.is_deleted == False))).scalar_one())  # noqa: E712
    live_now = int((await db.execute(select(func.count()).select_from(Agent).where(
        Agent.tenant_id == p.tenant_id, Agent.status == "live",
        Agent.is_deleted == False))).scalar_one())  # noqa: E712
    calls_today = int((await db.execute(select(func.coalesce(func.sum(AgentMetricsDaily.calls), 0)).where(
        AgentMetricsDaily.tenant_id == p.tenant_id, AgentMetricsDaily.date == date.today()))).scalar_one())
    return {"agents_defined": agents_defined, "live_now": live_now, "calls_today": calls_today}


class AgentCreate(BaseModel):
    name: str
    purpose_label: str | None = None
    direction: str = "inbound"
    template_version_id: uuid.UUID | None = None
    voice: str | None = None


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_agent(body: AgentCreate, p: Principal = Depends(require("agents:write")),
                       db: AsyncSession = Depends(get_db)):
    if body.direction not in DIRECTIONS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"direction must be one of {DIRECTIONS}")
    if body.template_version_id is not None:
        tv = (await db.execute(select(TemplateVersion).where(
            TemplateVersion.tenant_id == p.tenant_id,
            TemplateVersion.id == body.template_version_id))).scalar_one_or_none()
        if tv is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "template_version_id not found")

    agent = Agent(tenant_id=p.tenant_id, name=body.name, purpose_label=body.purpose_label,
                 direction=body.direction, template_version_id=body.template_version_id, voice=body.voice)
    db.add(agent)
    await db.flush()
    append_activity_event(db, p, "agents.create", resource_type="agent", resource_id=str(agent.id),
                          after={"name": agent.name, "direction": agent.direction})
    await db.commit()
    return _agent_out(agent, None)


class PersonalityIn(BaseModel):
    warmth: int | None = None
    formality: int | None = None
    verbosity: int | None = None
    persistence: int | None = None


class AgentPatch(BaseModel):
    status: str | None = None
    template_version_id: uuid.UUID | None = None
    # Behaviour tab
    name: str | None = None
    description: str | None = None
    language: str | None = None
    system_instructions: str | None = None
    personality: PersonalityIn | None = None
    greeting_message: str | None = None
    closing_message: str | None = None
    # Voice tab
    stt_model: str | None = None
    tts_model: str | None = None
    voice_accent: str | None = None
    speaking_speed: float | None = None
    response_latency_ms: int | None = None
    interruption_sensitivity: int | None = None
    silence_timeout_seconds: int | None = None
    background_noise: str | None = None
    call_quality: str | None = None


@router.patch("/{agent_id}")
async def update_agent(agent_id: uuid.UUID, body: AgentPatch, p: Principal = Depends(require("agents:write")),
                       db: AsyncSession = Depends(get_db)):
    agent = await _agent(db, p.tenant_id, agent_id)
    before = {"status": agent.status, "template_version_id": str(agent.template_version_id or "")}

    # template_version_id is applied BEFORE the status check below, so a single
    # request that sets both (assign a template and go live at once) validates
    # against the NEW template, not the stale one still on the row.
    if body.template_version_id is not None:
        tv = (await db.execute(select(TemplateVersion).where(
            TemplateVersion.tenant_id == p.tenant_id,
            TemplateVersion.id == body.template_version_id))).scalar_one_or_none()
        if tv is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "template_version_id not found")
        agent.template_version_id = body.template_version_id

    if body.status is not None:
        if body.status not in STATUSES:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"status must be one of {STATUSES}")
        if body.status == "live" and agent.template_version_id is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                "Cannot go live with no template_version_id assigned")
        agent.status = body.status

    if body.name is not None:
        if not body.name.strip():
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "name cannot be blank")
        agent.name = body.name.strip()

    # Behaviour/Voice tabs — plain field assignment, no cross-field validation needed.
    for field in ("description", "language", "system_instructions", "greeting_message",
                 "closing_message", "stt_model", "tts_model",
                 "voice_accent", "speaking_speed", "response_latency_ms",
                 "interruption_sensitivity", "silence_timeout_seconds",
                 "background_noise", "call_quality"):
        value = getattr(body, field)
        if value is not None:
            setattr(agent, field, value)
    if body.personality is not None:
        agent.personality = {**(agent.personality or {}),
                            **{k: v for k, v in body.personality.model_dump().items() if v is not None}}

    # Definition of Done: a status change is audited.
    append_activity_event(db, p, "agents.update", resource_type="agent", resource_id=str(agent.id),
                          before=before,
                          after={"status": agent.status, "template_version_id": str(agent.template_version_id or "")})
    await db.commit()
    # Explicit refresh: unlike a fresh INSERT (whose RETURNING clause already
    # carries every server-computed column back), an UPDATE only touches
    # `updated_at` via its onupdate=func.now() trigger — accessing it
    # unrefreshed on the in-memory object is what was crashing this endpoint
    # with a sync/async MissingGreenlet error. Same fix already used by
    # app/api/scheduling_admin.py's create_location/create_staff_member.
    await db.refresh(agent)
    return _agent_out(agent, None)


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(agent_id: uuid.UUID, p: Principal = Depends(require("agents:write")),
                       db: AsyncSession = Depends(get_db)):
    """Soft-delete (matches customers.py's is_deleted convention): agents are
    referenced by calls/tickets/agent_metrics_daily with a plain FK and no
    ondelete rule, so a hard delete would either violate those constraints or
    orphan call history. The row — and its history — stays; it just drops out
    of the roster/summary."""
    agent = await _agent(db, p.tenant_id, agent_id)
    agent.is_deleted = True
    append_activity_event(db, p, "agents.soft_delete", resource_type="agent",
                          resource_id=str(agent_id), before={"name": agent.name, "status": agent.status})
    await db.commit()


class AttachKBIn(BaseModel):
    knowledge_base_ids: list[uuid.UUID]


@router.post("/{agent_id}/knowledge-bases", status_code=status.HTTP_201_CREATED)
async def attach_knowledge_bases(agent_id: uuid.UUID, body: AttachKBIn,
                                 p: Principal = Depends(require("agents:write")),
                                 db: AsyncSession = Depends(get_db)):
    await _agent(db, p.tenant_id, agent_id)
    existing = {r.knowledge_base_id for r in (await db.execute(select(AgentKnowledgeBase).where(
        AgentKnowledgeBase.tenant_id == p.tenant_id, AgentKnowledgeBase.agent_id == agent_id))).scalars().all()}
    added = []
    for kb_id in body.knowledge_base_ids:
        if kb_id in existing:
            continue
        kb = (await db.execute(select(KnowledgeBase).where(
            KnowledgeBase.tenant_id == p.tenant_id, KnowledgeBase.id == kb_id))).scalar_one_or_none()
        if kb is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"knowledge_base {kb_id} not found")
        db.add(AgentKnowledgeBase(tenant_id=p.tenant_id, agent_id=agent_id, knowledge_base_id=kb_id))
        added.append(str(kb_id))
    if added:
        append_activity_event(db, p, "agents.attach_knowledge_base", resource_type="agent",
                              resource_id=str(agent_id), after={"knowledge_base_ids": added})
    await db.commit()
    rows = (await db.execute(select(AgentKnowledgeBase).where(
        AgentKnowledgeBase.tenant_id == p.tenant_id, AgentKnowledgeBase.agent_id == agent_id))).scalars().all()
    return [{"knowledge_base_id": str(r.knowledge_base_id)} for r in rows]


class AttachToolIn(BaseModel):
    tool_connection_ids: list[uuid.UUID]


@router.post("/{agent_id}/tools", status_code=status.HTTP_201_CREATED)
async def attach_tools(agent_id: uuid.UUID, body: AttachToolIn, p: Principal = Depends(require("agents:write")),
                       db: AsyncSession = Depends(get_db)):
    await _agent(db, p.tenant_id, agent_id)
    existing = {r.tool_connection_id for r in (await db.execute(select(AgentTool).where(
        AgentTool.tenant_id == p.tenant_id, AgentTool.agent_id == agent_id))).scalars().all()}
    added = []
    for conn_id in body.tool_connection_ids:
        if conn_id in existing:
            continue
        conn = (await db.execute(select(ToolConnection).where(
            ToolConnection.tenant_id == p.tenant_id, ToolConnection.id == conn_id))).scalar_one_or_none()
        if conn is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"tool_connection {conn_id} not found")
        db.add(AgentTool(tenant_id=p.tenant_id, agent_id=agent_id, tool_connection_id=conn_id))
        added.append(str(conn_id))
    if added:
        append_activity_event(db, p, "agents.attach_tool", resource_type="agent",
                              resource_id=str(agent_id), after={"tool_connection_ids": added})
    await db.commit()
    rows = (await db.execute(select(AgentTool).where(
        AgentTool.tenant_id == p.tenant_id, AgentTool.agent_id == agent_id))).scalars().all()
    return [{"tool_connection_id": str(r.tool_connection_id), "enabled": r.enabled} for r in rows]


class PhoneNumberIn(BaseModel):
    e164: str | None = None  # omit to provision a new mock number via Twilio
    area_code: str | None = None


@router.post("/{agent_id}/phone-numbers", status_code=status.HTTP_201_CREATED)
async def assign_phone_number(agent_id: uuid.UUID, body: PhoneNumberIn,
                              p: Principal = Depends(require("agents:write")),
                              db: AsyncSession = Depends(get_db)):
    agent = await _agent(db, p.tenant_id, agent_id)

    if body.e164:
        e164 = body.e164
    else:
        adapter = get_adapter("twilio", {}, {})
        result = adapter.execute("provision_number", {"area_code": body.area_code})
        if not result.success:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, result.error or "Number provisioning failed")
        e164 = result.data["e164"]

    phone = (await db.execute(select(PhoneNumber).where(
        PhoneNumber.tenant_id == p.tenant_id, PhoneNumber.e164 == e164))).scalar_one_or_none()
    if phone is not None and phone.assigned_to and phone.assigned_to != agent_id:
        raise HTTPException(status.HTTP_409_CONFLICT, "That number is already assigned to another agent")

    if phone is None:
        phone = PhoneNumber(tenant_id=p.tenant_id, e164=e164, provider="twilio", capability="both",
                            status="active", assigned_to=agent_id)
        db.add(phone)
        await db.flush()
    else:
        phone.assigned_to = agent_id

    agent.phone_number_id = phone.id
    append_activity_event(db, p, "agents.phone_assign", resource_type="agent", resource_id=str(agent_id),
                          after={"e164": phone.e164})
    await db.commit()
    return {"agent_id": str(agent_id), "phone_number_id": str(phone.id), "e164": phone.e164}


@router.get("/{agent_id}/metrics")
async def agent_metrics(agent_id: uuid.UUID, range: str = Query("7d", pattern=r"^\d+d$"),
                        p: Principal = Depends(require("agents:read")), db: AsyncSession = Depends(get_db)):
    """Sparkline data for the roster card — one row per day, zero-filled for a
    day with no rollup yet rather than a gap, so the frontend never has to
    special-case a missing point."""
    await _agent(db, p.tenant_id, agent_id)
    days = int(range[:-1])
    since = date.today() - timedelta(days=days - 1)
    rows = {r.date: r for r in (await db.execute(select(AgentMetricsDaily).where(
        AgentMetricsDaily.tenant_id == p.tenant_id, AgentMetricsDaily.agent_id == agent_id,
        AgentMetricsDaily.date >= since))).scalars().all()}
    out = []
    d = since
    today = date.today()
    while d <= today:
        r = rows.get(d)
        out.append({"date": str(d), "calls": r.calls if r else 0, "resolved": r.resolved if r else 0,
                   "avg_duration_seconds": r.avg_duration_seconds if r else 0,
                   "actions_taken": r.actions_taken if r else 0})
        d += timedelta(days=1)
    return {"agent_id": str(agent_id), "range": range, "series": out}


def _guardrail_out(g: AgentGuardrail) -> dict:
    return {"id": str(g.id), "agent_id": str(g.agent_id), "order_index": g.order_index,
           "condition_text": g.condition_text, "action_text": g.action_text, "active": g.active}


@router.get("/{agent_id}/guardrails")
async def list_guardrails(agent_id: uuid.UUID, p: Principal = Depends(require("agents:read")),
                          db: AsyncSession = Depends(get_db)):
    """Guardrails tab: this agent's intercept rules, in display order."""
    await _agent(db, p.tenant_id, agent_id)
    rows = (await db.execute(select(AgentGuardrail).where(
        AgentGuardrail.tenant_id == p.tenant_id, AgentGuardrail.agent_id == agent_id)
        .order_by(AgentGuardrail.order_index))).scalars().all()
    return [_guardrail_out(g) for g in rows]


class GuardrailCreate(BaseModel):
    condition_text: str
    action_text: str
    active: bool = True


@router.post("/{agent_id}/guardrails", status_code=status.HTTP_201_CREATED)
async def create_guardrail(agent_id: uuid.UUID, body: GuardrailCreate,
                           p: Principal = Depends(require("agents:write")),
                           db: AsyncSession = Depends(get_db)):
    await _agent(db, p.tenant_id, agent_id)
    max_order = (await db.execute(select(func.max(AgentGuardrail.order_index)).where(
        AgentGuardrail.tenant_id == p.tenant_id, AgentGuardrail.agent_id == agent_id))).scalar_one()
    rule = AgentGuardrail(tenant_id=p.tenant_id, agent_id=agent_id,
                          order_index=(max_order or 0) + 1,
                          condition_text=body.condition_text, action_text=body.action_text,
                          active=body.active)
    db.add(rule)
    await db.flush()
    append_activity_event(db, p, "agents.guardrail_create", resource_type="agent", resource_id=str(agent_id),
                          after={"condition_text": rule.condition_text, "action_text": rule.action_text})
    await db.commit()
    return _guardrail_out(rule)


class GuardrailPatch(BaseModel):
    condition_text: str | None = None
    action_text: str | None = None
    active: bool | None = None
    order_index: int | None = None


async def _guardrail(db: AsyncSession, tenant_id: uuid.UUID, agent_id: uuid.UUID,
                     rule_id: uuid.UUID) -> AgentGuardrail:
    row = (await db.execute(select(AgentGuardrail).where(
        AgentGuardrail.tenant_id == tenant_id, AgentGuardrail.agent_id == agent_id,
        AgentGuardrail.id == rule_id))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Guardrail rule not found")
    return row


@router.patch("/{agent_id}/guardrails/{rule_id}")
async def update_guardrail(agent_id: uuid.UUID, rule_id: uuid.UUID, body: GuardrailPatch,
                           p: Principal = Depends(require("agents:write")),
                           db: AsyncSession = Depends(get_db)):
    await _agent(db, p.tenant_id, agent_id)
    rule = await _guardrail(db, p.tenant_id, agent_id, rule_id)
    for field in ("condition_text", "action_text", "active", "order_index"):
        value = getattr(body, field)
        if value is not None:
            setattr(rule, field, value)
    await db.commit()
    await db.refresh(rule)
    return _guardrail_out(rule)


@router.delete("/{agent_id}/guardrails/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_guardrail(agent_id: uuid.UUID, rule_id: uuid.UUID,
                           p: Principal = Depends(require("agents:write")),
                           db: AsyncSession = Depends(get_db)):
    await _agent(db, p.tenant_id, agent_id)
    rule = await _guardrail(db, p.tenant_id, agent_id, rule_id)
    await db.delete(rule)
    append_activity_event(db, p, "agents.guardrail_delete", resource_type="agent",
                          resource_id=str(agent_id), before={"condition_text": rule.condition_text})
    await db.commit()
