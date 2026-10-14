"""Vocal Backend Tasksheet P3 (Templates): call-flow template CRUD, publish
(snapshot -> template_versions, compile, store hash), duplicate, and version
history. See app/engine/templates/ for the compiler/runtime this wraps.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import Agent, CallFlowStep, CallFlowTemplate, TemplateVersion
from app.engine.templates.compiler import HANDOFF, GraphCompiler
from app.security.deps import Principal, require
from app.services.activity import append_activity_event

router = APIRouter(prefix="/templates", tags=["templates"])

STEP_TYPES = ("SAY", "TOOL", "THINK", "RETRIEVE")


async def _template(db: AsyncSession, tenant_id: uuid.UUID, template_id: uuid.UUID) -> CallFlowTemplate:
    t = (await db.execute(select(CallFlowTemplate).where(
        CallFlowTemplate.tenant_id == tenant_id, CallFlowTemplate.id == template_id))).scalar_one_or_none()
    if t is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Template not found")
    return t


def _validate_step_config(key: str, step_type: str, config: dict) -> None:
    """Minimal shape validation, one rule per node factory's own hard
    requirement (see app/engine/templates/nodes.py) — not a full JSON Schema,
    just enough that a template fails to save with a clear message instead of
    failing to compile with a cryptic KeyError later.

    Only TOOL/RETRIEVE have a genuinely required field now (something to act
    on) — `action` is no longer required from the author: nodes.py auto-picks
    the connection's first catalog action when config omits one, since the
    editor UI dropped the separate "Action" input in favor of a single "Uses"
    picker. `agent_says`/`capture_key` used to be required for SAY/THINK too,
    but both are optional now that any step type can speak a line
    (`agent_says`) and/or record generic `captures`, independent of its type."""
    required = {
        "SAY": [],
        "TOOL": ["tool_connection_id"],
        "THINK": [],
        "RETRIEVE": ["kb_ids"],
    }[step_type]
    absent = [f for f in required if not config.get(f)]
    if absent:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"Step '{key}' ({step_type}) is missing required config field(s): {absent}")


def _template_out(t: CallFlowTemplate, latest_version: TemplateVersion | None, step_count: int) -> dict:
    return {"id": str(t.id), "name": t.name, "description": t.description,
           "is_system_template": t.is_system_template, "step_count": step_count,
           "latest_version": latest_version.version if latest_version else None,
           "created_at": t.created_at, "updated_at": t.updated_at}


def _step_out(s: CallFlowStep) -> dict:
    return {"id": str(s.id), "key": s.key, "step_order": s.step_order,
           "step_type": s.step_type, "config": s.config}


def _version_out(v: TemplateVersion) -> dict:
    return {"id": str(v.id), "template_id": str(v.template_id), "version": v.version,
           "compiled_graph_hash": v.compiled_graph_hash, "step_count": len(v.snapshot or []),
           "published_at": v.published_at, "published_by": str(v.published_by) if v.published_by else None}


class TemplateCreate(BaseModel):
    name: str
    description: str | None = None


@router.get("")
async def list_templates(p: Principal = Depends(require("templates:read")),
                         db: AsyncSession = Depends(get_db)):
    templates = (await db.execute(select(CallFlowTemplate).where(
        CallFlowTemplate.tenant_id == p.tenant_id).order_by(CallFlowTemplate.created_at.desc()))).scalars().all()
    step_counts = dict((await db.execute(
        select(CallFlowStep.template_id, func.count()).where(CallFlowStep.tenant_id == p.tenant_id)
        .group_by(CallFlowStep.template_id))).all())
    # One query for every template's latest version (Postgres DISTINCT ON), not
    # one query per template — a tenant with 50 templates previously cost 52
    # round trips here instead of 3.
    latest_versions = {v.template_id: v for v in (await db.execute(
        select(TemplateVersion).distinct(TemplateVersion.template_id)
        .where(TemplateVersion.tenant_id == p.tenant_id)
        .order_by(TemplateVersion.template_id, TemplateVersion.version.desc()))).scalars().all()}
    return [_template_out(t, latest_versions.get(t.id), int(step_counts.get(t.id, 0))) for t in templates]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_template(body: TemplateCreate, p: Principal = Depends(require("templates:write")),
                          db: AsyncSession = Depends(get_db)):
    t = CallFlowTemplate(tenant_id=p.tenant_id, name=body.name, description=body.description)
    db.add(t)
    await db.flush()
    append_activity_event(db, p, "templates.create", resource_type="call_flow_template",
                          resource_id=str(t.id), after={"name": t.name})
    await db.commit()
    return _template_out(t, None, 0)


@router.get("/{template_id}/steps")
async def get_steps(template_id: uuid.UUID, p: Principal = Depends(require("templates:read")),
                    db: AsyncSession = Depends(get_db)):
    await _template(db, p.tenant_id, template_id)
    steps = (await db.execute(select(CallFlowStep).where(
        CallFlowStep.tenant_id == p.tenant_id, CallFlowStep.template_id == template_id)
        .order_by(CallFlowStep.step_order))).scalars().all()
    return [_step_out(s) for s in steps]


class StepIn(BaseModel):
    key: str
    step_order: int
    step_type: str
    config: dict = {}


@router.put("/{template_id}/steps")
async def put_steps(template_id: uuid.UUID, body: list[StepIn],
                    p: Principal = Depends(require("templates:write")),
                    db: AsyncSession = Depends(get_db)):
    """Bulk upsert: the whole ordered step list replaces the template's current
    working copy atomically (a partial-update PATCH would leave routes.next
    references dangling mid-write). Does not touch template_versions — this is
    always the editable copy; publish snapshots it."""
    await _template(db, p.tenant_id, template_id)
    if not body:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "At least one step is required")

    keys = [s.key for s in body]
    if len(set(keys)) != len(keys):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Step keys must be unique within a template")
    key_set = set(keys)

    for s in body:
        if s.step_type not in STEP_TYPES:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                f"step_type must be one of {STEP_TYPES}")
        _validate_step_config(s.key, s.step_type, s.config)
        for route in (s.config.get("routes") or []):
            target = route.get("next")
            if target not in key_set and target != HANDOFF:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                    f"Step '{s.key}' routes to unknown step '{target}'")

    await db.execute(delete(CallFlowStep).where(
        CallFlowStep.tenant_id == p.tenant_id, CallFlowStep.template_id == template_id))
    for s in body:
        db.add(CallFlowStep(tenant_id=p.tenant_id, template_id=template_id, step_order=s.step_order,
                            key=s.key, step_type=s.step_type, config=s.config))
    append_activity_event(db, p, "templates.steps_update", resource_type="call_flow_template",
                          resource_id=str(template_id), after={"step_count": len(body)})
    await db.commit()

    steps = (await db.execute(select(CallFlowStep).where(
        CallFlowStep.tenant_id == p.tenant_id, CallFlowStep.template_id == template_id)
        .order_by(CallFlowStep.step_order))).scalars().all()
    return [_step_out(s) for s in steps]


@router.post("/{template_id}/publish", status_code=status.HTTP_201_CREATED)
async def publish_template(template_id: uuid.UUID, p: Principal = Depends(require("templates:write")),
                           db: AsyncSession = Depends(get_db)):
    """Snapshot the current steps into an immutable template_versions row, only
    once GraphCompiler.compile() confirms they actually compile — a template
    that doesn't compile is never publishable, so nothing can bind to a version
    that would fail at call time."""
    t = await _template(db, p.tenant_id, template_id)
    steps = (await db.execute(select(CallFlowStep).where(
        CallFlowStep.tenant_id == p.tenant_id, CallFlowStep.template_id == template_id)
        .order_by(CallFlowStep.step_order))).scalars().all()
    if not steps:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Cannot publish a template with no steps")

    step_dicts = [{"key": s.key, "step_order": s.step_order, "step_type": s.step_type,
                  "config": s.config} for s in steps]
    try:
        _, graph_hash = GraphCompiler.compile(step_dicts, p.tenant_id)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Template does not compile: {e}")

    last = (await db.execute(select(TemplateVersion).where(
        TemplateVersion.template_id == template_id).order_by(TemplateVersion.version.desc()))).scalars().first()
    next_version = (last.version + 1) if last else 1
    tv = TemplateVersion(tenant_id=p.tenant_id, template_id=template_id, version=next_version,
                        snapshot=step_dicts, compiled_graph_hash=graph_hash, published_by=p.user_id)
    db.add(tv)
    await db.flush()
    append_activity_event(db, p, "templates.publish", resource_type="template_version",
                          resource_id=str(tv.id),
                          after={"template": t.name, "version": next_version, "graph_hash": graph_hash})
    await db.commit()
    return _version_out(tv)


class DuplicateIn(BaseModel):
    name: str | None = None


@router.post("/{template_id}/duplicate", status_code=status.HTTP_201_CREATED)
async def duplicate_template(template_id: uuid.UUID, body: DuplicateIn,
                             p: Principal = Depends(require("templates:write")),
                             db: AsyncSession = Depends(get_db)):
    """Clones the working copy only (call_flow_steps) — never the source's
    published versions, since those belong to the original template's history."""
    src = await _template(db, p.tenant_id, template_id)
    new_t = CallFlowTemplate(tenant_id=p.tenant_id, name=body.name or f"{src.name} (copy)",
                             description=src.description)
    db.add(new_t)
    await db.flush()

    steps = (await db.execute(select(CallFlowStep).where(
        CallFlowStep.tenant_id == p.tenant_id, CallFlowStep.template_id == template_id))).scalars().all()
    for s in steps:
        db.add(CallFlowStep(tenant_id=p.tenant_id, template_id=new_t.id, step_order=s.step_order,
                            key=s.key, step_type=s.step_type, config=s.config))
    append_activity_event(db, p, "templates.duplicate", resource_type="call_flow_template",
                          resource_id=str(new_t.id), after={"source_template": src.name})
    await db.commit()
    return _template_out(new_t, None, len(steps))


@router.delete("/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_template(template_id: uuid.UUID, p: Principal = Depends(require("templates:write")),
                          db: AsyncSession = Depends(get_db)):
    """Refuses to delete a template with a published version some agent is
    currently bound to (agent.template_version_id) — deleting it would leave
    that agent's calls compiling against a graph that no longer exists. The
    working copy (call_flow_steps) and any unused versions are removed with it."""
    t = await _template(db, p.tenant_id, template_id)
    version_ids = (await db.execute(select(TemplateVersion.id).where(
        TemplateVersion.tenant_id == p.tenant_id, TemplateVersion.template_id == template_id))).scalars().all()
    if version_ids:
        in_use = (await db.execute(select(func.count()).where(
            Agent.tenant_id == p.tenant_id, Agent.template_version_id.in_(version_ids)))).scalar_one()
        if in_use:
            raise HTTPException(status.HTTP_409_CONFLICT,
                                f"{in_use} agent(s) are still using a published version of this template")

    await db.execute(delete(CallFlowStep).where(
        CallFlowStep.tenant_id == p.tenant_id, CallFlowStep.template_id == template_id))
    await db.execute(delete(TemplateVersion).where(
        TemplateVersion.tenant_id == p.tenant_id, TemplateVersion.template_id == template_id))
    await db.delete(t)
    append_activity_event(db, p, "templates.delete", resource_type="call_flow_template",
                          resource_id=str(template_id), before={"name": t.name})
    await db.commit()


@router.get("/{template_id}/versions")
async def list_versions(template_id: uuid.UUID, p: Principal = Depends(require("templates:read")),
                        db: AsyncSession = Depends(get_db)):
    await _template(db, p.tenant_id, template_id)
    versions = (await db.execute(select(TemplateVersion).where(
        TemplateVersion.tenant_id == p.tenant_id, TemplateVersion.template_id == template_id)
        .order_by(TemplateVersion.version.desc()))).scalars().all()
    return [_version_out(v) for v in versions]
