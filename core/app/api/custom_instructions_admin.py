"""Admin API for Custom Instructions (advisory, Phase 1) and Dynamic Knowledge.

Mirrors rules_admin.py conventions: permission-guarded, audited, versioned. The author
flow is: compile-preview -> create (draft, inactive) -> review the compiled form + test
scenarios -> activate. Update makes a new version; delete is a soft deactivate that keeps
the audit trail (an explicit ?purge=true hard-deletes, separately gated).
"""
import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.custom_instructions import caps, facts_guard, service
from app.db.base import get_db
from app.db.models import AuditEvent, CustomInstruction, ProjectFactDynamic
from app.providers.llm.base import LLMError
from app.security.deps import Principal, require

router = APIRouter(prefix="/custom-instructions", tags=["custom-instructions"])


# --------------------------------------------------------------------------- serializers
def _ci_out(r: CustomInstruction) -> dict:
    return {"id": str(r.id), "project_id": str(r.project_id) if r.project_id else None,
            "raw_text": r.raw_text, "kind": r.kind,
            "compiled_prompt": r.compiled_prompt, "scope": r.scope or {},
            "priority": r.priority, "is_active": r.is_active, "version": r.version,
            "root_id": str(r.root_id) if r.root_id else None,
            "preview": r.preview, "created_at": r.created_at, "updated_at": r.updated_at}


async def _get_ci(db: AsyncSession, tenant_id: uuid.UUID, ci_id: uuid.UUID) -> CustomInstruction:
    row = (await db.execute(select(CustomInstruction).where(
        CustomInstruction.tenant_id == tenant_id,
        CustomInstruction.id == ci_id))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Custom instruction not found")
    return row


# --------------------------------------------------------------------------- compile-only
class CompileBody(BaseModel):
    raw_text: str


@router.post("/compile-preview")
async def compile_preview(body: CompileBody,
                          p: Principal = Depends(require("instructions:write")),
                          db: AsyncSession = Depends(get_db)):
    """Run the classify+compile call WITHOUT saving. Powers the live authoring preview."""
    try:
        result = await service.compile_preview(db, p.tenant_id, body.raw_text)
    except LLMError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                            {"code": "COMPILE_FAILED", "detail": str(e)})
    return result.as_preview()


# --------------------------------------------------------------------------- CRUD
class CreateBody(BaseModel):
    raw_text: str
    project_id: uuid.UUID | None = None
    # Optional overrides; when omitted the compiler's suggested scope/priority are used.
    scope: dict | None = None
    priority: int | None = None


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(body: CreateBody,
                 p: Principal = Depends(require("instructions:write")),
                 db: AsyncSession = Depends(get_db)):
    """Compile the English and store an INACTIVE draft (version 1). Never auto-activates."""
    try:
        compiled = await service.compile_preview(db, p.tenant_id, body.raw_text)
    except LLMError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                            {"code": "COMPILE_FAILED", "detail": str(e)})

    row = CustomInstruction(
        tenant_id=p.tenant_id, project_id=body.project_id,
        raw_text=body.raw_text.strip(), kind=compiled.kind,
        compiled_prompt=compiled.compiled_prompt,
        scope=(body.scope if body.scope is not None else compiled.scope),
        priority=(body.priority if body.priority is not None else 100),
        preview=compiled.as_preview(), is_active=False, version=1,
        created_by=p.user_id)
    db.add(row)
    await db.flush()
    row.root_id = row.id  # first version is its own root
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="instructions.create",
                      resource_type="custom_instruction", resource_id=str(row.id),
                      after={"kind": compiled.kind, "activatable": compiled.activatable,
                             "grounding_gaps": compiled.grounding_gaps}))
    await db.commit()
    return _ci_out(row) | {"activatable": compiled.activatable}


@router.get("")
async def list_instructions(
        active: bool | None = None, kind: str | None = None,
        p: Principal = Depends(require("instructions:read")),
        db: AsyncSession = Depends(get_db)):
    """Latest version of each logical instruction (grouped by root_id)."""
    q = select(CustomInstruction).where(CustomInstruction.tenant_id == p.tenant_id)
    if kind:
        q = q.where(CustomInstruction.kind == kind)
    rows = (await db.execute(q.order_by(CustomInstruction.version.desc()))).scalars().all()
    latest: dict[uuid.UUID, CustomInstruction] = {}
    for r in rows:
        key = r.root_id or r.id
        if key not in latest:  # rows are version-desc, so the first seen is newest
            latest[key] = r
    out = [r for r in latest.values()]
    if active is not None:
        out = [r for r in out if r.is_active == active]
    out.sort(key=lambda r: (r.priority, r.created_at))
    return [_ci_out(r) for r in out]


@router.get("/{ci_id}")
async def get_instruction(ci_id: uuid.UUID,
                          p: Principal = Depends(require("instructions:read")),
                          db: AsyncSession = Depends(get_db)):
    return _ci_out(await _get_ci(db, p.tenant_id, ci_id))


@router.get("/{ci_id}/versions")
async def versions(ci_id: uuid.UUID,
                   p: Principal = Depends(require("instructions:read")),
                   db: AsyncSession = Depends(get_db)):
    row = await _get_ci(db, p.tenant_id, ci_id)
    root = row.root_id or row.id
    rows = (await db.execute(select(CustomInstruction).where(
        CustomInstruction.tenant_id == p.tenant_id,
        (CustomInstruction.root_id == root) | (CustomInstruction.id == root))
        .order_by(CustomInstruction.version.desc()))).scalars().all()
    return [{"id": str(r.id), "version": r.version, "is_active": r.is_active,
             "kind": r.kind, "created_at": r.created_at} for r in rows]


class UpdateBody(BaseModel):
    raw_text: str | None = None
    scope: dict | None = None
    priority: int | None = None


@router.put("/{ci_id}")
async def update(ci_id: uuid.UUID, body: UpdateBody,
                 p: Principal = Depends(require("instructions:write")),
                 db: AsyncSession = Depends(get_db)):
    """Create a NEW version (old retained, deactivated). Recompiles when raw_text changes."""
    prev = await _get_ci(db, p.tenant_id, ci_id)
    root = prev.root_id or prev.id
    raw_text = (body.raw_text or prev.raw_text).strip()

    if body.raw_text and body.raw_text.strip() != prev.raw_text:
        try:
            compiled = await service.compile_preview(db, p.tenant_id, raw_text)
        except LLMError as e:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                                {"code": "COMPILE_FAILED", "detail": str(e)})
        kind, compiled_prompt = compiled.kind, compiled.compiled_prompt
        preview, suggested_scope = compiled.as_preview(), compiled.scope
    else:
        kind, compiled_prompt = prev.kind, prev.compiled_prompt
        preview, suggested_scope = prev.preview, prev.scope or {}

    # Highest existing version in this lineage + 1.
    max_v = (await db.execute(select(CustomInstruction.version).where(
        CustomInstruction.tenant_id == p.tenant_id,
        (CustomInstruction.root_id == root) | (CustomInstruction.id == root))
        .order_by(CustomInstruction.version.desc()))).scalars().first() or prev.version

    prev.is_active = False  # a new version supersedes the old; admin re-activates explicitly
    row = CustomInstruction(
        tenant_id=p.tenant_id, project_id=prev.project_id, raw_text=raw_text, kind=kind,
        compiled_prompt=compiled_prompt,
        scope=(body.scope if body.scope is not None else suggested_scope),
        priority=(body.priority if body.priority is not None else prev.priority),
        preview=preview, is_active=False, version=max_v + 1, root_id=root,
        created_by=p.user_id)
    db.add(row)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="instructions.new_version",
                      resource_type="custom_instruction", resource_id=str(root),
                      before={"version": prev.version}, after={"version": max_v + 1}))
    await db.commit()
    return _ci_out(row)


class ActivateBody(BaseModel):
    # Destinations for transfer targets the routing instruction provisions, collected inline
    # at the review gate: {"<target_key>": {"kind": "phone"|"sip"|..., "value": "+91..."}}.
    # Omitted for advisory rows and for routing rows that reuse existing targets.
    target_destinations: dict[str, dict] | None = None


@router.post("/{ci_id}/activate")
async def activate(ci_id: uuid.UUID, body: ActivateBody | None = None,
                   p: Principal = Depends(require("instructions:write")),
                   db: AsyncSession = Depends(get_db)):
    """The review gate: flip is_active true after caps/grounding checks. Advisory rows join
    the responder prompt; routing rows provision any new intents/targets they need and build +
    activate their owned RuleSet. Deactivates any other active version in the same lineage so
    only one version is ever live."""
    row = await _get_ci(db, p.tenant_id, ci_id)

    if row.kind == "advisory":
        try:
            await service.assert_can_activate_advisory(db, p.tenant_id, row)
        except caps.CapExceeded as e:
            raise HTTPException(status.HTTP_409_CONFLICT,
                                {"code": "CAP_EXCEEDED", "detail": e.message})
        except ValueError as e:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                {"code": "NOT_ACTIVATABLE", "detail": str(e)})
    elif row.kind == "routing":
        try:
            await service.provision_and_activate(
                db, p.tenant_id, row, activated_by=p.user_id,
                target_destinations=(body.target_destinations if body else None))
        except ValueError as e:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                {"code": "NOT_GROUNDABLE", "detail": str(e)})
    else:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            {"code": "NOT_ACTIVATABLE",
                             "detail": f"'{row.kind}' instructions cannot be activated; "
                                       f"resolve the ambiguity and recompile."})

    root = row.root_id or row.id
    siblings = (await db.execute(select(CustomInstruction).where(
        CustomInstruction.tenant_id == p.tenant_id,
        (CustomInstruction.root_id == root) | (CustomInstruction.id == root)))).scalars().all()
    for s in siblings:
        s.is_active = (s.id == row.id)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="instructions.activate",
                      resource_type="custom_instruction", resource_id=str(row.id),
                      after={"version": row.version, "kind": row.kind,
                             "rule_set_id": str(row.rule_set_id) if row.rule_set_id else None}))
    await db.commit()
    return _ci_out(row)


@router.post("/{ci_id}/deactivate")
async def deactivate(ci_id: uuid.UUID,
                     p: Principal = Depends(require("instructions:write")),
                     db: AsyncSession = Depends(get_db)):
    row = await _get_ci(db, p.tenant_id, ci_id)
    row.is_active = False
    if row.kind == "routing":
        # Tear down the owned RuleSet in the SAME transaction, or a live rule is orphaned.
        await service.deactivate_owned_ruleset(db, p.tenant_id, row)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="instructions.deactivate",
                      resource_type="custom_instruction", resource_id=str(row.id)))
    await db.commit()
    return _ci_out(row)


@router.delete("/{ci_id}")
async def delete(ci_id: uuid.UUID, purge: bool = Query(default=False),
                 p: Principal = Depends(require("instructions:write")),
                 db: AsyncSession = Depends(get_db)):
    """Soft-delete by default (deactivate, keep the row + audit). ``?purge=true`` hard-
    deletes the whole lineage — genuine cleanup only, and always audited."""
    row = await _get_ci(db, p.tenant_id, ci_id)
    # Either way, a routing instruction's owned RuleSet must stop firing (soft: deactivate;
    # purge: same, plus the row is removed — the RuleSet is left inactive, not deleted, so
    # its audit/version history survives even a purge).
    if row.kind == "routing":
        await service.deactivate_owned_ruleset(db, p.tenant_id, row)
    if not purge:
        row.is_active = False
        db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="instructions.delete_soft",
                          resource_type="custom_instruction", resource_id=str(row.id)))
        await db.commit()
        return {"id": str(row.id), "deleted": "soft", "is_active": False}

    root = row.root_id or row.id
    siblings = (await db.execute(select(CustomInstruction).where(
        CustomInstruction.tenant_id == p.tenant_id,
        (CustomInstruction.root_id == root) | (CustomInstruction.id == root)))).scalars().all()
    for s in siblings:
        await db.delete(s)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="instructions.delete_purge",
                      resource_type="custom_instruction", resource_id=str(root),
                      before={"versions_removed": len(siblings)}))
    await db.commit()
    return {"root_id": str(root), "deleted": "purged", "versions_removed": len(siblings)}


# =========================================================================== dynamic facts
facts_router = APIRouter(prefix="/dynamic-facts", tags=["dynamic-facts"])


def _fact_out(r: ProjectFactDynamic) -> dict:
    return {"id": str(r.id), "project_id": str(r.project_id) if r.project_id else None,
            "label": r.label, "content": r.content,
            "valid_from": r.valid_from, "valid_until": r.valid_until,
            "priority": r.priority, "is_active": r.is_active,
            "created_at": r.created_at, "updated_at": r.updated_at}


async def _get_fact(db: AsyncSession, tenant_id: uuid.UUID, fid: uuid.UUID) -> ProjectFactDynamic:
    row = (await db.execute(select(ProjectFactDynamic).where(
        ProjectFactDynamic.tenant_id == tenant_id,
        ProjectFactDynamic.id == fid))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dynamic fact not found")
    return row


class FactBody(BaseModel):
    content: str
    label: str | None = None
    project_id: uuid.UUID | None = None
    valid_from: date | None = None
    valid_until: date | None = None
    priority: int = 100
    is_active: bool = True


@facts_router.get("")
async def list_facts(active: bool | None = None,
                     p: Principal = Depends(require("instructions:read")),
                     db: AsyncSession = Depends(get_db)):
    q = select(ProjectFactDynamic).where(ProjectFactDynamic.tenant_id == p.tenant_id)
    if active is not None:
        q = q.where(ProjectFactDynamic.is_active.is_(active))
    rows = (await db.execute(q.order_by(ProjectFactDynamic.priority.desc(),
                                        ProjectFactDynamic.created_at.asc()))).scalars().all()
    return [_fact_out(r) for r in rows]


async def _active_fact_contents(db: AsyncSession, tenant_id: uuid.UUID, *,
                                exclude_id: uuid.UUID | None) -> list[str]:
    """The OTHER active facts' contents for this tenant — the budget a new/edited active fact
    is checked against (excludes the row being saved so an edit doesn't count itself)."""
    q = select(ProjectFactDynamic.content).where(
        ProjectFactDynamic.tenant_id == tenant_id,
        ProjectFactDynamic.is_active.is_(True))
    if exclude_id is not None:
        q = q.where(ProjectFactDynamic.id != exclude_id)
    return [c for c in (await db.execute(q)).scalars().all() if c]


@facts_router.post("", status_code=status.HTTP_201_CREATED)
async def create_fact(body: FactBody,
                      p: Principal = Depends(require("instructions:write")),
                      db: AsyncSession = Depends(get_db)):
    if body.valid_from and body.valid_until and body.valid_until < body.valid_from:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "valid_until must not be before valid_from")
    # A dynamic fact enters the responder's TRUSTED tier verbatim, so screen + cap it here.
    try:
        content = facts_guard.sanitize_fact(body.content)
        facts_guard.check_caps(
            active_contents=await _active_fact_contents(db, p.tenant_id, exclude_id=None),
            new_content=content, is_active=body.is_active)
    except facts_guard.FactRejected as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            {"code": "FACT_REJECTED", "detail": e.message})
    row = ProjectFactDynamic(
        tenant_id=p.tenant_id, project_id=body.project_id,
        label=(body.label.strip() if body.label else None), content=content,
        valid_from=body.valid_from, valid_until=body.valid_until,
        priority=body.priority, is_active=body.is_active, created_by=p.user_id)
    db.add(row)
    await db.flush()  # assign row.id (Python-side default) before the audit references it
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="dynamic_fact.create",
                      resource_type="project_fact_dynamic", resource_id=str(row.id),
                      after={"content": row.content}))
    await db.commit()
    return _fact_out(row)


@facts_router.put("/{fid}")
async def update_fact(fid: uuid.UUID, body: FactBody,
                      p: Principal = Depends(require("instructions:write")),
                      db: AsyncSession = Depends(get_db)):
    if body.valid_from and body.valid_until and body.valid_until < body.valid_from:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "valid_until must not be before valid_from")
    row = await _get_fact(db, p.tenant_id, fid)
    try:
        content = facts_guard.sanitize_fact(body.content)
        facts_guard.check_caps(
            active_contents=await _active_fact_contents(db, p.tenant_id, exclude_id=row.id),
            new_content=content, is_active=body.is_active)
    except facts_guard.FactRejected as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            {"code": "FACT_REJECTED", "detail": e.message})
    row.label = body.label.strip() if body.label else None
    row.content = content
    row.project_id = body.project_id
    row.valid_from, row.valid_until = body.valid_from, body.valid_until
    row.priority, row.is_active = body.priority, body.is_active
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="dynamic_fact.update",
                      resource_type="project_fact_dynamic", resource_id=str(row.id)))
    await db.commit()
    return _fact_out(row)


@facts_router.delete("/{fid}")
async def delete_fact(fid: uuid.UUID, purge: bool = Query(default=False),
                      p: Principal = Depends(require("instructions:write")),
                      db: AsyncSession = Depends(get_db)):
    row = await _get_fact(db, p.tenant_id, fid)
    if not purge:
        row.is_active = False
        db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="dynamic_fact.delete_soft",
                          resource_type="project_fact_dynamic", resource_id=str(row.id)))
        await db.commit()
        return {"id": str(row.id), "deleted": "soft", "is_active": False}
    rid = str(row.id)
    await db.delete(row)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="dynamic_fact.delete_purge",
                      resource_type="project_fact_dynamic", resource_id=rid))
    await db.commit()
    return {"id": rid, "deleted": "purged"}


# =========================================================================== intent vocabulary
intents_router = APIRouter(prefix="/intent-vocabulary", tags=["intent-vocabulary"])


@intents_router.get("")
async def list_intents(p: Principal = Depends(require("instructions:read")),
                       db: AsyncSession = Depends(get_db)):
    """The full grounding vocabulary: code defaults merged with this tenant's rows. The
    compiler grounds routing rules against exactly this set."""
    from app.custom_instructions import intents as intents_mod
    vocab = await intents_mod.load_vocabulary(db, p.tenant_id)
    return [{"name": e.name, "description": e.description, "aliases": e.aliases,
             "examples": e.examples, "source": e.source} for e in vocab]


class IntentBody(BaseModel):
    name: str
    description: str | None = None
    aliases: list[str] = []
    examples: list[str] = []
    is_active: bool = True


def _norm_name(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")


@intents_router.put("/{name}")
async def upsert_intent(name: str, body: IntentBody,
                        p: Principal = Depends(require("instructions:write")),
                        db: AsyncSession = Depends(get_db)):
    """Create or update a tenant intent (canonical name + the aliases the planner's label is
    normalised through). Idempotent by name."""
    from app.db.models import IntentVocabularyEntry
    canonical = _norm_name(body.name or name)
    if not canonical:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "intent name is empty")
    row = (await db.execute(select(IntentVocabularyEntry).where(
        IntentVocabularyEntry.tenant_id == p.tenant_id,
        IntentVocabularyEntry.name == canonical))).scalar_one_or_none()
    aliases = sorted({str(a).strip().lower() for a in (body.aliases or []) if str(a).strip()})
    if row is None:
        row = IntentVocabularyEntry(tenant_id=p.tenant_id, name=canonical, created_by=p.user_id)
        db.add(row)
    row.description = (body.description or "").strip() or None
    row.aliases = aliases
    row.examples = [str(e).strip() for e in (body.examples or []) if str(e).strip()]
    row.is_active = body.is_active
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="intent_vocab.upsert",
                      resource_type="intent_vocabulary", resource_id=canonical,
                      after={"aliases": aliases}))
    await db.commit()
    return {"name": canonical, "description": row.description, "aliases": row.aliases,
            "examples": row.examples, "is_active": row.is_active, "source": "tenant"}


@intents_router.delete("/{name}")
async def delete_intent(name: str, p: Principal = Depends(require("instructions:write")),
                        db: AsyncSession = Depends(get_db)):
    """Remove a tenant intent. Code-default intents cannot be deleted (they have no row)."""
    from app.db.models import IntentVocabularyEntry
    canonical = _norm_name(name)
    row = (await db.execute(select(IntentVocabularyEntry).where(
        IntentVocabularyEntry.tenant_id == p.tenant_id,
        IntentVocabularyEntry.name == canonical))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            "No tenant intent by that name (code defaults are not deletable)")
    await db.delete(row)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="intent_vocab.delete",
                      resource_type="intent_vocabulary", resource_id=canonical))
    await db.commit()
    return {"name": canonical, "deleted": True}


# =========================================================================== transfer targets
targets_router = APIRouter(prefix="/transfer-targets", tags=["transfer-targets"])


def _target_out(r) -> dict:
    return {"id": str(r.id), "key": r.key, "display_name": r.display_name,
            "destination": r.destination or {}, "is_active": r.is_active,
            "created_at": r.created_at, "updated_at": r.updated_at}


class TargetBody(BaseModel):
    key: str
    display_name: str
    destination: dict = {}
    is_active: bool = True


@targets_router.get("")
async def list_targets(active: bool | None = None,
                       p: Principal = Depends(require("instructions:read")),
                       db: AsyncSession = Depends(get_db)):
    from app.db.models import TransferTarget
    q = select(TransferTarget).where(TransferTarget.tenant_id == p.tenant_id)
    if active is not None:
        q = q.where(TransferTarget.is_active.is_(active))
    rows = (await db.execute(q.order_by(TransferTarget.key))).scalars().all()
    return [_target_out(r) for r in rows]


@targets_router.put("/{key}")
async def upsert_target(key: str, body: TargetBody,
                        p: Principal = Depends(require("instructions:write")),
                        db: AsyncSession = Depends(get_db)):
    """Create or update a transfer target. Idempotent by key."""
    from app.db.models import TransferTarget
    canonical = _norm_name(body.key or key)
    if not canonical:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "target key is empty")
    row = (await db.execute(select(TransferTarget).where(
        TransferTarget.tenant_id == p.tenant_id,
        TransferTarget.key == canonical))).scalar_one_or_none()
    if row is None:
        row = TransferTarget(tenant_id=p.tenant_id, key=canonical, created_by=p.user_id)
        db.add(row)
    row.display_name = (body.display_name or canonical).strip()
    row.destination = body.destination or {}
    row.is_active = body.is_active
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="transfer_target.upsert",
                      resource_type="transfer_target", resource_id=canonical))
    await db.commit()
    return _target_out(row)


@targets_router.delete("/{key}")
async def delete_target(key: str, p: Principal = Depends(require("instructions:write")),
                        db: AsyncSession = Depends(get_db)):
    from app.db.models import TransferTarget
    canonical = _norm_name(key)
    row = (await db.execute(select(TransferTarget).where(
        TransferTarget.tenant_id == p.tenant_id,
        TransferTarget.key == canonical))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Transfer target not found")
    row.is_active = False  # soft: a routing rule may still reference it; deactivate, keep audit
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="transfer_target.delete",
                      resource_type="transfer_target", resource_id=canonical))
    await db.commit()
    return {"key": canonical, "deleted": "soft", "is_active": False}
