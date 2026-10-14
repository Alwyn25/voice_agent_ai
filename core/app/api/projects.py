"""Projects: the developments a tenant sells, each with its own knowledge, facts and number.

Multi-project-per-tenant (see app.db.models.project). A tenant with no projects is tenant-wide
and unchanged; creating projects lets retrieval, the prompt facts and bookings scope per
project, and lets an inbound call be resolved to one (by dialled number, the caller's history,
or an explicit mention).

Gated on the knowledge permissions: a project governs how a tenant's knowledge is scoped, and
the roles that may edit knowledge (admin, supervisor) are exactly those that may edit projects.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import AuditEvent, KnowledgeDocument, Location, Project
from app.security.deps import Principal, require

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectIn(BaseModel):
    name: str
    slug: str
    aliases: list[str] = []
    phone_did: str | None = None
    facts: list[dict] = []
    synonyms: list[list[str]] = []
    default_centre_id: uuid.UUID | None = None
    is_default: bool = False
    status: str = "active"


class ProjectPatch(BaseModel):
    name: str | None = None
    aliases: list[str] | None = None
    phone_did: str | None = None
    facts: list[dict] | None = None
    synonyms: list[list[str]] | None = None
    default_centre_id: uuid.UUID | None = None
    is_default: bool | None = None
    status: str | None = None


def _brief(p: Project, doc_count: int = 0) -> dict:
    return {"id": str(p.id), "name": p.name, "slug": p.slug, "status": p.status,
            "is_default": p.is_default, "phone_did": p.phone_did,
            "aliases": list(p.aliases or []), "facts": list(p.facts or []),
            "synonyms": list(p.synonyms or []),
            "default_centre_id": str(p.default_centre_id) if p.default_centre_id else None,
            "documents": doc_count, "created_at": p.created_at}


async def _get(db: AsyncSession, tenant_id: uuid.UUID, project_id: uuid.UUID) -> Project:
    proj = (await db.execute(select(Project).where(
        Project.tenant_id == tenant_id, Project.id == project_id))).scalar_one_or_none()
    if proj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return proj


async def _validate_centre(db: AsyncSession, tenant_id: uuid.UUID,
                           centre_id: uuid.UUID | None) -> None:
    if centre_id is None:
        return
    centre = (await db.execute(select(Location).where(
        Location.tenant_id == tenant_id, Location.id == centre_id))).scalar_one_or_none()
    if centre is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "default_centre_id does not name a centre in this tenant")


@router.get("")
async def list_projects(p: Principal = Depends(require("knowledge:read")),
                        db: AsyncSession = Depends(get_db)):
    projects = (await db.execute(select(Project).where(
        Project.tenant_id == p.tenant_id).order_by(Project.created_at))).scalars().all()
    counts = dict((await db.execute(
        select(KnowledgeDocument.project_id, func.count()).where(
            KnowledgeDocument.tenant_id == p.tenant_id,
            KnowledgeDocument.project_id.isnot(None))
        .group_by(KnowledgeDocument.project_id))).all())
    return [_brief(proj, int(counts.get(proj.id, 0))) for proj in projects]


@router.post("", status_code=201)
async def create_project(body: ProjectIn, p: Principal = Depends(require("knowledge:write")),
                         db: AsyncSession = Depends(get_db)):
    if body.status not in ("active", "inactive"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "status must be active|inactive")
    dup = (await db.execute(select(Project).where(
        Project.tenant_id == p.tenant_id, Project.slug == body.slug))).scalar_one_or_none()
    if dup is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"A project with slug '{body.slug}' exists")
    await _validate_centre(db, p.tenant_id, body.default_centre_id)
    proj = Project(tenant_id=p.tenant_id, name=body.name[:200], slug=body.slug[:100],
                   status=body.status, is_default=body.is_default, phone_did=body.phone_did,
                   aliases=[a.strip().lower() for a in body.aliases if a.strip()],
                   facts=body.facts, synonyms=body.synonyms,
                   default_centre_id=body.default_centre_id)
    db.add(proj)
    await db.flush()
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="project.create",
                      resource_type="project", resource_id=str(proj.id),
                      after={"name": proj.name, "slug": proj.slug}))
    await db.commit()
    return _brief(proj)


@router.get("/{project_id}")
async def get_project(project_id: uuid.UUID, p: Principal = Depends(require("knowledge:read")),
                      db: AsyncSession = Depends(get_db)):
    proj = await _get(db, p.tenant_id, project_id)
    count = (await db.execute(select(func.count()).select_from(KnowledgeDocument).where(
        KnowledgeDocument.tenant_id == p.tenant_id,
        KnowledgeDocument.project_id == project_id))).scalar_one()
    return _brief(proj, int(count))


@router.patch("/{project_id}")
async def update_project(project_id: uuid.UUID, body: ProjectPatch,
                         p: Principal = Depends(require("knowledge:write")),
                         db: AsyncSession = Depends(get_db)):
    proj = await _get(db, p.tenant_id, project_id)
    if body.status is not None:
        if body.status not in ("active", "inactive"):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "status must be active|inactive")
        proj.status = body.status
    if body.name is not None:
        proj.name = body.name[:200]
    if body.aliases is not None:
        proj.aliases = [a.strip().lower() for a in body.aliases if a.strip()]
    if body.phone_did is not None:
        proj.phone_did = body.phone_did or None
    if body.facts is not None:
        proj.facts = body.facts
    if body.synonyms is not None:
        proj.synonyms = body.synonyms
    if body.is_default is not None:
        proj.is_default = body.is_default
    if body.default_centre_id is not None:
        await _validate_centre(db, p.tenant_id, body.default_centre_id)
        proj.default_centre_id = body.default_centre_id
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="project.update",
                      resource_type="project", resource_id=str(proj.id),
                      after={"name": proj.name, "status": proj.status,
                             "is_default": proj.is_default}))
    await db.commit()
    return _brief(proj)
