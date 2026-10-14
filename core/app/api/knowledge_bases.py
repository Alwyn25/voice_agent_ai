"""Vocal Backend Tasksheet P2 (Knowledge Base): the knowledge_bases grouping layer
on top of the pre-existing flat knowledge_documents/knowledge_chunks tables (see
app/db/models/knowledge.py's docstrings for the reconciliation — knowledge_documents
IS the tasksheet's "knowledge_sources", reused rather than duplicated).

`router` (prefix /knowledge-bases) is the tenant-facing CRUD surface; `internal_router`
carries the one genuinely internal-only route, POST /internal/retrieve, kept in this
file because it is this module's own retrieval-scoping concern.
"""
import uuid
from datetime import date

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import Integer, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.knowledge import DOC_TYPES, extract_text
from app.db.base import get_db
from app.db.models import (AgentKnowledgeBase, KnowledgeBase, KnowledgeChunk, KnowledgeCitation,
                           KnowledgeDocument)
from app.engine.rag.ingestion import ingest_document
from app.engine.rag.kb_retrieval import retrieve_for_kb_ids
from app.security.deps import Principal, get_principal, require
from app.services.activity import append_activity_event

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])
internal_router = APIRouter(tags=["knowledge-internal"])


async def _kb(db: AsyncSession, tenant_id: uuid.UUID, kb_id: uuid.UUID) -> KnowledgeBase:
    kb = (await db.execute(select(KnowledgeBase).where(
        KnowledgeBase.tenant_id == tenant_id, KnowledgeBase.id == kb_id))).scalar_one_or_none()
    if kb is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Knowledge base not found")
    return kb


async def _chunk_and_citation_stats(
        db: AsyncSession, tenant_id: uuid.UUID) -> tuple[dict[uuid.UUID, int], dict[uuid.UUID, float]]:
    """Chunk counts + citation rates per knowledge_base_id, in one query (an
    outer join so a KB with chunks but no citations yet still gets a chunk
    count) rather than two separate GROUP BYs over the same table. See
    KnowledgeCitation's docstring for why was_cited defaults True today."""
    rows = (await db.execute(
        select(KnowledgeChunk.knowledge_base_id, func.count(func.distinct(KnowledgeChunk.id)),
              func.sum(cast(KnowledgeCitation.was_cited, Integer)), func.count(KnowledgeCitation.id))
        .outerjoin(KnowledgeCitation, KnowledgeCitation.chunk_id == KnowledgeChunk.id)
        .where(KnowledgeChunk.tenant_id == tenant_id, KnowledgeChunk.knowledge_base_id.is_not(None))
        .group_by(KnowledgeChunk.knowledge_base_id))).all()
    chunk_counts: dict[uuid.UUID, int] = {}
    rates: dict[uuid.UUID, float] = {}
    for kb_id, chunk_count, cited, total_citations in rows:
        chunk_counts[kb_id] = int(chunk_count)
        if total_citations:
            rates[kb_id] = float(cited or 0) / total_citations
    return chunk_counts, rates


async def _used_by_counts(db: AsyncSession, tenant_id: uuid.UUID) -> dict[uuid.UUID, int]:
    """How many agents have each knowledge base attached — the editor modal's
    footer ("used by N agents")."""
    rows = (await db.execute(
        select(AgentKnowledgeBase.knowledge_base_id, func.count(func.distinct(AgentKnowledgeBase.agent_id)))
        .where(AgentKnowledgeBase.tenant_id == tenant_id)
        .group_by(AgentKnowledgeBase.knowledge_base_id))).all()
    return {kb_id: int(n) for kb_id, n in rows}


def _kb_out(kb: KnowledgeBase, sources: int, chunks: int, citation_rate: float | None,
           used_by_agents: int = 0) -> dict:
    return {"id": str(kb.id), "name": kb.name, "scope_description": kb.scope_description,
           "status": kb.status, "reindex_policy": kb.reindex_policy, "sources": sources,
           "chunks_indexed": chunks,
           "citation_rate": round(citation_rate, 4) if citation_rate is not None else None,
           "used_by_agents": used_by_agents,
           "created_at": kb.created_at}


REINDEX_POLICIES = ("manual", "on_source_change", "daily")


class KBCreate(BaseModel):
    name: str
    scope_description: str | None = None


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(body: KBCreate, p: Principal = Depends(require("knowledge:write")),
                                db: AsyncSession = Depends(get_db)):
    kb = KnowledgeBase(tenant_id=p.tenant_id, name=body.name, scope_description=body.scope_description)
    db.add(kb)
    await db.flush()
    append_activity_event(db, p, "knowledge.kb_create", resource_type="knowledge_base",
                          resource_id=str(kb.id), after={"name": kb.name})
    await db.commit()
    return _kb_out(kb, sources=0, chunks=0, citation_rate=None)


class KBPatch(BaseModel):
    name: str | None = None
    scope_description: str | None = None
    reindex_policy: str | None = None


@router.patch("/{kb_id}")
async def update_knowledge_base(kb_id: uuid.UUID, body: KBPatch,
                                p: Principal = Depends(require("knowledge:write")),
                                db: AsyncSession = Depends(get_db)):
    kb = await _kb(db, p.tenant_id, kb_id)
    if body.name is not None:
        if not body.name.strip():
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "name cannot be blank")
        kb.name = body.name.strip()
    if body.scope_description is not None:
        kb.scope_description = body.scope_description
    if body.reindex_policy is not None:
        if body.reindex_policy not in REINDEX_POLICIES:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                f"reindex_policy must be one of: {', '.join(REINDEX_POLICIES)}")
        kb.reindex_policy = body.reindex_policy
    append_activity_event(db, p, "knowledge.kb_update", resource_type="knowledge_base",
                          resource_id=str(kb.id), after={"name": kb.name})
    await db.commit()
    await db.refresh(kb)
    chunk_counts, rates = await _chunk_and_citation_stats(db, p.tenant_id)
    used_by = await _used_by_counts(db, p.tenant_id)
    source_count = int((await db.execute(select(func.count()).select_from(KnowledgeDocument).where(
        KnowledgeDocument.tenant_id == p.tenant_id, KnowledgeDocument.knowledge_base_id == kb.id
        ))).scalar_one())
    return _kb_out(kb, source_count, chunk_counts.get(kb.id, 0), rates.get(kb.id), used_by.get(kb.id, 0))


@router.get("")
async def list_knowledge_bases(p: Principal = Depends(require("knowledge:read")),
                               db: AsyncSession = Depends(get_db)):
    kbs = (await db.execute(select(KnowledgeBase).where(KnowledgeBase.tenant_id == p.tenant_id)
                            .order_by(KnowledgeBase.created_at.desc()))).scalars().all()
    source_counts = dict((await db.execute(
        select(KnowledgeDocument.knowledge_base_id, func.count()).where(
            KnowledgeDocument.tenant_id == p.tenant_id, KnowledgeDocument.knowledge_base_id.is_not(None))
        .group_by(KnowledgeDocument.knowledge_base_id))).all())
    chunk_counts, rates = await _chunk_and_citation_stats(db, p.tenant_id)
    used_by = await _used_by_counts(db, p.tenant_id)
    return [_kb_out(kb, source_counts.get(kb.id, 0), chunk_counts.get(kb.id, 0), rates.get(kb.id),
                    used_by.get(kb.id, 0))
           for kb in kbs]


@router.get("/stats")
async def knowledge_base_stats(p: Principal = Depends(require("knowledge:read")),
                               db: AsyncSession = Depends(get_db)):
    """Org-wide aggregate header for the KB list page (UI shows KNOWLEDGE BASES /
    SOURCES / CHUNKS INDEXED / CITATION RATE as four stat tiles). One query with
    scalar subqueries, not four sequential round trips for four independent counts."""
    bases_sq = select(func.count()).select_from(KnowledgeBase).where(
        KnowledgeBase.tenant_id == p.tenant_id).scalar_subquery()
    sources_sq = select(func.count()).select_from(KnowledgeDocument).where(
        KnowledgeDocument.tenant_id == p.tenant_id,
        KnowledgeDocument.knowledge_base_id.is_not(None)).scalar_subquery()
    chunks_sq = select(func.count()).select_from(KnowledgeChunk).where(
        KnowledgeChunk.tenant_id == p.tenant_id,
        KnowledgeChunk.knowledge_base_id.is_not(None)).scalar_subquery()
    cited_sq = select(func.sum(cast(KnowledgeCitation.was_cited, Integer))).select_from(KnowledgeCitation).join(
        KnowledgeChunk, KnowledgeChunk.id == KnowledgeCitation.chunk_id).where(
        KnowledgeChunk.tenant_id == p.tenant_id, KnowledgeChunk.knowledge_base_id.is_not(None)).scalar_subquery()
    total_citations_sq = select(func.count()).select_from(KnowledgeCitation).join(
        KnowledgeChunk, KnowledgeChunk.id == KnowledgeCitation.chunk_id).where(
        KnowledgeChunk.tenant_id == p.tenant_id, KnowledgeChunk.knowledge_base_id.is_not(None)).scalar_subquery()

    total_bases, total_sources, total_chunks, cited, total_citations = (await db.execute(
        select(bases_sq, sources_sq, chunks_sq, cited_sq, total_citations_sq))).one()
    org_citation_rate = round(float(cited or 0) / total_citations, 4) if total_citations else None
    return {"total_bases": int(total_bases), "total_sources": int(total_sources),
           "total_chunks_indexed": int(total_chunks), "org_citation_rate": org_citation_rate}


@router.post("/{kb_id}/sources", status_code=status.HTTP_201_CREATED)
async def add_source(
        kb_id: uuid.UUID,
        p: Principal = Depends(require("knowledge:write")),
        db: AsyncSession = Depends(get_db),
        file: UploadFile | None = File(None),
        title: str | None = Form(None),
        doc_type: str = Form("faq"),
        source_type: str = Form("doc"),  # url|file|doc|integration
        source_uri: str | None = Form(None),
        content: str | None = Form(None),
        effective_from: date | None = Form(None),
        effective_to: date | None = Form(None)):
    """Upload a source into a knowledge base. A `file` (text/markdown/PDF/DOCX — see
    app/api/knowledge.py's extract_text(), shared with the pre-existing POST
    /knowledge/documents/upload so the two never drift again) is chunked and embedded
    immediately; otherwise `content` is used directly. `source_type="url"` labels
    provenance only — this endpoint does not crawl a URL itself, it stores
    `source_uri` and indexes whatever `content` the caller supplies. Real web
    crawling is a future integration, not a P2 schema/infra concern."""
    kb = await _kb(db, p.tenant_id, kb_id)

    if file is not None:
        raw = await file.read()
        text = extract_text(raw, file.filename or "", file.content_type or "")
        source_type = "file"
        source_uri = source_uri or file.filename
        title = title or file.filename or "untitled"
    else:
        text = content or ""
        if not text.strip():
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Either `file` or `content` is required")
        title = title or "untitled"

    if doc_type not in DOC_TYPES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"doc_type must be one of: {', '.join(DOC_TYPES)}")

    doc = KnowledgeDocument(tenant_id=p.tenant_id, knowledge_base_id=kb.id, title=title[:300],
                            doc_type=doc_type, source_type=source_type, source_uri=source_uri,
                            status="processing", effective_from=effective_from, effective_to=effective_to,
                            meta={"chars": len(text)})
    db.add(doc)
    await db.flush()
    report = await ingest_document(db, tenant_id=p.tenant_id, document_id=doc.id, text=text)
    append_activity_event(db, p, "knowledge.source_add", resource_type="knowledge_document",
                          resource_id=str(doc.id), after={"knowledge_base": kb.name, "title": doc.title})
    await db.commit()
    return {"id": str(doc.id), "knowledge_base_id": str(kb.id), "title": doc.title,
           "source_type": doc.source_type, "status": report.status,
           "chunks_created": report.chunks_created, "error": report.error}


@router.get("/{kb_id}/sources")
async def list_sources(kb_id: uuid.UUID, p: Principal = Depends(require("knowledge:read")),
                       db: AsyncSession = Depends(get_db)):
    kb = await _kb(db, p.tenant_id, kb_id)
    docs = (await db.execute(select(KnowledgeDocument).where(
        KnowledgeDocument.tenant_id == p.tenant_id, KnowledgeDocument.knowledge_base_id == kb.id)
        .order_by(KnowledgeDocument.created_at.desc()))).scalars().all()
    chunk_counts = dict((await db.execute(
        select(KnowledgeChunk.document_id, func.count()).where(
            KnowledgeChunk.tenant_id == p.tenant_id, KnowledgeChunk.knowledge_base_id == kb.id)
        .group_by(KnowledgeChunk.document_id))).all())
    return [{"id": str(d.id), "title": d.title, "doc_type": d.doc_type, "source_type": d.source_type,
            "source_uri": d.source_uri, "status": d.status, "chunks": int(chunk_counts.get(d.id, 0)),
            "last_indexed_at": d.last_indexed_at, "created_at": d.created_at} for d in docs]


class InternalRetrieveIn(BaseModel):
    query: str
    kb_ids: list[uuid.UUID]
    top_k: int | None = None
    project_id: uuid.UUID | None = None
    conversation_id: uuid.UUID | None = None


@internal_router.post("/internal/retrieve")
async def internal_retrieve(body: InternalRetrieveIn, p: Principal = Depends(get_principal),
                            db: AsyncSession = Depends(get_db)):
    """Internal-only — called by the Call Runtime (P6), never by the frontend
    directly. Gated to service (API-key) principals rather than exposed as a
    normal permission-gated user route, per the tasksheet's own 'not public API'
    note."""
    if p.kind != "api_key":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Internal endpoint — requires a service API key")

    chunks = await retrieve_for_kb_ids(db, tenant_id=p.tenant_id, query=body.query, kb_ids=body.kb_ids,
                                       top_k=body.top_k, project_id=body.project_id,
                                       conversation_id=body.conversation_id)
    await db.commit()
    return {"query": body.query, "results": [c.as_context() for c in chunks]}
