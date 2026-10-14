"""Knowledge base: upload, manage, and test what the agent would retrieve."""
import io
import uuid
from datetime import date

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from docx import Document as DocxDocument
from pypdf import PdfReader

from app.db.base import get_db
from app.db.models import AuditEvent, KnowledgeChunk, KnowledgeDocument, Project
from app.engine.rag.ingestion import ingest_document, ingest_task
from app.engine.rag.retrieval import retrieve
from app.security.deps import Principal, require

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

DOC_TYPES = ("brochure", "faq", "pricing", "policy", "promo", "playbook", "sop", "compliance")


class DocumentIn(BaseModel):
    title: str
    doc_type: str = "faq"
    content: str
    effective_from: date | None = None
    effective_to: date | None = None
    metadata: dict = {}
    ingest_now: bool = True          # False queues it in the background
    # Which project this document belongs to. Omit (null) for a tenant-wide document, which
    # every project shares — the pre-multi-project default.
    project_id: uuid.UUID | None = None


@router.post("/documents", status_code=201)
async def create_document(body: DocumentIn, background: BackgroundTasks,
                          p: Principal = Depends(require("knowledge:write")),
                          db: AsyncSession = Depends(get_db)):
    if body.doc_type not in DOC_TYPES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"doc_type must be one of: {', '.join(DOC_TYPES)}")
    if body.effective_from and body.effective_to and body.effective_from > body.effective_to:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "effective_from must be on or before effective_to")

    if body.project_id is not None:
        await _require_project(db, p.tenant_id, body.project_id)
    doc = KnowledgeDocument(tenant_id=p.tenant_id, title=body.title[:300],
                            doc_type=body.doc_type, status="processing",
                            project_id=body.project_id,
                            effective_from=body.effective_from, effective_to=body.effective_to,
                            meta={**(body.metadata or {}), "chars": len(body.content)})
    db.add(doc)
    await db.flush()
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="knowledge.upload",
                      resource_type="knowledge_document", resource_id=str(doc.id),
                      after={"title": doc.title, "doc_type": doc.doc_type,
                             "effective_from": str(body.effective_from) if body.effective_from else None,
                             "effective_to": str(body.effective_to) if body.effective_to else None}))

    if body.ingest_now:
        report = await ingest_document(db, tenant_id=p.tenant_id, document_id=doc.id,
                                       text=body.content)
        await db.commit()
        return {"id": str(doc.id), "title": doc.title, "status": report.status,
                "chunks_created": report.chunks_created, "headings": report.headings,
                "embedding_model": report.embedding_model, "error": report.error,
                "enrichment": {"chunks_with_price": report.chunks_with_price,
                               "chunks_with_area": report.chunks_with_area,
                               "chunks_with_bhk": report.chunks_with_bhk,
                               "failures": report.enrichment_failures},
                "parent_child": _pc_breakdown(report)}

    await db.commit()
    background.add_task(ingest_task, p.tenant_id, doc.id, body.content)
    return {"id": str(doc.id), "title": doc.title, "status": "processing",
            "note": "Ingestion queued; poll GET /knowledge/documents/{id} for status."}


def extract_text(raw: bytes, filename: str, content_type: str) -> str:
    """Plain-text/markdown decodes directly; PDF and DOCX are extracted to their text content.
    Dispatches on the filename extension first (what a caller actually typed/picked), falling
    back to content_type for a file picked without one (e.g. a renamed export) -- either way,
    an unsupported or unreadable file is rejected with a clear 422 rather than stored empty or
    garbled."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf" or content_type == "application/pdf":
        try:
            pages = PdfReader(io.BytesIO(raw)).pages
            text = "\n\n".join(page.extract_text() or "" for page in pages)
        except Exception as e:  # noqa: BLE001 -- a malformed PDF must 422, not 500
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                f"Could not read this PDF: {e.__class__.__name__}")
        if not text.strip():
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                "No extractable text found in this PDF -- it may be scanned "
                                "pages (images) rather than real text, which this does not OCR")
        return text
    docx_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if ext == "docx" or content_type == docx_type:
        try:
            paragraphs = DocxDocument(io.BytesIO(raw)).paragraphs
            text = "\n\n".join(para.text for para in paragraphs if para.text.strip())
        except Exception as e:  # noqa: BLE001 -- a malformed DOCX must 422, not 500
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                f"Could not read this DOCX file: {e.__class__.__name__}")
        if not text.strip():
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                "No extractable text found in this DOCX file")
        return text
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Unsupported file type -- only .txt, .md, .pdf and .docx are "
                            "supported")


@router.post("/documents/upload", status_code=201)
async def upload_document(p: Principal = Depends(require("knowledge:write")),
                          db: AsyncSession = Depends(get_db),
                          file: UploadFile = File(...),
                          title: str | None = Form(None),
                          doc_type: str = Form("faq"),
                          effective_from: date | None = Form(None),
                          effective_to: date | None = Form(None),
                          project_id: uuid.UUID | None = Form(None)):
    """Plain-text/markdown, PDF, and DOCX upload -- see extract_text() for how each is read."""
    raw = await file.read()
    content = extract_text(raw, file.filename or "", file.content_type or "")
    body = DocumentIn(title=title or file.filename or "untitled", doc_type=doc_type,
                      content=content, effective_from=effective_from, effective_to=effective_to,
                      metadata={"filename": file.filename}, project_id=project_id)
    return await create_document(body, BackgroundTasks(), p, db)


@router.get("/documents")
async def list_documents(p: Principal = Depends(require("knowledge:read")),
                         db: AsyncSession = Depends(get_db),
                         doc_type: str | None = None, include_expired: bool = True):
    q = select(KnowledgeDocument).where(KnowledgeDocument.tenant_id == p.tenant_id)
    if doc_type:
        q = q.where(KnowledgeDocument.doc_type == doc_type)
    docs = (await db.execute(q.order_by(KnowledgeDocument.created_at.desc()))).scalars().all()

    counts = dict((await db.execute(
        select(KnowledgeChunk.document_id, func.count()).where(
            KnowledgeChunk.tenant_id == p.tenant_id,
            # R5.03: count in pre-R5.03 units — children are search slices of their parents.
            KnowledgeChunk.chunk_role != "child").group_by(KnowledgeChunk.document_id)
    )).all())

    today = date.today()
    out = []
    for d in docs:
        expired = bool((d.effective_to and d.effective_to < today) or
                       (d.effective_from and d.effective_from > today))
        if expired and not include_expired:
            continue
        out.append({"id": str(d.id), "title": d.title, "doc_type": d.doc_type,
                    "status": d.status, "version": d.version,
                    "project_id": str(d.project_id) if d.project_id else None,
                    "chunks": int(counts.get(d.id, 0)),
                    "effective_from": d.effective_from, "effective_to": d.effective_to,
                    "retrievable_today": d.status == "ready" and not expired,
                    "created_at": d.created_at})
    return out


@router.get("/documents/{document_id}")
async def get_document(document_id: uuid.UUID,
                       p: Principal = Depends(require("knowledge:read")),
                       db: AsyncSession = Depends(get_db)):
    doc = await _doc(db, p.tenant_id, document_id)
    chunks = (await db.execute(select(KnowledgeChunk).where(
        KnowledgeChunk.tenant_id == p.tenant_id,
        KnowledgeChunk.document_id == document_id)
        .order_by(KnowledgeChunk.chunk_index))).scalars().all()
    return {"id": str(doc.id), "title": doc.title, "doc_type": doc.doc_type,
            "status": doc.status, "version": doc.version,
            "effective_from": doc.effective_from, "effective_to": doc.effective_to,
            "metadata": doc.meta,
            "chunks": [{"index": c.chunk_index, "heading": (c.meta or {}).get("heading"),
                        "content": c.content, "tokens": c.token_count,
                        "embedded": c.embedding is not None,
                        # R5.03: standalone | parent | child, and the child's parent.
                        "role": c.chunk_role,
                        "parent_id": str(c.parent_id) if c.parent_id else None}
                       for c in chunks]}


class DocumentPatch(BaseModel):
    title: str | None = None
    doc_type: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    clear_effective_dates: bool = False


@router.patch("/documents/{document_id}")
async def update_document(document_id: uuid.UUID, body: DocumentPatch,
                          p: Principal = Depends(require("knowledge:write")),
                          db: AsyncSession = Depends(get_db)):
    doc = await _doc(db, p.tenant_id, document_id)
    before = {"title": doc.title, "doc_type": doc.doc_type,
              "effective_from": str(doc.effective_from), "effective_to": str(doc.effective_to)}
    if body.title:
        doc.title = body.title[:300]
    if body.doc_type:
        if body.doc_type not in DOC_TYPES:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid doc_type")
        doc.doc_type = body.doc_type
    if body.clear_effective_dates:
        doc.effective_from = doc.effective_to = None
    else:
        if body.effective_from is not None:
            doc.effective_from = body.effective_from
        if body.effective_to is not None:
            doc.effective_to = body.effective_to
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="knowledge.update",
                      resource_type="knowledge_document", resource_id=str(document_id),
                      before=before,
                      after={"title": doc.title, "doc_type": doc.doc_type,
                             "effective_from": str(doc.effective_from),
                             "effective_to": str(doc.effective_to)}))
    await db.commit()
    return {"id": str(doc.id), "title": doc.title, "doc_type": doc.doc_type,
            "effective_from": doc.effective_from, "effective_to": doc.effective_to}


@router.delete("/documents/{document_id}", status_code=200)
async def delete_document(document_id: uuid.UUID,
                          p: Principal = Depends(require("knowledge:write")),
                          db: AsyncSession = Depends(get_db)):
    """Removes the document and its chunks — the content stops being retrievable at once."""
    doc = await _doc(db, p.tenant_id, document_id)
    removed = (await db.execute(select(func.count()).select_from(KnowledgeChunk).where(
        KnowledgeChunk.tenant_id == p.tenant_id,
        KnowledgeChunk.document_id == document_id))).scalar_one()
    await db.execute(delete(KnowledgeChunk).where(
        KnowledgeChunk.tenant_id == p.tenant_id, KnowledgeChunk.document_id == document_id))
    await db.delete(doc)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="knowledge.delete",
                      resource_type="knowledge_document", resource_id=str(document_id),
                      before={"title": doc.title, "chunks": int(removed)}))
    await db.commit()
    return {"deleted": True, "chunks_removed": int(removed)}


@router.post("/documents/{document_id}/reingest")
async def reingest(document_id: uuid.UUID, body: dict,
                   p: Principal = Depends(require("knowledge:write")),
                   db: AsyncSession = Depends(get_db)):
    """Re-chunk and re-embed with new content. Old chunks are replaced atomically."""
    await _doc(db, p.tenant_id, document_id)
    content = str(body.get("content") or "")
    if not content.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "content is required")
    report = await ingest_document(db, tenant_id=p.tenant_id, document_id=document_id,
                                   text=content)
    await db.commit()
    return {"document_id": str(document_id), "status": report.status,
            "chunks_created": report.chunks_created, "chunks_removed": report.chunks_removed,
            "error": report.error,
            "enrichment": {"chunks_with_price": report.chunks_with_price,
                           "chunks_with_area": report.chunks_with_area,
                           "chunks_with_bhk": report.chunks_with_bhk,
                           "failures": report.enrichment_failures},
            "parent_child": _pc_breakdown(report)}


class SearchIn(BaseModel):
    query: str
    top_k: int | None = None
    doc_type: str | None = None
    on_date: date | None = None      # simulate a future date to test expiry
    project_id: uuid.UUID | None = None   # scope retrieval to one project (as a call would)
    # R5.01: force the cross-encoder step on/off for this request (None = server setting), so
    # the same question can be compared with and without re-ranking.
    rerank: bool | None = None
    knowledge_base_id: uuid.UUID | None = None  # scope to one KB — the KB editor's own test box
    # R5.03: "flat" | "parent_child" to compare both on the same question; None = what a live
    # call for this tenant would use (tenant override, else global).
    granularity: str | None = None


@router.post("/search")
async def search(body: SearchIn, p: Principal = Depends(require("knowledge:read")),
                 db: AsyncSession = Depends(get_db)):
    """Exactly what the agent would retrieve for this question — the debugging tool
    for 'why did it say that?' and 'why doesn't it know this?'"""
    result = await retrieve(db, tenant_id=p.tenant_id, query=body.query,
                            top_k=body.top_k, doc_type=body.doc_type, on_date=body.on_date,
                            project_id=body.project_id, knowledge_base_id=body.knowledge_base_id,
                            rerank=body.rerank, granularity=body.granularity)
    return {"query": result.query, "as_of": body.on_date or date.today(),
            "vector_hits": result.vector_hits, "text_hits": result.text_hits,
            "excluded_by_effective_dates": result.excluded_expired,
            "latency_ms": result.latency_ms, "error": result.error,
            "metadata": result.metadata,
            "vector": result.vector_diagnostics(),
            "rerank": result.rerank_diagnostics(),
            "parent_child": result.parent_child_diagnostics(),
            "results": [{"chunk_id": c.chunk_id, "source": c.title, "heading": c.heading,
                         "doc_type": c.doc_type, "score": round(c.score, 4),
                         "vector_rank": c.vector_rank, "text_rank": c.text_rank,
                         "base_rank": c.base_rank,
                         "rerank_score": (round(c.rerank_score, 3)
                                          if c.rerank_score is not None else None),
                         "rerank_rank": c.rerank_rank,
                         "facts": c.facts, "meta_boosts": c.meta_boosts,
                         "content": c.content,
                         "matched_children": c.matched_children} for c in result.chunks]}


def _pc_breakdown(report) -> dict:
    """R5.03: how the stored rows split into standalone / parent / child chunks."""
    return {"standalone": report.standalone_created, "parents": report.parents_created,
            "children": report.children_created,
            "embeddings_failed": report.embeddings_failed}


@router.get("/reranker")
async def reranker_status(p: Principal = Depends(require("knowledge:read"))):
    """R5.01: is the cross-encoder loaded, which exact model file (sha256), and how calls have
    gone since this process started (per-status counts, latency p50/p95). Process-local: with
    several replicas, each reports its own."""
    from app.config import get_settings
    from app.engine.rag import reranker
    s = get_settings()
    return {"enabled": s.retrieval_rerank_enabled, "mode": s.retrieval_rerank_mode,
            "candidates": s.retrieval_rerank_candidates,
            **reranker.reranker_status(reranker.config_from_settings(s))}


async def _doc(db: AsyncSession, tenant_id: uuid.UUID, doc_id: uuid.UUID) -> KnowledgeDocument:
    doc = (await db.execute(select(KnowledgeDocument).where(
        KnowledgeDocument.tenant_id == tenant_id,
        KnowledgeDocument.id == doc_id))).scalar_one_or_none()
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return doc


async def _require_project(db: AsyncSession, tenant_id: uuid.UUID, project_id: uuid.UUID) -> Project:
    """A document may only be attached to a project that exists in the SAME tenant."""
    proj = (await db.execute(select(Project).where(
        Project.tenant_id == tenant_id, Project.id == project_id))).scalar_one_or_none()
    if proj is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "project_id does not name a project in this tenant")
    return proj
