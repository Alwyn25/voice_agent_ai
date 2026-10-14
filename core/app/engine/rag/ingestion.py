"""Document ingestion: chunk -> embed -> store.

Re-ingesting a document replaces its chunks atomically, so an updated price list
never leaves stale chunks behind that the agent could still quote.

R5.03: with CHUNK_PARENT_CHILD_ENABLED=true (or the tenant setting chunk_parent_child_enabled)
each chunk large enough to benefit is stored as a PARENT plus ~200-char CHILDREN (see
app/engine/rag/parent_child.py for why). With it off, the rows are all 'standalone'.

Extracted text (PDF/DOCX/OCR) is normalised before chunking (app/engine/rag/normalize.py);
embedding retries only the batch that failed (embed_resilient). Both are recorded on the report.
"""
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import KnowledgeChunk, KnowledgeDocument, Tenant
from app.engine.rag.chunkers import Chunk, chunk_text
from app.engine.rag.enrichment import enrich_chunk_meta
from app.engine.rag.normalize import normalize_extracted_text
from app.engine.rag.parent_child import (ROLE_CHILD, ROLE_PARENT, ROLE_STANDALONE,
                                         child_split_kwargs, plan_families)
from app.providers.embeddings.base import EmbeddingError
from app.providers.embeddings.router import get_embeddings

log = logging.getLogger(__name__)

# Texts per embedding request made by ingestion. The provider batches internally too (96); this
# is the unit of RETRY — a failed request costs a per-item retry of at most this many texts.
EMBED_BATCH = 64


@dataclass
class EmbeddedTexts:
    vectors: list                     # one per input; None where the text could not be embedded
    model: str = ""
    dimensions: int = 0
    failed: int = 0
    retried_batches: int = 0
    last_error: str | None = None


async def embed_resilient(embedder, texts: list[str], *,
                          batch_size: int = EMBED_BATCH) -> EmbeddedTexts:
    """Embed ``texts`` in batches; when a batch fails, retry ONLY that batch one text at a time.

    History: the whole document used to go in one call, and on failure EVERY chunk was retried
    one by one — fine for a 20-chunk brochure, thousands of sequential calls for a large extracted
    PDF (x4 with parent-child). A document that embeds 99 of 100 texts is still ingested (status
    'partial'); only a document where nothing embeds fails. Never raises EmbeddingError.
    """
    out = EmbeddedTexts(vectors=[])
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        try:
            res = await embedder.embed(batch)
            out.vectors.extend(res.vectors)
            out.model, out.dimensions = res.model, res.dimensions
            continue
        except EmbeddingError as e:
            out.retried_batches += 1
            out.last_error = str(e)
            log.warning("ingest.embed batch %d-%d failed (%s); retrying one at a time",
                        start, start + len(batch), e)
        for t in batch:
            try:
                one = await embedder.embed([t])
                out.vectors.append(one.vectors[0])
                out.model, out.dimensions = one.model, one.dimensions
            except EmbeddingError as e:
                out.vectors.append(None)
                out.failed += 1
                out.last_error = str(e)
    return out


async def _tenant_settings(db: AsyncSession, tenant_id: uuid.UUID) -> dict:
    """The tenant's settings JSON ({} when missing or malformed) — per-tenant overrides."""
    try:
        tenant = (await db.execute(select(Tenant).where(
            Tenant.id == tenant_id))).scalar_one_or_none()
    except Exception:  # noqa: BLE001 -- an override lookup must never fail an ingestion
        return {}
    return (tenant.settings if tenant and isinstance(tenant.settings, dict) else {}) or {}


def parent_child_enabled_for(s, tenant_settings: dict) -> bool:
    """Tenant setting ``chunk_parent_child_enabled`` (a real bool) overrides the global flag, so
    parent-child can be rolled out tenant by tenant, each after passing its own eval gate."""
    override = tenant_settings.get("chunk_parent_child_enabled")
    return override if isinstance(override, bool) else bool(s.chunk_parent_child_enabled)

@dataclass
class IngestionReport:
    document_id: uuid.UUID
    status: str
    chunks_created: int = 0
    chunks_removed: int = 0
    embedding_model: str = ""
    error: str | None = None
    headings: list[str] = field(default_factory=list)
    # R5.08 enrichment coverage — how many stored chunks state each kind of fact. A pricing
    # document that reports chunks_with_price=0 is the signal the extractor missed its format.
    chunks_with_price: int = 0
    chunks_with_area: int = 0
    chunks_with_bhk: int = 0
    enrichment_failures: int = 0
    # R5.03 breakdown of chunks_created (which counts every stored row, children included).
    parents_created: int = 0
    children_created: int = 0
    standalone_created: int = 0
    embeddings_failed: int = 0
    embedding_batches_retried: int = 0
    parent_child_enabled: bool = False
    # What normalize_extracted_text() changed (NormalizationReport.as_dict()).
    normalization: dict = field(default_factory=dict)


def _chunk_facts(text: str, document_id: uuid.UUID, index: int) -> dict | None:
    """R5.08 facts for one chunk, or None. Enrichment is an optimisation of ranking, never a
    precondition for storing knowledge: any failure stores the chunk untagged (retrieval treats
    untagged chunks as neutral) and is logged, so one odd document cannot fail an ingest."""
    try:
        return enrich_chunk_meta(text)
    except Exception as e:  # noqa: BLE001 — see docstring
        log.warning("rag.enrich failed doc=%s chunk=%d: %s: %s",
                    document_id, index, e.__class__.__name__, e)
        return None


async def ingest_document(db: AsyncSession, *, tenant_id: uuid.UUID,
                          document_id: uuid.UUID, text: str,
                          normalize_mode: str | None = None) -> IngestionReport:
    """Chunk, embed and store ``text`` as the document's chunks (atomic replace).

    ``normalize_mode`` — "auto" | "always" | "off" (normalize.py); None = INGEST_NORMALIZE_MODE.
    An ingestion service that KNOWS its text came from a PDF/OCR extractor should pass "always".
    """
    s = get_settings()
    doc = (await db.execute(select(KnowledgeDocument).where(
        KnowledgeDocument.tenant_id == tenant_id,
        KnowledgeDocument.id == document_id))).scalar_one_or_none()
    if doc is None:
        return IngestionReport(document_id=document_id, status="failed",
                               error="document not found")

    old = (await db.execute(select(KnowledgeChunk.id).where(
        KnowledgeChunk.tenant_id == tenant_id,
        KnowledgeChunk.document_id == document_id))).scalars().all()

    normalized = normalize_extracted_text(text, mode=normalize_mode or s.ingest_normalize_mode)
    if normalized.report.changed:
        log.info("ingest.normalize doc=%s %s", document_id, normalized.report.as_dict())
    text = normalized.text

    tenant_settings = await _tenant_settings(db, tenant_id)
    pc_enabled = parent_child_enabled_for(s, tenant_settings)

    chunks = chunk_text(text, target_chars=s.chunk_target_chars,
                        overlap_chars=s.chunk_overlap_chars,
                        list_group_size=s.chunk_list_group_size,
                        min_chars=s.chunk_min_chars)
    if not chunks:
        doc.status = "failed"
        await db.flush()
        return IngestionReport(document_id=document_id, status="failed",
                               error="no usable content found in the document")

    # R5.03: decide which chunks become parents with ~200-char children. With the flag off every
    # chunk is a standalone family and everything below is exactly the pre-R5.03 ingestion.
    families = plan_families(chunks, enabled=pc_enabled,
                             **child_split_kwargs(s, title=doc.title))
    # One flat list of everything to embed, in family order: parent, then its children. Parents
    # are embedded too, so "flat" retrieval keeps working on this document (parent_child.py).
    units: list[Chunk] = []
    for fam in families:
        units.append(fam.parent)
        units.extend(fam.children)

    # Embedding must not lose a whole document to one slow response — that failure is silent
    # from the outside (the endpoint still returns 201 and the agent simply cannot answer from
    # it; three DSR documents, including the 3 BHK cost sheet, once disappeared this way). A
    # failed batch is retried text by text; a document that embeds most texts is kept.
    result = await embed_resilient(get_embeddings(), [c.embed_text for c in units])
    if result.failed == len(units):
        doc.status = "failed"
        await db.flush()
        return IngestionReport(document_id=document_id, status="failed",
                               error=result.last_error or "embedding failed",
                               normalization=normalized.report.as_dict())

    # Deleting every chunk of the document removes parents AND children in one statement
    # (children would also go by the parent_id ON DELETE CASCADE).
    await db.execute(delete(KnowledgeChunk).where(
        KnowledgeChunk.tenant_id == tenant_id,
        KnowledgeChunk.document_id == document_id))

    enrich = s.ingestion_enrich_metadata
    report = IngestionReport(document_id=document_id, status="ready",
                             chunks_removed=len(old), embedding_model=result.model,
                             headings=sorted({c.heading for c in chunks if c.heading}))
    report.embeddings_failed = result.failed
    report.embedding_batches_retried = result.retried_batches
    report.parent_child_enabled = pc_enabled
    report.normalization = normalized.report.as_dict()

    def row(chunk: Chunk, vector, *, role: str, parent_id: uuid.UUID | None = None,
            chunk_index: int, extra_meta: dict | None = None) -> KnowledgeChunk:
        return KnowledgeChunk(
            id=uuid.uuid4(), tenant_id=tenant_id, document_id=document_id,
            chunk_index=chunk_index,
            # Denormalised from the document so retrieval can filter chunks by project
            # without joining. Re-ingest replaces chunks, so this stays in step with the doc.
            project_id=doc.project_id, knowledge_base_id=doc.knowledge_base_id,
            content=chunk.embed_text, embedding=vector,
            token_count=len(chunk.content) // 4,
            chunk_role=role, parent_id=parent_id,
            # `kind` (prose|list|table) is carried through from the chunker for observability
            # and any future kind-aware ranking; retrieval does not depend on it today.
            meta={"heading": chunk.heading, "doc_type": doc.doc_type, "title": doc.title,
                  "kind": chunk.metadata.get("kind"), **(extra_meta or {})})

    vectors = iter(result.vectors)
    for fam in families:
        parent_vec = next(vectors)
        child_vecs = [next(vectors) for _ in fam.children]
        kept = [(c, v) for c, v in zip(fam.children, child_vecs) if v is not None]

        def _enrich(chunk: Chunk) -> dict:
            if not enrich:
                return {}
            facts = _chunk_facts(chunk.embed_text, document_id, chunk.index)
            if facts is None:
                report.enrichment_failures += 1
                return {}
            report.chunks_with_price += facts["has_price"]
            report.chunks_with_area += facts["has_area"]
            report.chunks_with_bhk += bool(facts["bhk"])
            return {"facts": facts}

        if not kept:
            # No children (not split, or none could be embedded): an ordinary standalone
            # chunk — searchable in BOTH retrieval modes, exactly as before R5.03.
            if parent_vec is None:
                continue      # could not be embedded at all; the rest of the document stands
            db.add(row(fam.parent, parent_vec, role=ROLE_STANDALONE,
                       chunk_index=fam.parent.index,
                       extra_meta=_enrich(fam.parent) or None))
            report.standalone_created += 1
            continue

        # A parent is stored even if ITS embedding failed: its children are the search surface
        # in parent_child mode, and the parent's text is what they hand to the model.
        parent = row(fam.parent, parent_vec, role=ROLE_PARENT, chunk_index=fam.parent.index,
                     extra_meta={"children": len(kept)})
        db.add(parent)
        report.parents_created += 1
        for child, vec in kept:
            # chunk_index = the parent's, so ordering by chunk_index keeps a family together;
            # the child's position within its parent is in metadata.child_index.
            db.add(row(child, vec, role=ROLE_CHILD, parent_id=parent.id,
                       chunk_index=fam.parent.index,
                       extra_meta={"child_index": child.metadata.get("child_index"),
                                   **_enrich(child)}))
            report.children_created += 1

    report.chunks_created = (report.standalone_created + report.parents_created
                             + report.children_created)
    doc.status = "ready" if report.embeddings_failed == 0 else "partial"
    doc.version = (doc.version or 0) + 1
    doc.last_indexed_at = datetime.now(timezone.utc)
    report.status = doc.status
    await db.flush()
    log.info("ingest doc=%s tenant=%s status=%s standalone=%d parents=%d children=%d "
             "embed_failed=%d batches_retried=%d parent_child=%s normalized=%s",
             document_id, tenant_id, report.status, report.standalone_created,
             report.parents_created, report.children_created, report.embeddings_failed,
             report.embedding_batches_retried, pc_enabled, normalized.report.changed)

    # One line per document: grep "rag.enrich" to see what the extractor found in each doc.
    log.info("rag.enrich doc=%s title=%r chunks=%d price=%d area=%d bhk=%d failed=%d%s",
             document_id, doc.title, report.chunks_created, report.chunks_with_price,
             report.chunks_with_area, report.chunks_with_bhk, report.enrichment_failures,
             "" if enrich else " (disabled)")
    # The chunks just replaced may already be sitting in another caller's cached knowledge
    # lookup (Redis, 5-min TTL) -- bump the tenant's cache version so no request after this
    # point can be served a pre-ingestion answer for the rest of that window.
    from app.engine.rag.retrieval import invalidate_tenant_knowledge_cache
    await invalidate_tenant_knowledge_cache(tenant_id)

    return report


async def ingest_task(tenant_id: uuid.UUID, document_id: uuid.UUID, text: str) -> None:
    """Background entry point — opens its own session (Celery seam)."""
    from app.db.base import get_session_factory
    async with get_session_factory()() as db:
        try:
            await ingest_document(db, tenant_id=tenant_id, document_id=document_id, text=text)
            await db.commit()
        except Exception:
            await db.rollback()
