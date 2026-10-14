"""KB-scoped retrieval + citation logging (Vocal Backend Tasksheet P2/P3) — the
shared core behind POST /internal/retrieve (app/api/knowledge_bases.py) and P3's
RETRIEVE node factory (app/engine/templates/nodes.py). Lives in the engine layer,
not the API layer, since both an API route and an in-process graph node need to
call it without the engine ever importing from app.api.*.
"""
import asyncio
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.base import get_session_factory
from app.db.models import KnowledgeCitation
from app.engine.rag.retrieval import RetrievalResult, RetrievedChunk, retrieve


async def retrieve_for_kb_ids(citation_db: AsyncSession, *, tenant_id: uuid.UUID, query: str,
                              kb_ids: list[uuid.UUID], top_k: int | None = None,
                              project_id: uuid.UUID | None = None,
                              conversation_id: uuid.UUID | None = None) -> list[RetrievedChunk]:
    """Calls the shared retrieve() once per requested kb_id, each on its own DB
    session (an AsyncSession is not safe for concurrent queries — see
    app/engine/rag/retrieval.py's docstring on why this isn't a single query with
    a list param), merges by best score, and logs one knowledge_citations row per
    returned chunk using `citation_db` (a session the caller commits — this
    function flushes but does not commit, so the caller's own transaction
    boundary, whatever it is, is respected)."""
    top_k = top_k or get_settings().retrieval_top_k

    async def _retrieve_one(kb_id: uuid.UUID) -> RetrievalResult:
        async with get_session_factory()() as kb_db:
            return await retrieve(kb_db, tenant_id=tenant_id, query=query, top_k=top_k,
                                  project_id=project_id, knowledge_base_id=kb_id)

    results = await asyncio.gather(*[_retrieve_one(kb_id) for kb_id in kb_ids])
    merged: dict[str, RetrievedChunk] = {}
    for result in results:
        for c in result.chunks:
            if c.chunk_id not in merged or c.score > merged[c.chunk_id].score:
                merged[c.chunk_id] = c
    chunks = sorted(merged.values(), key=lambda c: c.score, reverse=True)[:top_k]

    for c in chunks:
        citation_db.add(KnowledgeCitation(tenant_id=tenant_id, chunk_id=uuid.UUID(c.chunk_id),
                                          conversation_id=conversation_id, relevance_score=c.relevance))
    await citation_db.flush()
    return chunks
