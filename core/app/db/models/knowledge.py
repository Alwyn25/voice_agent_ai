import uuid
from datetime import date, datetime

from sqlalchemy import Boolean, CheckConstraint, Date, DateTime, ForeignKey, Index, Integer, Numeric, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from app.db.base import Base, IdMixin, TimestampMixin

EMBEDDING_DIM = 1536


class KnowledgeBase(IdMixin, TimestampMixin, Base):
    """Vocal Backend Tasksheet P2 — a named grouping of knowledge_documents (the
    tasksheet's "knowledge_sources"; see KnowledgeDocument below for why that table
    is reused rather than duplicated). NULL knowledge_base_id on a document/chunk
    means the pre-existing tenant-wide, ungrouped case — every KB-scoping query
    treats that as "visible everywhere", exactly as project_id NULL already does."""
    __tablename__ = "knowledge_bases"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    scope_description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="active")  # active|indexing|error
    # UI-only setting for now (the Knowledge Base editor's "Re-index" dropdown) — stored
    # for real so it round-trips, but nothing currently schedules a reindex off of it;
    # a source is still only re-embedded via explicit POST .../reingest.
    reindex_policy: Mapped[str] = mapped_column(String(30), default="on_source_change")


class KnowledgeDocument(IdMixin, TimestampMixin, Base):
    """Also the tasksheet's "knowledge_sources" — reused rather than duplicated,
    the same reconciliation the P0/P1 build already established for tenants/
    tenant_pack_config-style overlaps. source_type/last_indexed_at are the two
    fields the tasksheet's knowledge_sources wanted that this table didn't
    already have; knowledge_base_id is the new grouping FK (nullable, so every
    pre-existing document stays valid and tenant-wide as before)."""
    __tablename__ = "knowledge_documents"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    knowledge_base_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_bases.id"), index=True)
    # Which project this document belongs to. NULL = tenant-wide (the pre-multi-project
    # default, and the fallback for single-project tenants). Retrieval filters on the chunk's
    # copy of this, not here, so it is denormalised onto KnowledgeChunk at ingest.
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"), index=True)
    title: Mapped[str] = mapped_column(String(300))
    doc_type: Mapped[str] = mapped_column(String(30))  # brochure|faq|pricing|policy|promo|playbook|sop|compliance
    source_type: Mapped[str] = mapped_column(String(20), default="doc")  # url|file|doc|integration
    source_uri: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="processing")  # processing|ready|failed
    version: Mapped[int] = mapped_column(Integer, default=1)
    last_indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    effective_from: Mapped[date | None] = mapped_column(Date)
    effective_to: Mapped[date | None] = mapped_column(Date)
    meta: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)


class KnowledgeChunk(IdMixin, TimestampMixin, Base):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        Index("ix_chunks_doc", "tenant_id", "document_id"),
        # Retrieval filters chunks by (tenant, project); index the pair it queries on.
        Index("ix_chunks_project", "tenant_id", "project_id"),
        # R5.03 — names match migration 0036 exactly, so a fresh database built by 0001's
        # create_all() and an upgraded one end up with identical constraints (0036's guards
        # then skip what already exists instead of adding duplicates).
        Index("ix_chunks_parent", "parent_id", postgresql_where=text("parent_id IS NOT NULL")),
        CheckConstraint("chunk_role IN ('standalone', 'parent', 'child')", name="ck_chunks_role"),
        CheckConstraint("(chunk_role = 'child') = (parent_id IS NOT NULL)",
                        name="ck_chunks_role_parent"),
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    # Denormalised from the parent document so retrieval can filter chunks by project without
    # a join. Kept in step with the document at ingest. NULL = tenant-wide (pre-multi-project).
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"))
    # Denormalised the same way, for the same reason: POST /internal/retrieve scopes to one or
    # more knowledge_base_id values without a join back through knowledge_documents.
    knowledge_base_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_bases.id"))
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_documents.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer)
    meta: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    # R5.03 parent-child chunking (migration 0036; app/engine/rag/parent_child.py).
    #   standalone — an ordinary chunk (everything ingested before R5.03, and small chunks)
    #   parent     — a ~900-char chunk that ALSO has children; searched only in "flat" mode
    #   child      — a ~200-char slice of a parent; searched only in "parent_child" mode
    # A child's parent_id is set; nothing else's is (enforced by ck_chunks_role_parent).
    # ON DELETE CASCADE: removing a parent can never strand children that point nowhere.
    chunk_role: Mapped[str] = mapped_column(String(12), server_default="standalone",
                                            default="standalone")
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_chunks.id", ondelete="CASCADE",
                                       name="fk_chunks_parent"))


class KnowledgeCitation(IdMixin, Base):
    """Per-retrieval log of which chunks were returned and (once a later wave can determine
    it from an actual generated reply) which were genuinely cited. Drives the Citation Rate
    stat on the KB list page.

    was_cited defaults True at write time: at retrieval time a returned chunk is the best
    available proxy for "used" until a finer signal (grounding check against the real reply)
    can update it.
    """
    __tablename__ = "knowledge_citations"
    __table_args__ = (Index("ix_knowledge_citation_chunk", "tenant_id", "chunk_id"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    chunk_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_chunks.id"), index=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id"))
    relevance_score: Mapped[float] = mapped_column(Numeric(6, 4))
    was_cited: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
