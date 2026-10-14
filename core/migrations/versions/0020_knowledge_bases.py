"""Vocal Backend Tasksheet P2 (Knowledge Base): knowledge_bases + knowledge_citations
as genuinely new tables; knowledge_documents (the tasksheet's "knowledge_sources")
gains knowledge_base_id/source_type/last_indexed_at; knowledge_chunks gains a
denormalised knowledge_base_id for join-free retrieval scoping.

Additive and idempotent (guarded), so a re-run is safe. Every new/altered column on
the two existing tables is nullable or has a server default, so this never breaks
an existing row.

Revision ID: 0020
Revises: 0019
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def _columns(table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def _has_index(name: str) -> bool:
    bind = op.get_bind()
    row = bind.execute(sa.text("SELECT 1 FROM pg_indexes WHERE indexname = :name"), {"name": name}).first()
    return row is not None


def upgrade() -> None:
    if not _has_table("knowledge_bases"):
        op.create_table(
            "knowledge_bases",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("name", sa.String(length=200), nullable=False),
            sa.Column("scope_description", sa.Text(), nullable=True),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )

    doc_cols = _columns("knowledge_documents")
    if "knowledge_base_id" not in doc_cols:
        op.add_column("knowledge_documents", sa.Column(
            "knowledge_base_id", UUID(as_uuid=True), sa.ForeignKey("knowledge_bases.id"), nullable=True))
        op.create_index("ix_knowledge_documents_kb", "knowledge_documents", ["knowledge_base_id"])
    if "source_type" not in doc_cols:
        op.add_column("knowledge_documents", sa.Column(
            "source_type", sa.String(length=20), nullable=False, server_default="doc"))
    if "last_indexed_at" not in doc_cols:
        op.add_column("knowledge_documents", sa.Column("last_indexed_at", sa.DateTime(timezone=True), nullable=True))

    chunk_cols = _columns("knowledge_chunks")
    if "knowledge_base_id" not in chunk_cols:
        op.add_column("knowledge_chunks", sa.Column(
            "knowledge_base_id", UUID(as_uuid=True), sa.ForeignKey("knowledge_bases.id"), nullable=True))
        op.create_index("ix_chunks_kb", "knowledge_chunks", ["tenant_id", "knowledge_base_id"])

    if not _has_table("knowledge_citations"):
        op.create_table(
            "knowledge_citations",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("chunk_id", UUID(as_uuid=True),
                      sa.ForeignKey("knowledge_chunks.id"), nullable=False, index=True),
            sa.Column("conversation_id", UUID(as_uuid=True), sa.ForeignKey("conversations.id"), nullable=True),
            sa.Column("relevance_score", sa.Numeric(6, 4), nullable=False),
            sa.Column("was_cited", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
        op.create_index("ix_knowledge_citation_chunk", "knowledge_citations", ["tenant_id", "chunk_id"])


def downgrade() -> None:
    if _has_table("knowledge_citations"):
        op.drop_index("ix_knowledge_citation_chunk", table_name="knowledge_citations")
        op.drop_table("knowledge_citations")
    chunk_cols = _columns("knowledge_chunks")
    if "knowledge_base_id" in chunk_cols:
        if _has_index("ix_chunks_kb"):
            op.drop_index("ix_chunks_kb", table_name="knowledge_chunks")
        op.drop_column("knowledge_chunks", "knowledge_base_id")
    doc_cols = _columns("knowledge_documents")
    if "last_indexed_at" in doc_cols:
        op.drop_column("knowledge_documents", "last_indexed_at")
    if "source_type" in doc_cols:
        op.drop_column("knowledge_documents", "source_type")
    if "knowledge_base_id" in doc_cols:
        if _has_index("ix_knowledge_documents_kb"):
            op.drop_index("ix_knowledge_documents_kb", table_name="knowledge_documents")
        op.drop_column("knowledge_documents", "knowledge_base_id")
    if _has_table("knowledge_bases"):
        op.drop_table("knowledge_bases")
