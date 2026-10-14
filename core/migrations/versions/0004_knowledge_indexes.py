"""Indexes for hybrid retrieval: HNSW for vector search, GIN for full-text.

Revision ID: 0004
Revises: 0003
"""
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Semantic search. HNSW gives good recall with fast queries; cosine matches the
    # distance operator used in retrieval (<=>).
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw
        ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)
    """)
    # Keyword search over the same chunks.
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_chunks_content_fts
        ON knowledge_chunks USING gin (to_tsvector('english', content))
    """)
    # Effective-date filtering happens on every retrieval.
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_documents_effective
        ON knowledge_documents (tenant_id, status, effective_from, effective_to)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_documents_effective")
    op.execute("DROP INDEX IF EXISTS ix_chunks_content_fts")
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
