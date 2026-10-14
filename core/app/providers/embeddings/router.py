from functools import lru_cache

from app.config import get_settings
from app.providers.embeddings.base import EmbeddingProvider


@lru_cache
def get_embeddings() -> EmbeddingProvider:
    s = get_settings()
    if s.embedding_provider == "mock" or s.llm_provider == "mock":
        from app.providers.embeddings.mock_embeddings import MockEmbeddings
        return MockEmbeddings()
    from app.providers.embeddings.openai_embeddings import OpenAIEmbeddings
    return OpenAIEmbeddings()
