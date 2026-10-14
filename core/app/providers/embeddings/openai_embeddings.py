"""OpenAI embeddings adapter. text-embedding-3-small -> 1536 dims (matches the schema)."""
from app.config import get_settings
from app.providers.embeddings.base import EmbeddingError, EmbeddingResult

BATCH_SIZE = 96


class OpenAIEmbeddings:
    """Works against any OpenAI-compatible embeddings endpoint.

    The database column is a fixed-width vector, so whatever model you point this at MUST
    return that many dimensions or nothing will insert. Some models (Gemini's embedding
    family, OpenAI's -3 series) accept a `dimensions` argument and will truncate to fit;
    EMBEDDING_DIMENSIONS asks for that.

    Changing the embedding MODEL invalidates every stored vector — they are only
    comparable to others from the same model — so re-run knowledge ingestion afterwards or
    retrieval will return confident nonsense.
    """

    name = "openai"
    dimensions = 1536

    def __init__(self) -> None:
        s = get_settings()
        if not (s.embedding_api_key or s.openai_api_key):
            raise EmbeddingError("OPENAI_API_KEY is not configured")
        from openai import AsyncOpenAI
        # Embedding a whole document is a far heavier request than a chat completion, and
        # it shares LLM_TIMEOUT_SECONDS with them. At 8 seconds three DSR documents were
        # cut off mid-flight and silently stored as zero chunks — invisible to the agent,
        # with the seeder reporting success. A floor of 60s here means a short chat timeout
        # cannot quietly destroy the knowledge base, and max_retries lets the SDK ride out
        # a slow response rather than losing the document.
        timeout = max(float(s.llm_timeout_seconds or 0), 60.0)
        self._client = AsyncOpenAI(
            api_key=s.embedding_api_key or s.openai_api_key,
            timeout=timeout,
            max_retries=max(int(s.llm_max_retries or 0), 3),
            **({"base_url": s.embedding_base_url} if s.embedding_base_url else {}))
        self._model = s.embedding_model
        self.dimensions = int(getattr(s, "embedding_dimensions", 0) or self.dimensions)
        self._ask_dims = bool(getattr(s, "embedding_dimensions", 0))

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=[], model=self._model, dimensions=self.dimensions)
        vectors: list[list[float]] = []
        total = 0
        for i in range(0, len(texts), BATCH_SIZE):
            batch = [t.replace("\n", " ")[:8000] for t in texts[i:i + BATCH_SIZE]]
            try:
                resp = await self._client.embeddings.create(
                    model=self._model, input=batch,
                    **({"dimensions": self.dimensions} if self._ask_dims else {}))
            except Exception as e:
                raise EmbeddingError(f"{e.__class__.__name__}: {e}") from e
            vectors.extend([d.embedding for d in resp.data])
            total += getattr(resp.usage, "total_tokens", 0) or 0
        return EmbeddingResult(vectors=vectors, model=self._model,
                               dimensions=self.dimensions, tokens={"total": total})