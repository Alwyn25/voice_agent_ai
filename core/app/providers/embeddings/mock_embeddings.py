"""Deterministic offline embeddings — lets ingestion and retrieval be tested without an API.

Hash-based bag-of-words projection. Not semantically meaningful, but stable and
similar texts share dimensions, which is enough to exercise the pipeline.
"""
import hashlib
import math
import re

from app.providers.embeddings.base import EmbeddingResult

DIMS = 1536


class MockEmbeddings:
    name = "mock"
    dimensions = DIMS

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        return EmbeddingResult(vectors=[_vector(t) for t in texts], model="mock-embed",
                               dimensions=DIMS, tokens={"total": 0})


def _vector(text: str) -> list[float]:
    vec = [0.0] * DIMS
    for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % DIMS] += 1.0
        vec[(h >> 16) % DIMS] += 0.5
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]
