"""Embedding provider interface."""
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    model: str
    dimensions: int
    tokens: dict[str, int] = field(default_factory=dict)


class EmbeddingError(Exception):
    pass


class EmbeddingProvider(Protocol):
    name: str
    dimensions: int

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        ...
