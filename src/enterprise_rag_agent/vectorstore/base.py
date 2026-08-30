from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..models import Chunk


@dataclass
class SearchHit:
    chunk_id: str
    score: float
    chunk: Chunk | None = None


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        raise NotImplementedError

    @abstractmethod
    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        raise NotImplementedError

    @abstractmethod
    def get_all_chunks(self) -> list[Chunk]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, document_id: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

