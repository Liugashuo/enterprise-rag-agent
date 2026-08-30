from __future__ import annotations

from typing import Any

from ..config import Settings
from ..embeddings import Embedder
from ..models import Chunk, SearchResult
from ..vectorstore import VectorStore
from .bm25 import BM25Index
from .reranker import Reranker


class HybridRetriever:
    """BM25 + 向量检索，使用 RRF 融合并可选 Rerank。"""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Reranker,
        top_k: int = 6,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.top_k = top_k
        self.bm25 = BM25Index()
        self.rebuild_index()

    def rebuild_index(self) -> None:
        self.bm25.build(self.vector_store.get_all_chunks())

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        top_k = top_k or self.top_k
        query_embedding = self.embedder.embed_query(query)
        vector_hits = self.vector_store.search(query_embedding, top_k=top_k * 2, filters=filters)
        bm25_hits = self.bm25.search(query, top_k=top_k * 2)

        fused = self._rrf_fusion(vector_hits, bm25_hits, k=60)
        ids = [chunk_id for chunk_id, _ in fused[: top_k * 2]]
        chunk_map = {c.id: c for c in self.vector_store.get_chunks(ids)}
        candidates: list[SearchResult] = []
        for chunk_id, score in fused[: top_k * 2]:
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                continue
            candidates.append(
                SearchResult(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    score=score,
                    metadata=chunk.metadata,
                )
            )
        candidates = self.reranker.rerank(query, candidates)
        return candidates[:top_k]

    @staticmethod
    def _rrf_fusion(
        vector_hits: list[Any],
        bm25_hits: list[tuple[str, float]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for rank, hit in enumerate(vector_hits):
            chunk_id = getattr(hit, "chunk_id", None)
            if chunk_id:
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        for rank, (chunk_id, _) in enumerate(bm25_hits):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

