from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import Chunk
from .base import SearchHit, VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, root: str | Path, collection_name: str = "enterprise_docs"):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("使用 ChromaVectorStore 需要安装 chromadb。") from exc
        self._client = chromadb.PersistentClient(path=str(root))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.content for c in chunks],
            embeddings=embeddings,
            metadatas=[{**c.metadata, "document_id": c.document_id} for c in chunks],
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        where = self._build_where(filters)
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        hits: list[SearchHit] = []
        for idx, chunk_id in enumerate(ids):
            distance = float(distances[idx]) if idx < len(distances) else 1.0
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            chunk = Chunk(
                id=chunk_id,
                document_id=str(metadata.get("document_id", "")),
                content=documents[idx] if idx < len(documents) else "",
                metadata=metadata,
            )
            hits.append(SearchHit(chunk_id=chunk_id, score=1.0 - distance, chunk=chunk))
        return hits

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        result = self._collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        chunks: list[Chunk] = []
        for idx, chunk_id in enumerate(result["ids"]):
            metadata = result["metadatas"][idx] or {}
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=str(metadata.get("document_id", "")),
                    content=result["documents"][idx] or "",
                    metadata=metadata,
                )
            )
        return chunks

    def get_all_chunks(self) -> list[Chunk]:
        result = self._collection.get(include=["documents", "metadatas"])
        chunks: list[Chunk] = []
        for idx, chunk_id in enumerate(result["ids"]):
            metadata = result["metadatas"][idx] or {}
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=str(metadata.get("document_id", "")),
                    content=result["documents"][idx] or "",
                    metadata=metadata,
                )
            )
        return chunks

    def delete(self, document_id: str) -> int:
        result = self._collection.get(
            where={"document_id": document_id},
            include=[],
        )
        ids = result.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def clear(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _build_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
        if not filters:
            return None
        conditions = [{"document_id": value} for value in filters.values()]
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

