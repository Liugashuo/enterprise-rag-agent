from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any

from ..models import Chunk
from .base import SearchHit, VectorStore


class JsonVectorStore(VectorStore):
    """零依赖 JSON 向量库，适合本地演示与单元测试。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.chunks_path = self.root / "chunks.json"
        self.vectors_path = self.root / "vectors.json"
        self._lock = threading.Lock()
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}
        self._load()

    def _load(self) -> None:
        if self.chunks_path.exists():
            try:
                raw = json.loads(self.chunks_path.read_text(encoding="utf-8"))
                self._chunks = {cid: Chunk(**item) for cid, item in raw.items()}
            except Exception:
                self._chunks = {}
        if self.vectors_path.exists():
            try:
                self._vectors = json.loads(self.vectors_path.read_text(encoding="utf-8"))
            except Exception:
                self._vectors = {}

    def _save(self) -> None:
        chunks = {cid: chunk.to_dict() for cid, chunk in self._chunks.items()}
        tmp_chunks = self.chunks_path.with_suffix(".tmp")
        tmp_vectors = self.vectors_path.with_suffix(".tmp")
        tmp_chunks.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        tmp_vectors.write_text(json.dumps(self._vectors, ensure_ascii=False), encoding="utf-8")
        tmp_chunks.replace(self.chunks_path)
        tmp_vectors.replace(self.vectors_path)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks 与 embeddings 数量不一致")
        with self._lock:
            for chunk, vector in zip(chunks, embeddings):
                self._chunks[chunk.id] = chunk
                self._vectors[chunk.id] = vector
            self._save()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        with self._lock:
            for chunk_id, vector in self._vectors.items():
                chunk = self._chunks.get(chunk_id)
                if chunk is None:
                    continue
                if not self._matches(chunk, filters):
                    continue
                score = self._cosine(query_embedding, vector)
                hits.append(SearchHit(chunk_id=chunk_id, score=score, chunk=chunk))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def get_all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def delete(self, document_id: str) -> int:
        with self._lock:
            ids = [cid for cid, chunk in self._chunks.items() if chunk.document_id == document_id]
            for cid in ids:
                self._chunks.pop(cid, None)
                self._vectors.pop(cid, None)
            if ids:
                self._save()
            return len(ids)

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._vectors.clear()
            self._save()

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _matches(chunk: Chunk, filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        return all(chunk.metadata.get(key) == value for key, value in filters.items())

