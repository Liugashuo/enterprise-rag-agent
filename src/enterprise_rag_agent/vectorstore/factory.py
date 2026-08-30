from __future__ import annotations

from ..config import Settings
from .base import VectorStore
from .json_store import JsonVectorStore


def get_vector_store(settings: Settings) -> VectorStore:
    store_type = settings.vector_store_type.lower()
    if store_type == "json":
        return JsonVectorStore(settings.vector_store_path)
    if store_type == "chroma":
        from .chroma_store import ChromaVectorStore

        return ChromaVectorStore(settings.vector_store_path)
    raise ValueError(f"不支持的向量库类型: {settings.vector_store_type}")

