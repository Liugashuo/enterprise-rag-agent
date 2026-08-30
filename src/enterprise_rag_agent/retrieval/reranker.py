from __future__ import annotations

from abc import ABC, abstractmethod

from ..embeddings import tokenize
from ..models import SearchResult


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        raise NotImplementedError


class IdentityReranker(Reranker):
    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        return candidates


class KeywordReranker(Reranker):
    """轻量关键词重合度重排，用于无 GPU/无模型环境。"""

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        query_terms = set(tokenize(query))
        for item in candidates:
            content_terms = set(tokenize(item.content))
            overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
            item.score = 0.65 * item.score + 0.35 * overlap
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name)
        except ImportError as exc:
            raise RuntimeError("使用 CrossEncoder 重排需要安装 sentence-transformers。") from exc

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        pairs = [(query, item.content) for item in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        for item, score in zip(candidates, scores):
            item.score = float(score)
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates


def get_reranker(enabled: bool, model_name: str = "") -> Reranker:
    if not enabled:
        return IdentityReranker()
    if model_name and model_name.lower() not in {"keyword", "none", "identity"}:
        try:
            return CrossEncoderReranker(model_name)
        except RuntimeError:
            return KeywordReranker()
    return KeywordReranker()

