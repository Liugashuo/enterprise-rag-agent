from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Iterable

from .config import Settings


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """对中英文做轻量分词；BM25 与 HashEmbedder 共用。"""
    text = text.lower()
    tokens: list[str] = []
    for match in _WORD_RE.finditer(text):
        tokens.append(match.group())
    for char in text:
        if _CJK_RE.match(char):
            tokens.append(char)
    # 为中文补充二元组，提高短词召回。
    cjk = [c for c in text if _CJK_RE.match(c)]
    tokens.extend("".join(cjk[i : i + 2]) for i in range(len(cjk) - 1))
    return tokens


class Embedder(ABC):
    dimension: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HashEmbedder(Embedder):
    """零依赖确定性嵌入，仅用于本地演示与测试。

    生产环境请使用 OpenAI、BGE、M3E 等语义嵌入模型。
    """

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = tokenize(text) or [text]
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str, api_key: str, base_url: str | None = None, dimension: int = 1536):
        self.model = model
        self.dimension = dimension
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url or None)
        except ImportError as exc:  # pragma: no cover - 运行时依赖错误
            raise RuntimeError("使用 OpenAIEmbedder 需要安装 openai 包。") from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=texts)
        data = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in data]


def get_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider.lower()
    if provider in {"openai", "remote", "ollama"}:
        return OpenAIEmbedder(
            model=settings.embedding_model,
            api_key=settings.api_key,
            base_url=settings.embedding_base_url or settings.model_base_url or None,
            dimension=settings.embedding_dimension,
        )
    if provider in {"hash", "local", "dummy"}:
        return HashEmbedder(settings.embedding_dimension)
    raise ValueError(f"不支持的 Embedding Provider: {settings.embedding_provider}")
