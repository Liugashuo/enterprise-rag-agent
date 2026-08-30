from __future__ import annotations

import math
from collections import Counter

from ..embeddings import tokenize
from ..models import Chunk


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._corpus: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._doc_freqs: Counter[str] = Counter()
        self._avgdl = 0.0
        self._idf: dict[str, float] = {}

    def build(self, chunks: list[Chunk]) -> None:
        self._corpus = list(chunks)
        self._tokens = [tokenize(c.content) for c in self._corpus]
        self._doc_freqs = Counter()
        for tokens in self._tokens:
            self._doc_freqs.update(set(tokens))
        total_docs = len(self._corpus)
        self._avgdl = (
            sum(len(t) for t in self._tokens) / total_docs if total_docs else 0.0
        )
        self._idf = {
            term: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in self._doc_freqs.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._corpus:
            return []
        query_tokens = tokenize(query)
        scores: list[tuple[str, float]] = []
        for idx, tokens in enumerate(self._tokens):
            length = len(tokens)
            tf = Counter(tokens)
            score = 0.0
            for term in query_tokens:
                if term not in self._idf:
                    continue
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                denominator = freq + self.k1 * (1 - self.b + self.b * length / max(self._avgdl, 1e-9))
                score += self._idf[term] * freq * (self.k1 + 1) / denominator
            if score > 0:
                scores.append((self._corpus[idx].id, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

