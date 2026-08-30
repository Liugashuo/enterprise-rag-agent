from __future__ import annotations

from abc import ABC, abstractmethod
import re

from ..models import ChatMessage


class QueryRewriter(ABC):
    @abstractmethod
    def rewrite(self, query: str, history: list[ChatMessage]) -> str:
        raise NotImplementedError


class RuleBasedQueryRewriter(QueryRewriter):
    """无 LLM 时的指代消解与上下文补全。"""

    _REFERENCE_RE = re.compile(r"(这个|那个|它|其|该|上述|前面|之前|this|that|it)")

    def rewrite(self, query: str, history: list[ChatMessage]) -> str:
        if not history:
            return query
        recent_user = [m.content for m in history[-4:] if m.role == "user"]
        if not recent_user:
            return query
        if self._REFERENCE_RE.search(query) or len(query.strip()) <= 12:
            context = "；".join(recent_user)
            if context:
                return f"{query}\n（上文背景：{context}）"
        return query


class LLMQueryRewriter(QueryRewriter):
    def __init__(self, llm):
        self._llm = llm

    def rewrite(self, query: str, history: list[ChatMessage]) -> str:
        history_text = "\n".join(f"{m.role}: {m.content}" for m in history[-4:])
        messages = [
            {
                "role": "system",
                "content": "[REWRITE] 你是查询重写器。把当前问题改写为独立、完整、适合知识库检索的问题。只输出问题本身。",
            },
            {"role": "user", "content": f"历史对话：\n{history_text}\n\n当前问题：{query}"},
        ]
        rewritten = self._llm.complete(messages, temperature=0.0).strip()
        return rewritten or query


def get_query_rewriter(llm, use_llm: bool = True) -> QueryRewriter:
    if use_llm:
        return LLMQueryRewriter(llm)
    return RuleBasedQueryRewriter()
