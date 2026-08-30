from __future__ import annotations

import json
from typing import Any

from ..retrieval import HybridRetriever


class KnowledgeSearchTool:
    name = "search_knowledge"
    description = "在企业知识库中检索与问题相关的文档片段。适合制度、流程、产品、技术等知识型问题。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "改写后的独立检索问题"},
            "top_k": {"type": "integer", "description": "返回片段数量", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def run(self, query: str, top_k: int = 5) -> str:
        results = self.retriever.search(query, top_k=top_k)
        payload = {
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                    "content": r.content,
                    "score": round(r.score, 4),
                    "metadata": r.metadata,
                    "source": r.source,
                }
                for r in results
            ]
        }
        return json.dumps(payload, ensure_ascii=False)


def create_knowledge_tool(retriever: HybridRetriever) -> Any:
    tool = KnowledgeSearchTool(retriever)
    from .registry import Tool

    return Tool(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        function=tool.run,
    )
