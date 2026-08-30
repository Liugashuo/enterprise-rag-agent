from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ..config import Settings
from ..llm import LLMClient
from ..memory import MemoryManager
from ..models import AgentResult, SearchResult
from ..retrieval import HybridRetriever, QueryRewriter
from ..tools import ToolRegistry
from .nodes import GraphNodes
from .state import AgentState


class AgentWorkflow:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        memory: MemoryManager,
        retriever: HybridRetriever,
        query_rewriter: QueryRewriter,
        tools: ToolRegistry,
    ):
        self.settings = settings
        self.llm = llm
        self.memory = memory
        self.retriever = retriever
        self.nodes = GraphNodes(
            settings=settings,
            llm=llm,
            memory=memory,
            retriever=retriever,
            query_rewriter=query_rewriter,
            tools=tools,
        )
        self._graph = self._build()

    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("preprocess", self.nodes.preprocess)
        graph.add_node("classify_intent", self.nodes.classify_intent)
        graph.add_node("plan", self.nodes.plan)
        graph.add_node("tool_agent", self.nodes.tool_agent)
        graph.add_node("validate", self.nodes.validate)
        graph.add_node("generate", self.nodes.generate)
        graph.add_node("finalize", self.nodes.finalize)

        graph.add_edge(START, "preprocess")
        graph.add_edge("preprocess", "classify_intent")
        graph.add_edge("classify_intent", "plan")
        graph.add_edge("plan", "tool_agent")
        graph.add_edge("tool_agent", "validate")
        graph.add_edge("validate", "generate")
        graph.add_edge("generate", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def run(self, session_id: str, query: str) -> AgentResult:
        initial: AgentState = {
            "session_id": session_id,
            "query": query,
            "intent": "",
            "plan": [],
            "rewritten_query": query,
            "history": [],
            "summary": "",
            "system_prompt": "",
            "search_results": [],
            "history_results": "",
            "draft_answer": "",
            "tool_calls": [],
            "iterations": 0,
            "confidence": 1.0,
            "low_confidence": False,
            "final_answer": "",
            "error": "",
        }
        final = self._graph.invoke(initial)
        sources = [SearchResult(**item) for item in final.get("search_results", [])]
        return AgentResult(
            session_id=session_id,
            answer=final.get("final_answer", ""),
            intent=final.get("intent", "chat"),
            plan=final.get("plan", []),
            sources=sources,
            confidence=final.get("confidence", 1.0),
            low_confidence=final.get("low_confidence", False),
        )

