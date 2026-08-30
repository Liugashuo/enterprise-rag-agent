from __future__ import annotations

import json
from typing import Any

from ..config import Settings
from ..llm import LLMClient
from ..memory import MemoryManager
from ..models import ChatMessage
from ..retrieval import HybridRetriever, QueryRewriter
from ..tools import ToolRegistry
from .state import AgentState


class GraphNodes:
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
        self.query_rewriter = query_rewriter
        self.tools = tools

    def preprocess(self, state: AgentState) -> dict[str, Any]:
        context = self.memory.prepare(state["session_id"], state["query"])
        return {
            "history": context.messages,
            "summary": context.summary,
            "system_prompt": context.system_prompt,
        }

    def classify_intent(self, state: AgentState) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "[INTENT] 你是意图识别器。只能输出 knowledge、history、chat、task 之一。"
                    "knowledge=需要查企业知识库；history=需要查历史对话；"
                    "chat=普通闲聊；task=需要规划步骤并可能调用工具。"
                ),
            },
            {"role": "user", "content": state["query"]},
        ]
        raw = self.llm.complete(messages, temperature=0.0).strip().lower()
        intent = self._normalize_intent(raw) or self._fallback_intent(state["query"])
        return {"intent": intent}

    def plan(self, state: AgentState) -> dict[str, Any]:
        intent = state.get("intent", "chat")
        plans = {
            "chat": ["直接回答用户问题"],
            "history": ["调用历史消息检索工具", "基于历史对话回答"],
            "knowledge": ["改写查询", "调用知识库检索工具", "校验证据", "生成答案"],
            "task": ["识别任务目标", "调用必要工具", "规划执行步骤", "生成答案"],
        }
        plan = plans.get(intent, plans["chat"])
        rewritten = state["query"]
        if intent in {"knowledge", "task"}:
            history_messages = [
                ChatMessage(**item) for item in state.get("history", []) if item.get("role") in {"user", "assistant"}
            ]
            rewritten = self.query_rewriter.rewrite(state["query"], history_messages)
        return {"plan": plan, "rewritten_query": rewritten}

    def tool_agent(self, state: AgentState) -> dict[str, Any]:
        intent = state.get("intent", "chat")
        tools_names = self._tool_names_for_intent(intent)
        tools_spec = self.tools.to_openai(tools_names) if tools_names else None
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是企业文档智能 Agent。根据用户问题选择工具，工具结果作为后续回答的证据。"
                    "如果没有合适工具，直接给出诚实回答。"
                ),
            }
        ]
        messages.extend(state.get("history", []))

        search_results: list[dict[str, Any]] = list(state.get("search_results", []))
        tool_calls: list[dict[str, Any]] = list(state.get("tool_calls", []))
        history_results = state.get("history_results", "")
        draft_answer = ""

        for _ in range(max(self.settings.max_agent_steps, 1)):
            response = self.llm.chat(messages, tools=tools_spec, temperature=0.0)
            assistant_message: dict[str, Any] = {"role": "assistant", "content": response.content}
            if response.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
                    }
                    for call in response.tool_calls
                ]
            messages.append(assistant_message)

            if not response.tool_calls:
                draft_answer = response.content
                break

            for call in response.tool_calls:
                if call.name == "search_history" and "session_id" not in call.arguments:
                    call.arguments["session_id"] = state["session_id"]
                try:
                    result = self.tools.execute(call)
                except Exception as exc:
                    result = json.dumps({"error": str(exc)}, ensure_ascii=False)
                tool_calls.append({"name": call.name, "arguments": call.arguments, "result": result})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                if call.name == "search_knowledge":
                    search_results.extend(self._parse_search_results(result))
                elif call.name == "search_history":
                    history_results = result

        return {
            "search_results": search_results,
            "history_results": history_results,
            "tool_calls": tool_calls,
            "draft_answer": draft_answer,
            "iterations": state.get("iterations", 0) + 1,
        }

    def validate(self, state: AgentState) -> dict[str, Any]:
        intent = state.get("intent", "chat")
        results = list(state.get("search_results", []))
        if intent not in {"knowledge", "task"}:
            return {"confidence": 1.0, "low_confidence": False}
        if not results:
            fallback = self.retriever.search(state.get("rewritten_query") or state["query"], top_k=self.settings.top_k)
            results = [r.to_dict() for r in fallback]
        if not results:
            return {"search_results": [], "confidence": 0.2, "low_confidence": True}
        unique_docs = {item.get("document_id") for item in results if item.get("document_id")}
        max_score = max((float(item.get("score", 0.0)) for item in results), default=0.0)
        below_threshold = max_score < self.settings.evidence_min_score
        confidence = min(0.98, 0.5 + 0.12 * len(unique_docs) + 0.2 * min(max_score * 20.0, 1.0))
        if below_threshold:
            confidence = min(confidence, 0.4)
        return {
            "search_results": results,
            "confidence": round(confidence, 4),
            "low_confidence": confidence < 0.45 or below_threshold,
        }

    def generate(self, state: AgentState) -> dict[str, Any]:
        evidence = self._build_evidence(state)
        system = f"{state.get('system_prompt', '')}\n[GENERATE]\n{evidence}"
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        messages.extend(state.get("history", []))
        answer = self.llm.complete(messages, temperature=0.2).strip()
        if not answer:
            answer = state.get("draft_answer") or "抱歉，暂时无法生成答案。"
        if state.get("low_confidence"):
            answer = f"⚠️ 当前答案证据较弱，请谨慎参考。\n\n{answer}"
        return {"final_answer": answer}

    def finalize(self, state: AgentState) -> dict[str, Any]:
        return {
            "final_answer": state.get("final_answer", ""),
            "confidence": state.get("confidence", 1.0),
            "low_confidence": state.get("low_confidence", False),
        }

    def _tool_names_for_intent(self, intent: str) -> list[str]:
        if intent == "knowledge":
            return ["search_knowledge"]
        if intent == "history":
            return ["search_history"]
        if intent == "task":
            return ["search_knowledge", "search_history"]
        return []

    @staticmethod
    def _normalize_intent(raw: str) -> str | None:
        mapping = {
            "knowledge": "knowledge",
            "knowledge_qa": "knowledge",
            "知识": "knowledge",
            "history": "history",
            "历史": "history",
            "chat": "chat",
            "闲聊": "chat",
            "task": "task",
            "任务": "task",
        }
        return mapping.get(raw)

    @staticmethod
    def _fallback_intent(query: str) -> str:
        if any(word in query for word in ("之前", "上次", "刚才", "历史", "以前", "我说过")):
            return "history"
        if any(word in query for word in ("计划", "步骤", "方案", "制定")):
            return "task"
        if any(
            word in query
            for word in (
                "文档",
                "知识",
                "规定",
                "政策",
                "制度",
                "手册",
                "流程",
                "如何",
                "什么",
                "哪些",
                "怎么",
                "多少",
                "几天",
                "哪",
                "是否",
                "吗",
                "查询",
            )
        ):
            return "knowledge"
        return "chat"

    @staticmethod
    def _parse_search_results(raw: str) -> list[dict[str, Any]]:
        try:
            payload = json.loads(raw)
            return list(payload.get("results", []))
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _build_evidence(state: AgentState) -> str:
        parts: list[str] = []
        for item in state.get("search_results", [])[:8]:
            source = item.get("document_id", "未知文档")
            score = item.get("score", 0.0)
            content = item.get("content", "")
            parts.append(f"[{source} | score={score:.4f}] {content}")
        if state.get("history_results"):
            parts.append(f"[历史消息]\n{state['history_results']}")
        if state.get("draft_answer"):
            parts.append(f"[Agent草稿]\n{state['draft_answer']}")
        return "证据资料：\n" + "\n\n".join(parts) if parts else "证据资料：\n（无可用证据）"
