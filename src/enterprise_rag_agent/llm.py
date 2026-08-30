from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import Settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


class LLMClient:
    def complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> str:
        raise NotImplementedError

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        raise NotImplementedError


class OpenAIChatLLM(LLMClient):
    def __init__(self, settings: Settings):
        self.settings = settings
        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=settings.api_key or "EMPTY",
                base_url=settings.model_base_url or None,
            )
        except ImportError as exc:
            raise RuntimeError("使用 OpenAIChatLLM 需要安装 openai。") from exc

    def complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> str:
        response = self.chat(messages, temperature=temperature)
        return response.content

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.settings.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
        completion = self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        message = choice.message
        tool_calls: list[ToolCall] = []
        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(
                    ToolCall(
                        id=call.id,
                        name=call.function.name,
                        arguments=arguments,
                    )
                )
        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
        )


class DeepSeekChatLLM(OpenAIChatLLM):
    """DeepSeek 官方 API 适配层。

    DeepSeek 使用 OpenAI 兼容协议，因此复用 OpenAIChatLLM，
    只在未显式配置时补齐 DeepSeek 默认模型与地址。
    """

    def __init__(self, settings: Settings):
        if not settings.model_name or settings.model_name in {"gpt-4o-mini", "gpt-3.5-turbo"}:
            settings.model_name = "deepseek-chat"
        if not settings.model_base_url:
            settings.model_base_url = "https://api.deepseek.com/v1"
        super().__init__(settings)


class MockLLM(LLMClient):
    """无 API Key 时的确定性演示模型，便于本地跑通整个链路。"""

    def complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> str:
        system = self._system(messages)
        user = self._latest_user(messages)
        if "[INTENT]" in system:
            return self._classify(user)
        if "[PLAN]" in system:
            return "1. 理解问题并确定任务目标\n2. 检索企业知识库并收集证据\n3. 综合证据生成可信答案"
        if "[REWRITE]" in system:
            return user
        if "[SUMMARIZE]" in system:
            text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
            return f"对话摘要：{text[:200]}"
        if "[GENERATE]" in system:
            return self._generate(user, messages)
        return user

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        user = self._latest_user(messages)
        tool_names = {t.get("function", {}).get("name") for t in (tools or [])}
        if any(m.get("role") == "tool" for m in messages):
            return LLMResponse(content=self.complete(messages, temperature=temperature))
        if tool_names and self._is_history_query(user) and "search_history" in tool_names:
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="search_history",
                        arguments={"query": user, "limit": 5},
                    )
                ],
            )
        if tool_names and "search_knowledge" in tool_names and not self._is_pure_chat(user):
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_2",
                        name="search_knowledge",
                        arguments={"query": user, "top_k": 5},
                    )
                ],
            )
        return LLMResponse(content=self.complete(messages, temperature=temperature))

    @staticmethod
    def _system(messages: list[dict[str, Any]]) -> str:
        for message in messages:
            if message.get("role") == "system":
                return str(message.get("content", ""))
        return ""

    @staticmethod
    def _latest_user(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    @staticmethod
    def _classify(query: str) -> str:
        q = query.lower()
        history_words = ("之前", "上次", "刚才", "历史", "以前", "我说过", "你刚才")
        if any(word in query for word in history_words):
            return "history"
        if any(
            word in q or word in query
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
                "介绍",
                "查询",
            )
        ):
            return "knowledge"
        if any(word in query for word in ("计划", "步骤", "方案", "制定")):
            return "task"
        return "chat"

    @staticmethod
    def _is_history_query(query: str) -> bool:
        return any(word in query for word in ("之前", "上次", "刚才", "历史", "以前", "我说过"))

    @staticmethod
    def _is_pure_chat(query: str) -> bool:
        chat_words = ("你好", "谢谢", "再见", "你是谁", "能做什么")
        return any(word in query for word in chat_words)

    def _generate(self, user: str, messages: list[dict[str, Any]]) -> str:
        context_parts = [
            str(m.get("content", ""))
            for m in messages
            if m.get("role") in {"system", "tool"} and "证据" in str(m.get("content", ""))
        ]
        evidence_parts: list[str] = []
        for part in context_parts:
            if "证据资料：" in part:
                evidence_parts.append(part.split("证据资料：", 1)[1].strip())
            else:
                evidence_parts.append(part.strip())
        context = "\n".join(evidence_parts)
        if context:
            return f"根据企业文档资料，关于“{user}”的答案是：{self._shorten(context)}"
        return f"我目前没有在企业知识库中找到与“{user}”直接相关的证据，建议补充文档或换一种提问方式。"

    @staticmethod
    def _shorten(text: str, limit: int = 240) -> str:
        text = text.strip()
        return text[:limit] + ("..." if len(text) > limit else "")


def get_llm(settings: Settings) -> LLMClient:
    provider = settings.llm_provider.lower()
    if provider == "mock":
        return MockLLM()
    if provider == "deepseek":
        if settings.api_key:
            return DeepSeekChatLLM(settings)
        return MockLLM()
    if provider in {"openai", "openai_compatible", "ollama", "vllm", "azure"}:
        if settings.api_key or settings.model_base_url:
            return OpenAIChatLLM(settings)
        return MockLLM()
    if settings.api_key or settings.model_base_url:
        return OpenAIChatLLM(settings)
    return MockLLM()
