from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ..llm import ToolCall


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., str]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        return self.function(**arguments)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def to_openai(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        selected = names or list(self._tools)
        return [self._tools[name].to_openai() for name in selected if name in self._tools]

    def execute(self, call: ToolCall) -> str:
        tool = self.get(call.name)
        if tool is None:
            raise ValueError(f"未注册的工具: {call.name}")
        return tool.execute(call.arguments)

