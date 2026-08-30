from __future__ import annotations

import json
from typing import Any

from ..memory import SessionStore


class HistorySearchTool:
    name = "search_history"
    description = "查询当前会话的历史消息，用于回答“你刚才说了什么”“之前讨论过什么”等问题。"
    parameters = {
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "会话 ID"},
            "query": {"type": "string", "description": "要检索的历史关键词或问题"},
            "limit": {"type": "integer", "description": "返回条数", "default": 5},
        },
        "required": ["session_id", "query"],
    }

    def __init__(self, store: SessionStore):
        self.store = store

    def run(self, session_id: str, query: str, limit: int = 5) -> str:
        messages = self.store.search(session_id, query, limit=limit)
        messages = [
            message
            for message in messages
            if not (message.role == "user" and message.content.strip() == query.strip())
        ]
        if not messages:
            all_messages = self.store.get_messages(session_id)
            candidates = all_messages
            if (
                all_messages
                and all_messages[-1].role == "user"
                and all_messages[-1].content.strip() == query.strip()
            ):
                candidates = all_messages[:-1]
            messages = candidates[-limit:]
        payload = {
            "results": [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                for m in messages
            ]
        }
        return json.dumps(payload, ensure_ascii=False)


def create_history_tool(store: SessionStore) -> Any:
    tool = HistorySearchTool(store)
    from .registry import Tool

    return Tool(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        function=tool.run,
    )
