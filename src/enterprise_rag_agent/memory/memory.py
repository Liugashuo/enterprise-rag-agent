from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Settings
from ..models import ChatMessage


def estimate_tokens(text: str) -> int:
    """轻量 Token 估算：中文约 1 字 1 token，英文约 4 字符 1 token。"""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = max(len(text) - cjk, 0)
    return cjk + max(1, other // 4)


@dataclass
class Session:
    session_id: str
    messages: list[ChatMessage] = field(default_factory=list)
    summary: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SessionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                for item in raw.get("sessions", []):
                    messages = [ChatMessage(**m) for m in item.get("messages", [])]
                    session = Session(
                        session_id=item["session_id"],
                        messages=messages,
                        summary=item.get("summary", ""),
                        created_at=item.get("created_at", ""),
                        updated_at=item.get("updated_at", ""),
                    )
                    self._sessions[session.session_id] = session
            except Exception:
                self._sessions = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "sessions": [
                {
                    "session_id": s.session_id,
                    "messages": [asdict(m) for m in s.messages],
                    "summary": s.summary,
                    "created_at": s.created_at,
                    "updated_at": s.updated_at,
                }
                for s in self._sessions.values()
            ]
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def get_or_create(self, session_id: str) -> Session:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(session_id=session_id)
                self._save()
            return self._sessions[session_id]

    def add_message(self, session_id: str, message: ChatMessage) -> None:
        with self._lock:
            session = self.get_or_create(session_id)
            session.messages.append(message)
            session.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()

    def get_messages(self, session_id: str) -> list[ChatMessage]:
        return list(self.get_or_create(session_id).messages)

    def update_summary(self, session_id: str, summary: str) -> None:
        with self._lock:
            session = self.get_or_create(session_id)
            session.summary = summary
            session.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()

    def get_summary(self, session_id: str) -> str:
        return self.get_or_create(session_id).summary

    def delete(self, session_id: str) -> bool:
        with self._lock:
            existed = session_id in self._sessions
            self._sessions.pop(session_id, None)
            if existed:
                self._save()
            return existed

    def search(self, session_id: str, query: str, limit: int = 5) -> list[ChatMessage]:
        messages = self.get_messages(session_id)
        terms = [t for t in query.lower().split() if t]
        scored: list[tuple[int, ChatMessage]] = []
        for message in messages:
            content = message.content.lower()
            score = sum(1 for term in terms if term in content)
            if score:
                scored.append((score, message))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [message for _, message in scored[:limit]]


@dataclass
class SessionContext:
    messages: list[dict[str, Any]]
    summary: str
    system_prompt: str


class MemoryManager:
    def __init__(
        self,
        store: SessionStore,
        settings: Settings,
        llm,
    ):
        self.store = store
        self.settings = settings
        self.llm = llm
        self.system_prompt = (
            "你是一名严谨的企业文档智能助手。回答必须基于提供的证据；"
            "证据不足时明确说明不确定，不要编造。"
        )

    def prepare(self, session_id: str, query: str) -> SessionContext:
        messages = self.store.get_messages(session_id)
        summary = self.store.get_summary(session_id)
        if self.settings.summary_enabled and len(messages) > self.settings.max_history_messages:
            summary = self._summarize(messages[: -self.settings.max_history_messages], summary)
            self.store.update_summary(session_id, summary)
        recent = messages[-self.settings.max_history_messages :]
        context_messages: list[dict[str, Any]] = []
        if summary:
            context_messages.append({"role": "system", "content": f"历史会话摘要：{summary}"})
        context_messages.extend(m.to_dict() for m in recent)
        context_messages.append({"role": "user", "content": query})
        context_messages = self._compress(context_messages)
        return SessionContext(
            messages=context_messages,
            summary=summary,
            system_prompt=self.system_prompt,
        )

    def _summarize(self, old_messages: list[ChatMessage], current: str) -> str:
        if not old_messages:
            return current
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "[SUMMARIZE] 请压缩历史对话，保留用户目标、关键事实和未完成任务。",
            },
            {
                "role": "user",
                "content": "\n".join(f"{m.role}: {m.content}" for m in old_messages),
            },
        ]
        if current:
            messages.append({"role": "assistant", "content": current})
        try:
            result = self.llm.complete(messages, temperature=0.0).strip()
            return result or current
        except Exception:
            return current or "；".join(m.content for m in old_messages[-2:])

    def _compress(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        budget = self.settings.context_max_tokens
        result: list[dict[str, Any]] = []
        used = estimate_tokens(self.system_prompt)
        for message in reversed(messages):
            tokens = estimate_tokens(str(message.get("content", "")))
            if used + tokens > budget and result:
                break
            result.append(message)
            used += tokens
        result.reverse()
        return result
