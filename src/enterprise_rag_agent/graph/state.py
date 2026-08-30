from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    session_id: str
    query: str
    intent: str
    plan: list[str]
    rewritten_query: str
    history: list[dict[str, Any]]
    summary: str
    system_prompt: str
    search_results: list[dict[str, Any]]
    history_results: str
    draft_answer: str
    tool_calls: list[dict[str, Any]]
    iterations: int
    confidence: float
    low_confidence: bool
    final_answer: str
    error: str

