from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    app_name: str = "Enterprise RAG Agent"
    llm_provider: str = "deepseek"
    model_name: str = "deepseek-chat"
    model_base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    embedding_provider: str = "hash"
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str = ""
    embedding_dimension: int = 256
    vector_store_type: str = "json"
    vector_store_path: str = "data/vector_store"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 6
    hybrid_alpha: float = 0.5
    rerank_enabled: bool = True
    reranker_model: str = "keyword"
    max_agent_steps: int = 4
    context_max_tokens: int = 4000
    max_history_messages: int = 12
    summary_enabled: bool = True
    evidence_min_score: float = 0.01
    host: str = "0.0.0.0"
    port: int = 8000
    data_dir: str = "data"

    @classmethod
    def from_env(cls) -> "Settings":
        values: dict[str, Any] = {}
        for field in fields(cls):
            env_name = field.name.upper()
            if field.name == "api_key":
                value = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
            else:
                value = os.getenv(env_name)
            if value is None:
                continue
            if field.type is bool or field.type == "bool":
                values[field.name] = _as_bool(value)
            elif field.type is int or field.type == "int":
                values[field.name] = _as_int(value, getattr(cls, field.name))
            elif field.type is float or field.type == "float":
                values[field.name] = _as_float(value, getattr(cls, field.name))
            else:
                values[field.name] = value
        return cls(**values)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Settings":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        settings = cls.from_env()
        candidates = [Path(path)] if path else [Path("config.yaml"), Path("config.yml")]
        for candidate in candidates:
            if candidate and candidate.exists():
                try:
                    import yaml

                    data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
                    file_settings = cls.from_mapping(data)
                    env_data: dict[str, Any] = {}
                    for f in fields(cls):
                        if f.name == "api_key":
                            has_value = (
                                os.getenv("DEEPSEEK_API_KEY") is not None
                                or os.getenv("OPENAI_API_KEY") is not None
                                or os.getenv("API_KEY") is not None
                            )
                        else:
                            has_value = os.getenv(f.name.upper()) is not None
                        if has_value:
                            env_data[f.name] = getattr(settings, f.name)
                    merged = {**file_settings.__dict__, **env_data}
                    settings = cls(**merged)
                except Exception:
                    # 配置解析失败时保留环境变量配置，避免服务无法启动。
                    pass
                break
        settings._ensure_dirs()
        return settings

    def _ensure_dirs(self) -> None:
        root = Path(self.vector_store_path).parent
        root.mkdir(parents=True, exist_ok=True)
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
