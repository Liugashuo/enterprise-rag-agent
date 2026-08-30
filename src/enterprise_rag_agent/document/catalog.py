from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable

from ..models import DocumentRecord


class DocumentCatalog:
    """文档目录，用于增量更新去重和文档级删除。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._records: dict[str, DocumentRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._records = {
                    item["document_id"]: DocumentRecord(**item)
                    for item in raw.get("documents", [])
                }
            except Exception:
                self._records = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        data = {"documents": [record.to_dict() for record in self._records.values()]}
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, document_id: str) -> DocumentRecord | None:
        return self._records.get(document_id)

    def list(self) -> list[DocumentRecord]:
        return list(self._records.values())

    def upsert(self, record: DocumentRecord) -> None:
        with self._lock:
            self._records[record.document_id] = record
            self._save()

    def delete(self, document_id: str) -> bool:
        with self._lock:
            existed = document_id in self._records
            self._records.pop(document_id, None)
            self._save()
            return existed

    def all_ids(self) -> set[str]:
        return set(self._records)

