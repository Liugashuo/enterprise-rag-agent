from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..models import Chunk
from .loaders import ParsedDocument


_PARA_RE = re.compile(r"\n\s*\n|\r\n\s*\r\n")
_SENT_RE = re.compile(r"(?<=[。！？!?；;])\s*")


@dataclass
class RecursiveTextChunker:
    chunk_size: int = 800
    chunk_overlap: int = 120

    def split_document(self, document: ParsedDocument, document_id: str) -> list[Chunk]:
        pieces = self._split_text(document.text)
        chunks: list[Chunk] = []
        for index, piece in enumerate(pieces):
            chunk_id = hashlib.sha256(
                f"{document_id}:{index}:{piece}".encode("utf-8")
            ).hexdigest()
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    content=piece,
                    metadata={
                        **document.metadata,
                        "filename": document.filename,
                        "chunk_index": index,
                    },
                    index=index,
                )
            )
        return chunks

    def _split_text(self, text: str) -> list[str]:
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            return []
        paragraphs = [p.strip() for p in _PARA_RE.split(text) if p.strip()]
        sentences: list[str] = []
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                sentences.append(para)
                continue
            parts = [s.strip() for s in _SENT_RE.split(para) if s.strip()]
            if not parts:
                parts = [para]
            sentences.extend(parts)

        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            if len(sentence) > self.chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._split_long(sentence))
                continue
            if not current:
                current = sentence
            elif len(current) + len(sentence) + 1 <= self.chunk_size:
                current = f"{current}\n{sentence}"
            else:
                chunks.append(current)
                overlap = current[-self.chunk_overlap :] if self.chunk_overlap > 0 else ""
                current = f"{overlap}\n{sentence}" if overlap else sentence
        if current.strip():
            chunks.append(current)
        return [c.strip() for c in chunks if c.strip()]

    def _split_long(self, text: str) -> list[str]:
        step = max(self.chunk_size - self.chunk_overlap, 1)
        return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]
