from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import Settings
from .document import DocumentCatalog, DocumentLoader, RecursiveTextChunker
from .embeddings import get_embedder
from .graph import AgentWorkflow
from .llm import get_llm
from .memory import MemoryManager, SessionStore
from .models import ChatMessage, DocumentRecord
from .retrieval import HybridRetriever, get_query_rewriter, get_reranker
from .tools import ToolRegistry, create_history_tool, create_knowledge_tool
from .vectorstore import get_vector_store


class RAGAgentService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.load()
        self.loader = DocumentLoader()
        self.chunker = RecursiveTextChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.embedder = get_embedder(self.settings)
        self.vector_store = get_vector_store(self.settings)
        self.catalog = DocumentCatalog(Path(self.settings.data_dir) / "documents.json")
        self.session_store = SessionStore(Path(self.settings.data_dir) / "sessions.json")
        self.llm = get_llm(self.settings)
        self.memory = MemoryManager(self.session_store, self.settings, self.llm)

        reranker = get_reranker(self.settings.rerank_enabled, self.settings.reranker_model)
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            reranker=reranker,
            top_k=self.settings.top_k,
        )
        self.query_rewriter = get_query_rewriter(
            self.llm,
            use_llm=bool(self.settings.api_key or self.settings.model_base_url),
        )
        self.tools = ToolRegistry()
        self.tools.register(create_knowledge_tool(self.retriever))
        self.tools.register(create_history_tool(self.session_store))
        self.workflow = AgentWorkflow(
            settings=self.settings,
            llm=self.llm,
            memory=self.memory,
            retriever=self.retriever,
            query_rewriter=self.query_rewriter,
            tools=self.tools,
        )

    def ingest_file(self, path: str | Path) -> dict[str, Any]:
        parsed = self.loader.load(path)
        return self._ingest(parsed.text, parsed.filename, parsed.metadata)

    def ingest_bytes(self, filename: str, data: bytes) -> dict[str, Any]:
        parsed = self.loader.load_bytes(data, filename)
        return self._ingest(parsed.text, parsed.filename, parsed.metadata)

    def _ingest(self, text: str, filename: str, metadata: dict[str, Any]) -> dict[str, Any]:
        document_id = hashlib.sha256(filename.strip().lower().encode("utf-8")).hexdigest()
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing = self.catalog.get(document_id)
        if existing and existing.content_hash == content_hash:
            return existing.to_dict()

        if existing:
            self.vector_store.delete(document_id)

        parsed = self._parsed_from_text(text, filename, metadata)
        chunks = self.chunker.split_document(parsed, document_id)
        if not chunks:
            raise ValueError(f"文档 {filename} 未解析出可用文本")
        embeddings = self.embedder.embed_documents([chunk.content for chunk in chunks])
        self.vector_store.add(chunks, embeddings)

        now = datetime.now(timezone.utc).isoformat()
        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            content_hash=content_hash,
            chunk_count=len(chunks),
            created_at=existing.created_at if existing else now,
            updated_at=now,
            metadata=metadata,
        )
        self.catalog.upsert(record)
        self.retriever.rebuild_index()
        return record.to_dict()

    @staticmethod
    def _parsed_from_text(text: str, filename: str, metadata: dict[str, Any]):
        from .document import ParsedDocument

        return ParsedDocument(text=text, filename=filename, metadata=metadata)

    def list_documents(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.catalog.list()]

    def delete_document(self, document_id: str) -> bool:
        record = self.catalog.get(document_id)
        if not record:
            return False
        self.vector_store.delete(document_id)
        self.catalog.delete(document_id)
        self.retriever.rebuild_index()
        return True

    def chat(self, message: str, session_id: str | None = None) -> dict[str, Any]:
        session_id = session_id or uuid4().hex
        self.session_store.add_message(
            session_id,
            ChatMessage(
                role="user",
                content=message,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
        )
        result = self.workflow.run(session_id, message)
        self.session_store.add_message(
            session_id,
            ChatMessage(
                role="assistant",
                content=result.answer,
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "source_count": len(result.sources),
                },
            ),
        )
        return {
            "session_id": session_id,
            "answer": result.answer,
            "intent": result.intent,
            "plan": result.plan,
            "confidence": result.confidence,
            "low_confidence": result.low_confidence,
            "sources": [source.to_dict() for source in result.sources],
        }

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self.session_store.get_messages(session_id)]

    def delete_session(self, session_id: str) -> bool:
        return self.session_store.delete(session_id)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "app": self.settings.app_name,
            "vector_store": self.settings.vector_store_type,
            "embedding_provider": self.settings.embedding_provider,
            "documents": len(self.catalog.list()),
        }
