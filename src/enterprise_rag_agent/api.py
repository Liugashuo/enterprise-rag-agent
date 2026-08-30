from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .config import Settings
from .service import RAGAgentService


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None


class IngestRequest(BaseModel):
    path: str


class CreateAppResult:
    def __init__(self, app: FastAPI, service: RAGAgentService):
        self.app = app
        self.service = service


def create_app(settings: Settings | None = None) -> CreateAppResult:
    service = RAGAgentService(settings)
    app = FastAPI(title=service.settings.app_name, version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return service.health()

    @app.post("/v1/chat")
    def chat(request: ChatRequest) -> dict[str, Any]:
        try:
            return service.chat(message=request.message, session_id=request.session_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/v1/documents")
    async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
        try:
            data = await file.read()
            return service.ingest_bytes(file.filename or "upload", data)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/documents/ingest")
    def ingest_document(request: IngestRequest) -> dict[str, Any]:
        path = Path(request.path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        try:
            return service.ingest_file(path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/documents")
    def list_documents() -> list[dict[str, Any]]:
        return service.list_documents()

    @app.delete("/v1/documents/{document_id}")
    def delete_document(document_id: str) -> dict[str, Any]:
        deleted = service.delete_document(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="文档不存在")
        return {"deleted": True, "document_id": document_id}

    @app.get("/v1/sessions/{session_id}/messages")
    def get_messages(session_id: str) -> list[dict[str, Any]]:
        return service.get_history(session_id)

    @app.delete("/v1/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, Any]:
        deleted = service.delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"deleted": True, "session_id": session_id}

    return CreateAppResult(app=app, service=service)


result = create_app()
app = result.app
service = result.service

