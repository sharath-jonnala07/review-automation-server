"""Google Docs MCP-compatible server."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import structlog

logger = structlog.get_logger()
app = FastAPI(title="Pulse Docs MCP Server")

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

_DOCS_SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]


class MCPToolRequest(BaseModel):
    """MCP HTTP wrapper request."""

    arguments: dict[str, Any] = Field(default_factory=dict)


def _format_google_error(exc: Exception) -> str:
    """Return a readable Google API error instead of an empty 500 detail."""
    status = getattr(getattr(exc, "resp", None), "status", None)
    content = getattr(exc, "content", None)
    detail = str(exc).strip()

    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")

    if not detail and content:
        detail = str(content).strip()

    if status and detail:
        return f"Google API error {status}: {detail}"
    if status:
        return f"Google API error {status}"
    return detail or exc.__class__.__name__


def _load_service_account_info() -> dict[str, Any]:
    raw_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw_json:
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON") from exc

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        path = Path(credentials_path)
        if path.is_dir():
            json_files = sorted(path.glob("*.json"))
            if len(json_files) == 1:
                path = json_files[0]
    else:
        path = Path(__file__).resolve().parents[1] / "service-account.json"

    if not path.exists() and path.suffix != ".json":
        json_sibling = path.with_suffix(".json")
        if json_sibling.exists():
            path = json_sibling

    if not path.exists():
        raise RuntimeError(
            "Google service account credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS "
            "or GOOGLE_SERVICE_ACCOUNT_JSON."
        )

    return json.loads(path.read_text(encoding="utf-8"))


def _build_clients() -> tuple[Any, Any]:
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Google API dependencies are missing. Install google-api-python-client and google-auth."
        ) from exc

    credentials = Credentials.from_service_account_info(
        _load_service_account_info(),
        scopes=_DOCS_SCOPES,
    )
    docs_client = build("docs", "v1", credentials=credentials, cache_discovery=False)
    drive_client = build("drive", "v3", credentials=credentials, cache_discovery=False)
    return docs_client, drive_client


def _get_docs_client() -> Any:
    return app.state.docs_client


def _get_drive_client() -> Any:
    return app.state.drive_client


@app.on_event("startup")
def startup() -> None:
    docs_client, drive_client = _build_clients()
    app.state.docs_client = docs_client
    app.state.drive_client = drive_client
    logger.info("Docs MCP server ready")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tools/search_documents")
async def search_documents(payload: MCPToolRequest) -> dict[str, Any]:
    query = str(payload.arguments.get("query", "")).strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    escaped_query = query.replace("'", "\\'")

    try:
        response = (
            _get_drive_client()
            .files()
            .list(
                q=(
                    "mimeType='application/vnd.google-apps.document' and trashed=false "
                    f"and name contains '{escaped_query}'"
                ),
                spaces="drive",
                fields="files(id,name,createdTime,webViewLink)",
                pageSize=10,
            )
            .execute()
        )
        return {"documents": response.get("files", [])}
    except Exception as exc:
        logger.exception("search_documents failed")
        raise HTTPException(status_code=500, detail=_format_google_error(exc)) from exc


@app.post("/tools/create_document")
async def create_document(payload: MCPToolRequest) -> dict[str, Any]:
    title = str(payload.arguments.get("title", "")).strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")

    try:
        result = _get_docs_client().documents().create(body={"title": title}).execute()
        return {"documentId": result["documentId"]}
    except Exception as exc:
        logger.exception("create_document failed")
        raise HTTPException(status_code=500, detail=_format_google_error(exc)) from exc


@app.post("/tools/get_document")
async def get_document(payload: MCPToolRequest) -> dict[str, Any]:
    document_id = str(payload.arguments.get("documentId", "")).strip()
    if not document_id:
        raise HTTPException(status_code=400, detail="documentId is required")

    try:
        return _get_docs_client().documents().get(documentId=document_id).execute()
    except Exception as exc:
        logger.exception("get_document failed")
        raise HTTPException(status_code=500, detail=_format_google_error(exc)) from exc


@app.post("/tools/batch_update")
async def batch_update(payload: MCPToolRequest) -> dict[str, Any]:
    document_id = str(payload.arguments.get("documentId", "")).strip()
    requests = payload.arguments.get("requests")
    if not document_id:
        raise HTTPException(status_code=400, detail="documentId is required")
    if not isinstance(requests, list):
        raise HTTPException(status_code=400, detail="requests must be a list")

    try:
        return (
            _get_docs_client()
            .documents()
            .batchUpdate(documentId=document_id, body={"requests": requests})
            .execute()
        )
    except Exception as exc:
        logger.exception("batch_update failed")
        raise HTTPException(status_code=500, detail=_format_google_error(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mcp_servers.docs_server:app", host="0.0.0.0", port=5000, reload=False)
