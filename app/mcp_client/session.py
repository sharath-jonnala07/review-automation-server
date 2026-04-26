"""MCP session management for Google Workspace servers."""

from typing import Any

import httpx
import structlog

from app.config import get_settings
from app.core.exceptions import MCPError

logger = structlog.get_logger()


class MCPSession:
    """HTTP-based MCP client session.

    This is a simplified MCP client for SSE/HTTP transport.
    In production, this would use the official MCP SDK.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = base_url or ""
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "MCPSession":
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client:
            await self._client.aclose()

    async def call_tool(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool via HTTP POST.

        Args:
            tool: Tool name
            arguments: Tool arguments

        Returns:
            Tool result

        Raises:
            MCPError: On connection or response error
        """
        if not self._client:
            raise MCPError("Session not started. Use async context manager.")

        url = f"{self.base_url}/tools/{tool}"
        payload = {"arguments": arguments}

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                payload = e.response.json()
                detail = str(payload.get("detail") or "").strip()
            except ValueError:
                detail = e.response.text.strip()

            suffix = f" Detail: {detail}" if detail else ""
            raise MCPError(
                f"MCP tool call failed: {e}{suffix}",
                context={"tool": tool, "url": url, "status_code": e.response.status_code},
            ) from e
        except httpx.HTTPError as e:
            raise MCPError(
                f"MCP tool call failed: {e}",
                context={"tool": tool, "url": url},
            ) from e


class MCPConnectionManager:
    """Manages connections to multiple MCP servers."""

    def __init__(self) -> None:
        settings = get_settings()
        self.docs_url = settings.docs_mcp_url
        self.gmail_url = settings.gmail_mcp_url
        self._docs: MCPSession | None = None
        self._gmail: MCPSession | None = None

    async def __aenter__(self) -> "MCPConnectionManager":
        if self.docs_url:
            self._docs = MCPSession(self.docs_url)
            await self._docs.__aenter__()
            logger.info("Connected to Docs MCP", url=self.docs_url)
        if self.gmail_url:
            self._gmail = MCPSession(self.gmail_url)
            await self._gmail.__aenter__()
            logger.info("Connected to Gmail MCP", url=self.gmail_url)
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._docs:
            await self._docs.__aexit__(*args)
        if self._gmail:
            await self._gmail.__aexit__(*args)

    @property
    def docs(self) -> MCPSession:
        if not self._docs:
            raise MCPError("Docs MCP not configured")
        return self._docs

    @property
    def gmail(self) -> MCPSession:
        if not self._gmail:
            raise MCPError("Gmail MCP not configured")
        return self._gmail
