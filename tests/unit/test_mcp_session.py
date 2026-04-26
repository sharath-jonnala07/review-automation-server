import httpx
import pytest

from app.core.exceptions import MCPError
from app.mcp_client.session import MCPSession


@pytest.mark.anyio
async def test_call_tool_includes_server_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "Google API error 403: permission denied"})

    transport = httpx.MockTransport(handler)
    session = MCPSession("http://example.test")

    async with httpx.AsyncClient(transport=transport) as client:
        session._client = client
        with pytest.raises(MCPError) as exc:
            await session.call_tool("create_document", {"title": "x"})

    assert "permission denied" in str(exc.value)