"""LangChain tools wrapping MCP operations."""

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.mcp_client.docs_ops import DocsPublisher
from app.mcp_client.gmail_ops import GmailPublisher
from app.mcp_client.session import MCPConnectionManager


class PublishDocsInput(BaseModel):
    """Input for publishing to Google Docs."""

    product: str = Field(description="Product key")
    anchor: str = Field(description="Idempotency anchor string")


class PublishDocsTool(BaseTool):
    """Tool to publish a pulse report to Google Docs."""

    name: str = "publish_docs"
    description: str = "Append a pulse report section to a Google Doc"
    args_schema: type[BaseModel] = PublishDocsInput

    async def _arun(self, product: str, _anchor: str) -> str:
        async with MCPConnectionManager() as mcp:
            publisher = DocsPublisher(mcp)
            doc_id = await publisher.resolve_document(product)
            return f"Published to doc {doc_id}"

    def _run(self, product: str, anchor: str) -> str:
        raise NotImplementedError("Use async")


class SendEmailInput(BaseModel):
    """Input for sending stakeholder email."""

    run_id: str = Field(description="Run identifier")
    to: str = Field(description="Recipient email")
    subject: str = Field(description="Email subject")
    html_body: str = Field(description="HTML body")
    text_body: str = Field(description="Plain text body")
    product: str = Field(description="Product key")


class SendEmailTool(BaseTool):
    """Tool to send stakeholder email via Gmail MCP."""

    name: str = "send_email"
    description: str = "Send a stakeholder email via Gmail"
    args_schema: type[BaseModel] = SendEmailInput

    async def _arun(self, **kwargs: Any) -> str:
        async with MCPConnectionManager() as mcp:
            publisher = GmailPublisher(mcp)
            message_id = await publisher.publish(**kwargs)
            return f"Sent message {message_id}"

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")
