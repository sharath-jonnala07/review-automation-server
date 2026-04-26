"""Google Docs MCP operations with idempotency."""

from typing import Any

import structlog

from app.core.exceptions import MCPError
from app.mcp_client.session import MCPConnectionManager

logger = structlog.get_logger()


class DocsPublisher:
    """Publish pulse reports to Google Docs with idempotent appends."""

    def __init__(self, mcp: MCPConnectionManager) -> None:
        self.mcp = mcp

    async def resolve_document(self, product: str, title: str | None = None) -> str:
        """Find or create the per-product Google Doc.

        Returns:
            Google Doc ID
        """
        doc_title = title or f"Weekly Review Pulse - {product}"

        # Try to find existing doc
        try:
            result = await self.mcp.docs.call_tool(
                "search_documents",
                {"query": doc_title},
            )
            documents = result.get("documents", [])
            if documents:
                doc_id = str(documents[0]["id"])
                logger.info("Found existing Doc", title=doc_title, doc_id=doc_id)
                return doc_id
        except MCPError:
            logger.debug("No existing Doc found, creating new one", title=doc_title)

        # Create new doc
        result = await self.mcp.docs.call_tool(
            "create_document",
            {"title": doc_title},
        )
        doc_id = str(result["documentId"])
        logger.info("Created new Doc", title=doc_title, doc_id=doc_id)
        return doc_id

    async def check_anchor(self, doc_id: str, anchor: str) -> bool:
        """Check if an anchor string already exists in the Doc.

        Returns:
            True if anchor found (skip append)
        """
        try:
            result = await self.mcp.docs.call_tool(
                "get_document",
                {"documentId": doc_id},
            )
            body = result.get("body", {}).get("content", [])
            text = ""
            for elem in body:
                if "paragraph" in elem:
                    for run in elem["paragraph"].get("elements", []):
                        text += run.get("textRun", {}).get("content", "")
            return anchor in text
        except MCPError as e:
            logger.warning("Failed to check anchor", doc_id=doc_id, error=str(e))
            return False

    async def append_section(
        self,
        doc_id: str,
        requests: list[dict[str, object]],
        anchor: str,
    ) -> str:
        """Append a new section to the Doc.

        Returns:
            headingId of the new section for deep-linking
        """
        # Get current end index
        doc = await self.mcp.docs.call_tool(
            "get_document",
            {"documentId": doc_id},
        )
        end_index = doc.get("body", {}).get("content", [{}])[-1].get("endIndex", 1)

        # Insert page break at end
        batch_requests = [
            {
                "insertPageBreak": {
                    "location": {"index": end_index - 1},
                }
            },
            *requests,
        ]

        await self.mcp.docs.call_tool(
            "batch_update",
            {
                "documentId": doc_id,
                "requests": batch_requests,
            },
        )

        # Re-read to get headingId
        updated = await self.mcp.docs.call_tool(
            "get_document",
            {"documentId": doc_id},
        )

        # Find heading by anchor text
        heading_id = self._extract_heading_id(updated, anchor)

        logger.info(
            "Appended section to Doc",
            doc_id=doc_id,
            anchor=anchor,
            heading_id=heading_id,
        )
        return str(heading_id)

    def _extract_heading_id(self, doc: dict[str, Any], anchor: str) -> str:
        """Extract headingId from document by searching for anchor text."""
        body = doc.get("body", {}).get("content", [])
        for elem in body:
            if "paragraph" in elem:
                style = elem["paragraph"].get("paragraphStyle", {})
                if style.get("namedStyleType", "").startswith("HEADING"):
                    text = ""
                    for run in elem["paragraph"].get("elements", []):
                        text += run.get("textRun", {}).get("content", "")
                    if anchor in text:
                        return str(
                            elem["paragraph"].get("paragraphStyle", {}).get("headingId", "")
                        )
        return ""

    async def publish(
        self,
        product: str,
        doc_requests: list[dict[str, object]],
        anchor: str,
        doc_id: str | None = None,
    ) -> tuple[str, str]:
        """Idempotently publish to Google Docs.

        Returns:
            Tuple of (doc_id, heading_id)
        """
        resolved_doc_id = doc_id or await self.resolve_document(product)

        if await self.check_anchor(resolved_doc_id, anchor):
            logger.info("Anchor already exists, skipping", anchor=anchor)
            # Still try to get headingId
            doc = await self.mcp.docs.call_tool(
                "get_document",
                {"documentId": resolved_doc_id},
            )
            heading_id = self._extract_heading_id(doc, anchor)
            return resolved_doc_id, heading_id

        heading_id = await self.append_section(resolved_doc_id, doc_requests, anchor)
        return resolved_doc_id, heading_id


def build_deep_link(doc_id: str, heading_id: str) -> str:
    """Build a Google Doc deep-link URL."""
    return f"https://docs.google.com/document/d/{doc_id}/edit#heading={heading_id}"
