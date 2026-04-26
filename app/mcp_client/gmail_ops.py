"""Gmail MCP operations with idempotency."""

import structlog

from app.config import get_settings
from app.core.exceptions import MCPError
from app.mcp_client.session import MCPConnectionManager

logger = structlog.get_logger()


class GmailPublisher:
    """Publish pulse emails via Gmail MCP with idempotent sends."""

    def __init__(self, mcp: MCPConnectionManager) -> None:
        self.mcp = mcp

    async def check_existing(self, run_id: str) -> str | None:
        """Check if an email was already sent for this run.

        Returns:
            Existing message ID, or None if not found
        """
        try:
            result = await self.mcp.gmail.call_tool(
                "search_messages",
                {"query": f"from:me X-Pulse-Run-Id:{run_id}"},
            )
            messages = result.get("messages", [])
            if messages:
                msg_id = str(messages[0]["id"])
                logger.info("Found existing email", run_id=run_id, message_id=msg_id)
                return msg_id
        except MCPError as e:
            logger.warning("Failed to search Gmail", run_id=run_id, error=str(e))
        return None

    async def create_draft(
        self,
        run_id: str,
        to: str,
        subject: str,
        html_body: str,
        text_body: str,
        product: str,
    ) -> str:
        """Create a draft email.

        Returns:
            Draft ID
        """
        result = await self.mcp.gmail.call_tool(
            "create_draft",
            {
                "message": {
                    "headers": [
                        {"name": "X-Pulse-Run-Id", "value": run_id},
                    ],
                    "labelIds": [f"Pulse/{product}"],
                    "to": to,
                    "subject": subject,
                    "body": {
                        "html": html_body,
                        "text": text_body,
                    },
                }
            },
        )
        draft_id = str(result["id"])
        logger.info("Created draft", run_id=run_id, draft_id=draft_id)
        return draft_id

    async def send_draft(self, draft_id: str) -> str:
        """Send a draft email.

        Returns:
            Sent message ID
        """
        result = await self.mcp.gmail.call_tool(
            "send_message",
            {"draftId": draft_id},
        )
        message_id = str(result["id"])
        logger.info("Sent email", draft_id=draft_id, message_id=message_id)
        return message_id

    async def publish(
        self,
        run_id: str,
        to: str,
        subject: str,
        html_body: str,
        text_body: str,
        product: str,
    ) -> str | None:
        """Idempotently send a pulse email.

        Returns:
            Message ID if sent, None if draft-only or already sent
        """
        settings = get_settings()

        # Check for existing send
        existing = await self.check_existing(run_id)
        if existing:
            return existing

        # Create draft
        draft_id = await self.create_draft(
            run_id=run_id,
            to=to,
            subject=subject,
            html_body=html_body,
            text_body=text_body,
            product=product,
        )

        # Send if confirmed
        if settings.confirm_send:
            message_id = await self.send_draft(draft_id)
            return message_id

        logger.info("Draft created, confirm_send=false", run_id=run_id, draft_id=draft_id)
        return None
