"""Publishing node for LangGraph."""

import structlog

from app.agent.state import AgentState
from app.config import load_products_config
from app.mcp_client.docs_ops import DocsPublisher, build_deep_link
from app.mcp_client.gmail_ops import GmailPublisher
from app.mcp_client.session import MCPConnectionManager
from app.renderer.email_renderer import render_email
from app.services.run_progress import persist_run_stage

logger = structlog.get_logger()


async def publish_docs_node(state: AgentState) -> AgentState:
    """Publish to Google Docs."""
    await persist_run_stage(state["run_id"], "publishing")

    if state.get("dry_run"):
        logger.info("Skipping Docs publish for dry run", run_id=state["run_id"])
        return {
            "gdoc_id": None,
            "gdoc_heading_id": None,
            "status": "completed",
        }

    async with MCPConnectionManager() as mcp:
        publisher = DocsPublisher(mcp)
        anchor = f"pulse-{state['product_key']}-{state['iso_week']}"

        try:
            doc_id, heading_id = await publisher.publish(
                product=state["product_key"],
                doc_requests=state["doc_requests"],
                anchor=anchor,
                doc_id=state.get("configured_gdoc_id"),
            )
        except Exception as exc:
            logger.warning(
                "Docs publish failed; continuing to Gmail",
                run_id=state["run_id"],
                error=str(exc),
            )
            return {
                "gdoc_id": None,
                "gdoc_heading_id": None,
                "doc_publish_error": str(exc),
                "status": "publishing",
            }

        logger.info(
            "Published to Docs",
            run_id=state["run_id"],
            doc_id=doc_id,
            heading_id=heading_id,
        )

        return {
            "gdoc_id": doc_id,
            "gdoc_heading_id": heading_id,
            "doc_publish_error": None,
            "status": "publishing",
        }


async def publish_gmail_node(state: AgentState) -> AgentState:
    """Publish to Gmail."""
    await persist_run_stage(state["run_id"], "publishing")

    async with MCPConnectionManager() as mcp:
        publisher = GmailPublisher(mcp)

        products = load_products_config()
        config = next(
            (p for p in products if p.key == state["product_key"]), None
        )
        to = state.get("gmail_to") or (config.gmail_to or "" if config else "")

        deep_link = ""
        if state.get("gdoc_id") and state.get("gdoc_heading_id"):
            deep_link = build_deep_link(
                state.get("gdoc_id") or "",
                state.get("gdoc_heading_id") or "",
            )

        if state.get("summary"):
            html, text = render_email(
                state["summary"],
                doc_deep_link=deep_link,
                run_id=state["run_id"],
            )
        else:
            html = state["email_html"]
            text = state["email_text"]

        message_id = await publisher.publish(
            run_id=state["run_id"],
            to=to,
            subject=f"[Weekly Pulse] {state['product_key']} - {state['iso_week']}",
            html_body=html,
            text_body=text,
            product=state["product_key"],
        )

        logger.info(
            "Published to Gmail",
            run_id=state["run_id"],
            message_id=message_id,
        )

        return {
            "gmail_message_id": message_id,
            "status": "completed",
        }
