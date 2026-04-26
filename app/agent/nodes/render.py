"""Rendering node for LangGraph."""

import structlog

from app.agent.state import AgentState
from app.core.exceptions import RenderError
from app.renderer.docs_tree import build_doc_requests
from app.renderer.email_renderer import render_email
from app.services.run_progress import persist_run_stage

logger = structlog.get_logger()


async def render_node(state: AgentState) -> AgentState:
    """Render Doc requests and email bodies."""
    summary = state.get("summary")
    if not summary:
        raise RenderError("No summary to render")

    await persist_run_stage(state["run_id"], "rendering")
    logger.info("Rendering started", run_id=state["run_id"])

    anchor = f"pulse-{state['product_key']}-{state['iso_week']}"
    doc_requests = build_doc_requests(summary, anchor)

    # Placeholder deep link - will be replaced after Docs publish
    html, text = render_email(
        summary,
        doc_deep_link="{DOC_DEEP_LINK}",
        run_id=state["run_id"],
    )

    logger.info("Rendering complete", run_id=state["run_id"])

    return {
        "doc_requests": doc_requests,
        "email_html": html,
        "email_text": text,
        "status": "rendering",
    }
