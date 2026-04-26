"""Summarization node for LangGraph."""

from datetime import date

import structlog

from app.agent.state import AgentState
from app.core.exceptions import SummarizationError
from app.core.models import Window
from app.core.types import ProductKey
from app.summarization.engine import SummarizationEngine

logger = structlog.get_logger()


async def summarize_node(state: AgentState) -> AgentState:
    """Generate themes, quotes, and action ideas."""
    reviews = state.get("reviews", [])
    clusters = state.get("clusters", [])

    if not clusters:
        raise SummarizationError("No clusters to summarize")

    logger.info("Summarization started", run_id=state["run_id"])

    engine = SummarizationEngine()
    window = Window(
        start=state.get("window_start", date.today()),
        end=state.get("window_end", date.today()),
        weeks=10,
    )

    summary = await engine.summarize(
        product=ProductKey(state["product_key"]),
        window=window,
        reviews=reviews,
        clusters=clusters,
    )

    logger.info(
        "Summarization complete",
        run_id=state["run_id"],
        themes=len(summary.top_themes),
    )

    return {
        "summary": summary,
        "status": "summarizing",
    }
