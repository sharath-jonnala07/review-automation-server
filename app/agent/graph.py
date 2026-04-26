"""LangGraph state machine for the Pulse Agent."""

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agent.nodes.cluster import cluster_node
from app.agent.nodes.ingest import ingest_node
from app.agent.nodes.publish import publish_docs_node, publish_gmail_node
from app.agent.nodes.render import render_node
from app.agent.nodes.summarize import summarize_node
from app.agent.state import AgentState
from app.config import get_settings

logger = structlog.get_logger()


def _should_cluster(state: AgentState) -> str:
    """Route to cluster or end on insufficient reviews."""
    reviews = state.get("reviews", [])
    min_reviews = get_settings().min_reviews_per_run
    if len(reviews) >= min_reviews:
        return "cluster"
    logger.warning("Insufficient reviews", count=len(reviews), minimum=min_reviews)
    return END


def _should_summarize(state: AgentState) -> str:
    """Route to summarize whenever clustering produced at least one cluster."""
    clusters = state.get("clusters", [])
    if clusters:
        return "summarize"
    logger.warning("Insufficient clusters", count=len(clusters))
    return END


def _should_render(state: AgentState) -> str:
    """Route to render if summary exists."""
    if state.get("summary"):
        return "render"
    return END


def _should_publish_gmail(state: AgentState) -> str:
    """Route to Gmail for publish runs even if Docs publication failed."""
    if not state.get("dry_run") and state.get("email_html"):
        return "publish_gmail"
    return END


async def build_graph() -> object:
    """Build and compile the Pulse Agent LangGraph.

    Returns:
        Compiled StateGraph with checkpointing
    """
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("ingest", ingest_node)
    builder.add_node("cluster", cluster_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("render", render_node)
    builder.add_node("publish_docs", publish_docs_node)
    builder.add_node("publish_gmail", publish_gmail_node)

    # Entry point
    builder.set_entry_point("ingest")

    # Conditional edges
    builder.add_conditional_edges(
        "ingest",
        _should_cluster,
        {"cluster": "cluster", END: END},
    )

    builder.add_conditional_edges(
        "cluster",
        _should_summarize,
        {"summarize": "summarize", END: END},
    )

    builder.add_conditional_edges(
        "summarize",
        _should_render,
        {"render": "render", END: END},
    )

    builder.add_edge("render", "publish_docs")

    builder.add_conditional_edges(
        "publish_docs",
        _should_publish_gmail,
        {"publish_gmail": "publish_gmail", END: END},
    )

    builder.add_edge("publish_gmail", END)

    # Compile with in-memory checkpointing
    # In production, swap for SqliteSaver or PostgresSaver
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("Pulse Agent graph compiled with checkpointing")
    return graph
