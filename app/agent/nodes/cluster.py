"""Clustering node for LangGraph."""

import asyncio

import structlog

from app.agent.state import AgentState
from app.clustering.clusterer import ReviewClusterer
from app.clustering.embeddings import get_embedding_provider
from app.core.exceptions import ClusteringError
from app.services.run_progress import persist_run_stage

logger = structlog.get_logger()


async def cluster_node(state: AgentState) -> AgentState:
    """Embed and cluster reviews."""
    reviews = state.get("reviews", [])
    if len(reviews) < 8:
        raise ClusteringError(f"Not enough reviews: {len(reviews)}")

    await persist_run_stage(state["run_id"], "clustering")
    logger.info("Clustering started", run_id=state["run_id"], count=len(reviews))

    # Get embeddings
    provider = get_embedding_provider()
    texts = [r.body for r in reviews]
    embeddings = await provider.embed(texts)

    # Cluster
    clusterer = ReviewClusterer()
    clusters, _labels = await asyncio.to_thread(
        clusterer.cluster_reviews,
        embeddings,
        texts,
        3,
    )

    logger.info(
        "Clustering complete",
        run_id=state["run_id"],
        clusters=len(clusters),
    )

    return {
        "embeddings": embeddings,
        "clusters": clusters,
        "status": "clustering",
    }
