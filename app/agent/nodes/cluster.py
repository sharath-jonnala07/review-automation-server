"""Clustering node for LangGraph."""

import structlog

from app.agent.state import AgentState
from app.clustering.clusterer import ReviewClusterer
from app.clustering.embeddings import get_embedding_provider
from app.core.exceptions import ClusteringError

logger = structlog.get_logger()


async def cluster_node(state: AgentState) -> AgentState:
    """Embed and cluster reviews."""
    reviews = state.get("reviews", [])
    if len(reviews) < 8:
        raise ClusteringError(f"Not enough reviews: {len(reviews)}")

    logger.info("Clustering started", run_id=state["run_id"], count=len(reviews))

    # Get embeddings
    provider = get_embedding_provider()
    texts = [r.body for r in reviews]
    embeddings = await provider.embed(texts)

    # Cluster
    clusterer = ReviewClusterer()
    clusters, _labels = clusterer.cluster_reviews(
        embeddings,
        texts,
        target_clusters=3,
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
