"""Ingestion node for LangGraph."""


import structlog

from app.agent.state import AgentState
from app.config import load_products_config
from app.core.exceptions import IngestionError
from app.core.types import ProductKey
from app.ingestion.appstore import fetch_appstore_reviews
from app.ingestion.playstore import fetch_playstore_reviews

logger = structlog.get_logger()


async def ingest_node(state: AgentState) -> AgentState:
    """Fetch reviews from App Store and Play Store."""
    product = state["product_key"]
    weeks = 10  # Configurable via state

    logger.info("Ingestion started", product=product, run_id=state["run_id"])

    existing_reviews = state.get("reviews", [])
    if state.get("skip_ingest") and existing_reviews:
        logger.info(
            "Using preloaded reviews",
            product=product,
            run_id=state["run_id"],
            count=len(existing_reviews),
        )
        return {
            "reviews": existing_reviews,
            "status": "ingesting",
        }

    products = load_products_config()
    config = next((p for p in products if p.key == product), None)
    if not config:
        raise IngestionError(f"Product {product} not found")

    all_reviews = []

    if config.appstore_id:
        reviews = await fetch_appstore_reviews(
            ProductKey(product), config.appstore_id, weeks=weeks
        )
        all_reviews.extend(reviews)

    if config.play_package:
        reviews = await fetch_playstore_reviews(
            ProductKey(product), config.play_package, weeks=weeks
        )
        all_reviews.extend(reviews)

    logger.info("Ingestion complete", product=product, count=len(all_reviews))

    return {
        "reviews": all_reviews,
        "status": "ingesting",
    }
