"""LangGraph state definition for the Pulse Agent."""

from datetime import date
from typing import TypedDict

from app.clustering.clusterer import Cluster
from app.core.models import PulseSummary, RawReview


# Reducer: append lists
def _append_list(left: list[object], right: list[object]) -> list[object]:
    return left + right


class AgentState(TypedDict, total=False):
    """Complete state for a pulse run.

    Every field is optional (total=False) so nodes only return what they update.
    """

    # Run identification
    run_id: str
    product_key: str
    iso_week: str
    window_start: date
    window_end: date

    # Phase outputs
    reviews: list[RawReview]
    embeddings: list[list[float]]
    clusters: list[Cluster]
    summary: PulseSummary | None
    doc_requests: list[dict[str, object]]
    email_html: str
    email_text: str

    # Delivery tracking
    gdoc_id: str | None
    gdoc_heading_id: str | None
    gmail_message_id: str | None
    doc_publish_error: str | None
    configured_gdoc_id: str | None

    # Control
    status: str
    error: str | None
    retry_count: int
    llm_tokens_used: int
    llm_cost_usd: float
    skip_ingest: bool
    dry_run: bool
    gmail_to: str | None
