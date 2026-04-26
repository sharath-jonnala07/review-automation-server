"""App Store review ingestion via iTunes RSS feed."""

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import structlog

from app.core.exceptions import IngestionError
from app.core.models import RawReview
from app.core.types import ProductKey, ReviewId

logger = structlog.get_logger()

ITUNES_RSS_URL = "https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/sortBy=mostRecent/page={page}/json"
MAX_PAGES = 10
REVIEWS_PER_PAGE = 50


def _parse_itunes_date(date_str: str) -> datetime:
    """Parse iTunes RSS date format to datetime."""
    # iTunes dates are ISO 8601: 2026-04-20T10:30:00-07:00
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    return dt.astimezone(UTC)


def _extract_reviews_from_feed(
    feed: dict[str, Any],
    product_key: ProductKey,
    _app_id: str,
) -> list[RawReview]:
    """Extract RawReview objects from iTunes RSS feed JSON."""
    entries = feed.get("entry", [])
    if not entries:
        return []

    # First entry is usually the app metadata, skip it if it has no content
    reviews: list[RawReview] = []
    for entry in entries:
        # Skip the app metadata entry (it has no content label)
        content = entry.get("content", {})
        if not content or not content.get("label"):
            continue

        review_id = str(entry.get("id", {}).get("label", ""))
        if not review_id:
            continue

        # Extract rating from im:rating
        rating_label = entry.get("im:rating", {}).get("label", "5")
        try:
            rating = int(rating_label)
        except ValueError:
            rating = 5

        # Extract version
        version = entry.get("im:version", {}).get("label")

        # Extract title
        title = entry.get("title", {}).get("label", "")

        # Extract body
        body = content.get("label", "")

        # Extract date
        updated = entry.get("updated", {}).get("label", "")
        posted_at = _parse_itunes_date(updated) if updated else datetime.now(UTC)

        reviews.append(
            RawReview(
                id=ReviewId(f"appstore-{review_id}"),
                product_key=product_key,
                source="appstore",
                rating=rating,
                title=title or None,
                body=body,
                posted_at=posted_at,
                version=version,
                language="en",
                country="in",
            )
        )

    return reviews


async def fetch_appstore_reviews(
    product_key: ProductKey,
    app_id: str,
    *,
    weeks: int = 10,
    country: str = "in",
    max_pages: int = MAX_PAGES,
) -> list[RawReview]:
    """Fetch App Store reviews for a product.

    Args:
        product_key: Product identifier
        app_id: iTunes App Store ID
        weeks: How many weeks back to fetch
        country: App Store country code
        max_pages: Maximum pages to fetch

    Returns:
        List of RawReview objects

    Raises:
        IngestionError: If the feed cannot be fetched or parsed
    """
    if not app_id:
        return []

    cutoff_date = datetime.now(UTC) - timedelta(weeks=weeks)
    all_reviews: list[RawReview] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for page in range(1, max_pages + 1):
            url = ITUNES_RSS_URL.format(country=country, app_id=app_id, page=page)
            logger.debug("Fetching App Store page", url=url, page=page, product=product_key)

            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise IngestionError(
                    f"Failed to fetch App Store reviews: {e}",
                    context={"product": product_key, "app_id": app_id, "page": page},
                ) from e

            data = response.json()
            feed = data.get("feed", {})

            reviews = _extract_reviews_from_feed(feed, product_key, app_id)
            if not reviews:
                break

            # Filter by date
            page_reviews = [r for r in reviews if r.posted_at >= cutoff_date]
            all_reviews.extend(page_reviews)

            # If any review on this page is too old, stop paginating
            if len(page_reviews) < len(reviews):
                break

            logger.debug(
                "Fetched App Store page",
                page=page,
                reviews=len(page_reviews),
                product=product_key,
            )

    logger.info(
        "App Store ingestion complete",
        product=product_key,
        total_reviews=len(all_reviews),
        weeks=weeks,
    )
    return all_reviews
