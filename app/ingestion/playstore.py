"""Google Play Store review ingestion."""

import asyncio
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

from app.core.models import RawReview
from app.core.types import ProductKey, ReviewId

logger = structlog.get_logger()

MAX_PAGES = 20


def _parse_play_date(date_str: str) -> datetime:
    """Parse Google Play date format to datetime.

    Google Play returns relative dates like "1 week ago", "2 days ago",
    or absolute dates like "Apr 20, 2026".
    """
    date_str = date_str.strip()
    now = datetime.now(UTC)

    # Try relative patterns
    relative_patterns: list[tuple[str, Callable[[re.Match[str]], datetime]]] = [
        (r"(\d+)\s+weeks?\s+ago", lambda m: now - timedelta(weeks=int(m.group(1)))),
        (r"(\d+)\s+days?\s+ago", lambda m: now - timedelta(days=int(m.group(1)))),
        (r"(\d+)\s+hours?\s+ago", lambda m: now - timedelta(hours=int(m.group(1)))),
        (r"(\d+)\s+minutes?\s+ago", lambda m: now - timedelta(minutes=int(m.group(1)))),
        (r"just\s+now", lambda _m: now),
    ]

    for pattern, handler in relative_patterns:
        match = re.match(pattern, date_str, re.IGNORECASE)
        if match:
            return handler(match)

    # Try absolute date patterns
    abs_patterns = [
        "%b %d, %Y",
        "%B %d, %Y",
        "%d %b %Y",
        "%d %B %Y",
    ]
    for fmt in abs_patterns:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            continue

    # Fallback: return now
    logger.warning("Could not parse Play Store date", date_str=date_str)
    return now


def _extract_reviews_from_html(
    html: str,
    product_key: ProductKey,
    _package: str,
) -> list[RawReview]:
    """Extract reviews from Google Play HTML response.

    This is a simplified parser. In production, consider using
    google-play-scraper for more robust handling.
    """
    reviews: list[RawReview] = []

    # Try to find review data in the page
    # Google Play embeds review data in script tags or JSON
    # For this implementation, we'll use a regex-based approach
    # on the rendered HTML structure

    # Alternative: look for JSON data in script tags
    json_pattern = re.compile(
        r'"reviewId":"([^"]+)".*?'
        r'"userComment":\{[^\}]*"text":"([^"]+)".*?'
        r'"starRating":(\d)',
        re.DOTALL,
    )

    matches = json_pattern.findall(html)
    for match in matches:
        review_id, body, rating = match
        # Unescape JSON string
        body = body.encode().decode("unicode_escape")

        reviews.append(
            RawReview(
                id=ReviewId(f"playstore-{review_id}"),
                product_key=product_key,
                source="playstore",
                rating=int(rating),
                title=None,
                body=body,
                posted_at=datetime.now(UTC),
                version=None,
                language="en",
                country="in",
            )
        )

    if not reviews:
        # Fallback: try simpler text extraction
        logger.warning(
            "No reviews extracted from Play Store HTML, using fallback",
            product=product_key,
        )

    return reviews


def _extract_reviews_from_scraper_result(
    results: list[dict[str, Any]],
    product_key: ProductKey,
    cutoff_date: datetime,
) -> list[RawReview]:
    """Convert google-play-scraper results to RawReview models."""
    reviews: list[RawReview] = []

    for entry in results:
        body = str(entry.get("content") or "").strip()
        review_id = str(entry.get("reviewId") or "").strip()
        posted_at = entry.get("at")
        if not body or not review_id or not isinstance(posted_at, datetime):
            continue

        posted_at_utc = posted_at.astimezone(UTC) if posted_at.tzinfo else posted_at.replace(tzinfo=UTC)
        if posted_at_utc < cutoff_date:
            continue

        score = entry.get("score", 5)
        try:
            rating = int(score)
        except (TypeError, ValueError):
            rating = 5

        reviews.append(
            RawReview(
                id=ReviewId(f"playstore-{review_id}"),
                product_key=product_key,
                source="playstore",
                rating=rating,
                title=None,
                body=body,
                posted_at=posted_at_utc,
                version=str(entry.get("reviewCreatedVersion") or "") or None,
                language="en",
                country="in",
            )
        )

    return reviews


async def fetch_playstore_reviews(
    product_key: ProductKey,
    package: str,
    *,
    weeks: int = 10,
    max_pages: int = MAX_PAGES,
) -> list[RawReview]:
    """Fetch Google Play Store reviews for a product.

    Args:
        product_key: Product identifier
        package: Android package name
        weeks: How many weeks back to fetch
        max_pages: Maximum pages to fetch

    Returns:
        List of RawReview objects

    Raises:
        IngestionError: If reviews cannot be fetched
    """
    if not package:
        return []

    try:
        from google_play_scraper import Sort, reviews
    except ImportError as e:
        logger.warning("google-play-scraper not installed", product=product_key)
        raise RuntimeError(
            "google-play-scraper is required for Play Store ingestion"
        ) from e

    cutoff_date = datetime.now(UTC) - timedelta(weeks=weeks)
    all_reviews: list[RawReview] = []
    continuation_token: str | None = None

    logger.info(
        "Play Store ingestion started",
        product=product_key,
        package=package,
        weeks=weeks,
    )

    for page in range(1, max_pages + 1):
        result, continuation_token = await asyncio.to_thread(
            reviews,
            package,
            lang="en",
            country="in",
            sort=Sort.NEWEST,
            count=100,
            continuation_token=continuation_token,
        )

        page_reviews = _extract_reviews_from_scraper_result(result, product_key, cutoff_date)
        all_reviews.extend(page_reviews)

        logger.debug(
            "Fetched Play Store page",
            product=product_key,
            page=page,
            reviews=len(page_reviews),
        )

        if not continuation_token or len(page_reviews) < len(result):
            break

    logger.info(
        "Play Store ingestion complete",
        product=product_key,
        total_reviews=len(all_reviews),
        weeks=weeks,
    )
    return all_reviews
