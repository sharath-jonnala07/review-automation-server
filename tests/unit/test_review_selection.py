"""Tests for selecting high-value reviews for a run."""

from datetime import UTC, datetime, timedelta

from app.core.models import RawReview
from app.core.types import ProductKey, ReviewId
from app.services.pipeline import prepare_reviews_for_run
from app.services.review_selection import is_low_signal_review


def _review(review_id: str, body: str, *, rating: int = 3, days_ago: int = 0) -> RawReview:
    return RawReview(
        id=ReviewId(review_id),
        product_key=ProductKey("groww"),
        source="playstore",
        rating=rating,
        body=body,
        posted_at=datetime.now(UTC) - timedelta(days=days_ago),
    )


def test_low_signal_reviews_are_detected() -> None:
    assert is_low_signal_review(_review("1", "Great")) is True
    assert is_low_signal_review(_review("2", "cool app")) is True
    assert is_low_signal_review(_review("3", "The app crashes during payment and I cannot complete UPI transfers")) is False


def test_prepare_reviews_for_run_caps_at_200() -> None:
    reviews = [
        _review(
            str(index),
            f"The app crashes during payment step {index} and login fails after the update",
            rating=1,
            days_ago=index % 7,
        )
        for index in range(230)
    ]

    selected, metrics = prepare_reviews_for_run(reviews, max_reviews=200)

    assert len(selected) == 200
    assert metrics["reviewsAvailable"] == 230
    assert metrics["reviewsSelected"] == 200
    assert metrics["lowSignalDropped"] == 0
    assert metrics["highSignalOverflow"] == 30


def test_prepare_reviews_for_run_drops_low_signal_reviews() -> None:
    reviews = [
        _review("1", "great"),
        _review("1b", "bad"),
        _review("2", "cool"),
        _review("3", "good app"),
        _review("4", "Need an option to export portfolio data and fix the slow loading dashboard"),
        _review("5", "KYC verification keeps failing after update and support is not responding"),
    ]

    selected, metrics = prepare_reviews_for_run(reviews, max_reviews=200)

    assert {review.id for review in selected} == {ReviewId("4"), ReviewId("5")}
    assert metrics["lowSignalDropped"] == 4
    assert metrics["highSignalOverflow"] == 0