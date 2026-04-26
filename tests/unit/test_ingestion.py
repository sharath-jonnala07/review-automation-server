"""Tests for review ingestion and PII scrubbing."""


from datetime import UTC

from app.core.models import RawReview
from app.core.types import ProductKey, ReviewId
from app.ingestion.scrubber import scrub_text


class TestPIIScrubber:
    """PII scrubbing tests."""

    def test_scrub_email(self) -> None:
        """Should remove email addresses."""
        text = "Contact me at john.doe@example.com for details"
        result = scrub_text(text)
        assert "john.doe@example.com" not in result
        assert "[REDACTED]" in result

    def test_scrub_phone(self) -> None:
        """Should remove phone numbers."""
        text = "Call me at +91 98765 43210"
        result = scrub_text(text)
        assert "98765 43210" not in result
        assert "[REDACTED]" in result

    def test_scrub_aadhaar(self) -> None:
        """Should remove Aadhaar numbers."""
        text = "My Aadhaar is 1234 5678 9012"
        result = scrub_text(text)
        assert "1234 5678 9012" not in result
        assert "[REDACTED]" in result

    def test_scrub_multiple(self) -> None:
        """Should remove multiple PII types in one text."""
        text = "Email: test@mail.com, Phone: 9876543210, Aadhaar: 1234-5678-9012"
        result = scrub_text(text)
        assert result.count("[REDACTED]") == 3

    def test_no_pii_unchanged(self) -> None:
        """Should leave clean text unchanged."""
        text = "This app is great and works perfectly!"
        result = scrub_text(text)
        assert result == text


class TestRawReviewModel:
    """RawReview model tests."""

    def test_review_creation(self) -> None:
        """Should create a valid RawReview."""
        from datetime import datetime

        review = RawReview(
            id=ReviewId("playstore-abc123"),
            product_key=ProductKey("groww"),
            source="playstore",
            rating=4,
            body="Great app for investing!",
            posted_at=datetime(2026, 4, 1, 10, 0, 0, tzinfo=UTC),
        )
        assert review.rating == 4
        assert review.source == "playstore"
