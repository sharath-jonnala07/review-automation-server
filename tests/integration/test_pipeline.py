"""Integration tests for the full pipeline with mocked services."""

from datetime import UTC, date
from pathlib import Path

import pytest

from app.clustering.cache import EmbeddingCache
from app.core.models import RawReview, Window
from app.core.types import ProductKey, ReviewId
from app.ingestion.scrubber import scrub_review_body
from app.renderer.docs_tree import build_doc_requests
from app.renderer.email_renderer import render_email


class TestPipeline:
    """End-to-end pipeline tests with fixtures."""

    @pytest.fixture
    def sample_reviews(self) -> list[RawReview]:
        """Generate sample reviews for testing."""
        from datetime import datetime

        return [
            RawReview(
                id=ReviewId(f"rev-{i}"),
                product_key=ProductKey("groww"),
                source="appstore",
                rating=4 if i % 3 == 0 else 2,
                body=f"This is review number {i} about app performance and bugs",
                posted_at=datetime(2026, 4, 1, 0, 0, 0, tzinfo=UTC),
            )
            for i in range(60)
        ]

    def test_scrub_then_embed(self) -> None:
        """Reviews should be scrubbed before embedding."""
        dirty = "Contact me at test@email.com for help"
        clean = scrub_review_body(dirty)
        assert "test@email.com" not in clean
        assert "[REDACTED]" in clean

    def test_embedding_cache(self, tmp_path: Path) -> None:
        """Embedding cache should deduplicate by text content."""
        cache = EmbeddingCache(cache_dir=tmp_path / "embeddings")
        text = "Great app for investing in mutual funds"
        emb = [0.1] * 384

        cache.set(text, emb)
        hit = cache.get(text)
        assert hit == emb

        miss = cache.get("different text")
        assert miss is None

    def test_render_doc_requests(self) -> None:
        """Doc requests should be generated from summary."""
        from app.core.models import (
            ActionIdea,
            AudienceValue,
            PulseStats,
            PulseSummary,
            Quote,
            Theme,
        )
        from app.core.types import ThemeId

        summary = PulseSummary(
            product=ProductKey("groww"),
            window=Window(start=date(2026, 4, 1), end=date(2026, 4, 7), weeks=1),
            stats=PulseStats(total_reviews=100, avg_rating=4.2),
            top_themes=[
                Theme(
                    id=ThemeId("t1"),
                    rank=1,
                    label="App Performance",
                    description="Users report lag and crashes",
                    sentiment="negative",
                    review_count=45,
                    representative_review_ids=[ReviewId("r1")],
                )
            ],
            quotes=[Quote(text="App crashes daily", review_id=ReviewId("r1"))],
            action_ideas=[
                ActionIdea(theme_id=ThemeId("t1"), title="Fix crashes", description="Investigate")
            ],
            what_this_solves=[
                AudienceValue(audience="Product", value="Prioritize fixes")
            ],
        )

        requests = build_doc_requests(summary, "pulse-groww-2026-W16")
        assert len(requests) > 0
        # Should have insertText requests
        insert_requests = [r for r in requests if "insertText" in r]
        assert len(insert_requests) > 0
        assert all("insertHorizontalRule" not in request for request in requests)

    def test_render_email(self) -> None:
        """Email should render HTML and text."""
        from app.core.models import (
            ActionIdea,
            AudienceValue,
            PulseStats,
            PulseSummary,
            Quote,
            Theme,
        )
        from app.core.types import ThemeId

        summary = PulseSummary(
            product=ProductKey("groww"),
            window=Window(start=date(2026, 4, 1), end=date(2026, 4, 7), weeks=1),
            stats=PulseStats(total_reviews=100, avg_rating=4.2),
            top_themes=[
                Theme(
                    id=ThemeId("t1"),
                    rank=1,
                    label="Performance",
                    description="Lag issues",
                    sentiment="negative",
                    review_count=45,
                    representative_review_ids=[ReviewId("r1")],
                )
            ],
            quotes=[Quote(text="Slow app", review_id=ReviewId("r1"))],
            action_ideas=[
                ActionIdea(theme_id=ThemeId("t1"), title="Optimize", description="Speed up")
            ],
            what_this_solves=[
                AudienceValue(audience="Product", value="Fix it")
            ],
        )

        html, text = render_email(summary, "https://docs.google.com/doc", "run-123")
        assert "Performance" in html
        assert "Slow app" in text
        assert "https://docs.google.com/doc" in html
