"""Regression tests for summarization orchestration."""

import asyncio
from datetime import UTC, date, datetime
from typing import Any

from app.clustering.clusterer import Cluster
from app.core.models import RawReview, Window
from app.core.types import ProductKey, ReviewId
from app.summarization.engine import SummarizationEngine


class FakeLLMClient:
    """Small structured-output fake for summarization tests."""

    async def structured_call(
        self,
        prompt_template: str,
        output_schema: type[Any],
        variables: dict[str, str],
        *,
        max_retries: int = 3,
    ) -> Any:
        if output_schema.__name__ == "ThemeOutput":
            return output_schema.model_validate(
                {
                    "label": "Performance",
                    "description": "Customers mention slow loading and chart delays.",
                    "sentiment": "negative",
                }
            )
        if output_schema.__name__ == "QuoteOutput":
            return output_schema.model_validate(
                {
                    "quotes": [
                        "The app is slow during market hours",
                        "Charts take too long to load",
                    ]
                }
            )
        if output_schema.__name__ == "ActionOutput":
            action_count = max(1, variables.get("themes", "").count("\n\n") + 1)
            return output_schema.model_validate(
                {
                    "actions": [
                        {
                            "title": "Improve market-hour performance",
                            "description": "Profile slow screens and prioritize chart loading fixes.",
                        }
                        for _ in range(action_count)
                    ]
                }
            )
        raise AssertionError(f"Unexpected schema: {output_schema.__name__}")


def test_summarize_sets_quote_theme_ids_without_mutating_frozen_models() -> None:
    """Quote models are frozen, so theme IDs must be attached via copies."""
    reviews = [
        RawReview(
            id=ReviewId("review-1"),
            product_key=ProductKey("groww"),
            source="playstore",
            rating=1,
            body="The app is slow during market hours and charts take too long to load",
            posted_at=datetime(2026, 4, 26, tzinfo=UTC),
        ),
        RawReview(
            id=ReviewId("review-2"),
            product_key=ProductKey("groww"),
            source="playstore",
            rating=2,
            body="Charts take too long to load when the market is open",
            posted_at=datetime(2026, 4, 26, tzinfo=UTC),
        ),
    ]
    clusters = [
        Cluster(
            id=0,
            review_indices=[0, 1],
            medoid_index=0,
            keyphrases=["market performance", "chart loading"],
        )
    ]
    engine = SummarizationEngine(llm_client=FakeLLMClient())

    summary = asyncio.run(
        engine.summarize(
            product=ProductKey("groww"),
            window=Window(start=date(2026, 4, 20), end=date(2026, 4, 26), weeks=1),
            reviews=reviews,
            clusters=clusters,
        )
    )

    assert summary.top_themes[0].id == "theme-groww-1"
    assert summary.quotes
    assert {quote.theme_id for quote in summary.quotes} == {summary.top_themes[0].id}


def test_summarize_backfills_actions_for_each_theme() -> None:
    """Action generation should still return one action per theme when the LLM under-returns."""
    reviews = [
        RawReview(
            id=ReviewId(f"review-{index}"),
            product_key=ProductKey("groww"),
            source="playstore",
            rating=1 + (index % 2),
            body=f"Review body {index} about platform stability and charts",
            posted_at=datetime(2026, 4, 26, tzinfo=UTC),
        )
        for index in range(6)
    ]
    clusters = [
        Cluster(id=0, review_indices=[0, 1], medoid_index=0, keyphrases=["stability"]),
        Cluster(id=1, review_indices=[2, 3], medoid_index=2, keyphrases=["support"]),
        Cluster(id=2, review_indices=[4, 5], medoid_index=4, keyphrases=["performance"]),
    ]
    engine = SummarizationEngine(llm_client=FakeLLMClient())

    async def no_quotes(
        theme_label: str,
        theme_description: str,
        cluster_reviews: list[RawReview],
    ) -> list[object]:
        return []

    engine.select_quotes = no_quotes  # type: ignore[method-assign]

    summary = asyncio.run(
        engine.summarize(
            product=ProductKey("groww"),
            window=Window(start=date(2026, 4, 20), end=date(2026, 4, 26), weeks=1),
            reviews=reviews,
            clusters=clusters,
        )
    )

    assert len(summary.top_themes) == 3
    assert len(summary.action_ideas) == 3
    assert summary.action_ideas[0].title == "Improve market-hour performance"
    assert summary.action_ideas[1].theme_id == summary.top_themes[1].id
    assert summary.action_ideas[2].theme_id == summary.top_themes[2].id


def test_summarize_includes_all_clusters_as_themes() -> None:
    """All generated clusters should be promoted into themes, not just the first three."""
    reviews = [
        RawReview(
            id=ReviewId(f"review-{index}"),
            product_key=ProductKey("groww"),
            source="playstore",
            rating=4,
            body=f"Review body {index} about theme {index}",
            posted_at=datetime(2026, 4, 26, tzinfo=UTC),
        )
        for index in range(8)
    ]
    clusters = [
        Cluster(id=index, review_indices=[index, index + 1], medoid_index=index, keyphrases=[f"theme-{index}"])
        for index in range(4)
    ]
    engine = SummarizationEngine(llm_client=FakeLLMClient())

    async def no_quotes(
        theme_label: str,
        theme_description: str,
        cluster_reviews: list[RawReview],
    ) -> list[object]:
        return []

    engine.select_quotes = no_quotes  # type: ignore[method-assign]

    summary = asyncio.run(
        engine.summarize(
            product=ProductKey("groww"),
            window=Window(start=date(2026, 4, 20), end=date(2026, 4, 26), weeks=1),
            reviews=reviews,
            clusters=clusters,
        )
    )

    assert len(summary.top_themes) == 4
    assert len(summary.action_ideas) == 4
    assert [theme.rank for theme in summary.top_themes] == [1, 2, 3, 4]