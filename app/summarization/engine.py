"""Main summarization engine: themes, quotes, and action ideas."""

import asyncio
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

from app.clustering.clusterer import Cluster
from app.core.models import (
    ActionIdea,
    AudienceValue,
    PulseStats,
    PulseSummary,
    Quote,
    RawReview,
    Theme,
    Window,
)
from app.core.types import ProductKey, ReviewId, Sentiment, ThemeId
from app.ingestion.scrubber import scrub_review_body
from app.summarization.llm_client import LLMClient
from app.summarization.validators import validate_quotes

logger = structlog.get_logger()

# What this solves - static mapping for now
AUDIENCE_VALUES = [
    AudienceValue(
        audience="Product",
        value="Prioritize roadmap from recurring themes",
    ),
    AudienceValue(
        audience="Support",
        value="Spot repeating complaints and quality issues",
    ),
    AudienceValue(
        audience="Leadership",
        value="Fast health snapshot tied to customer voice",
    ),
]


class ThemeOutput(BaseModel):
    """Structured output for theme labeling."""

    label: str = Field(..., min_length=3, max_length=100)
    description: str = Field(..., min_length=10, max_length=500)
    sentiment: Sentiment


class QuoteOutput(BaseModel):
    """Structured output for quote selection."""

    quotes: list[str] = Field(..., min_length=1, max_length=5)


class ActionOutput(BaseModel):
    """Structured output for action idea generation."""

    actions: list[dict[str, str]] = Field(..., min_length=1)


class SummarizationEngine:
    """Orchestrates LLM calls to produce PulseSummary from clustered reviews."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client or LLMClient()
        self.prompts_dir = Path(__file__).parent / "prompts"

    def _load_prompt(self, name: str) -> str:
        """Load a prompt template from disk."""
        path = self.prompts_dir / f"{name}.txt"
        return path.read_text(encoding="utf-8")

    async def label_theme(
        self,
        keyphrases: list[str],
        representative_reviews: list[RawReview],
    ) -> ThemeOutput:
        """Label a cluster with a theme."""
        # Re-scrub before LLM
        reviews_text = "\n\n".join(
            f"- {scrub_review_body(r.body)}" for r in representative_reviews
        )

        prompt = self._load_prompt("label_theme")
        return await self.llm.structured_call(
            prompt_template=prompt,
            output_schema=ThemeOutput,
            variables={
                "keyphrases": ", ".join(keyphrases),
                "reviews": reviews_text,
            },
        )

    async def select_quotes(
        self,
        theme_label: str,
        theme_description: str,
        cluster_reviews: list[RawReview],
    ) -> list[Quote]:
        """Select verified verbatim quotes for a theme."""
        # Re-scrub before LLM
        reviews_text = "\n\n".join(
            f"- {scrub_review_body(r.body)}" for r in cluster_reviews
        )

        prompt = self._load_prompt("select_quotes")
        result = await self.llm.structured_call(
            prompt_template=prompt,
            output_schema=QuoteOutput,
            variables={
                "theme_label": theme_label,
                "theme_description": theme_description,
                "reviews": reviews_text,
            },
        )

        # Validate quotes against source reviews
        validated = validate_quotes(
            result.quotes,
            cluster_reviews,
            drop_invalid=True,
            raise_on_all_invalid=False,
        )

        return [
            Quote(text=q, review_id=ReviewId(review.id))
            for q, review in validated
        ]

    async def generate_actions(
        self,
        themes: list[Theme],
    ) -> list[ActionIdea]:
        """Generate action ideas from themes."""
        themes_text = "\n\n".join(
            f"{i+1}. {t.label}\n{t.description}"
            for i, t in enumerate(themes)
        )

        prompt = self._load_prompt("generate_actions")
        result = await self.llm.structured_call(
            prompt_template=prompt,
            output_schema=ActionOutput,
            variables={"themes": themes_text},
        )

        actions: list[ActionIdea] = []
        for i, theme in enumerate(themes):
            action_dict = result.actions[i] if i < len(result.actions) else {}
            title = action_dict.get("title") or f"Investigate {theme.label}"
            description = action_dict.get("description") or (
                f"Turn the {theme.label.lower()} theme into a concrete remediation plan. "
                f"Current signal: {theme.description}"
            )
            actions.append(
                ActionIdea(
                    theme_id=theme.id,
                    title=title,
                    description=description,
                )
            )

        return actions

    async def summarize(
        self,
        product: ProductKey,
        window: Window,
        reviews: list[RawReview],
        clusters: list[Cluster],
    ) -> PulseSummary:
        """Run full summarization pipeline.

        Args:
            product: Product key
            window: Time window
            reviews: All raw reviews
            clusters: Cluster objects with review_indices, keyphrases, medoid_index

        Returns:
            PulseSummary with themes, quotes, and actions
        """
        # Calculate stats
        avg_rating = sum(r.rating for r in reviews) / len(reviews) if reviews else 0.0
        appstore_count = sum(1 for r in reviews if r.source == "appstore")
        playstore_count = sum(1 for r in reviews if r.source == "playstore")

        stats = PulseStats(
            total_reviews=len(reviews),
            avg_rating=round(avg_rating, 2),
            appstore_reviews=appstore_count,
            playstore_reviews=playstore_count,
        )

        themes: list[Theme] = []
        all_quotes: list[Quote] = []

        async def build_theme(rank: int, cluster: Cluster) -> tuple[Theme, list[Quote]]:
            cluster_reviews = [reviews[i] for i in cluster.review_indices]
            representative = [
                reviews[cluster.medoid_index],
                *[
                    reviews[i]
                    for i in cluster.review_indices[:3]
                    if i != cluster.medoid_index
                ],
            ]

            theme_output = await self.label_theme(
                keyphrases=cluster.keyphrases,
                representative_reviews=representative,
            )

            theme = Theme(
                id=ThemeId(f"theme-{product}-{rank}"),
                rank=rank,
                label=theme_output.label,
                description=theme_output.description,
                sentiment=theme_output.sentiment,
                review_count=len(cluster.review_indices),
                representative_review_ids=[
                    ReviewId(reviews[i].id) for i in cluster.review_indices[:5]
                ],
                keyphrases=cluster.keyphrases,
            )

            quotes = await self.select_quotes(
                theme_label=theme.label,
                theme_description=theme.description,
                cluster_reviews=cluster_reviews,
            )
            themed_quotes = [quote.model_copy(update={"theme_id": theme.id}) for quote in quotes]
            return theme, themed_quotes

        theme_results = await asyncio.gather(
            *(build_theme(rank, cluster) for rank, cluster in enumerate(clusters[:3], 1))
        )
        for theme, quotes in theme_results:
            themes.append(theme)
            all_quotes.extend(quotes)

        # Generate action ideas
        actions = await self.generate_actions(themes)

        return PulseSummary(
            product=product,
            window=window,
            stats=stats,
            top_themes=themes,
            quotes=all_quotes,
            action_ideas=actions,
            what_this_solves=AUDIENCE_VALUES,
        )
