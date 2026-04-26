"""Canonical Pydantic domain models."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.types import ProductKey, ReviewId, Sentiment, ThemeId


class Window(BaseModel):
    """Time window for review analysis."""

    model_config = ConfigDict(frozen=True)

    start: date
    end: date
    weeks: int


class RawReview(BaseModel):
    """A single review from App Store or Play Store."""

    model_config = ConfigDict(frozen=True)

    id: ReviewId
    product_key: ProductKey
    source: Literal["appstore", "playstore"]
    rating: int = Field(..., ge=1, le=5)
    title: str | None = None
    body: str
    posted_at: datetime
    version: str | None = None
    language: str = "en"
    country: str = "in"


class Quote(BaseModel):
    """Verified verbatim quote from a review."""

    model_config = ConfigDict(frozen=True)

    text: str
    review_id: ReviewId
    theme_id: ThemeId | None = None


class ActionIdea(BaseModel):
    """Actionable idea derived from a theme."""

    model_config = ConfigDict(frozen=True)

    theme_id: ThemeId
    title: str
    description: str


class AudienceValue(BaseModel):
    """Who this insight helps and how."""

    model_config = ConfigDict(frozen=True)

    audience: str
    value: str


class Theme(BaseModel):
    """A clustered theme with metadata."""

    model_config = ConfigDict(frozen=True)

    id: ThemeId
    rank: int = Field(..., ge=1)
    label: str
    description: str
    sentiment: Sentiment
    review_count: int = Field(..., ge=1)
    representative_review_ids: list[ReviewId]
    keyphrases: list[str] = Field(default_factory=list)


class PulseStats(BaseModel):
    """Statistics for a pulse run."""

    model_config = ConfigDict(frozen=True)

    total_reviews: int
    avg_rating: float = Field(..., ge=1.0, le=5.0)
    rating_delta_vs_prev: float | None = None
    appstore_reviews: int = 0
    playstore_reviews: int = 0


class PulseMetrics(BaseModel):
    """Operational metrics for a run."""

    model_config = ConfigDict(frozen=True)

    reviews_ingested: int = 0
    clusters_formed: int = 0
    llm_tokens: int = 0
    llm_cost_usd: float = 0.0
    duration_seconds: float = 0.0


class PulseSummary(BaseModel):
    """The final generated pulse report."""

    model_config = ConfigDict(frozen=True)

    product: ProductKey
    window: Window
    stats: PulseStats
    top_themes: list[Theme]
    quotes: list[Quote]
    action_ideas: list[ActionIdea]
    what_this_solves: list[AudienceValue]


class ProductConfig(BaseModel):
    """Configuration for a tracked product."""

    model_config = ConfigDict(frozen=True)

    key: ProductKey
    display_name: str
    appstore_id: str | None = None
    play_package: str | None = None
    gdoc_id: str | None = None
    gmail_to: str | None = None
    is_active: bool = True
