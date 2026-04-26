"""Review selection heuristics for clustering and summarization."""

from __future__ import annotations

import re
from collections import Counter

from app.core.models import RawReview

LOW_SIGNAL_PATTERNS = {
    "bad",
    "good",
    "great",
    "nice",
    "cool",
    "awesome",
    "super",
    "amazing",
    "best",
    "ok",
    "okay",
    "fine",
    "poor",
    "worst",
    "love it",
    "very good",
    "excellent",
}

DECISION_KEYWORDS = {
    "bug",
    "bugs",
    "crash",
    "crashes",
    "stuck",
    "issue",
    "issues",
    "problem",
    "problems",
    "slow",
    "lag",
    "freeze",
    "error",
    "failed",
    "login",
    "payment",
    "upi",
    "bank",
    "transaction",
    "portfolio",
    "support",
    "feature",
    "features",
    "option",
    "options",
    "improve",
    "improvement",
    "update",
    "performance",
    "ui",
    "ux",
    "design",
    "loading",
    "verification",
    "kyc",
    "withdrawal",
    "deposit",
}


def _normalize_text(text: str) -> str:
    """Normalize text for heuristic checks."""
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def is_low_signal_review(review: RawReview) -> bool:
    """Return true for short praise or otherwise non-actionable reviews."""
    normalized = _normalize_text(review.body)
    words = normalized.split()
    unique_words = set(words)

    if not normalized:
        return True
    if normalized in LOW_SIGNAL_PATTERNS:
        return True
    if len(words) <= 3 and normalized.replace(" ", "") in LOW_SIGNAL_PATTERNS:
        return True
    if len(words) < 5 and len(unique_words) <= 3 and not (unique_words & DECISION_KEYWORDS):
        return True
    if len(normalized) < 20 and not (unique_words & DECISION_KEYWORDS):
        return True
    return False


def review_value_score(review: RawReview) -> int:
    """Score reviews by decision-making value for product teams."""
    normalized = _normalize_text(review.body)
    words = normalized.split()
    word_counts = Counter(words)
    keyword_hits = sum(1 for word in set(words) if word in DECISION_KEYWORDS)
    repeated_word_penalty = sum(count - 1 for count in word_counts.values() if count > 2)

    score = 0
    score += min(len(words), 40)
    score += keyword_hits * 8
    score += 5 if review.rating <= 2 else 0
    score += 2 if review.rating >= 4 and keyword_hits > 0 else 0
    score += 3 if any(token in normalized for token in {"should", "please", "need", "wish"}) else 0
    score -= repeated_word_penalty * 2

    return score


def select_reviews_for_processing(
    reviews: list[RawReview],
    *,
    max_reviews: int,
) -> tuple[list[RawReview], int]:
    """Filter low-signal reviews and cap the remaining set for processing."""
    informative_reviews = [review for review in reviews if not is_low_signal_review(review)]
    low_signal_dropped = len(reviews) - len(informative_reviews)

    ranked_reviews = sorted(
        informative_reviews,
        key=lambda review: (review_value_score(review), review.posted_at),
        reverse=True,
    )
    return ranked_reviews[:max_reviews], low_signal_dropped