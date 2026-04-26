"""Output validators for LLM-generated content."""

import structlog

from app.core.exceptions import QuoteValidationError
from app.core.models import RawReview

logger = structlog.get_logger()


def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace."""
    return " ".join(text.lower().split())


def validate_quote(quote_text: str, reviews: list[RawReview]) -> RawReview | None:
    """Check that a quote appears verbatim in at least one review.

    Args:
        quote_text: The proposed quote
        reviews: Pool of reviews to search

    Returns:
        The matching review, or None if no match found
    """
    normalized_quote = _normalize(quote_text)

    for review in reviews:
        normalized_body = _normalize(review.body)
        if normalized_quote in normalized_body:
            return review

    return None


def validate_quotes(
    quotes: list[str],
    reviews: list[RawReview],
    *,
    drop_invalid: bool = True,
    raise_on_all_invalid: bool = False,
) -> list[tuple[str, RawReview]]:
    """Validate multiple quotes against review bodies.

    Args:
        quotes: Proposed quotes
        reviews: Pool of reviews
        drop_invalid: If True, drop non-matching quotes. If False, raise.
        raise_on_all_invalid: If True and all quotes are invalid, raise.

    Returns:
        List of (quote, matching_review) tuples

    Raises:
        QuoteValidationError: If a quote fails validation and drop_invalid is False
    """
    validated: list[tuple[str, RawReview]] = []

    for quote in quotes:
        match = validate_quote(quote, reviews)
        if match:
            validated.append((quote, match))
        else:
            logger.warning("Quote failed verbatim validation", quote=quote[:100])
            if not drop_invalid:
                raise QuoteValidationError(
                    f"Quote not found in any review: {quote[:200]}"
                )

    if raise_on_all_invalid and not validated and quotes:
        raise QuoteValidationError("All proposed quotes failed verbatim validation")

    return validated
