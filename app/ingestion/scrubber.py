"""PII scrubber for review text before LLM and publishing."""

import re

import structlog

logger = structlog.get_logger()

# Regex patterns for PII detection
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(
    r"(?:\+91[-\s]?)?\d{10}|(?:\+91[-\s]?)?\d{5}[-\s]?\d{5}"
)
AADHAAR_PATTERN = re.compile(
    r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
)
CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
)
# Indian PAN number
PAN_PATTERN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")

REPLACEMENT = "[REDACTED]"


def scrub_text(text: str) -> str:
    """Remove PII from text using regex patterns.

    Args:
        text: Raw text that may contain PII

    Returns:
        Text with PII replaced by [REDACTED]
    """
    if not text:
        return text

    original = text
    text = EMAIL_PATTERN.sub(REPLACEMENT, text)
    text = PHONE_PATTERN.sub(REPLACEMENT, text)
    text = AADHAAR_PATTERN.sub(REPLACEMENT, text)
    text = CREDIT_CARD_PATTERN.sub(REPLACEMENT, text)
    text = PAN_PATTERN.sub(REPLACEMENT, text)

    if text != original:
        logger.debug("Scrubbed PII from text", replacements=text.count(REPLACEMENT))

    return text


def scrub_review_body(body: str) -> str:
    """Scrub PII from a review body specifically.

    This is a convenience wrapper around scrub_text that may
    add review-specific logic in the future.
    """
    return scrub_text(body)
