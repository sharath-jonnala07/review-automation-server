"""Custom exceptions with structured error context."""


class PulseError(Exception):
    """Base exception for all pulse agent errors."""

    def __init__(self, message: str, *, context: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}


class IngestionError(PulseError):
    """Failed to fetch reviews from a store."""


class ClusteringError(PulseError):
    """Embedding or clustering pipeline failure."""


class SummarizationError(PulseError):
    """LLM summarization or validation failure."""


class QuoteValidationError(SummarizationError):
    """A proposed quote failed verbatim validation."""


class RenderError(PulseError):
    """Report or email rendering failure."""


class PublishError(PulseError):
    """MCP publication failure."""


class MCPError(PublishError):
    """MCP server communication error."""


class CostExceededError(PulseError):
    """LLM cost cap exceeded for this run."""


class IdempotencyError(PulseError):
    """Idempotency check detected a duplicate."""
