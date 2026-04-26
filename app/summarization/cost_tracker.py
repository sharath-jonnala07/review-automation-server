"""Per-run LLM cost tracking with hard cap."""

import structlog

from app.config import get_settings
from app.core.exceptions import CostExceededError

logger = structlog.get_logger()

# Cost per 1M tokens (input / output) — Groq pricing
# https://groq.com/pricing/
MODEL_COSTS: dict[str, dict[str, float]] = {
    # Groq models
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.3-70b-specdec": {"input": 0.59, "output": 0.99},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},
    # Optional OpenAI embedding fallback
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    # Legacy OpenAI fallback
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.5, "output": 10.0},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD for an LLM call."""
    costs = MODEL_COSTS.get(model, MODEL_COSTS["llama-3.3-70b-versatile"])
    input_cost = (prompt_tokens / 1_000_000) * costs["input"]
    output_cost = (completion_tokens / 1_000_000) * costs["output"]
    return input_cost + output_cost


class CostTracker:
    """Tracks LLM spend per run with a hard cap."""

    def __init__(self, max_usd: float | None = None) -> None:
        settings = get_settings()
        self.max_usd = max_usd or settings.llm_max_cost_usd
        self.spent_usd = 0.0
        self.total_tokens = 0

    def charge(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Record cost for an LLM call.

        Raises:
            CostExceededError: If the charge would exceed the cap.
        """
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        new_total = self.spent_usd + cost

        if new_total > self.max_usd:
            raise CostExceededError(
                f"Cost cap exceeded: ${new_total:.4f} > ${self.max_usd:.4f} "
                f"(model={model}, tokens={prompt_tokens + completion_tokens})"
            )

        self.spent_usd = new_total
        self.total_tokens += prompt_tokens + completion_tokens

        logger.debug(
            "LLM cost charged",
            model=model,
            cost_usd=round(cost, 6),
            total_usd=round(self.spent_usd, 6),
            tokens=prompt_tokens + completion_tokens,
        )

    def get_metrics(self) -> dict[str, float]:
        """Return current cost metrics."""
        return {
            "spent_usd": round(self.spent_usd, 6),
            "max_usd": self.max_usd,
            "total_tokens": self.total_tokens,
        }
