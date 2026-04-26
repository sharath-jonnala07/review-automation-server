"""Unit tests for LLM backend selection and heuristic fallback."""

import asyncio

from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
import pytest

from app.config import get_settings
from app.core.exceptions import SummarizationError
from app.summarization.llm_client import LLMClient


class ThemeLikeOutput(BaseModel):
    """Minimal schema matching the theme-label contract."""

    label: str
    description: str
    sentiment: str


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """Reset cached settings between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_llm_client_uses_heuristic_fallback_without_remote_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dev fallback should keep summarization working without remote credentials."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("LLM_ALLOW_HEURISTIC_FALLBACK", "true")

    client = LLMClient()

    result = asyncio.run(
        client.structured_call(
            prompt_template="{format_instructions}",
            output_schema=ThemeLikeOutput,
            variables={
                "keyphrases": "authentication 2fa, login flow, groww app",
                "reviews": "- Login keeps asking for 2FA again\n\n- The new login flow is slow and frustrating",
            },
        )
    )

    assert "Authentication" in result.label
    assert "customers repeatedly mention" in result.description.lower()
    assert result.sentiment == "negative"
    assert client.backend_name == "heuristic"


def test_llm_client_preflight_succeeds_with_heuristic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight should pass when heuristic fallback is the only available mode."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("LLM_ALLOW_HEURISTIC_FALLBACK", "true")

    client = LLMClient()

    backend = asyncio.run(client.ensure_ready())

    assert backend == "heuristic"


def test_llm_client_blocks_heuristic_after_remote_auth_failure_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad live key should fail loudly instead of silently using heuristics."""
    monkeypatch.setenv("GROQ_API_KEY", "bad-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_ALLOW_HEURISTIC_FALLBACK", "true")
    monkeypatch.setenv("LLM_ALLOW_HEURISTIC_AUTH_FALLBACK", "false")

    client = LLMClient()
    remaining_backends = [backend for backend in client._backends if backend.kind == "heuristic"]

    with pytest.raises(SummarizationError, match="remote LLM rejected authentication"):
        client._raise_if_heuristic_auth_fallback_is_disabled(
            RuntimeError("Error code: 401 invalid_api_key"),
            remaining_backends,
        )


def test_llm_client_uses_heuristic_after_remote_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rate-limited local runs should continue with heuristic fallback."""
    monkeypatch.setenv("GROQ_API_KEY", "rate-limited-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_ALLOW_HEURISTIC_FALLBACK", "true")

    client = LLMClient()

    def raise_rate_limit(_: object) -> object:
        raise RuntimeError(
            "Error code: 429 - {'error': {'message': 'Rate limit reached', 'code': 'rate_limit_exceeded'}}"
        )

    monkeypatch.setattr(client, "_get_remote_llm", lambda backend: RunnableLambda(raise_rate_limit))

    result = asyncio.run(
        client.structured_call(
            prompt_template="{format_instructions}",
            output_schema=ThemeLikeOutput,
            variables={
                "keyphrases": "support quality, login issues, trading delays",
                "reviews": "- Customer support never resolves my issue\n\n- The app feels slow when the market opens",
            },
            max_retries=1,
        )
    )

    assert result.label
    assert client.backend_name == "heuristic"
