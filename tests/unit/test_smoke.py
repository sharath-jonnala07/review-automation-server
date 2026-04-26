"""Smoke tests to verify the codebase skeleton is functional."""

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.core.models import ProductConfig, RawReview, Window
from app.core.types import ProductKey, ReviewId, RunId


class TestSettings:
    """Configuration smoke tests."""

    def test_settings_defaults(self) -> None:
        """Settings should load with sensible defaults."""
        settings = Settings(_env_file=None)
        assert settings.app_name == "pulse-agent"
        assert settings.llm_provider == "auto"
        assert settings.llm_model == "llama-3.3-70b-versatile"
        assert settings.groq_base_url == "https://api.groq.com/openai/v1"
        assert settings.llm_max_cost_usd == 0.5
        assert settings.confirm_send is False
        assert settings.embedding_backend == "huggingface-local"
        assert settings.embedding_model == "Qwen/Qwen3-Embedding-0.6B"
        assert settings.max_reviews_per_run == 200
        assert settings.preferred_llm_provider == "unconfigured"

    def test_settings_prefer_openai_for_gpt_models(self) -> None:
        """OpenAI should be preferred when the model targets GPT-family endpoints."""
        settings = Settings(
            _env_file=None,
            OPENAI_API_KEY="sk-openai-test",
            LLM_MODEL="gpt-4o-mini",
        )

        assert settings.preferred_llm_provider == "openai"
        assert settings.llm_ready is True

    def test_settings_prefer_openai_for_gpt_models(self) -> None:
        """OpenAI should be preferred when the model targets GPT-family endpoints."""
        settings = Settings(
            _env_file=None,
            OPENAI_API_KEY="sk-openai-test",
            LLM_MODEL="gpt-4o-mini",
        )

        assert settings.preferred_llm_provider == "openai"
        assert settings.llm_ready is True

    def test_settings_creates_directories(self) -> None:
        """Settings should ensure data directories exist."""
        settings = Settings()
        assert settings.data_dir.exists()
        assert settings.raw_data_dir.exists()
        assert settings.artifacts_dir.exists()

    def test_get_settings_cached(self) -> None:
        """get_settings should return cached instance."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2


class TestDomainModels:
    """Domain model validation smoke tests."""

    def test_raw_review_valid(self) -> None:
        """RawReview should accept valid data."""
        review = RawReview(
            id=ReviewId("rev-001"),
            product_key=ProductKey("groww"),
            source="playstore",
            rating=5,
            body="Great app!",
            posted_at=datetime(2026, 4, 1, 0, 0, 0),
        )
        assert review.rating == 5
        assert review.source == "playstore"

    def test_raw_review_invalid_rating(self) -> None:
        """RawReview should reject invalid ratings."""
        with pytest.raises(ValidationError):
            RawReview(
                id=ReviewId("rev-001"),
                product_key=ProductKey("groww"),
                source="playstore",
                rating=6,
                body="Great app!",
                posted_at=datetime(2026, 4, 1, 0, 0, 0),
            )

    def test_window_frozen(self) -> None:
        """Window should be frozen/immutable."""
        window = Window(start=date(2026, 1, 1), end=date(2026, 1, 7), weeks=1)
        assert window.start == date(2026, 1, 1)

    def test_product_config(self) -> None:
        """ProductConfig should validate."""
        product = ProductConfig(
            key=ProductKey("groww"),
            display_name="Groww",
            appstore_id="12345",
            play_package="com.nextbillion.groww",
        )
        assert product.is_active is True
        assert product.play_package == "com.nextbillion.groww"


class TestTypes:
    """Type system smoke tests."""

    def test_branded_types_are_strings(self) -> None:
        """Branded types should behave as strings."""
        run_id: RunId = RunId("run-123")
        review_id: ReviewId = ReviewId("rev-456")
        assert isinstance(run_id, str)
        assert isinstance(review_id, str)
        assert str(run_id) != str(review_id)

    def test_branded_type_values(self) -> None:
        """Branded types carry their string values."""
        run_id: RunId = RunId("run-123")
        assert run_id == "run-123"


class TestCLI:
    """CLI smoke tests."""

    def test_cli_imports(self) -> None:
        """CLI module should import without errors."""
        from app.__main__ import app

        assert app is not None
