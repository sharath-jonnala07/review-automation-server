"""Application configuration loaded from environment."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.models import ProductConfig
from app.core.types import ProductKey


class Settings(BaseSettings):
    """Application settings with validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="pulse-agent")
    app_version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)
    allowed_origins_csv: str | None = Field(default=None, alias="ALLOWED_ORIGINS")
    allowed_origin_regex: str | None = Field(default=None, alias="ALLOWED_ORIGIN_REGEX")

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value: object) -> object:
        """Accept common deployment strings for DEBUG."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "production", "prod"}:
                return False
            if normalized in {"development", "dev"}:
                return True
        return value

    @field_validator("llm_provider", mode="before")
    @classmethod
    def parse_llm_provider(cls, value: object) -> object:
        """Normalize supported LLM provider names."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            aliases = {
                "": "auto",
                "default": "auto",
                "automatic": "auto",
                "openai-compatible": "custom",
            }
            return aliases.get(normalized, normalized)
        return value

    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/pulse.db",
        alias="DATABASE_URL",
    )

    # LLM via Groq / OpenAI-compatible APIs
    llm_provider: str = Field(default="auto", alias="LLM_PROVIDER")
    llm_api_key: str | None = Field(default=None, alias="LLM_API_KEY")
    llm_base_url: str | None = Field(default=None, alias="LLM_BASE_URL")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL"
    )
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    llm_model: str = Field(default="llama-3.3-70b-versatile", alias="LLM_MODEL")
    llm_max_cost_usd: float = Field(default=0.5, alias="LLM_MAX_COST_USD")
    llm_temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE")
    llm_allow_heuristic_fallback: bool = Field(
        default=False,
        alias="LLM_ALLOW_HEURISTIC_FALLBACK",
    )
    llm_allow_heuristic_auth_fallback: bool = Field(
        default=False,
        alias="LLM_ALLOW_HEURISTIC_AUTH_FALLBACK",
    )

    # Embeddings
    embedding_backend: str = Field(default="huggingface-local", alias="EMBEDDING_BACKEND")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    huggingface_api_key: str | None = Field(default=None, alias="HUGGINGFACE_API_KEY")
    huggingface_api_url: str = Field(
        default="https://router.huggingface.co/hf-inference/models",
        alias="HUGGINGFACE_API_URL",
    )
    embedding_model: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        alias="EMBEDDING_MODEL",
    )

    # Pipeline controls
    min_reviews_per_run: int = Field(default=200, alias="MIN_REVIEWS_PER_RUN")
    max_reviews_per_run: int = Field(default=200, alias="MAX_REVIEWS_PER_RUN")

    # MCP Servers
    docs_mcp_url: str | None = Field(default=None, alias="DOCS_MCP_URL")
    gmail_mcp_url: str | None = Field(default=None, alias="GMAIL_MCP_URL")
    confirm_send: bool = Field(default=False, alias="CONFIRM_SEND")

    # Observability
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="pulse-agent", alias="LANGSMITH_PROJECT")
    langsmith_tracing: bool = Field(default=True, alias="LANGSMITH_TRACING")
    otlp_endpoint: str | None = Field(default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT")

    # Paths
    data_dir: Path = Field(default=Path("./data"))
    raw_data_dir: Path = Field(default=Path("./data/raw"))
    artifacts_dir: Path = Field(default=Path("./data/artifacts"))

    @field_validator("embedding_backend", mode="before")
    @classmethod
    def parse_embedding_backend(cls, value: object) -> object:
        """Normalize supported embedding backend names."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            aliases = {
                "huggingface": "huggingface-local",
                "huggingface-api": "huggingface-api",
                "huggingface-inference": "huggingface-api",
                "hf": "huggingface-local",
                "hf-api": "huggingface-api",
                "hf-inference": "huggingface-api",
                "local": "huggingface-local",
                "sentence-transformers": "huggingface-local",
                "inference-api": "huggingface-api",
                "openai": "openai",
            }
            return aliases.get(normalized, normalized)
        return value

    def model_post_init(self, __context: object) -> None:
        """Ensure directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def heuristic_llm_enabled(self) -> bool:
        """Return whether deterministic local summarization is allowed."""
        return self.llm_allow_heuristic_fallback or self.debug

    @property
    def allowed_origins(self) -> list[str]:
        """Return allowed CORS origins from a comma-separated env var."""
        if not self.allowed_origins_csv:
            return []
        return [origin.strip() for origin in self.allowed_origins_csv.split(",") if origin.strip()]

    @property
    def preferred_llm_provider(self) -> Literal[
        "auto",
        "custom",
        "groq",
        "heuristic",
        "openai",
        "unconfigured",
    ]:
        """Return the provider that should be tried first."""
        if self.llm_provider != "auto":
            if self.llm_provider == "custom" and self.llm_api_key:
                return "custom"
            if self.llm_provider == "groq" and self.groq_api_key:
                return "groq"
            if self.llm_provider == "openai" and self.openai_api_key:
                return "openai"
            if self.heuristic_llm_enabled:
                return "heuristic"
            return "unconfigured"

        if self.llm_api_key:
            return "custom"

        model_name = self.llm_model.strip().lower()
        if model_name.startswith(("gpt", "o1", "o3", "o4")) and self.openai_api_key:
            return "openai"
        if self.groq_api_key:
            return "groq"
        if self.openai_api_key:
            return "openai"
        if self.heuristic_llm_enabled:
            return "heuristic"
        return "unconfigured"

    @property
    def llm_ready(self) -> bool:
        """Return whether at least one LLM execution path is configured."""
        return self.preferred_llm_provider != "unconfigured"


def load_products_config(path: Path | None = None) -> list[ProductConfig]:
    """Load product configurations from YAML file."""
    if path is None:
        path = Path("data/products.yaml")

    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    products = data.get("products", [])
    return [
        ProductConfig(
            key=ProductKey(p["key"]),
            display_name=p["display_name"],
            appstore_id=p.get("appstore_id") or None,
            play_package=p.get("play_package") or None,
            gdoc_id=p.get("gdoc_id") or None,
            gmail_to=p.get("gmail_to") or None,
            is_active=p.get("is_active", True),
        )
        for p in products
    ]


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()


async def sync_products_config() -> None:
    """Upsert product metadata from YAML into the live database."""
    from sqlalchemy import select

    from app.db.models import Product as ProductORM
    from app.db.session import db_session

    products = load_products_config()
    if not products:
        return

    async with db_session() as session:
        for product in products:
            existing = await session.execute(
                select(ProductORM).where(ProductORM.key == product.key)
            )
            row = existing.scalar_one_or_none()
            if row is None:
                session.add(
                    ProductORM(
                        key=product.key,
                        display_name=product.display_name,
                        appstore_id=product.appstore_id,
                        play_package=product.play_package,
                        gdoc_id=product.gdoc_id,
                        gmail_to=product.gmail_to,
                        is_active=product.is_active,
                    )
                )
                continue

            row.display_name = product.display_name
            row.appstore_id = product.appstore_id
            row.play_package = product.play_package
            row.gdoc_id = product.gdoc_id
            row.gmail_to = product.gmail_to
            row.is_active = product.is_active
