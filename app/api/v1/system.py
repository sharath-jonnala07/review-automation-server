"""System readiness endpoints."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.config import Settings, get_settings

router = APIRouter()


class SystemReadiness(BaseModel):
    """Frontend-safe system readiness state."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    database: bool
    llm_ready: bool
    llm_provider: str
    groq_key_present: bool
    openai_key_present: bool
    docs_mcp_configured: bool
    gmail_mcp_configured: bool
    confirm_send: bool
    min_reviews_per_run: int
    max_reviews_per_run: int
    llm_max_cost_usd: float
    llm_model: str
    embedding_backend: str
    embedding_model: str
    heuristic_llm_enabled: bool
    openai_fallback_configured: bool


@router.get("/readiness", response_model=SystemReadiness)
async def readiness(
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> SystemReadiness:
    """Return configuration readiness without exposing secret values."""
    database_ready = True
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        database_ready = False

    return SystemReadiness(
        database=database_ready,
        llm_ready=settings.llm_ready,
        llm_provider=settings.preferred_llm_provider,
        groq_key_present=bool(settings.groq_api_key),
        openai_key_present=bool(settings.openai_api_key),
        docs_mcp_configured=bool(settings.docs_mcp_url),
        gmail_mcp_configured=bool(settings.gmail_mcp_url),
        confirm_send=settings.confirm_send,
        min_reviews_per_run=settings.min_reviews_per_run,
        max_reviews_per_run=settings.max_reviews_per_run,
        llm_max_cost_usd=settings.llm_max_cost_usd,
        llm_model=settings.llm_model,
        embedding_backend=settings.embedding_backend,
        embedding_model=settings.embedding_model,
        heuristic_llm_enabled=settings.heuristic_llm_enabled,
        openai_fallback_configured=bool(settings.openai_api_key),
    )