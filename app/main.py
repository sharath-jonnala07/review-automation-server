"""FastAPI application factory."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import get_settings, sync_products_config
from app.db.session import init_db
from app.services.pipeline import recover_interrupted_runs, run_product_pipeline

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    """Application lifespan events."""
    # Startup
    settings = get_settings()
    recovery_tasks: set[asyncio.Task[None]] = set()
    app.state.recovery_tasks = recovery_tasks
    logger.info(
        "Starting Pulse Agent",
        version=settings.app_version,
        debug=settings.debug,
    )
    await init_db()
    await sync_products_config()
    logger.info("Database initialized")

    resumable_runs = await recover_interrupted_runs()
    if resumable_runs:
        logger.warning(
            "Resuming interrupted runs after restart",
            count=len(resumable_runs),
            run_ids=[run.run_id for run in resumable_runs],
        )
        for resumable_run in resumable_runs:
            task = asyncio.create_task(
                run_product_pipeline(
                    resumable_run.product_key,
                    resumable_run.iso_week,
                    resumable_run.weeks,
                    resumable_run.dry_run,
                ),
                name=f"resume-{resumable_run.run_id}",
            )
            recovery_tasks.add(task)
            task.add_done_callback(recovery_tasks.discard)

    yield
    # Shutdown
    for task in list(recovery_tasks):
        if not task.done():
            task.cancel()
    logger.info("Shutting down Pulse Agent")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    allowed_origins = settings.allowed_origins or [
        "http://localhost:3000",
        "https://localhost:3000",
        "http://localhost:3001",
        "https://localhost:3001",
    ]

    app = FastAPI(
        title="Pulse Agent API",
        description="Weekly Product Review Pulse — AI Agent",
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check
    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": settings.app_version}

    # API routes
    app.include_router(api_router)

    return app


app = create_app()
