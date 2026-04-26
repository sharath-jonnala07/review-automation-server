"""Runs API endpoints."""

from datetime import date, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.db.models import AuditLog as AuditLogORM
from app.db.models import Product as ProductORM
from app.db.models import Run as RunORM
from app.db.models import Theme as ThemeORM
from app.services.pipeline import create_or_reset_run, run_product_pipeline

router = APIRouter()


class RunCreatePayload(BaseModel):
    """Start-run request."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    product_key: str = Field(..., min_length=1)
    iso_week: str | None = Field(default=None, pattern=r"^\d{4}-W\d{2}$")
    weeks: int = Field(default=10, ge=1, le=52)
    dry_run: bool = True


class RunResponse(BaseModel):
    """Run API response."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: str
    product_key: str
    product_name: str | None = None
    iso_week: str
    status: str
    metrics: dict[str, object]
    gdoc_id: str | None = None
    gdoc_heading_id: str | None = None
    gmail_message_id: str | None = None
    error_message: str | None = None
    window_start: date
    window_end: date
    created_at: datetime
    updated_at: datetime


class ThemeResponse(BaseModel):
    """Theme response."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: str
    rank: int
    label: str
    description: str
    sentiment: str
    review_count: int
    representative_review_ids: list[str]
    action_ideas: list[dict[str, str]]
    created_at: datetime


class AuditResponse(BaseModel):
    """Audit log response."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: int
    run_id: str | None
    event_type: str
    event_data: dict[str, object]
    created_at: datetime


def _run_response(row: RunORM, product_name: str | None = None) -> RunResponse:
    return RunResponse(
        id=row.id,
        product_key=row.product_key,
        product_name=product_name,
        iso_week=row.iso_week,
        status=row.status,
        metrics=row.metrics_json or {},
        gdoc_id=row.gdoc_id,
        gdoc_heading_id=row.gdoc_heading_id,
        gmail_message_id=row.gmail_message_id,
        error_message=row.error_message,
        window_start=row.window_start,
        window_end=row.window_end,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.get("", response_model=list[RunResponse])
async def list_runs(
    product_key: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[RunResponse]:
    """List pulse runs, optionally filtered by product."""
    query = select(RunORM, ProductORM.display_name).join(ProductORM, RunORM.product_key == ProductORM.key)
    if product_key:
        query = query.where(RunORM.product_key == product_key)
    query = query.order_by(RunORM.created_at.desc())
    result = await db.execute(query)
    return [_run_response(row, product_name) for row, product_name in result.all()]


@router.post("", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_run(
    payload: RunCreatePayload,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> RunResponse:
    """Start a pipeline run in the background."""
    product = await db.get(ProductORM, payload.product_key)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    if not product.is_active:
        raise HTTPException(status_code=409, detail="Product is inactive")

    run_id = await create_or_reset_run(
        payload.product_key,
        payload.iso_week,
        payload.weeks,
        payload.dry_run,
    )
    background_tasks.add_task(
        run_product_pipeline,
        payload.product_key,
        payload.iso_week,
        payload.weeks,
        payload.dry_run,
    )
    row = await db.get(RunORM, run_id)
    if row is None:
        raise HTTPException(status_code=500, detail="Run could not be created")
    return _run_response(row, product.display_name)


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> RunResponse:
    """Get a single run by ID."""
    result = await db.execute(
        select(RunORM, ProductORM.display_name)
        .join(ProductORM, RunORM.product_key == ProductORM.key)
        .where(RunORM.id == run_id)
    )
    record = result.one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    row, product_name = record
    return _run_response(row, product_name)


@router.get("/{run_id}/status", response_model=RunResponse)
async def get_run_status(
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> RunResponse:
    """Poll a single run status."""
    return await get_run(run_id, db)


@router.post("/{run_id}/retry", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
async def retry_run(
    run_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> RunResponse:
    """Retry a failed or completed run using its original product/week."""
    row = await db.get(RunORM, run_id)
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    weeks = int((row.metrics_json or {}).get("weeks", 10))
    dry_run = bool((row.metrics_json or {}).get("dryRun", True))
    await create_or_reset_run(row.product_key, row.iso_week, weeks, dry_run)
    background_tasks.add_task(run_product_pipeline, row.product_key, row.iso_week, weeks, dry_run)
    product = await db.get(ProductORM, row.product_key)
    refreshed = await db.get(RunORM, run_id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Run could not be retried")
    return _run_response(refreshed, product.display_name if product else None)


@router.get("/{run_id}/themes", response_model=list[ThemeResponse])
async def list_run_themes(
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> list[ThemeResponse]:
    """List generated themes for a run."""
    result = await db.execute(
        select(ThemeORM).where(ThemeORM.run_id == run_id).order_by(ThemeORM.rank.asc())
    )
    return [
        ThemeResponse(
            id=row.id,
            rank=row.rank,
            label=row.label,
            description=row.description,
            sentiment=row.sentiment,
            review_count=row.review_count,
            representative_review_ids=row.representative_review_ids_json or [],
            action_ideas=row.action_ideas_json or [],
            created_at=row.created_at,
        )
        for row in result.scalars().all()
    ]


@router.get("/{run_id}/audit", response_model=list[AuditResponse])
async def list_run_audit(
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> list[AuditResponse]:
    """List audit events for a run."""
    result = await db.execute(
        select(AuditLogORM)
        .where(AuditLogORM.run_id == run_id)
        .order_by(AuditLogORM.created_at.asc(), AuditLogORM.id.asc())
    )
    return [
        AuditResponse(
            id=row.id,
            run_id=row.run_id,
            event_type=row.event_type,
            event_data=row.event_data,
            created_at=row.created_at,
        )
        for row in result.scalars().all()
    ]
