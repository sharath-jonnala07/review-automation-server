"""Reusable pipeline orchestration for CLI and API entry points."""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import structlog
from sqlalchemy import select

from app.agent.graph import build_graph
from app.agent.state import AgentState
from app.config import get_settings
from app.core.models import RawReview
from app.core.types import ProductKey, ReviewId
from app.db.models import AuditLog, Product, Review, Run, Theme
from app.db.session import db_session
from app.ingestion.appstore import fetch_appstore_reviews
from app.ingestion.playstore import fetch_playstore_reviews
from app.ingestion.scrubber import scrub_review_body
from app.services.review_selection import select_reviews_for_processing
from app.summarization.llm_client import LLMClient

logger = structlog.get_logger()

ACTIVE_RUN_STATUSES = {
    "pending",
    "ingesting",
    "clustering",
    "summarizing",
    "rendering",
    "publishing",
}


@dataclass(frozen=True)
class ResumableRun:
    """Persisted run metadata needed to resume work after a restart."""

    run_id: str
    product_key: str
    iso_week: str
    weeks: int
    dry_run: bool


def stable_review_id(source: str, external_id: str) -> ReviewId:
    """Generate a stable review ID from source and external ID."""
    digest = hashlib.sha1(f"{source}:{external_id}".encode()).hexdigest()[:16]
    return ReviewId(f"{source}-{digest}")


def run_id_for(product: str, iso_week: str) -> str:
    """Generate deterministic run ID."""
    return hashlib.sha1(f"{product}:{iso_week}".encode()).hexdigest()[:16]


def _resumable_run_from_row(row: Run) -> ResumableRun:
    """Extract restart-safe run arguments from a persisted run row."""
    metrics = row.metrics_json or {}
    weeks_raw = metrics.get("weeks", 10)
    dry_run_raw = metrics.get("dryRun", True)

    try:
        weeks = int(weeks_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        weeks = 10

    if isinstance(dry_run_raw, bool):
        dry_run = dry_run_raw
    else:
        dry_run = str(dry_run_raw).strip().lower() in {"1", "true", "yes", "on"}

    return ResumableRun(
        run_id=row.id,
        product_key=row.product_key,
        iso_week=row.iso_week,
        weeks=weeks,
        dry_run=dry_run,
    )


async def recover_interrupted_runs() -> list[ResumableRun]:
    """Reset and return unfinished runs so startup can resume them."""
    async with db_session() as session:
        result = await session.execute(
            select(Run)
            .where(Run.status.in_(ACTIVE_RUN_STATUSES))
            .order_by(Run.created_at.asc())
        )
        rows = result.scalars().all()
        resumable_runs: list[ResumableRun] = []

        for row in rows:
            previous_status = row.status
            resumable_runs.append(_resumable_run_from_row(row))
            row.status = "pending"
            row.error_message = None
            session.add(
                AuditLog(
                    run_id=row.id,
                    event_type="run.recovered",
                    event_data={"statusBeforeRestart": previous_status},
                )
            )

        return resumable_runs


def iso_week_for_date(dt: datetime) -> str:
    """Get ISO week string from datetime."""
    return dt.strftime("%G-W%V")


def window_for_weeks(weeks: int) -> tuple[date, date]:
    """Return the current trailing review window."""
    window_end = datetime.now(UTC).date()
    return window_end - timedelta(weeks=weeks), window_end


async def _audit(run_id: str, event_type: str, event_data: dict[str, object]) -> None:
    async with db_session() as session:
        session.add(AuditLog(run_id=run_id, event_type=event_type, event_data=event_data))


async def _upsert_run(
    *,
    run_id: str,
    product_key: str,
    iso_week: str,
    window_start: date,
    window_end: date,
    status: str,
    metrics: dict[str, object] | None = None,
    error_message: str | None = None,
    gdoc_id: str | None = None,
    gdoc_heading_id: str | None = None,
    gmail_message_id: str | None = None,
) -> None:
    async with db_session() as session:
        row = await session.get(Run, run_id)
        if row is None:
            row = Run(
                id=run_id,
                product_key=product_key,
                iso_week=iso_week,
                window_start=window_start,
                window_end=window_end,
            )
            session.add(row)

        row.status = status
        row.window_start = window_start
        row.window_end = window_end
        row.error_message = error_message
        if metrics is not None:
            row.metrics_json = metrics
        row.gdoc_id = gdoc_id
        row.gdoc_heading_id = gdoc_heading_id
        row.gmail_message_id = gmail_message_id


async def create_or_reset_run(
    product_key: str,
    iso_week: str | None = None,
    weeks: int = 10,
    dry_run: bool = True,
) -> str:
    """Create the database run row before background execution starts."""
    settings = get_settings()
    resolved_week = iso_week or iso_week_for_date(datetime.now(UTC))
    run_id = run_id_for(product_key, resolved_week)
    window_start, window_end = window_for_weeks(weeks)
    metrics = {
        "weeks": weeks,
        "dryRun": dry_run,
        "minReviews": settings.min_reviews_per_run,
        "reviewsIngested": 0,
        "clustersFormed": 0,
        "llmTokens": 0,
        "llmCostUsd": 0,
    }
    await _upsert_run(
        run_id=run_id,
        product_key=product_key,
        iso_week=resolved_week,
        window_start=window_start,
        window_end=window_end,
        status="pending",
        metrics=metrics,
        error_message=None,
    )
    await _audit(run_id, "run.queued", {"productKey": product_key, "isoWeek": resolved_week})
    return run_id


async def _get_product(product_key: str) -> Product:
    async with db_session() as session:
        row = await session.get(Product, product_key)
        if row is None:
            raise ValueError(f"Product '{product_key}' was not found")
        if not row.is_active:
            raise ValueError(f"Product '{product_key}' is not active")
        return Product(
            key=row.key,
            display_name=row.display_name,
            appstore_id=row.appstore_id,
            play_package=row.play_package,
            gdoc_id=row.gdoc_id,
            gmail_to=row.gmail_to,
            is_active=row.is_active,
        )


async def ingest_reviews_for_product(product: Product, weeks: int) -> int:
    """Fetch store reviews and persist newly seen rows."""
    all_reviews: list[RawReview] = []

    if product.appstore_id:
        all_reviews.extend(
            await fetch_appstore_reviews(ProductKey(product.key), product.appstore_id, weeks=weeks)
        )

    if product.play_package:
        all_reviews.extend(
            await fetch_playstore_reviews(ProductKey(product.key), product.play_package, weeks=weeks)
        )

    async with db_session() as session:
        inserted = 0
        for review in all_reviews:
            review_id = stable_review_id(review.source, review.id)
            existing = await session.get(Review, review_id)
            if existing is not None:
                continue
            session.add(
                Review(
                    id=review_id,
                    product_key=product.key,
                    source=review.source,
                    external_id=review.id,
                    rating=review.rating,
                    title=review.title,
                    body=scrub_review_body(review.body),
                    posted_at=review.posted_at,
                    version=review.version,
                    language=review.language,
                    country=review.country,
                    raw_json=review.model_dump_json(),
                )
            )
            inserted += 1

    if all_reviews:
        snapshot_id = f"ingest-{product.key}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        raw_path = Path(f"data/raw/{product.key}/{snapshot_id}.jsonl")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w", encoding="utf-8") as file:
            for review in all_reviews:
                file.write(json.dumps(review.model_dump(mode="json")) + "\n")

    return inserted


async def load_reviews(product_key: str) -> list[RawReview]:
    """Load persisted reviews in graph-ready form."""
    async with db_session() as session:
        result = await session.execute(select(Review).where(Review.product_key == product_key))
        rows = result.scalars().all()

    return [
        RawReview(
            id=ReviewId(row.id),
            product_key=ProductKey(row.product_key),
            source=row.source,  # type: ignore[arg-type]
            rating=row.rating,
            title=row.title,
            body=row.body,
            posted_at=row.posted_at,
            version=row.version,
            language=row.language,
            country=row.country,
        )
        for row in rows
    ]


def prepare_reviews_for_run(
    reviews: list[RawReview],
    *,
    max_reviews: int,
) -> tuple[list[RawReview], dict[str, int]]:
    """Filter and cap reviews before clustering and summarization."""
    selected_reviews, low_signal_dropped = select_reviews_for_processing(
        reviews,
        max_reviews=max_reviews,
    )
    informative_count = len(reviews) - low_signal_dropped
    return selected_reviews, {
        "reviewsAvailable": len(reviews),
        "reviewsSelected": len(selected_reviews),
        "lowSignalDropped": low_signal_dropped,
        "highSignalOverflow": max(0, informative_count - len(selected_reviews)),
    }


def metrics_for_reviews(
    reviews: list[RawReview], weeks: int, dry_run: bool, min_reviews: int
) -> dict[str, object]:
    """Build frontend-friendly run metrics."""
    appstore_reviews = sum(1 for review in reviews if review.source == "appstore")
    playstore_reviews = sum(1 for review in reviews if review.source == "playstore")
    avg_rating = round(sum(review.rating for review in reviews) / len(reviews), 2) if reviews else 0
    return {
        "weeks": weeks,
        "dryRun": dry_run,
        "minReviews": min_reviews,
        "reviewsIngested": len(reviews),
        "appstoreReviews": appstore_reviews,
        "playstoreReviews": playstore_reviews,
        "avgRating": avg_rating,
        "clustersFormed": 0,
        "llmTokens": 0,
        "llmCostUsd": 0,
    }


async def persist_summary(run_id: str, result_state: AgentState) -> None:
    """Persist themes generated by the graph."""
    summary = result_state.get("summary")
    if summary is None:
        return

    async with db_session() as session:
        existing = await session.execute(select(Theme).where(Theme.run_id == run_id))
        for row in existing.scalars().all():
            await session.delete(row)

        for theme in summary.top_themes:
            actions = [
                {"title": action.title, "description": action.description}
                for action in summary.action_ideas
                if action.theme_id == theme.id
            ]
            session.add(
                Theme(
                    id=f"{run_id}-{theme.id}",
                    run_id=run_id,
                    rank=theme.rank,
                    label=theme.label,
                    description=theme.description,
                    sentiment=theme.sentiment,
                    review_count=theme.review_count,
                    representative_review_ids_json=[str(review_id) for review_id in theme.representative_review_ids],
                    action_ideas_json=actions,
                )
            )


async def run_product_pipeline(
    product_key: str,
    iso_week: str | None = None,
    weeks: int = 10,
    dry_run: bool = True,
) -> None:
    """Run ingestion, validation, graph execution, and persistence."""
    settings = get_settings()
    resolved_week = iso_week or iso_week_for_date(datetime.now(UTC))
    run_id = run_id_for(product_key, resolved_week)
    window_start, window_end = window_for_weeks(weeks)
    started_at = datetime.now(UTC)

    try:
        product = await _get_product(product_key)
        llm_backend = await LLMClient().ensure_ready()
        await _audit(run_id, "llm.ready", {"backend": llm_backend})
        await _upsert_run(
            run_id=run_id,
            product_key=product_key,
            iso_week=resolved_week,
            window_start=window_start,
            window_end=window_end,
            status="ingesting",
            metrics={"weeks": weeks, "dryRun": dry_run, "minReviews": settings.min_reviews_per_run},
        )
        await _audit(run_id, "ingest.started", {"weeks": weeks})

        inserted = await ingest_reviews_for_product(product, weeks)
        all_reviews = await load_reviews(product_key)
        selected_reviews, selection_metrics = prepare_reviews_for_run(
            all_reviews,
            max_reviews=settings.max_reviews_per_run,
        )
        metrics = metrics_for_reviews(all_reviews, weeks, dry_run, settings.min_reviews_per_run)
        metrics.update(selection_metrics)
        metrics["llmBackend"] = llm_backend
        metrics["maxReviews"] = settings.max_reviews_per_run
        metrics["reviewsInserted"] = inserted
        await _upsert_run(
            run_id=run_id,
            product_key=product_key,
            iso_week=resolved_week,
            window_start=window_start,
            window_end=window_end,
            status="clustering",
            metrics=metrics,
        )
        await _audit(
            run_id,
            "ingest.completed",
            {
                "reviewsAvailable": len(all_reviews),
                "reviewsSelected": len(selected_reviews),
                "inserted": inserted,
                "lowSignalDropped": selection_metrics["lowSignalDropped"],
                "highSignalOverflow": selection_metrics["highSignalOverflow"],
            },
        )

        if len(selected_reviews) < settings.min_reviews_per_run:
            message = (
                f"Only {len(selected_reviews)} quality reviews available. At least "
                f"{settings.min_reviews_per_run} reviews are required to run Pulse."
            )
            await _upsert_run(
                run_id=run_id,
                product_key=product_key,
                iso_week=resolved_week,
                window_start=window_start,
                window_end=window_end,
                status="failed",
                metrics=metrics,
                error_message=message,
            )
            await _audit(run_id, "run.failed", {"reason": message})
            return

        initial_state: AgentState = {
            "run_id": run_id,
            "product_key": product_key,
            "iso_week": resolved_week,
            "window_start": window_start,
            "window_end": window_end,
            "reviews": selected_reviews,
            "status": "pending",
            "retry_count": 0,
            "llm_tokens_used": 0,
            "llm_cost_usd": 0.0,
            "skip_ingest": True,
            "dry_run": dry_run,
            "gmail_to": product.gmail_to,
            "configured_gdoc_id": product.gdoc_id,
        }

        graph = await build_graph()
        result_state = await graph.ainvoke(
            initial_state,
            {"configurable": {"thread_id": run_id}},
        )
        metrics["clustersFormed"] = len(result_state.get("clusters", []))
        metrics["llmTokens"] = result_state.get("llm_tokens_used", 0)
        metrics["llmCostUsd"] = result_state.get("llm_cost_usd", 0.0)
        metrics["durationSeconds"] = round((datetime.now(UTC) - started_at).total_seconds(), 2)
        metrics["docPublished"] = bool(result_state.get("gdoc_id"))
        metrics["emailSent"] = bool(result_state.get("gmail_message_id"))

        await persist_summary(run_id, result_state)
        doc_publish_error = result_state.get("doc_publish_error")
        error_message = result_state.get("error")
        if not error_message and doc_publish_error:
            error_message = f"Docs publish warning: {doc_publish_error}"
        await _upsert_run(
            run_id=run_id,
            product_key=product_key,
            iso_week=resolved_week,
            window_start=window_start,
            window_end=window_end,
            status="completed" if not result_state.get("error") else "failed",
            metrics=metrics,
            error_message=error_message,
            gdoc_id=result_state.get("gdoc_id"),
            gdoc_heading_id=result_state.get("gdoc_heading_id"),
            gmail_message_id=result_state.get("gmail_message_id"),
        )
        await _audit(
            run_id,
            "run.completed",
            {
                "dryRun": dry_run,
                "docPublished": bool(result_state.get("gdoc_id")),
                "emailSent": bool(result_state.get("gmail_message_id")),
                "docPublishError": doc_publish_error,
            },
        )
    except Exception as exc:
        logger.exception("Pipeline failed", run_id=run_id, product_key=product_key)
        await _upsert_run(
            run_id=run_id,
            product_key=product_key,
            iso_week=resolved_week,
            window_start=window_start,
            window_end=window_end,
            status="failed",
            metrics={"weeks": weeks, "dryRun": dry_run, "minReviews": settings.min_reviews_per_run},
            error_message=str(exc),
        )
        await _audit(run_id, "run.failed", {"reason": str(exc)})