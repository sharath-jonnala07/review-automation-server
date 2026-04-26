"""CLI entry point for the Pulse Agent."""

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import typer
from sqlalchemy import select

from app.agent.graph import build_graph
from app.config import load_products_config, sync_products_config

if TYPE_CHECKING:
    from app.agent.state import AgentState
from app.core.types import ProductKey, ReviewId
from app.db.models import Product as ProductORM
from app.db.models import Review as ReviewORM
from app.db.models import Run as RunORM
from app.db.session import db_session, init_db
from app.ingestion.appstore import fetch_appstore_reviews
from app.ingestion.playstore import fetch_playstore_reviews
from app.ingestion.scrubber import scrub_review_body
from app.services.pipeline import iso_week_for_date, run_id_for, run_product_pipeline

if TYPE_CHECKING:
    from app.core.models import RawReview

logger = structlog.get_logger()
app = typer.Typer(
    name="pulse",
    help="Weekly Product Review Pulse - AI Agent CLI",
    no_args_is_help=True,
)


def _stable_id(source: str, external_id: str) -> ReviewId:
    """Generate a stable review ID from source and external ID."""
    h = hashlib.sha1(f"{source}:{external_id}".encode()).hexdigest()[:16]
    return ReviewId(f"{source}-{h}")


def _run_id(product: str, iso_week: str) -> str:
    """Generate deterministic run ID."""
    return run_id_for(product, iso_week)


def _iso_week_for_date(dt: datetime) -> str:
    """Get ISO week string from datetime."""
    return iso_week_for_date(dt)


@app.command()
def init_db_cmd() -> None:
    """Initialize the SQLite database with all tables."""

    async def _init() -> None:
        await init_db()
        products = load_products_config()
        async with db_session() as session:
            for product in products:
                existing = await session.execute(
                    select(ProductORM).where(ProductORM.key == product.key)
                )
                if existing.scalar_one_or_none() is None:
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
                    logger.info("Seeded product", product=product.key)
        typer.echo("Database initialized successfully.")

    asyncio.run(_init())


@app.command()
def ingest(
    product: str = typer.Option(..., "--product", "-p", help="Product key to ingest"),
    weeks: int = typer.Option(10, "--weeks", "-w", help="Number of weeks to look back"),
) -> None:
    """Ingest reviews from App Store and Play Store."""

    async def _ingest() -> None:
        products = load_products_config()
        product_config = next(
            (p for p in products if p.key == product), None
        )
        if not product_config:
            typer.echo(f"Product '{product}' not found in config", err=True)
            raise typer.Exit(1)

        all_reviews: list[RawReview] = []

        if product_config.appstore_id:
            typer.echo(f"Fetching App Store reviews for {product}...")
            appstore_reviews = await fetch_appstore_reviews(
                ProductKey(product),
                product_config.appstore_id,
                weeks=weeks,
            )
            all_reviews.extend(appstore_reviews)
            typer.echo(f"  -> {len(appstore_reviews)} App Store reviews")

        if product_config.play_package:
            typer.echo(f"Fetching Play Store reviews for {product}...")
            playstore_reviews = await fetch_playstore_reviews(
                ProductKey(product),
                product_config.play_package,
                weeks=weeks,
            )
            all_reviews.extend(playstore_reviews)
            typer.echo(f"  -> {len(playstore_reviews)} Play Store reviews")

        async with db_session() as session:
            inserted = 0
            for review in all_reviews:
                scrubbed_body = scrub_review_body(review.body)
                review_id = _stable_id(review.source, review.id)

                existing = await session.execute(
                    select(ReviewORM).where(ReviewORM.id == review_id)
                )
                if existing.scalar_one_or_none() is not None:
                    continue

                session.add(
                    ReviewORM(
                        id=review_id,
                        product_key=product,
                        source=review.source,
                        external_id=review.id,
                        rating=review.rating,
                        title=review.title,
                        body=scrubbed_body,
                        posted_at=review.posted_at,
                        version=review.version,
                        language=review.language,
                        country=review.country,
                        raw_json=review.model_dump_json(),
                    )
                )
                inserted += 1

        typer.echo(f"Total: {len(all_reviews)} reviews, {inserted} new inserted")

        run_id = f"ingest-{product}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        raw_path = Path(f"data/raw/{product}/{run_id}.jsonl")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w", encoding="utf-8") as f:
            for review in all_reviews:
                f.write(json.dumps(review.model_dump(mode="json")) + "\n")
        typer.echo(f"Raw snapshot saved to {raw_path}")

    asyncio.run(_ingest())


@app.command()
def run_pipeline(
    product: str = typer.Option(..., "--product", "-p", help="Product key"),
    week: str | None = typer.Option(None, "--week", help="ISO week (e.g. 2026-W16)"),
    weeks: int = typer.Option(10, "--weeks", "-w", help="Ingestion window in weeks"),
    _dry_run: bool = typer.Option(False, "--dry-run", help="Skip MCP publish"),
) -> None:
    """Run the full pipeline: ingest -> cluster -> summarize -> render -> publish."""

    async def _run() -> None:
        await init_db()
        await sync_products_config()
        iso_week = week or _iso_week_for_date(datetime.now(UTC))
        run_id = _run_id(product, iso_week)

        typer.echo(f"Running pipeline for {product} week {iso_week} (run_id={run_id})")
        await run_product_pipeline(product, iso_week, weeks, _dry_run)

        async with db_session() as session:
            run = await session.get(RunORM, run_id)
            if run is None:
                typer.echo("Pipeline finished without a persisted run", err=True)
                raise typer.Exit(1)
            typer.echo(f"Pipeline complete. Status: {run.status}")
            if run.error_message:
                typer.echo(f"  Error: {run.error_message}")
            if run.gdoc_id:
                typer.echo(f"  Doc: {run.gdoc_id}")
            if run.gmail_message_id:
                typer.echo(f"  Email: {run.gmail_message_id}")

    asyncio.run(_run())


@app.command()
def backfill(
    product: str = typer.Option(..., "--product", "-p", help="Product key"),
    week: str = typer.Option(..., "--week", help="ISO week to backfill"),
) -> None:
    """Backfill a specific ISO week (idempotent)."""
    # Re-use run_pipeline with explicit week
    asyncio.run(_run_backfill(product, week))


async def _run_backfill(product: str, week: str) -> None:
    await init_db()
    await sync_products_config()
    run_id = _run_id(product, week)
    typer.echo(f"Backfilling {product} for week {week} (run_id={run_id})")

    async with db_session() as session:
        existing = await session.execute(
            select(RunORM).where(RunORM.id == run_id)
        )
        if existing.scalar_one_or_none():
            typer.echo(f"Run {run_id} already exists. Skipping.")
            return

    await run_product_pipeline(product, week, 10, True)
    async with db_session() as session:
        run = await session.get(RunORM, run_id)
        typer.echo(f"Backfill complete. Status: {run.status if run else 'unknown'}")


@app.command()
def status(
    product: str = typer.Option(..., "--product", "-p", help="Product key"),
) -> None:
    """Show latest run status for a product."""

    async def _status() -> None:
        async with db_session() as session:
            result = await session.execute(
                select(RunORM)
                .where(RunORM.product_key == product)
                .order_by(RunORM.created_at.desc())
                .limit(1)
            )
            run = result.scalar_one_or_none()
            if not run:
                typer.echo(f"No runs found for {product}")
                return
            typer.echo(f"Latest run: {run.id}")
            typer.echo(f"  Week: {run.iso_week}")
            typer.echo(f"  Status: {run.status}")
            typer.echo(f"  Doc: {run.gdoc_id or 'N/A'}")
            typer.echo(f"  Email: {run.gmail_message_id or 'N/A'}")

    asyncio.run(_status())


def main() -> None:
    """Main entry point."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    app()


if __name__ == "__main__":
    main()
