from datetime import date

from app.db.models import Run
from app.services.pipeline import _resumable_run_from_row


def test_resumable_run_from_row_extracts_run_arguments() -> None:
    row = Run(
        id="run-123",
        product_key="groww",
        iso_week="2026-W17",
        window_start=date(2026, 4, 20),
        window_end=date(2026, 4, 26),
        status="publishing",
        metrics_json={"weeks": "8", "dryRun": False},
    )

    resumable = _resumable_run_from_row(row)

    assert resumable.run_id == "run-123"
    assert resumable.product_key == "groww"
    assert resumable.iso_week == "2026-W17"
    assert resumable.weeks == 8
    assert resumable.dry_run is False


def test_resumable_run_from_row_falls_back_to_safe_defaults() -> None:
    row = Run(
        id="run-456",
        product_key="indmoney",
        iso_week="2026-W18",
        window_start=date(2026, 4, 27),
        window_end=date(2026, 5, 3),
        status="pending",
        metrics_json={"weeks": "invalid", "dryRun": "yes"},
    )

    resumable = _resumable_run_from_row(row)

    assert resumable.weeks == 10
    assert resumable.dry_run is True