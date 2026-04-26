"""Helpers for persisting in-flight run stage changes."""

from app.db.models import AuditLog, Run
from app.db.session import db_session


async def persist_run_stage(run_id: str, status: str) -> None:
    """Update a run's current stage without disturbing other persisted fields."""
    async with db_session() as session:
        row = await session.get(Run, run_id)
        if row is None or row.status == status:
            return

        previous_status = row.status
        row.status = status
        session.add(
            AuditLog(
                run_id=run_id,
                event_type="run.stage_changed",
                event_data={
                    "from": previous_status,
                    "to": status,
                },
            )
        )