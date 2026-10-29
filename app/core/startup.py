"""Application startup tasks."""

from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.core.security import ensure_systemadmin
from app.core.workspace import current_workspace
from app.db.database import init_db
from app.services import ingest_jobs
from app.services.assessment import runner as assessment_runner
from app.services.documents import resync_approved_faqs


def on_startup() -> None:
    """Prepare the database, administrator, indexes, and upload directory."""
    init_db()
    ensure_systemadmin()
    resync_approved_faqs()
    ingest_jobs.recover()
    ingest_jobs.start()
    assessment_runner.recover()
    (Path(settings.upload_dir) / str(current_workspace())).mkdir(parents=True, exist_ok=True)
