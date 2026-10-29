"""Read an uploaded file in the background and report the job it came from.

The request only validates the file, writes it and queues the job. The worker
opens its own session, reads the stored file, finds the passages that are not
already in the workspace, embeds them a batch at a time and records the result
on the job row. A stopped or failed run leaves no rows behind, and the file
stays on disk so a retry can read it again.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException, UploadFile
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.core.extraction import extract_document
from app.core.workspace import current_workspace
from app.db import database
from app.db.database import Document, IngestJob, UploadedFile, User
from app.services import chunk_store, documents, ingest_jobs, vector_store
from app.services.chunk_store import IngestCancelledError
from app.services.dedupe import Collapse
from app.services.quality.pipeline import prepare_document

logger = logging.getLogger(__name__)
SAFE_FAILURE = "The file could not be indexed. Retry or upload it again."


def queue_upload(user: User, upload: UploadFile, db: Session) -> dict:
    """Validate the upload, queue it, and answer at once with the job id."""
    workspace = current_workspace()
    try:
        accepted = documents.accept_upload(db, workspace, upload)
    except documents.DuplicateUploadError as duplicate:
        return {**duplicate.response, "job_id": None}
    db.add(
        UploadedFile(
            source=accepted.filename,
            content_hash=accepted.digest,
            size_bytes=len(accepted.content),
            tenant_id=workspace,
        )
    )
    job = IngestJob(
        source=accepted.filename,
        tenant_id=workspace,
        status="queued",
        step=ingest_jobs.WAITING,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    ingest_jobs.submit(job.id, lambda: run_upload_job(job.id))
    return {"job_id": job.id, "source": accepted.filename, "status": "queued"}


def retry_upload(job_id: int, tenant_id: int | None) -> dict:
    """Queue a failed or stopped upload again, then return the new state."""
    ingest_jobs.reset(job_id, tenant_id)
    ingest_jobs.submit(job_id, lambda: run_upload_job(job_id))
    return ingest_jobs.snapshot(job_id, tenant_id)


def run_upload_job(job_id: int) -> None:
    """Read the stored file and store its passages, reporting on the job row."""
    with ingest_jobs.run_lock(job_id):
        _run_once(job_id)


def _run_once(job_id: int) -> None:
    """One attempt at the file, holding the job to itself for the whole read."""
    with database.SessionLocal() as db:
        job = db.get(IngestJob, job_id)
        if job is None or job.status != "queued":
            return
        ingest_jobs.begin(job_id)
        job.status = "running"
        job.step = "Reading the file"
        job.started_at = datetime.utcnow()
        db.commit()
        try:
            stored = _index(db, job)
        except IngestCancelledError:
            db.rollback()
            _finish(job_id, "cancelled", "Stopped by the trainer.")
        except HTTPException as error:
            db.rollback()
            _finish(job_id, "failed", "Failed", error=str(error.detail))
        except Exception:
            # The worker boundary: any unplanned failure must end on the job row
            # as a retryable error rather than disappear on the thread.
            db.rollback()
            logger.exception("Ingest job %s failed", job_id)
            _finish(job_id, "failed", "Failed", error=SAFE_FAILURE)
        else:
            db.commit()
            _finish(job_id, "done", "Finished", stored=stored)


def _index(db: Session, job: IngestJob) -> Collapse:
    """Clear an earlier attempt, then read, chunk and embed the stored file."""
    _clear(db, job.tenant_id, job.source)
    ingest_jobs.publish(job.id, "Reading the file", 0, 0)
    extracted = prepare_document(extract_document(str(_path(job))))
    ingest_jobs.publish(job.id, "Finding the passages", 0, 0)
    record = _record(db, job)
    if record is None:
        raise HTTPException(404, "The uploaded file is missing.")
    return chunk_store.store_chunks(
        chunk_store.Sink(
            db=db,
            workspace=job.tenant_id,
            source=job.source,
            record=record,
            progress=lambda done, total: ingest_jobs.publish(
                job.id, "Storing the passages", done, total
            ),
            cancelled=lambda: ingest_jobs.was_cancelled(job.id),
        ),
        extracted,
    )


def _clear(db: Session, tenant_id: int | None, source: str) -> None:
    """Remove what an interrupted attempt may have left under this source."""
    condition = Document.source == source
    if tenant_id is not None:
        condition = condition & (Document.tenant_id == tenant_id)
    db.execute(delete(Document).where(condition))
    db.commit()
    vector_store.delete_chunks_by_source(source, tenant_id)


def _record(db: Session, job: IngestJob) -> UploadedFile | None:
    statement = select(UploadedFile).where(UploadedFile.source == job.source)
    if job.tenant_id is not None:
        statement = statement.where(UploadedFile.tenant_id == job.tenant_id)
    return db.execute(statement).scalars().first()


def _path(job: IngestJob) -> Path:
    workspace = 0 if job.tenant_id is None else job.tenant_id
    return Path(settings.upload_dir) / str(workspace) / Path(job.source).name


def _finish(
    job_id: int,
    status: str,
    step: str,
    stored: Collapse | None = None,
    error: str | None = None,
) -> None:
    """Record the terminal state of the job and drop its live progress."""
    with database.SessionLocal() as db:
        job = db.get(IngestJob, job_id)
        if job is None or job.status == "queued":
            # A retry has taken the job over, so its own run is the one that reports.
            return
        job.status = status
        job.step = step
        job.error = error
        job.finished_at = datetime.utcnow()
        if stored is not None:
            job.chunks = stored.stored
            job.exact_duplicates = stored.exact
            job.near_duplicates = stored.near
            job.done = stored.stored
            job.total = stored.stored
            job.progress = 1.0
        db.commit()
    ingest_jobs.finish_live(job_id)
