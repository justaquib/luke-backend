"""The queue that keeps reading a large upload off the request thread.

There is no broker: one bounded pool in this process reads the file, cuts it
into passages and stores them. The job row is what the page reads, while the
live step and the passages stored so far stay in memory, so the worker never
has to open a second write transaction to report itself. A cancel is a flag the
worker checks before each batch, and a retry starts the same file again. Only
one run of a job is in the file at a time: the stopped run keeps its cancel
flag until the retried run has taken the job over, so the two never write the
same source at once.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select

from app.config import settings
from app.db import database
from app.db.database import IngestJob

logger = logging.getLogger(__name__)

TERMINAL = frozenset({"done", "duplicate", "failed", "cancelled"})
RETRYABLE = frozenset({"failed", "cancelled"})
WAITING = "Waiting to start"
SAFE_FAILURE = "The file could not be indexed. Retry or upload it again."
INTERRUPTED = "Interrupted by a restart. Retry to read the file again."

# How often the event stream repeats the job, and how long it stays open.
POLL_SECONDS = 0.5
MAX_POLLS = 600
RECENT_JOBS = 20

_POOLS: list[ThreadPoolExecutor] = []
_LIVE: dict[int, dict[str, Any]] = {}
_STOPPED: set[int] = set()
_RUNNING: dict[int, list] = {}
_LOCK = threading.Lock()


def start() -> None:
    """Start the bounded pool once."""
    with _LOCK:
        if not _POOLS:
            _POOLS.append(
                ThreadPoolExecutor(
                    max_workers=max(1, settings.ingest_workers), thread_name_prefix="ingest"
                )
            )


def shutdown() -> None:
    """Stop accepting work and wait for the running jobs to finish."""
    with _LOCK:
        pool = _POOLS.pop() if _POOLS else None
    if pool is not None:
        pool.shutdown(wait=True)


def submit(job_id: int, work: Callable[[], None]) -> None:
    """Run one job on the pool, and mark it failed if the work escapes."""
    start()
    with _LOCK:
        pool = _POOLS[0] if _POOLS else None
    if pool is not None:
        pool.submit(_guarded, job_id, work)


def _guarded(job_id: int, work: Callable[[], None]) -> None:
    try:
        work()
    except Exception:
        # The pool boundary: anything the job did not handle itself still ends
        # on the job row as a failure instead of being lost on a thread.
        logger.exception("Ingest job %s failed", job_id)
        mark_failed(job_id, SAFE_FAILURE)


def recover() -> int:
    """Mark the jobs a previous process left queued or running as retryable."""
    with database.SessionLocal() as db:
        statement = select(IngestJob).where(IngestJob.status.in_(("queued", "running")))
        jobs = list(db.execute(statement).scalars())
        for job in jobs:
            job.status = "failed"
            job.step = "Interrupted"
            job.error = INTERRUPTED
            job.finished_at = datetime.utcnow()
        db.commit()
    return len(jobs)


def publish(job_id: int, step: str, done: int, total: int) -> None:
    """Record the measured step and the passages stored so far."""
    progress = round(done / total, 4) if total else 0.0
    with _LOCK:
        _LIVE[job_id] = {"step": step, "done": done, "total": total, "progress": progress}


def finish_live(job_id: int) -> None:
    """Drop the in-memory progress of a job that reached a terminal state."""
    with _LOCK:
        _LIVE.pop(job_id, None)


def was_cancelled(job_id: int) -> bool:
    """Whether the trainer stopped this job."""
    with _LOCK:
        return job_id in _STOPPED


def begin(job_id: int) -> None:
    """Take the job over: a retried run starts without the previous stop."""
    with _LOCK:
        _STOPPED.discard(job_id)


@contextmanager
def run_lock(job_id: int) -> Iterator[None]:
    """Keep two runs of one job out of the same file at the same time."""
    with _LOCK:
        entry = _RUNNING.get(job_id)
        if entry is None:
            entry = _RUNNING[job_id] = [threading.Lock(), 0]
        entry[1] += 1
    try:
        with entry[0]:
            yield
    finally:
        with _LOCK:
            entry[1] -= 1
            if entry[1] == 0:
                _RUNNING.pop(job_id, None)


def as_dict(job: IngestJob) -> dict[str, Any]:
    """The job in the shape the page and the event stream read."""
    with _LOCK:
        live = dict(_LIVE.get(job.id, {}))
    done = int(live.get("done", job.done))
    total = int(live.get("total", job.total))
    return {
        "id": job.id,
        "source": job.source,
        "status": job.status,
        "step": str(live.get("step", job.step)),
        "progress": float(live.get("progress", job.progress)),
        "done": done,
        "total": total,
        "chunks": job.chunks,
        "duplicates": {
            "stored": job.chunks,
            "exact": job.exact_duplicates,
            "near": job.near_duplicates,
        },
        "error": job.error,
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "finished_at": _iso(job.finished_at),
    }


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment is not None else None


def snapshot(job_id: int, tenant_id: int | None) -> dict[str, Any]:
    """The job as it is now, with the live progress of this process."""
    with database.SessionLocal() as db:
        return as_dict(_owned(db, job_id, tenant_id))


def recent(tenant_id: int | None, limit: int = RECENT_JOBS) -> list[dict[str, Any]]:
    """The most recent jobs of the workspace, newest first."""
    statement = select(IngestJob).order_by(IngestJob.id.desc()).limit(limit)
    if tenant_id is not None:
        statement = statement.where(IngestJob.tenant_id == tenant_id)
    with database.SessionLocal() as db:
        return [as_dict(job) for job in db.execute(statement).scalars()]


def owns(job_id: int, tenant_id: int | None) -> None:
    """Raise 404 unless the workspace owns the job."""
    with database.SessionLocal() as db:
        _owned(db, job_id, tenant_id)


def cancel(job_id: int, tenant_id: int | None) -> dict[str, Any]:
    """Stop a queued or running job; a finished job is returned unchanged."""
    with database.SessionLocal() as db:
        job = _owned(db, job_id, tenant_id)
        if job.status not in TERMINAL:
            job.status = "cancelled"
            job.step = "Stopped by the trainer."
            job.finished_at = datetime.utcnow()
            with _LOCK:
                _STOPPED.add(job_id)
            db.commit()
        finish_live(job_id)
        return as_dict(job)


def reset(job_id: int, tenant_id: int | None) -> dict[str, Any]:
    """Put a failed or stopped job back in the queue, ready for another run.

    The stop flag stays where it is: a run that is still in the file has to
    finish backing itself out before the retried run takes over in `begin`.
    """
    with database.SessionLocal() as db:
        job = _owned(db, job_id, tenant_id)
        if job.status not in RETRYABLE:
            raise HTTPException(409, "Only a failed or stopped upload can be retried.")
        job.status = "queued"
        job.step = WAITING
        job.progress = 0.0
        job.done = 0
        job.total = 0
        job.chunks = 0
        job.exact_duplicates = 0
        job.near_duplicates = 0
        job.error = None
        job.started_at = None
        job.finished_at = None
        db.commit()
        return as_dict(job)


def mark_failed(job_id: int, message: str) -> None:
    """Record a failure that escaped the job's own handling."""
    with database.SessionLocal() as db:
        job = db.get(IngestJob, job_id)
        if job is None or job.status in TERMINAL or job.status == "queued":
            return
        job.status = "failed"
        job.step = "Failed"
        job.error = message
        job.finished_at = datetime.utcnow()
        db.commit()
    finish_live(job_id)


def events(job_id: int, tenant_id: int | None) -> Iterator[tuple[str, dict[str, Any]]]:
    """The job as server-sent events, from the first read to its terminal state."""
    for _ in range(MAX_POLLS):
        payload = snapshot(job_id, tenant_id)
        yield "job", payload
        if payload["status"] in TERMINAL:
            return
        time.sleep(POLL_SECONDS)


def _owned(db: Any, job_id: int, tenant_id: int | None) -> IngestJob:
    job = db.get(IngestJob, job_id)
    if job is None or (tenant_id is not None and job.tenant_id != tenant_id):
        raise HTTPException(404, "Ingest job not found")
    return job
