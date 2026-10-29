"""An ingest job is reportable, stoppable, retryable and private to its workspace."""

from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Any

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import database
from app.db.database import IngestJob, Tenant
from app.services import ingest_jobs

WORKSPACE = 0


@pytest.fixture()
def db() -> Session:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    database.Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False)()
    session.add_all(
        [
            Tenant(id=WORKSPACE, name="Luke Lab", slug="luke-lab"),
            Tenant(id=1, name="Other", slug="other"),
        ]
    )
    session.commit()
    yield session
    session.close()


def _job(db: Session, **fields: Any) -> IngestJob:
    values = {"source": "notes.txt", "tenant_id": WORKSPACE, "status": "queued", **fields}
    job = IngestJob(**values)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _use(db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(database, "SessionLocal", lambda: nullcontext(db))


def test_a_restart_marks_an_unfinished_job_as_retryable(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    queued = _job(db)
    running = _job(db, status="running", progress=0.4)
    finished = _job(db, status="done", progress=1.0)
    _use(db, monkeypatch)

    assert ingest_jobs.recover() == 2
    assert queued.status == running.status == "failed"
    assert queued.error == ingest_jobs.INTERRUPTED
    assert finished.status == "done"
    assert finished.error is None


def test_a_finished_job_cannot_be_retried(db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    job = _job(db, status="done")
    _use(db, monkeypatch)

    with pytest.raises(HTTPException) as error:
        ingest_jobs.reset(job.id, WORKSPACE)

    assert error.value.status_code == 409


def test_a_stopped_job_is_queued_again_by_a_retry(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job(db, status="running", step="Storing the passages", progress=0.5, done=8, total=16)
    _use(db, monkeypatch)

    stopped = ingest_jobs.cancel(job.id, WORKSPACE)
    assert stopped["status"] == "cancelled"
    assert ingest_jobs.was_cancelled(job.id)

    again = ingest_jobs.reset(job.id, WORKSPACE)
    assert again["status"] == "queued"
    assert (again["progress"], again["done"], again["total"]) == (0.0, 0, 0)
    assert again["step"] == ingest_jobs.WAITING
    assert again["error"] is None
    # The stopped run keeps its stop until the retried run takes the file over.
    assert ingest_jobs.was_cancelled(job.id)

    ingest_jobs.begin(job.id)
    assert not ingest_jobs.was_cancelled(job.id)


def test_a_retried_run_waits_for_the_stopped_one_to_leave_the_file() -> None:
    order: list[str] = []
    first_inside = threading.Event()
    release_first = threading.Event()
    second_inside = threading.Event()

    def first_run() -> None:
        with ingest_jobs.run_lock(7):
            order.append("first in")
            first_inside.set()
            release_first.wait(5)
            order.append("first out")

    def second_run() -> None:
        with ingest_jobs.run_lock(7):
            order.append("second in")
            second_inside.set()

    first = threading.Thread(target=first_run)
    first.start()
    assert first_inside.wait(5)
    second = threading.Thread(target=second_run)
    second.start()
    assert not second_inside.wait(0.3), "both runs were in the file at once"
    release_first.set()
    assert second_inside.wait(5)
    first.join(5)
    second.join(5)

    assert order == ["first in", "first out", "second in"]


def test_a_job_belongs_to_its_workspace(db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    job = _job(db)
    _use(db, monkeypatch)

    with pytest.raises(HTTPException) as error:
        ingest_jobs.snapshot(job.id, 1)

    assert error.value.status_code == 404


def test_the_stream_ends_when_the_job_is_terminal(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job(db, status="done", chunks=3)
    _use(db, monkeypatch)

    frames = list(ingest_jobs.events(job.id, WORKSPACE))

    assert len(frames) == 1
    event, payload = frames[0]
    assert (event, payload["status"], payload["chunks"]) == ("job", "done", 3)


def test_live_progress_is_reported_before_the_row_changes(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job(db)
    _use(db, monkeypatch)

    ingest_jobs.publish(job.id, "Storing the passages", 8, 16)
    shown = ingest_jobs.snapshot(job.id, WORKSPACE)

    assert shown["step"] == "Storing the passages"
    assert (shown["done"], shown["total"], shown["progress"]) == (8, 16, 0.5)
    assert job.done == 0
    ingest_jobs.finish_live(job.id)
    assert ingest_jobs.snapshot(job.id, WORKSPACE)["done"] == 0


def test_recent_lists_only_the_workspace_newest_first(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _job(db, source="first.txt")
    second = _job(db, source="second.txt")
    _job(db, source="elsewhere.txt", tenant_id=1)
    _use(db, monkeypatch)

    result = ingest_jobs.recent(WORKSPACE)

    assert [row["id"] for row in result] == [second.id, first.id]
