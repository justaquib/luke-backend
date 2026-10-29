"""Uploads belong to the one workspace and never overwrite each other."""

from __future__ import annotations

import io
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException, UploadFile
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.extraction import ExtractedText, PageText
from app.db import database
from app.db.database import Document, Faq, Intent, PendingQuestion, Tenant, UploadedFile, User
from app.schemas import IngestRequest
from app.services import chunk_store, documents, ingest_jobs, ingestion, upload_jobs
from app.services.nlp import ChunkSpan
from app.services.provenance import content_hash

WORKSPACE = 0


class _VectorStore:
    """A vector store that records what the service asked of it."""

    def __init__(self) -> None:
        self.faqs = 0
        self.intents = 0
        self.calls: list[tuple] = []
        self.notes: list[tuple] = []

    def chunk_id(self, tenant_id: int | None, source: str, text: str) -> str:
        return f"{tenant_id}:{source}:{text}"

    def add_chunks(
        self,
        source: str,
        chunks: list[str],
        ids: list[str],
        tenant_id: int | None,
        notes: Any = None,
    ) -> None:
        self.calls.append(("add", source, ids, tenant_id))
        self.notes.append((source, notes))

    def delete_chunks_by_source(self, source: str, tenant_id: int | None = None) -> None:
        self.calls.append(("delete", source, tenant_id))

    def upsert_faq(self, *args: Any) -> None:
        self.faqs += 1
        self.calls.append(("faq", *args))

    def index_pending(self, *args: Any) -> None:
        self.intents += 1
        self.calls.append(("pending", *args))

    def upsert_intent(self, *args: Any) -> None:
        self.intents += 1
        self.calls.append(("intent", *args))

    def faq_size(self) -> int:
        return self.faqs

    def intents_size(self) -> int:
        return self.intents


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


def _prepare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, vector: _VectorStore) -> None:
    settings = SimpleNamespace(
        upload_dir=str(tmp_path),
        max_upload_bytes=64,
        chroma_persist_dir=str(tmp_path / ".chroma"),
        pii_scrub_enabled=False,
    )
    monkeypatch.setattr(documents, "settings", settings)
    monkeypatch.setattr(upload_jobs, "settings", settings)
    monkeypatch.setattr(
        upload_jobs,
        "extract_document",
        lambda path: ExtractedText(text="alpha beta", pages=(PageText(None, "alpha beta", 0, 10),)),
    )
    monkeypatch.setattr(chunk_store, "chunk_spans", lambda text: [ChunkSpan("alpha beta", 0, 10)])
    monkeypatch.setattr(documents, "vector_store", vector)
    monkeypatch.setattr(chunk_store, "chunk_vectors", vector)
    monkeypatch.setattr(chunk_store, "vector_store", vector)
    monkeypatch.setattr(ingestion, "chunk_spans", lambda text: [ChunkSpan("alpha beta", 0, 10)])
    monkeypatch.setattr(ingestion, "vector_store", vector)
    monkeypatch.setattr(ingestion, "chunk_vectors", vector)
    monkeypatch.setattr(ingest_jobs, "submit", lambda job_id, work: None)


def _upload(name: str = "notes.txt", content: bytes = b"alpha beta") -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(content))


def _user() -> User:
    return User(id=1, email="person@example.com", name="Person")


def _upload_now(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str = "notes.txt",
    content: bytes = b"alpha beta",
) -> tuple[dict, _VectorStore]:
    """Queue one upload and run its worker in this thread."""
    vector = _VectorStore()
    _prepare(monkeypatch, tmp_path, vector)
    monkeypatch.setattr(database, "SessionLocal", lambda: nullcontext(db))
    queued = upload_jobs.queue_upload(_user(), _upload(name, content), db)
    if queued.get("job_id") is not None:
        upload_jobs.run_upload_job(queued["job_id"])
        return ingest_jobs.snapshot(queued["job_id"], WORKSPACE), vector
    return queued, vector


def _stored(db: Session) -> list[Document]:
    return list(db.execute(select(Document)).scalars())


def test_an_upload_lands_in_the_workspace_and_keeps_its_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db: Session
) -> None:
    job, vector = _upload_now(db, monkeypatch, tmp_path)

    assert (job["source"], job["chunks"], job["status"]) == ("notes.txt", 1, "done")
    assert job["duplicates"] == {"stored": 1, "exact": 0, "near": 0}
    assert (tmp_path / str(WORKSPACE) / "notes.txt").read_bytes() == b"alpha beta"
    assert [row.tenant_id for row in _stored(db)] == [WORKSPACE]
    assert ("add", "notes.txt", [f"{WORKSPACE}:notes.txt:alpha beta"], WORKSPACE) in vector.calls
    record = db.execute(select(UploadedFile)).scalar_one()
    assert record.tenant_id == WORKSPACE
    assert record.size_bytes == len(b"alpha beta")


def test_an_uploaded_chunk_records_where_it_came_from(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db: Session
) -> None:
    _upload_now(db, monkeypatch, tmp_path)

    row = _stored(db)[0]
    record = db.execute(select(UploadedFile)).scalar_one()
    assert row.document_id == record.id
    assert (row.char_start, row.char_end) == (0, 10)
    assert row.page is None
    assert row.content_hash == content_hash("alpha beta")
    assert row.simhash is not None
    assert 0.0 <= row.quality <= 1.0


def test_pasted_text_records_its_offsets_without_an_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db: Session
) -> None:
    vector = _VectorStore()
    _prepare(monkeypatch, tmp_path, vector)

    documents.ingest_document(_user(), IngestRequest(source="paste", text="alpha beta"), db)

    row = _stored(db)[0]
    assert row.document_id is None
    assert (row.char_start, row.char_end) == (0, 10)
    assert row.content_hash == content_hash("alpha beta")


def test_the_same_content_is_a_no_op(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db: Session
) -> None:
    _upload_now(db, monkeypatch, tmp_path)

    result, _ = _upload_now(db, monkeypatch, tmp_path, name="copy.txt")

    assert result["source"] == "notes.txt"
    assert result["status"] == "duplicate"
    assert result["job_id"] is None
    assert not (tmp_path / str(WORKSPACE) / "copy.txt").exists()
    assert len(_stored(db)) == 1


def test_a_used_name_with_new_content_is_refused_with_a_rename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db: Session
) -> None:
    _upload_now(db, monkeypatch, tmp_path)

    with pytest.raises(HTTPException) as error:
        _upload_now(db, monkeypatch, tmp_path, content=b"gamma delta")

    assert error.value.status_code == 409
    assert "notes (2).txt" in str(error.value.detail)
    assert (tmp_path / str(WORKSPACE) / "notes.txt").read_bytes() == b"alpha beta"


def test_a_file_over_the_cap_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db: Session
) -> None:
    with pytest.raises(HTTPException) as error:
        _upload_now(db, monkeypatch, tmp_path, content=b"x" * 65)

    assert error.value.status_code == 413
    assert not (tmp_path / str(WORKSPACE)).exists()


def test_a_pdf_that_is_not_a_pdf_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db: Session
) -> None:
    with pytest.raises(HTTPException) as error:
        _upload_now(db, monkeypatch, tmp_path, name="scan.pdf")

    assert error.value.status_code == 415


def test_a_repeated_passage_from_another_file_is_not_stored_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db: Session
) -> None:
    _upload_now(db, monkeypatch, tmp_path)

    job, _ = _upload_now(db, monkeypatch, tmp_path, name="second.txt", content=b"alpha beta ")

    assert job["chunks"] == 0
    assert job["duplicates"] == {"stored": 0, "exact": 1, "near": 0}
    assert len(_stored(db)) == 1


def test_a_broken_read_ends_as_a_retryable_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db: Session
) -> None:
    vector = _VectorStore()
    _prepare(monkeypatch, tmp_path, vector)

    def _explode(path: str) -> Any:
        raise ValueError("unreadable")

    monkeypatch.setattr(upload_jobs, "extract_document", _explode)
    monkeypatch.setattr(database, "SessionLocal", lambda: nullcontext(db))
    queued = upload_jobs.queue_upload(_user(), _upload(), db)

    upload_jobs.run_upload_job(queued["job_id"])

    job = ingest_jobs.snapshot(queued["job_id"], WORKSPACE)
    assert (job["status"], job["error"]) == ("failed", upload_jobs.SAFE_FAILURE)
    assert _stored(db) == []


def test_a_stopped_job_keeps_no_rows_and_no_vectors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db: Session
) -> None:
    vector = _VectorStore()
    _prepare(monkeypatch, tmp_path, vector)
    monkeypatch.setattr(ingest_jobs, "was_cancelled", lambda job_id: True)
    monkeypatch.setattr(database, "SessionLocal", lambda: nullcontext(db))
    queued = upload_jobs.queue_upload(_user(), _upload(), db)

    upload_jobs.run_upload_job(queued["job_id"])

    job = ingest_jobs.snapshot(queued["job_id"], WORKSPACE)
    assert job["status"] == "cancelled"
    assert _stored(db) == []
    assert ("delete", "notes.txt", WORKSPACE) in vector.calls


def test_a_retry_after_a_stop_indexes_the_file_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db: Session
) -> None:
    vector = _VectorStore()
    _prepare(monkeypatch, tmp_path, vector)
    monkeypatch.setattr(database, "SessionLocal", lambda: nullcontext(db))
    queued = upload_jobs.queue_upload(_user(), _upload(), db)
    job_id = queued["job_id"]
    ingest_jobs.cancel(job_id, WORKSPACE)
    ingest_jobs.reset(job_id, WORKSPACE)

    upload_jobs.run_upload_job(job_id)

    job = ingest_jobs.snapshot(job_id, WORKSPACE)
    assert (job["status"], job["chunks"]) == ("done", 1)
    assert len(_stored(db)) == 1


def test_a_late_report_from_a_stopped_run_leaves_a_retry_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db: Session
) -> None:
    _prepare(monkeypatch, tmp_path, _VectorStore())
    monkeypatch.setattr(database, "SessionLocal", lambda: nullcontext(db))
    queued = upload_jobs.queue_upload(_user(), _upload(), db)
    job_id = queued["job_id"]
    ingest_jobs.cancel(job_id, WORKSPACE)
    ingest_jobs.reset(job_id, WORKSPACE)

    upload_jobs._finish(job_id, "cancelled", "Stopped by the trainer.")

    assert ingest_jobs.snapshot(job_id, WORKSPACE)["status"] == "queued"


def test_pasted_text_is_ingested_for_the_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db: Session
) -> None:
    vector = _VectorStore()
    _prepare(monkeypatch, tmp_path, vector)

    result = documents.ingest_document(
        _user(), IngestRequest(source="paste", text="alpha beta"), db
    )

    assert result["source"] == "paste"
    assert result["chunks"] == 1
    assert result["duplicates"] == {"stored": 1, "exact": 0, "near": 0}
    assert [row.tenant_id for row in _stored(db)] == [WORKSPACE]
    assert ("add", "paste", [f"{WORKSPACE}:paste:alpha beta"], WORKSPACE) in vector.calls


def test_a_file_is_only_served_when_the_workspace_owns_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db: Session
) -> None:
    _upload_now(db, monkeypatch, tmp_path)

    assert documents.uploaded_file(_user(), "notes.txt", db).is_file()
    with pytest.raises(HTTPException) as error:
        documents.uploaded_file(_user(), "other.txt", db)
    assert error.value.status_code == 404


def test_deleting_removes_the_rows_the_file_and_the_vectors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db: Session
) -> None:
    _upload_now(db, monkeypatch, tmp_path)

    result = documents.delete_document(_user(), "notes.txt", db)

    assert result == {"source": "notes.txt", "deleted_chunks": 1, "status": "deleted"}
    assert _stored(db) == []
    assert not (tmp_path / str(WORKSPACE) / "notes.txt").exists()


def test_resync_writes_the_workspace_and_only_changed_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db: Session
) -> None:
    vector = _VectorStore()
    _prepare(monkeypatch, tmp_path, vector)
    db.add_all(
        [
            Faq(question="What is rain?", answer="Water falling.", status="approved"),
            PendingQuestion(question="Who are you?"),
            Intent(name="greeting", response_template="Hello."),
        ]
    )
    db.commit()
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=db.get_bind()))

    documents.resync_approved_faqs()
    first = list(vector.calls)
    vector.calls.clear()
    documents.resync_approved_faqs()

    assert len(first) == 3
    assert all(call[-1] == WORKSPACE for call in first)
    assert vector.calls == []


def test_listing_counts_only_the_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db: Session
) -> None:
    _upload_now(db, monkeypatch, tmp_path)
    db.add(Document(source="foreign.txt", chunk_index=0, text="elsewhere", tenant_id=1))
    db.commit()

    result = documents.list_documents(_user(), db)

    assert [(row["source"], row["chunks"]) for row in result] == [("notes.txt", 1)]
    assert db.execute(select(func.count(Document.id))).scalar_one() == 2
