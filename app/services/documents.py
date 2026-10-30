"""Document ingestion, listing, deletion, and feedback persistence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.engine import ScalarResult
from sqlalchemy.orm import Session

from app.config import settings
from app.core.files import ALLOWED_UPLOAD_EXTENSIONS, read_upload, sniff_extension
from app.core.workspace import current_workspace
from app.db.database import (
    Document,
    Faq,
    Feedback,
    PendingQuestion,
    UploadedFile,
    User,
)
from app.schemas import FeedbackCreate, IngestRequest, IngestResponse
from app.services import vector_store
from app.services.ingestion import ingest_text

_SYNC_FILE = "faq-sync.json"


@dataclass(frozen=True)
class AcceptedUpload:
    """One upload that passed validation and is now on disk."""

    filename: str
    content: bytes
    digest: str


class DuplicateUploadError(Exception):
    """The same bytes are already stored under another name."""

    def __init__(self, response: dict[str, Any]) -> None:
        super().__init__(response["source"])
        self.response = response


def _workspace(user: User) -> int:
    """The workspace new material belongs to; login is still required."""
    return current_workspace()


def ingest_document(user: User, req: IngestRequest, db: Session) -> IngestResponse:
    """Ingest pasted text for the one workspace."""
    return ingest_text(db, _workspace(user), req.source, req.text)


def accept_upload(db: Session, workspace: int, upload: UploadFile) -> AcceptedUpload:
    """Check one upload and keep it on disk, or raise the duplicate it matched."""
    filename = Path(upload.filename or "upload").name
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type: {extension}")
    content = read_upload(upload, settings.max_upload_bytes)
    if sniff_extension(filename, content) != extension:
        raise HTTPException(415, f"{filename} does not look like a readable {extension} file.")
    digest = hashlib.sha256(content).hexdigest()
    duplicate = _same_content(db, workspace, digest)
    if duplicate is not None:
        raise DuplicateUploadError(
            {
                "source": duplicate.source,
                "chunks": _chunk_count(db, workspace, duplicate.source),
                "status": "duplicate",
            }
        )
    _refuse_name_conflict(db, workspace, filename)
    destination = _upload_path(workspace, filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return AcceptedUpload(filename, content, digest)


def _same_content(db: Session, workspace: int, digest: str) -> UploadedFile | None:
    statement = select(UploadedFile).where(
        UploadedFile.tenant_id == workspace,
        UploadedFile.content_hash == digest,
    )
    return db.execute(statement).scalars().first()


def _chunk_count(db: Session, workspace: int, source: str) -> int:
    statement = select(func.count(Document.id)).where(
        Document.tenant_id == workspace, Document.source == source
    )
    return int(db.execute(statement).scalar_one())


def _refuse_name_conflict(db: Session, workspace: int, filename: str) -> None:
    statement = select(func.count(UploadedFile.id)).where(
        UploadedFile.tenant_id == workspace, UploadedFile.source == filename
    )
    recorded = db.execute(statement).scalar_one()
    if recorded or _chunk_count(db, workspace, filename):
        raise HTTPException(
            409,
            f"{filename} already exists with different content. "
            f"Rename it to {_suggest_name(filename)} and upload again.",
        )


def _suggest_name(filename: str) -> str:
    path = Path(filename)
    return f"{path.stem} (2){path.suffix}"


def list_documents(user: User, db: Session) -> list[dict]:
    """List indexed sources and their chunk counts for the workspace."""
    workspace = _workspace(user)
    statement = (
        select(Document.source, func.count(Document.id), func.min(Document.created_at))
        .where(Document.tenant_id == workspace)
        .group_by(Document.source)
        .order_by(Document.source)
    )
    return [
        {"source": source, "chunks": count, "created_at": created_at}
        for source, count, created_at in db.execute(statement).all()
    ]


def delete_document(user: User, filename: str, db: Session) -> dict:
    """Delete an indexed source, its chunks, its record, and its upload."""
    workspace = _workspace(user)
    safe = Path(filename).name
    count = _chunk_count(db, workspace, safe)
    db.execute(delete(Document).where(Document.tenant_id == workspace, Document.source == safe))
    db.execute(
        delete(UploadedFile).where(UploadedFile.tenant_id == workspace, UploadedFile.source == safe)
    )
    db.commit()
    vector_store.delete_chunks_by_source(safe, workspace)
    upload = _upload_path(workspace, safe)
    if upload.exists():
        upload.unlink()
    return {"source": safe, "deleted_chunks": count, "status": "deleted"}


def _upload_path(workspace: int, filename: str) -> Path:
    return Path(settings.upload_dir) / str(workspace) / Path(filename).name


def uploaded_file(user: User, filename: str, db: Session) -> Path:
    """The stored upload, only when the workspace owns that source."""
    workspace = _workspace(user)
    safe = Path(filename).name
    recorded = _chunk_count(db, workspace, safe)
    statement = select(func.count(UploadedFile.id)).where(
        UploadedFile.tenant_id == workspace, UploadedFile.source == safe
    )
    if not recorded and not db.execute(statement).scalar_one():
        raise HTTPException(404, "File not found")
    path = _upload_path(workspace, safe)
    if not path.is_file():
        raise HTTPException(404, "File not found")
    return path


def create_feedback(user: User, req: FeedbackCreate, db: Session) -> Feedback:
    """Store feedback left in the one workspace."""
    from app.services import vector_store
    from app.services.pending_questions import record_pending
    from app.services.query import _find_matching_faq

    tenant_id = _workspace(user)
    feedback = Feedback(
        message_id=req.message_id,
        question=req.question,
        answer=req.answer,
        source=req.source,
        rating=req.rating,
        tenant_id=tenant_id,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    if req.source in ("faq", "trained"):
        faq = _find_matching_faq(db, tenant_id, req.question, req.answer)
        if faq is not None:
            confidence_step = settings.confidence_step
            if req.rating == "like":
                faq.confidence = min(1.0, faq.confidence + confidence_step)
            elif req.rating == "dislike":
                faq.confidence = max(
                    settings.base_answer_confidence, faq.confidence - confidence_step
                )
                note = "User disliked this answer. Review and correct."
                record_pending(db, req.question, tenant_id, note)
            db.commit()
            db.refresh(faq)
            vector_store.upsert_faq(faq.id, faq.question, faq.answer, faq.status, tenant_id)

    return feedback


def resync_approved_faqs() -> None:
    """Rebuild the FAQ and intent indexes from the rows that changed."""
    from app.db.database import Intent, SessionLocal

    workspace = current_workspace()
    with SessionLocal() as db:
        synced_at = _synced_at()
        full = synced_at is None or vector_store.faq_size() == 0 or vector_store.intents_size() == 0
        approved = select(Faq).where(Faq.status == "approved")
        for faq in _changed(db, approved, Faq.updated_at, synced_at, full):
            vector_store.upsert_faq(faq.id, faq.question, faq.answer, faq.status, workspace)
        pending = select(PendingQuestion).where(PendingQuestion.resolved.is_(False))
        for question in _changed(db, pending, PendingQuestion.last_seen_at, synced_at, full):
            vector_store.index_pending(question.id, question.question, workspace)
        intents = select(Intent)
        for intent in _changed(db, intents, Intent.updated_at, synced_at, full):
            vector_store.upsert_intent(
                intent.id, intent.name, intent.response_template, intent.faq_id, workspace
            )
    _record_sync(datetime.utcnow())


def _changed(
    db: Session, statement: Any, updated: Any, synced_at: datetime | None, full: bool
) -> ScalarResult:
    """The rows that changed since the last sync, or all of them on a full sync."""
    if not full and synced_at is not None:
        statement = statement.where(updated > synced_at)
    return db.execute(statement).scalars()


def _sync_path() -> Path:
    return Path(settings.chroma_persist_dir) / _SYNC_FILE


def _synced_at() -> datetime | None:
    try:
        return datetime.fromisoformat(json.loads(_sync_path().read_text())["synced_at"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _record_sync(moment: datetime) -> None:
    path = _sync_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"synced_at": moment.isoformat()}))
