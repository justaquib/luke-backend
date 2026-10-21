"""Document ingestion, listing, deletion, and feedback persistence."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.core.files import ALLOWED_UPLOAD_EXTENSIONS, extract_text_from_file
from app.db.database import Document, Faq, Feedback, PendingQuestion, User
from app.schemas import (
    FeedbackCreate,
    IngestRequest,
    IngestResponse,
)
from app.services import vector_store
from app.services.ingestion import ingest_text
from app.services.nlp import chunk_text


def _tenant_id(user: User, action: str) -> int:
    if user.is_systemadmin:
        raise HTTPException(403, f"System admin cannot {action}")
    if not user.tenant_id:
        raise HTTPException(403, "User is not assigned to a tenant")
    return user.tenant_id


def ingest_document(user: User, req: IngestRequest, db: Session) -> IngestResponse:
    """Ingest pasted text for the current tenant."""
    tenant_id = _tenant_id(user, "ingest documents")
    return ingest_text(db, tenant_id, req.source, req.text)


def upload_document(user: User, upload: UploadFile, db: Session) -> dict:
    """Store an uploaded file and index its chunks for the current tenant."""
    tenant_id = _tenant_id(user, "upload documents")
    filename = Path(upload.filename or "upload").name
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type: {extension}")
    destination = Path(settings.upload_dir) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(upload.file.read())
    text = extract_text_from_file(str(destination), upload.content_type)
    chunks = chunk_text(text)
    if not chunks:
        return {"source": filename, "chunks": 0, "status": "uploaded"}
    start_id = vector_store.collection_size()
    ids = [f"{destination.name}-{start_id + index}" for index in range(len(chunks))]
    vector_store.add_chunks(destination.name, chunks, ids, tenant_id)
    for index, chunk in enumerate(chunks):
        db.add(
            Document(
                source=destination.name,
                chunk_index=start_id + index,
                text=chunk,
                tenant_id=tenant_id,
            )
        )
    db.commit()
    return {"source": destination.name, "chunks": len(chunks), "status": "ingested"}


def list_documents(user: User, db: Session) -> list[dict]:
    """List indexed sources and their chunk counts."""
    statement = select(
        Document.source, func.count(Document.id), func.min(Document.created_at)
    ).group_by(Document.source).order_by(Document.source)
    if not user.is_systemadmin:
        statement = statement.where(Document.tenant_id == user.tenant_id)
    return [
        {"source": source, "chunks": count, "created_at": created_at}
        for source, count, created_at in db.execute(statement).all()
    ]


def delete_document(user: User, filename: str, db: Session) -> dict:
    """Delete an indexed source, its chunks, and its upload."""
    if user.is_systemadmin:
        condition = Document.source == filename
    else:
        condition = (Document.source == filename) & (Document.tenant_id == user.tenant_id)
    count = db.execute(select(func.count(Document.id)).where(condition)).scalar_one()
    db.execute(delete(Document).where(condition))
    db.commit()
    vector_store.delete_chunks_by_source(filename)
    upload = Path(settings.upload_dir) / filename
    if upload.exists():
        upload.unlink()
    return {"source": filename, "deleted_chunks": count, "status": "deleted"}


def uploaded_file(filename: str) -> Path:
    """Return the upload path after checking that the file exists."""
    path = Path(settings.upload_dir) / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return path


def create_feedback(user: User, req: FeedbackCreate, db: Session) -> Feedback:
    """Store feedback left by a tenant user."""
    tenant_id = _tenant_id(user, "create feedback")
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
    return feedback


def resync_approved_faqs() -> None:
    """Rebuild the FAQ and intent indexes from the database on startup."""
    from app.db.database import SessionLocal, Intent

    with SessionLocal() as db:
        for faq in db.execute(select(Faq).where(Faq.status == "approved")).scalars():
            vector_store.upsert_faq(faq.id, faq.question, faq.answer, faq.status)
        pending = db.execute(
            select(PendingQuestion).where(PendingQuestion.resolved.is_(False))
        ).scalars()
        for question in pending:
            vector_store.index_pending(question.id, question.question)
        for intent in db.execute(select(Intent)).scalars():
            vector_store.upsert_intent(
                intent.id, intent.name, intent.response_template, intent.faq_id
            )
