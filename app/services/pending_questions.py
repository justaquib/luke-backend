"""Unanswered questions that a trainer still has to look at."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import PendingQuestion
from app.services import vector_store


def record_pending(
    db: Session,
    question: str,
    tenant_id: int | None = None,
    note: str | None = None,
) -> PendingQuestion:
    """Log an unanswered question, counting repeat occurrences.

    A note marks a question the trainer has to decide, such as an answer that
    contradicts one already approved; it is part of the identity of the entry,
    so the same conflict counts up instead of filling the inbox.
    """
    normalized = question.strip()
    conditions = [
        PendingQuestion.question == normalized,
        PendingQuestion.resolved.is_(False),
    ]
    if tenant_id is not None:
        conditions.append(PendingQuestion.tenant_id == tenant_id)
    conditions.append(
        PendingQuestion.note.is_(None) if note is None else PendingQuestion.note == note
    )
    existing = db.execute(select(PendingQuestion).where(*conditions)).scalar_one_or_none()
    if existing is not None:
        existing.occurrences += 1
        existing.last_seen_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()
        db.refresh(existing)
        vector_store.index_pending(existing.id, existing.question, tenant_id)
        return existing
    pending = PendingQuestion(question=normalized, note=note, tenant_id=tenant_id)
    db.add(pending)
    db.commit()
    db.refresh(pending)
    vector_store.index_pending(pending.id, pending.question, tenant_id)
    return pending
