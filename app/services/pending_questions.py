"""Unanswered questions that a trainer still has to look at."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import PendingQuestion
from app.services import vector_store


def record_pending(db: Session, question: str, tenant_id: int | None = None) -> PendingQuestion:
    """Log an unanswered question, counting repeat occurrences."""
    normalized = question.strip()
    existing = db.execute(
        select(PendingQuestion).where(
            PendingQuestion.question == normalized,
            PendingQuestion.resolved.is_(False),
        )
    ).scalar_one_or_none()
    if existing is not None:
        existing.occurrences += 1
        existing.last_seen_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()
        db.refresh(existing)
        vector_store.index_pending(existing.id, existing.question, tenant_id)
        return existing
    pending = PendingQuestion(question=normalized, tenant_id=tenant_id)
    db.add(pending)
    db.commit()
    db.refresh(pending)
    vector_store.index_pending(pending.id, pending.question, tenant_id)
    return pending
