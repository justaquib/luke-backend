"""System health and statistics."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.database import PendingQuestion
from app.schemas import StatsResponse
from app.services import vector_store


def health_status() -> dict:
    """Return a simple readiness payload."""
    return {"status": "ok"}


def stats(db: Session) -> StatsResponse:
    """Return document, FAQ, and pending-question counts."""
    pending_count = db.execute(
        select(func.count(PendingQuestion.id)).where(PendingQuestion.resolved.is_(False))
    ).scalar_one()
    return StatsResponse(
        chunks=vector_store.collection_size(),
        faqs=vector_store.faq_size(),
        pending=pending_count,
    )
