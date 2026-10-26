"""Question answering flow for the chat endpoint."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import Faq, User
from app.schemas import (
    QueryRequest,
    QueryResponse,
)
from app.services import answer_flow, language_logic, vector_store
from app.services.language import answering
from app.services.pending_questions import record_pending

UserLookup = Callable[[str, Session], User]
DefaultResponse = Callable[..., QueryResponse]


def _default_response(
    db: Session,
    req: QueryRequest,
    entities: list,
    disabled_external_sources: list[str] | None,
    tenant_id: int | None = None,
) -> QueryResponse:
    """Return the trainer-notified response for an unanswered question."""
    pending = record_pending(db, req.question, tenant_id)
    return QueryResponse(
        question=req.question,
        source="pending",
        answer=answering.NOT_LEARNED,
        hits=[],
        entities=entities,
        pending_id=pending.id,
        disabled_external_sources=disabled_external_sources,
    )


def tenant_for_chat(user: User) -> int | None:
    """The tenant the chat answers from; the system administrator has none."""
    if user.is_systemadmin:
        # The system administrator has no tenant of their own: they look after
        # every tenant, so the assistant answers them from all of it, and from
        # the lessons they taught themselves.
        return None
    if not user.tenant_id:
        raise HTTPException(403, "User is not assigned to a tenant")
    return user.tenant_id


def answer_question(
    req: QueryRequest,
    authorization: str,
    db: Session,
    lookup_user: UserLookup,
    default_response: DefaultResponse,
) -> QueryResponse:
    """Answer a chat question using FAQs, documents, intents, or integrations."""
    tenant_id = tenant_for_chat(lookup_user(authorization, db))
    return answer_flow.answer_for_tenant(req, db, tenant_id, default_response)


def faq_confidence_rank(hit: dict) -> float:
    """A lower rank means a more trustworthy FAQ hit."""
    distance = float(hit.get("distance", 1.0))
    confidence = float(hit.get("confidence", 0.0))
    return distance - confidence * 0.1


def confirm_existing_answer(tenant_id: int | None, question: str, answer: str, db: Session) -> dict:
    """Keep a trainer confirmation without duplicating a known answer."""
    cleaned_answer = answer.strip()
    lesson = language_logic.find_lesson_by_answer(tenant_id, cleaned_answer)
    if lesson is not None:
        card = language_logic.bump_lesson_confidence(tenant_id, cleaned_answer, question)
        return {
            "target": "lesson",
            "faq_id": None,
            "lesson_id": lesson.get("id"),
            "confidence": (card or lesson).get("confidence"),
        }
    faq = _find_matching_faq(db, tenant_id, question, cleaned_answer)
    if faq is not None:
        faq.confidence = min(
            settings.max_answer_confidence, float(faq.confidence) + settings.confidence_step
        )
        db.commit()
        db.refresh(faq)
        vector_store.upsert_faq(faq.id, faq.question, faq.answer, faq.status, tenant_id)
        return {"target": "trained", "faq_id": faq.id, "confidence": faq.confidence}
    faq = Faq(
        question=question.strip(),
        answer=cleaned_answer,
        status="pending",
        tenant_id=tenant_id,
        confidence=settings.base_answer_confidence,
    )
    db.add(faq)
    db.commit()
    db.refresh(faq)
    vector_store.upsert_faq(faq.id, faq.question, faq.answer, faq.status, tenant_id)
    return {"target": "trained", "faq_id": faq.id, "confidence": faq.confidence}


def _find_matching_faq(
    db: Session, tenant_id: int | None, question: str, answer: str
) -> Faq | None:
    stmt = select(Faq).where(
        func.lower(Faq.question) == question.strip().lower(),
        func.lower(Faq.answer) == answer.lower(),
    )
    if tenant_id is None:
        stmt = stmt.where(Faq.tenant_id.is_(None))
    else:
        stmt = stmt.where(Faq.tenant_id == tenant_id)
    return db.execute(stmt).scalars().first()
