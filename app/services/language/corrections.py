"""Teaching the assistant the right answer when it answered wrongly.

A correction does three things:

* it is stored as a trained answer, so the assistant answers the question
  correctly from then on,
* it is kept in the brain's own memory, so the heads learn from it,
* it is checked straight away, so the trainer can see whether the assistant
  really keeps it.

The correction is not eager. When the assistant was already sure of the
answer it gave, the correction is kept for review instead of being used at
once: a well-learned answer is not replaced without a second look, and the
trainer is the one who looks.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import Faq
from app.services import brain as brain_service
from app.services import vector_store


def _existing_faq(db: Session, tenant_id: int | None, question: str) -> Faq | None:
    """The trained answer that already covers this question."""
    statement = select(Faq).where(func.lower(Faq.question) == question.strip().lower())
    if tenant_id is None:
        statement = statement.where(Faq.tenant_id.is_(None))
    else:
        statement = statement.where(Faq.tenant_id == tenant_id)
    return db.execute(statement).scalars().first()


def _needs_review(faq: Faq | None) -> bool:
    """Whether the correction is kept for review instead of being used at once.

    The gate is the confidence the assistant already had for this question.
    A low confidence means the assistant was guessing, and the correction
    settles it at once. A high confidence means the answer was well learned,
    and the correction waits for a second look.
    """
    if faq is None:
        return False
    return float(faq.confidence) >= settings.correction_review_confidence


def _teach_answer(
    db: Session,
    tenant_id: int | None,
    question: str,
    answer: str,
    review: bool,
) -> Faq:
    """Store the right answer as a trained answer of this tenant."""
    faq = _existing_faq(db, tenant_id, question)
    status = "pending" if review else "approved"
    if faq is None:
        faq = Faq(
            question=question,
            answer=answer,
            status=status,
            tenant_id=tenant_id,
            confidence=settings.max_answer_confidence,
        )
        db.add(faq)
    else:
        faq.answer = answer
        faq.status = status
        faq.confidence = settings.max_answer_confidence
    db.commit()
    db.refresh(faq)
    vector_store.upsert_faq(faq.id, faq.question, faq.answer, faq.status, tenant_id)
    return faq


def correct(
    tenant_id: int | None,
    question: str,
    wrong_answer: str,
    correction: str,
    db: Session,
) -> dict[str, Any]:
    """Teach the assistant the right answer to a question it answered wrongly.

    The answer the trainer gives is the approval: it is stored as a trained
    answer, so the next question of that kind is answered correctly. When the
    assistant was already sure of the answer it gave, the correction is kept
    for review instead, and the trainer is told that it is waiting.
    """
    clean_question = (question or "").strip()
    clean_correction = (correction or "").strip()
    if not clean_question:
        raise ValueError("The question is needed to teach the correction.")
    if not clean_correction:
        raise ValueError("Write the answer that should have been given.")
    existing = _existing_faq(db, tenant_id, clean_question)
    confidence = float(existing.confidence) if existing is not None else 0.0
    review = _needs_review(existing)
    faq = _teach_answer(db, tenant_id, clean_question, clean_correction, review=review)
    brain_service.record_feedback(
        tenant_id,
        brain_service.Feedback(
            question=clean_question,
            answer=(wrong_answer or "").strip(),
            correct=False,
            correction=clean_correction,
        ),
    )
    check = brain_service.verify_correction(tenant_id, clean_question, clean_correction)
    if check["learned"]:
        brain_service.mark_correction_learned(tenant_id, clean_question, clean_correction)
    return {
        "faq_id": faq.id,
        "question": faq.question,
        "answer": faq.answer,
        "status": faq.status,
        "needs_review": review,
        "confidence": round(confidence, 3),
        "learned": check["learned"],
        "answer_now": check["answer"],
        "corrections": brain_service.learned_corrections(tenant_id),
    }
