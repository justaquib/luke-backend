"""Teaching the assistant the right answer when it answered wrongly.

A correction does three things:

* it is stored as a trained answer, so the assistant answers the question
  correctly from then on,
* it is kept in the brain's own memory, so the writing model learns the right
  wording and the heads learn from it,
* it is checked straight away, so the trainer can see whether the assistant
  really keeps it.
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


def _teach_answer(db: Session, tenant_id: int | None, question: str, answer: str) -> Faq:
    """Store the right answer as a trained answer of this tenant."""
    faq = _existing_faq(db, tenant_id, question)
    if faq is None:
        faq = Faq(
            question=question,
            answer=answer,
            status="approved",
            tenant_id=tenant_id,
            confidence=settings.max_answer_confidence,
        )
        db.add(faq)
    else:
        faq.answer = answer
        faq.status = "approved"
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
    answer, so the next question of that kind is answered correctly.
    """
    clean_question = (question or "").strip()
    clean_correction = (correction or "").strip()
    if not clean_question:
        raise ValueError("The question is needed to teach the correction.")
    if not clean_correction:
        raise ValueError("Write the answer that should have been given.")
    faq = _teach_answer(db, tenant_id, clean_question, clean_correction)
    brain_service.record_feedback(
        tenant_id,
        clean_question,
        (wrong_answer or "").strip(),
        correct=False,
        correction=clean_correction,
    )
    check = brain_service.verify_correction(tenant_id, clean_question, clean_correction)
    if check["learned"]:
        brain_service.mark_correction_learned(tenant_id, clean_question, clean_correction)
    return {
        "faq_id": faq.id,
        "question": faq.question,
        "answer": faq.answer,
        "learned": check["learned"],
        "answer_now": check["answer"],
        "corrections": brain_service.learned_corrections(tenant_id),
    }
