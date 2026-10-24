"""What the learner and the brain pages read: the approved answers, the close
matches for a question, and the numbers that explain the brain.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import Faq, Intent
from app.services import brain as brain_service
from app.services import vector_store

MAX_CANDIDATES = 3
APPROVED_CONFIDENCE = 0.75


def brain_state(tenant_id: int | None, force: bool = False) -> dict[str, Any]:
    """The brain's own numbers with the level it has reached alongside.

    The brain page reads both from one place, so the level sits next to the
    numbers it explains.
    """
    state = dict(brain_service.state(tenant_id))
    state["model_level"] = brain_service.model_level(tenant_id, force)
    return state


def approved_faqs(db: Session, tenant_id: int | None) -> list[Faq]:
    """Every approved answer of this tenant."""
    statement = select(Faq).where(Faq.status == "approved")
    if tenant_id is not None:
        statement = statement.where(Faq.tenant_id == tenant_id)
    return list(db.execute(statement).scalars().all())


def trainer_answers(db: Session, tenant_id: int | None) -> list[dict[str, Any]]:
    """The approved answers, in the shape the learner page shows."""
    return [
        {
            "id": faq.id,
            "question": faq.question,
            "answer": faq.answer,
            "confidence": APPROVED_CONFIDENCE,
        }
        for faq in approved_faqs(db, tenant_id)
    ]


def candidate_answers(question: str, db: Session, tenant_id: int | None) -> list[dict[str, Any]]:
    """The approved answers that look like the question that was asked."""
    faqs = approved_faqs(db, tenant_id)
    if not faqs:
        return []
    hits = vector_store.search_faq_in(
        question,
        [faq.id for faq in faqs],
        n_results=MAX_CANDIDATES,
    )
    known = {faq.id for faq in faqs}
    candidates = [
        {
            "faq_id": hit["faq_id"],
            "question": hit["question"],
            "answer": hit["answer"],
            "distance": hit["distance"],
        }
        for hit in hits
        if hit.get("faq_id") in known
        and hit.get("distance") is not None
        and hit["distance"] <= settings.faq_match_threshold
    ]
    candidates.sort(
        key=lambda candidate: candidate["distance"] if candidate["distance"] is not None else 1
    )
    return candidates[:MAX_CANDIDATES]


def exact_candidate(question: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The approved answer taught with this exact question, if there is one."""
    key = question.strip().casefold()
    return next(
        (
            candidate
            for candidate in candidates
            if str(candidate.get("question") or "").strip().casefold() == key
        ),
        None,
    )


def trained_answer(
    question: str, db: Session, tenant_id: int | None
) -> tuple[str, str, str] | None:
    """The exact approved answer, as the grade test reports it."""
    exact = exact_candidate(question, candidate_answers(question, db, tenant_id))
    if exact is None:
        return None
    return str(exact["answer"]), "faq", "Trainer-approved answer"


def matching_intents(question: str, db: Session, tenant_id: int | None) -> list[dict[str, Any]]:
    """The saved intents that already cover this question."""
    statement = select(Intent)
    if tenant_id is not None:
        statement = statement.where(Intent.tenant_id == tenant_id)
    intents = list(db.execute(statement).scalars().all())
    terms = {word for word in question.lower().split() if len(word) > 3}
    matches = [
        {
            "name": intent.name,
            "response_template": intent.response_template,
        }
        for intent in intents
        if terms and terms & set(intent.name.lower().split())
    ]
    return matches[:MAX_CANDIDATES]
