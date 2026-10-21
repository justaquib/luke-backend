"""FAQ, inbox, merge, and intent-suggestion flows."""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.core.security import require_tenant_user
from app.db.database import Faq, Intent, PendingQuestion, User
from app.schemas import (
    FaqCreate,
    FaqMergePreview,
    FaqMergeRequest,
    FaqMergeResult,
    FaqOut,
    FaqStatusUpdate,
    FaqSuggestion,
    FaqUpdate,
    IntentSuggestions,
    MergeMatch,
    MergePreview,
    MergeRequest,
    MergeResult,
    PendingOut,
    PendingSuggestion,
)
from app.services import vector_store

FAQ_STATUSES = {"pending", "approved", "declined", "duplicate"}


def _tenant(user: User, action: str) -> int:
    return require_tenant_user(user, action)


def _faq_for(user: User, faq_id: int, db: Session, missing: str = "FAQ not found") -> Faq:
    faq = db.get(Faq, faq_id)
    if faq is None or faq.tenant_id != user.tenant_id:
        raise HTTPException(404, missing)
    return faq


def _pending_for(user: User, pending_id: int, db: Session) -> PendingQuestion:
    pending = db.get(PendingQuestion, pending_id)
    if pending is None or pending.tenant_id != user.tenant_id:
        raise HTTPException(404, "Pending question not found")
    return pending


def create_faq(user: User, req: FaqCreate, db: Session) -> Faq:
    """Create a pending FAQ for the current tenant."""
    tenant_id = _tenant(user, "create FAQs")
    faq = Faq(
        question=req.question.strip(),
        answer=req.answer.strip(),
        status="pending",
        tenant_id=tenant_id,
    )
    db.add(faq)
    db.commit()
    db.refresh(faq)
    vector_store.upsert_faq(faq.id, faq.question, faq.answer, faq.status, tenant_id)
    return faq


def list_faqs(user: User, status: str | None, db: Session) -> list[Faq]:
    """List FAQs visible to the current user."""
    statement = select(Faq).order_by(Faq.updated_at.desc())
    if not user.is_systemadmin:
        statement = statement.where(Faq.tenant_id == user.tenant_id)
    if status:
        statement = statement.where(Faq.status == status)
    return list(db.execute(statement).scalars().all())


def update_faq(user: User, faq_id: int, req: FaqUpdate, db: Session) -> Faq:
    """Replace the answer of an existing FAQ."""
    _tenant(user, "update FAQs")
    faq = _faq_for(user, faq_id, db)
    faq.answer = req.answer.strip()
    db.commit()
    db.refresh(faq)
    vector_store.upsert_faq(faq.id, faq.question, faq.answer, faq.status, faq.tenant_id)
    return faq


def update_faq_status(user: User, faq_id: int, req: FaqStatusUpdate, db: Session) -> Faq:
    """Change the approval status of an FAQ."""
    _tenant(user, "update FAQ status")
    if req.status not in FAQ_STATUSES:
        raise HTTPException(400, f"status must be one of {sorted(FAQ_STATUSES)}")
    faq = _faq_for(user, faq_id, db)
    faq.status = req.status
    db.commit()
    db.refresh(faq)
    if faq.status == "approved":
        vector_store.upsert_faq(faq.id, faq.question, faq.answer, faq.status, faq.tenant_id)
    else:
        vector_store.delete_faq(faq.id)
    return faq


def remove_faq(user: User, faq_id: int, db: Session) -> dict:
    """Delete an FAQ and its vector entry."""
    _tenant(user, "delete FAQs")
    faq = _faq_for(user, faq_id, db)
    db.delete(faq)
    db.commit()
    vector_store.delete_faq(faq_id)
    return {"deleted": faq_id}


def list_inbox(user: User, resolved: bool, db: Session) -> list[PendingQuestion]:
    """List pending questions for the current tenant."""
    tenant_id = _tenant(user, "view inbox")
    return list(
        db.execute(
            select(PendingQuestion)
            .where(
                PendingQuestion.resolved == resolved,
                PendingQuestion.tenant_id == tenant_id,
            )
            .order_by(PendingQuestion.occurrences.desc(), PendingQuestion.last_seen_at.desc())
        )
        .scalars()
        .all()
    )


def answer_pending(user: User, pending_id: int, req: FaqUpdate, db: Session) -> Faq:
    """Turn a pending question into a pending FAQ."""
    tenant_id = _tenant(user, "answer pending questions")
    pending = _pending_for(user, pending_id, db)
    faq = Faq(
        question=pending.question,
        answer=req.answer.strip(),
        status="pending",
        tenant_id=tenant_id,
    )
    db.add(faq)
    pending.resolved = True
    db.commit()
    db.refresh(faq)
    vector_store.upsert_faq(faq.id, faq.question, faq.answer, faq.status, tenant_id)
    vector_store.delete_pending_index(pending_id)
    return faq


def log_question(user: User, question: str, db: Session) -> PendingQuestion:
    """Log a question manually from the trainer."""
    from app.services.query import _record_pending

    tenant_id = _tenant(user, "log questions")
    return _record_pending(db, question, tenant_id)


def delete_pending(user: User, pending_id: int, db: Session) -> dict:
    """Delete a pending question and its index entry."""
    _tenant(user, "delete pending questions")
    pending = _pending_for(user, pending_id, db)
    db.delete(pending)
    db.commit()
    vector_store.delete_pending_index(pending_id)
    return {"deleted": pending_id}


def merge_preview(user: User, pending_id: int, db: Session) -> MergePreview:
    """Suggest pending questions that repeat the selected one."""
    tenant_id = _tenant(user, "view merge preview")
    canonical = _pending_for(user, pending_id, db)
    candidates = list(
        db.execute(
            select(PendingQuestion).where(
                PendingQuestion.resolved.is_(False),
                PendingQuestion.id != canonical.id,
                PendingQuestion.tenant_id == tenant_id,
            )
        )
        .scalars()
        .all()
    )
    texts = [candidate.question for candidate in candidates]
    if not texts:
        return MergePreview(
            canonical_id=canonical.id, canonical_question=canonical.question, answer="", matches=[]
        )
    hits = vector_store.search_pending(
        canonical.question, texts, n_results=max(50, len(texts)), tenant_id=tenant_id
    )
    by_question = {hit["question"]: hit for hit in hits}
    matches = [
        MergeMatch(
            pending_id=candidate.id,
            question=candidate.question,
            occurrences=candidate.occurrences,
            distance=hit["distance"],
        )
        for candidate in candidates
        if (hit := by_question.get(candidate.question))
        and hit["distance"] is not None
        and hit["distance"] <= settings.intent_merge_threshold
    ]
    matches.sort(key=lambda match: match.distance)
    return MergePreview(
        canonical_id=canonical.id,
        canonical_question=canonical.question,
        answer="",
        matches=matches,
    )


def _merge_targets(user: User, req: MergeRequest, canonical: PendingQuestion, db: Session):
    targets: list[PendingQuestion] = []
    for pending_id in req.pending_ids:
        target = db.get(PendingQuestion, pending_id)
        if (
            target is not None
            and not target.resolved
            and target.id != canonical.id
            and target.tenant_id == user.tenant_id
        ):
            targets.append(target)
    return targets


def merge_pending(user: User, pending_id: int, req: MergeRequest, db: Session) -> MergeResult:
    """Merge pending questions into one FAQ answer."""
    tenant_id = _tenant(user, "merge pending questions")
    canonical = _pending_for(user, pending_id, db)
    answer = req.answer.strip()
    if not answer:
        raise HTTPException(400, "answer is required")
    targets = _merge_targets(user, req, canonical, db)
    faq = Faq(
        question=canonical.question.strip(), answer=answer, status="pending", tenant_id=tenant_id
    )
    db.add(faq)
    db.commit()
    db.refresh(faq)
    vector_store.upsert_faq(faq.id, faq.question, faq.answer, faq.status, tenant_id)
    created = _create_aliases(targets, answer, tenant_id, db)
    canonical.resolved = True
    vector_store.delete_pending_index(canonical.id)
    db.commit()
    return MergeResult(
        canonical_faq_id=faq.id, created=created + 1, resolved=len(targets) + 1
    )


def _create_aliases(
    targets: list[PendingQuestion], answer: str, tenant_id: int, db: Session
) -> int:
    for target in targets:
        alias = Faq(
            question=target.question.strip(),
            answer=answer,
            status="pending",
            tenant_id=tenant_id,
        )
        db.add(alias)
        db.flush()
        vector_store.upsert_faq(alias.id, alias.question, alias.answer, alias.status, tenant_id)
        target.resolved = True
        vector_store.delete_pending_index(target.id)
    return len(targets)


def intent_suggestions(user: User, question: str, db: Session) -> IntentSuggestions:
    """Suggest FAQs and pending questions that match a typed question."""
    tenant_id = _tenant(user, "view intent suggestions")
    query = question.strip()
    faq_hits = vector_store.search_faq_loose(query, n_results=10, tenant_id=tenant_id)
    pending_hits = vector_store.search_pending_all(query, n_results=10, tenant_id=tenant_id)
    return IntentSuggestions(
        query=query,
        faqs=[
            FaqSuggestion(
                faq_id=hit["faq_id"],
                question=hit["question"],
                answer=hit.get("answer", ""),
                status=hit.get("status") or "unknown",
                distance=hit.get("distance") if hit.get("distance") is not None else 1.0,
            )
            for hit in faq_hits
        ],
        pending=[
            PendingSuggestion(
                pending_id=hit["pending_id"],
                question=hit["question"],
                occurrences=_occurrences(db, hit["pending_id"]),
                distance=hit.get("distance") if hit.get("distance") is not None else 1.0,
            )
            for hit in pending_hits
        ],
    )


def _occurrences(db: Session, pending_id: int) -> int:
    pending = db.get(PendingQuestion, pending_id)
    return pending.occurrences if pending is not None else 1


def faq_merge_preview(user: User, faq_id: int, db: Session) -> FaqMergePreview:
    """Suggest FAQs that duplicate the selected FAQ."""
    tenant_id = _tenant(user, "view FAQ merge preview")
    canonical = _faq_for(user, faq_id, db)
    other_ids = [
        faq.id
        for faq in db.execute(
            select(Faq).where(Faq.id != canonical.id, Faq.tenant_id == tenant_id)
        ).scalars()
    ]
    if not other_ids:
        return FaqMergePreview(
            canonical_id=canonical.id,
            canonical_question=canonical.question,
            canonical_answer=canonical.answer,
            matches=[],
        )
    hits = vector_store.search_faq_in(
        canonical.question, other_ids, n_results=max(50, len(other_ids))
    )
    matches = [
        FaqSuggestion(
            faq_id=hit["faq_id"],
            question=hit["question"],
            answer=hit.get("answer", ""),
            status=hit.get("status") or "unknown",
            distance=hit["distance"],
        )
        for hit in hits
        if hit.get("distance") is not None
        and hit["distance"] <= settings.intent_merge_threshold
    ]
    matches.sort(key=lambda match: match.distance)
    return FaqMergePreview(
        canonical_id=canonical.id,
        canonical_question=canonical.question,
        canonical_answer=canonical.answer,
        matches=matches,
    )


def faq_merge(user: User, faq_id: int, req: FaqMergeRequest, db: Session) -> FaqMergeResult:
    """Merge FAQs into one answer while keeping aliases for former questions."""
    tenant_id = _tenant(user, "merge FAQs")
    canonical = _faq_for(user, faq_id, db)
    target = db.get(Faq, req.target_id or canonical.id)
    if target is None or target.tenant_id != tenant_id:
        raise HTTPException(404, "Target FAQ not found")
    _apply_merge_answer(canonical, target, req.answer)
    db.flush()
    absorbed_questions = _absorb_faqs(user, target, req.faq_ids, db)
    db.commit()
    db.refresh(target)
    vector_store.upsert_faq(target.id, target.question, target.answer, target.status)
    recreated = _recreate_aliases(absorbed_questions, target.answer, tenant_id, db)
    db.commit()
    return FaqMergeResult(
        canonical_faq_id=target.id,
        absorbed=len(absorbed_questions),
        recreated=recreated,
    )


def _apply_merge_answer(canonical: Faq, target: Faq, answer: str | None) -> None:
    if answer is None:
        return
    new_answer = answer.strip()
    if not new_answer:
        raise HTTPException(400, "answer cannot be empty")
    if target.id == canonical.id:
        target.answer = new_answer
    else:
        canonical.answer = new_answer


def _absorb_faqs(user: User, target: Faq, faq_ids: list[int], db: Session) -> list[str]:
    questions: list[str] = []
    for faq_id in faq_ids:
        other = db.get(Faq, faq_id)
        if other is None or other.id == target.id or other.tenant_id != user.tenant_id:
            continue
        questions.append(other.question)
        db.delete(other)
        vector_store.delete_faq(other.id)
    return questions


def _recreate_aliases(
    questions: list[str], answer: str, tenant_id: int, db: Session
) -> int:
    for question in questions:
        alias = Faq(
            question=question.strip(), answer=answer, status="pending", tenant_id=tenant_id
        )
        db.add(alias)
        db.flush()
        vector_store.upsert_faq(alias.id, alias.question, alias.answer, alias.status)
    return len(questions)
