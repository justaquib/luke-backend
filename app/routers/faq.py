"""FAQ, inbox, merge, and suggestion routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header
from sqlalchemy.orm import Session

from app.core.security import get_bearer_user
from app.db.database import get_session
from app.schemas import (
    FaqCreate,
    FaqMergePreview,
    FaqMergeRequest,
    FaqMergeResult,
    FaqOut,
    FaqStatusUpdate,
    FaqUpdate,
    IntentSuggestions,
    MergePreview,
    MergeRequest,
    MergeResult,
    PendingLog,
    PendingOut,
)
from app.services import faq

router = APIRouter(tags=["faq"])


def _user(authorization: str, db: Session):
    return get_bearer_user(authorization, db)


@router.post("/faq", response_model=FaqOut)
def create_faq(
    req: FaqCreate, authorization: str = Header(""), db: Session = Depends(get_session)
) -> FaqOut:
    return faq.create_faq(_user(authorization, db), req, db)


@router.get("/faq", response_model=list[FaqOut])
def list_faqs(
    status: str | None = None,
    authorization: str = Header(""),
    db: Session = Depends(get_session),
) -> list[FaqOut]:
    return faq.list_faqs(_user(authorization, db), status, db)


@router.get("/faq/inbox", response_model=list[PendingOut])
def list_inbox(
    resolved: bool = False,
    authorization: str = Header(""),
    db: Session = Depends(get_session),
) -> list[PendingOut]:
    return faq.list_inbox(_user(authorization, db), resolved, db)


@router.post("/faq/inbox/{pending_id}/answer", response_model=FaqOut)
def answer_pending(
    pending_id: int,
    req: FaqUpdate,
    authorization: str = Header(""),
    db: Session = Depends(get_session),
) -> FaqOut:
    return faq.answer_pending(_user(authorization, db), pending_id, req, db)


@router.post("/faq/log", response_model=PendingOut)
def log_question(
    req: PendingLog, authorization: str = Header(""), db: Session = Depends(get_session)
) -> PendingOut:
    return faq.log_question(_user(authorization, db), req.question, db)


@router.delete("/faq/inbox/{pending_id}")
def delete_pending(
    pending_id: int, authorization: str = Header(""), db: Session = Depends(get_session)
) -> dict:
    return faq.delete_pending(_user(authorization, db), pending_id, db)


@router.get("/faq/pending/{pending_id}/merge-preview", response_model=MergePreview)
def merge_preview(
    pending_id: int, authorization: str = Header(""), db: Session = Depends(get_session)
) -> MergePreview:
    return faq.merge_preview(_user(authorization, db), pending_id, db)


@router.post("/faq/pending/{pending_id}/merge", response_model=MergeResult)
def merge_pending(
    pending_id: int,
    req: MergeRequest,
    authorization: str = Header(""),
    db: Session = Depends(get_session),
) -> MergeResult:
    return faq.merge_pending(_user(authorization, db), pending_id, req, db)


@router.get("/faq/intent-suggestions", response_model=IntentSuggestions)
def intent_suggestions(
    question: str, authorization: str = Header(""), db: Session = Depends(get_session)
) -> IntentSuggestions:
    return faq.intent_suggestions(_user(authorization, db), question, db)


@router.get("/faq/{faq_id}/merge-preview", response_model=FaqMergePreview)
def faq_merge_preview(
    faq_id: int, authorization: str = Header(""), db: Session = Depends(get_session)
) -> FaqMergePreview:
    return faq.faq_merge_preview(_user(authorization, db), faq_id, db)


@router.post("/faq/{faq_id}/merge", response_model=FaqMergeResult)
def faq_merge(
    faq_id: int,
    req: FaqMergeRequest,
    authorization: str = Header(""),
    db: Session = Depends(get_session),
) -> FaqMergeResult:
    return faq.faq_merge(_user(authorization, db), faq_id, req, db)


@router.patch("/faq/{faq_id}", response_model=FaqOut)
def update_faq(
    faq_id: int,
    req: FaqUpdate,
    authorization: str = Header(""),
    db: Session = Depends(get_session),
) -> FaqOut:
    return faq.update_faq(_user(authorization, db), faq_id, req, db)


@router.patch("/faq/{faq_id}/status", response_model=FaqOut)
def update_faq_status(
    faq_id: int,
    req: FaqStatusUpdate,
    authorization: str = Header(""),
    db: Session = Depends(get_session),
) -> FaqOut:
    return faq.update_faq_status(_user(authorization, db), faq_id, req, db)


@router.delete("/faq/{faq_id}")
def remove_faq(
    faq_id: int, authorization: str = Header(""), db: Session = Depends(get_session)
) -> dict:
    return faq.remove_faq(_user(authorization, db), faq_id, db)
