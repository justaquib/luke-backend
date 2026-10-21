"""Intent administration routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header
from sqlalchemy.orm import Session

from app.core.security import get_bearer_user
from app.db.database import get_session
from app.schemas import IntentCreate, IntentOut
from app.services import intents

router = APIRouter(tags=["intents"])


@router.post("/intents", response_model=IntentOut)
def create_intent(
    req: IntentCreate, authorization: str = Header(""), db: Session = Depends(get_session)
) -> IntentOut:
    return intents.create_intent(get_bearer_user(authorization, db), req, db)


@router.get("/intents", response_model=list[IntentOut])
def list_intents(
    authorization: str = Header(""), db: Session = Depends(get_session)
) -> list[IntentOut]:
    return intents.list_intents(get_bearer_user(authorization, db), db)


@router.patch("/intents/{intent_id}", response_model=IntentOut)
def update_intent(
    intent_id: int,
    req: IntentCreate,
    authorization: str = Header(""),
    db: Session = Depends(get_session),
) -> IntentOut:
    return intents.update_intent(get_bearer_user(authorization, db), intent_id, req, db)


@router.delete("/intents/{intent_id}")
def remove_intent(
    intent_id: int, authorization: str = Header(""), db: Session = Depends(get_session)
) -> dict:
    return intents.remove_intent(get_bearer_user(authorization, db), intent_id, db)
