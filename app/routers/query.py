"""Chat query route."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header
from sqlalchemy.orm import Session

from app.core.security import get_bearer_user
from app.db.database import get_session
from app.schemas import QueryRequest, QueryResponse
from app.services import query as query_service

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    authorization: str = Header(""),
    db: Session = Depends(get_session),
) -> QueryResponse:
    return query_service.answer_question(
        req,
        authorization,
        db,
        get_bearer_user,
        query_service._default_response,
    )
