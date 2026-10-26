"""Chat query route."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.security import get_bearer_user
from app.db.database import User, get_session
from app.schemas import QueryRequest, QueryResponse
from app.services import answer_streams, streaming
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


@router.post("/query/stream")
def query_stream(
    req: QueryRequest,
    user: User = Depends(get_bearer_user),
) -> StreamingResponse:
    """The chat answer as server-sent events, one real stage at a time."""
    tenant_id = query_service.tenant_for_chat(user)
    return StreamingResponse(
        streaming.frames(answer_streams.query_stream(req, tenant_id)),
        media_type="text/event-stream",
        headers=streaming.SSE_HEADERS,
    )
