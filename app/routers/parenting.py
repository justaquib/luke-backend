"""HTTP routes for the parenting coaching chat.

The routes stay thin: they check the caller, read the request, and hand the
turn to the coaching engine that owns the coach.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.core.security import get_bearer_user, tenant_id_of
from app.db.database import User
from app.schemas import (
    CoachingAnswerOut,
    CoachingScaffoldOut,
    CoachingSessionOut,
    CoachingTurnOut,
    ParentingCorrection,
    ParentingSessionCreate,
    ParentingTurnBody,
)
from app.services import parenting as parenting_service
from app.services.language import lessons as lesson_service
from app.services.parenting.chat_engine import CoachingTurnInput

router = APIRouter(prefix="/parenting", tags=["parenting"])


def _session_out(session: Any) -> CoachingSessionOut:
    """The coaching session as the chat page reads it."""
    return CoachingSessionOut(
        id=session.id,
        topic=session.topic,
        child_age_band=session.child_age_band,
        status=session.status,
        turns=session.turns,
        corrections_used=session.corrections_used,
        scaffold_failures=session.scaffold_failures,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def _answer_out(answer: Any) -> CoachingAnswerOut | None:
    """The coach's answer, as the chat page reads it."""
    if answer is None:
        return None
    return CoachingAnswerOut(
        text=answer.text,
        confidence=answer.confidence,
        grounded=answer.grounded,
        source=answer.source,
        grammar_notes=list(answer.grammar_notes),
    )


def _scaffold_out(scaffold: Any) -> CoachingScaffoldOut | None:
    """The correction the coach still needs, as the chat page reads it."""
    if scaffold is None:
        return None
    return CoachingScaffoldOut(
        suggestions=list(scaffold.suggestions),
        retry_context=scaffold.retry_context,
        failure=scaffold.failure,
    )


def _turn_out(result: Any) -> CoachingTurnOut:
    """The coach's answer to one turn, with the scaffold it still needs."""
    return CoachingTurnOut(
        status=result.status,
        answer=_answer_out(result.answer),
        scaffold=_scaffold_out(result.scaffold),
    )


@router.post("/chat/session", response_model=CoachingSessionOut)
def create_parenting_chat_session(
    body: ParentingSessionCreate,
    user: User = Depends(get_bearer_user),
) -> CoachingSessionOut:
    """Start a coaching session between the parent and the coach."""
    session = parenting_service.start_coaching(body.topic, body.child_age_band)
    return _session_out(session)


@router.get("/chat/session/{session_id}", response_model=CoachingSessionOut)
def read_parenting_chat_session(
    session_id: str,
    user: User = Depends(get_bearer_user),
) -> CoachingSessionOut:
    """Read one coaching session."""
    session = parenting_service.get_coaching(session_id)
    if session is None:
        raise HTTPException(404, "Coaching session not found.")
    return _session_out(session)


@router.post("/chat/session/{session_id}/turn", response_model=CoachingTurnOut)
def coaching_chat_turn(
    session_id: str,
    body: ParentingTurnBody,
    user: User = Depends(get_bearer_user),
) -> CoachingTurnOut:
    """The coach answers the parent's turn, or asks for a correction."""
    session = parenting_service.get_coaching(session_id)
    if session is None:
        raise HTTPException(404, "Coaching session not found.")
    cards = lesson_service.all_lesson_cards(tenant_id_of(user))
    result = parenting_service.coaching_turn(
        CoachingTurnInput(
            tenant_id=tenant_id_of(user),
            session=session,
            message=body.message,
            cards=cards,
        ),
        correction="",
    )
    return _turn_out(result)


@router.post("/chat/session/{session_id}/correction", response_model=CoachingTurnOut)
def coaching_chat_correction(
    session_id: str,
    body: ParentingCorrection,
    user: User = Depends(get_bearer_user),
) -> CoachingTurnOut:
    """The coach retries with the correction the parent gave."""
    session = parenting_service.get_coaching(session_id)
    if session is None:
        raise HTTPException(404, "Coaching session not found.")
    cards = lesson_service.all_lesson_cards(tenant_id_of(user))
    result = parenting_service.coaching_turn(
        CoachingTurnInput(
            tenant_id=tenant_id_of(user),
            session=session,
            message=session.last_message,
            cards=cards,
        ),
        correction=body.correction,
    )
    return _turn_out(result)
