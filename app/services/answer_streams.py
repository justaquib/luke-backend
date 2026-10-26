"""The three answer streams: the chat, the learner preview and the coach.

Each stream emits the stages the backend actually reached, in the same order
and with the same data the JSON answer carries. The frames are produced by
`streaming.frames`, so a failure ends the stream with one safe error.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from app.db import database
from app.schemas import CoachingAnswerOut, CoachingTurnOut, QueryRequest
from app.services import answer_flow, parenting
from app.services import query as query_service
from app.services.language import lessons as lesson_service
from app.services.language import preview as preview_service
from app.services.nlu import QuestionReading, read_question
from app.services.parenting.sessions import CoachingSession

PREVIEW_PLAN = ("Trainer-approved answers", "Taught lessons and the offline dictionary")
COACHING_PLAN = ("Taught lessons",)


def _thinking(reading: QuestionReading, plan: tuple[str, ...]) -> dict:
    return {
        "question_type": reading.question_type,
        "key_words": list(reading.key_words),
        "plan": list(plan),
    }


def query_stream(req: QueryRequest, tenant_id: int | None) -> Iterator[tuple[str, dict]]:
    """The chat answer, stage by stage, from the sources the backend read."""
    db: Session = database.SessionLocal()
    try:
        yield from answer_flow.iter_answer_events(
            req, db, tenant_id, query_service._default_response
        )
    finally:
        db.close()


def preview_stream(
    question: str, tenant_id: int | None, skip_trainer: bool = False
) -> Iterator[tuple[str, dict]]:
    """The learner preview, read and connected the same way as the chat."""
    db: Session = database.SessionLocal()
    try:
        reading = read_question(question)
        yield "reading", reading.as_dict()
        yield "thinking", _thinking(reading, PREVIEW_PLAN)
        result = preview_service.preview_payload(question, tenant_id, db, skip_trainer)
        for connection in result.connections:
            yield "connecting", connection.as_dict()
        answer = str(result.payload.get("answer", ""))
        understanding = answer_flow.build_understanding(
            reading,
            list(PREVIEW_PLAN),
            list(result.connections),
            answer,
            (result.verified, result.verification),
        )
        payload = {**result.payload, "understanding": understanding.model_dump(mode="json")}
        yield "answer", payload
        yield "done", {}
    finally:
        db.close()


def coaching_stream(
    session: CoachingSession, message: str, tenant_id: int | None
) -> Iterator[tuple[str, dict]]:
    """One coaching turn, read and connected the same way as the chat."""
    reading = read_question(message)
    yield "reading", reading.as_dict()
    yield "thinking", _thinking(reading, COACHING_PLAN)
    cards = lesson_service.all_lesson_cards(tenant_id)
    turn = parenting.coaching_turn(
        parenting.CoachingTurnInput(
            tenant_id=tenant_id,
            session=session,
            message=message,
            cards=cards,
        )
    )
    answer = turn.answer
    connections = answer.evidence if answer is not None else ()
    for connection in connections:
        yield "connecting", connection.as_dict()
    verified = bool(answer and answer.grounded)
    verification = (
        "The coach answered from a taught lesson."
        if verified
        else "No taught lesson covered the message."
    )
    understanding = answer_flow.build_understanding(
        reading,
        list(COACHING_PLAN),
        list(connections),
        answer.text if answer else "",
        (verified, verification),
    )
    output = CoachingTurnOut(
        status=turn.status,
        answer=(
            CoachingAnswerOut(
                text=answer.text,
                confidence=answer.confidence,
                grounded=answer.grounded,
                source=answer.source,
            )
            if answer is not None
            else None
        ),
        understanding=understanding,
    )
    yield "answer", output.model_dump(mode="json")
    yield "done", {}
