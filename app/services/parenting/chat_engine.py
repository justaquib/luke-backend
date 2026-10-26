"""The coaching turn of the parenting chat: the coach answers from the lessons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.connections import Connection
from app.services.language import answering, finishing, matching
from app.services.language import requests as request_service
from app.services.parenting import sessions as session_service
from app.services.parenting.sessions import CoachingSession

ANSWERED = "answered"

# The parent's message is a sentence, not a lesson question, so the bar for
# the card it answers from is a little lower than a lesson lookup. The card
# still has to share most of its words with the message.
COACH_CARD_MATCH = 0.4


@dataclass(frozen=True)
class CoachingAnswer:
    """The coach's answer, and how sure the coach is of it."""

    text: str
    confidence: float
    grounded: bool
    source: str
    evidence: tuple[Connection, ...] = ()


@dataclass(frozen=True)
class CoachingTurnInput:
    """The turn the parent gave, and the material the coach answers from."""

    tenant_id: int | None
    session: CoachingSession
    message: str
    cards: list[dict[str, Any]]


@dataclass(frozen=True)
class CoachingTurn:
    """What the coach did with one turn of the parenting chat."""

    status: str
    answer: CoachingAnswer | None


def start_coaching(topic: str = "General", child_age_band: str = "") -> CoachingSession:
    """Start a coaching session on a topic and an age band."""
    return session_service.coaching_store().create(topic, child_age_band)


def get_coaching(session_id: str) -> CoachingSession | None:
    """The coaching session with this id, or None when it is not held."""
    return session_service.coaching_store().get(session_id)


def coaching_turn(payload: CoachingTurnInput) -> CoachingTurn:
    """Answer one turn of the parenting chat from the material the tenant taught.

    The coach never writes an answer of its own: it gives back the lesson that
    covers the parent's message, or says that the lesson was not taught.
    """
    session = payload.session
    clean_message = (payload.message or "").strip()
    session.last_message = clean_message
    session.turns += 1
    session_service.coaching_store().save(session)

    taught = _taught_answer(clean_message, payload.tenant_id)
    if taught is not None:
        session.status = ANSWERED
        return CoachingTurn(status=ANSWERED, answer=taught)

    if request_service.parse_request(clean_message) is not None:
        # The material does not answer this request, so there is nothing to
        # answer from. A card about another subject would invent an answer.
        session.status = ANSWERED
        return CoachingTurn(status=ANSWERED, answer=_not_learned_answer())

    session.status = ANSWERED
    return CoachingTurn(
        status=ANSWERED,
        answer=_coach_answer(clean_message, payload.cards),
    )


def _taught_answer(message: str, tenant_id: int | None) -> CoachingAnswer | None:
    """The taught lesson that answers the message, read with full confidence."""
    if not answering.is_question_or_request(message):
        return None
    taught = answering.answer_from_curriculum(message, tenant_id)
    if taught is None:
        return None
    evidence = taught.evidence or (
        Connection(
            taught.source,
            "match",
            f"{taught.detail} answers this.",
            taught.matched_question,
        ),
    )
    return CoachingAnswer(
        text=taught.answer,
        confidence=1.0,
        grounded=True,
        source=taught.source,
        evidence=evidence,
    )


def _coach_answer(message: str, cards: list[dict[str, Any]]) -> CoachingAnswer:
    """The coached lesson that covers the message, or the not-learned answer."""
    card = matching.lesson_covering(message, cards or [], minimum=COACH_CARD_MATCH)
    taught = str((card or {}).get("answer") or "").strip()
    if not taught:
        return _not_learned_answer()
    matched_question = str((card or {}).get("question") or "")
    detail = (
        f"Lesson “{matched_question}” covers the message."
        if matched_question
        else "A taught lesson covers the message."
    )
    return CoachingAnswer(
        text=finishing.finish(taught),
        confidence=1.0,
        grounded=True,
        source="taught",
        evidence=(Connection("taught", "match", detail, matched_question),),
    )


def _not_learned_answer() -> CoachingAnswer:
    """What the coach says when the material holds no answer."""
    return CoachingAnswer(
        text=answering.NOT_LEARNED,
        confidence=0.0,
        grounded=False,
        source="not_learned",
        evidence=(Connection("taught", "miss", "No taught lesson covers the message."),),
    )


def _not_learned_answer() -> CoachingAnswer:
    """What the coach says when the material holds no answer."""
    return CoachingAnswer(
        text=answering.NOT_LEARNED,
        confidence=0.0,
        grounded=False,
        source="not_learned",
    )
