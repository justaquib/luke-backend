"""The coaching turn of the parenting chat: the coach learns from the lessons."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.services import brain
from app.services.language import answering
from app.services.parenting import scaffold as scaffold_service
from app.services.parenting import sessions as session_service
from app.services.parenting.sessions import CoachingSession

logger = logging.getLogger(__name__)

ANSWERED = "answered"
NEEDS_CORRECTION = "needs_correction"
ESCALATED = "escalated"

COACHING_MAX_WORDS = 48
MAX_SCAFFOLD_ATTEMPTS = 2


@dataclass(frozen=True)
class CoachingAnswer:
    """The coach's answer, and how sure the coach is of it."""

    text: str
    confidence: float
    grounded: bool
    source: str
    grammar_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoachingTurnInput:
    """The turn the parent gave, and the material the coach answers from."""

    tenant_id: int | None
    session: CoachingSession
    message: str
    cards: list[dict[str, Any]]
    writer: Any = None


@dataclass(frozen=True)
class CoachingTurn:
    """What the coach did with one turn of the parenting chat."""

    status: str
    answer: CoachingAnswer | None
    scaffold: scaffold_service.CoachingScaffold | None = None


def start_coaching(topic: str = "General", child_age_band: str = "") -> CoachingSession:
    """Start a coaching session on a topic and an age band."""
    return session_service.coaching_store().create(topic, child_age_band)


def get_coaching(session_id: str) -> CoachingSession | None:
    """The coaching session with this id, or None when it is not held."""
    return session_service.coaching_store().get(session_id)


def coaching_turn(payload: CoachingTurnInput, *, correction: str = "") -> CoachingTurn:
    """Answer one turn of the parenting chat, and let the coach correct itself.

    The coach answers from the material the tenant taught, and the answer is
    read with the same eyes the writer wrote it with: its own confidence, the
    words it was unsure about, the grammar and the loop. When the answer is not
    good enough, the turn ends in a scaffold the parent can coach from, and the
    retry after a correction is re-read the same way.
    """
    tenant_id = payload.tenant_id
    session = payload.session
    clean_message = (payload.message or "").strip()
    clean_correction = (correction or "").strip()
    session.last_message = clean_message
    session.turns += 1
    session_service.coaching_store().save(session)
    if clean_correction:
        session.last_correction = clean_correction
        session.corrections_used += 1

    writer = payload.writer if payload.writer is not None else brain.load_writer(payload.tenant_id)
    taught = _taught_answer(clean_message, payload.tenant_id)
    if taught is not None:
        session.status = ANSWERED
        session.last_correction = ""
        return CoachingTurn(status=ANSWERED, answer=taught, scaffold=None)

    instruction = _instruction(clean_message, session.child_age_band, clean_correction)
    answer = _coach_answer(instruction, session.topic, payload.cards, writer)
    is_writer_answer = (
        answer.source == "writer" and writer is not None and bool(getattr(writer, "trained", False))
    )
    if is_writer_answer:
        critique = writer.last_critique
        problems = [str(problem) for problem in (getattr(critique, "problems", None) or [])]
        if problems:
            scaffold = scaffold_service.build(answer.text, problems)
            session.scaffold_failures += 1
            status = (
                ESCALATED
                if session.scaffold_failures >= MAX_SCAFFOLD_ATTEMPTS
                else NEEDS_CORRECTION
            )
            session.status = status
            return CoachingTurn(status=status, answer=answer, scaffold=scaffold)

    if answer.text:
        session.status = ANSWERED
        session.last_correction = ""
        _record_accepted(tenant_id, session.topic, answer)
    return CoachingTurn(status=session.status, answer=answer, scaffold=None)


def _taught_answer(message: str, tenant_id: int | None) -> CoachingAnswer | None:
    """The taught lesson that answers the message, read with full confidence."""
    if not answering.is_question_or_request(message):
        return None
    taught = answering.answer_from_curriculum(message, tenant_id)
    if taught is None:
        return None
    return CoachingAnswer(
        text=taught.answer,
        confidence=1.0,
        grounded=True,
        source=taught.source,
    )


def _coach_answer(
    instruction: str,
    topic: str,
    cards: list[dict[str, Any]],
    writer: Any = None,
) -> CoachingAnswer:
    """The coach's answer, written from the taught material, or the taught one."""
    card = _card_for(topic, cards)
    context = _context_of(card)
    taught = str((card or {}).get("answer") or "").strip()
    text: str | None = None
    source = "none"
    confidence = 0.0
    grounded = False
    grammar_notes: tuple[str, ...] = ()
    if writer is not None and bool(getattr(writer, "trained", False)) and writer.model is not None:
        try:
            written = writer.respond(instruction, context=context, max_words=COACHING_MAX_WORDS)
        except (RuntimeError, ValueError, TypeError) as error:
            logger.warning("The coaching answer failed: %s", error)
            written = None
        if written and written.strip():
            text = written.strip()
            source = "writer"
            confidence = float(getattr(writer, "last_confidence", 0.0) or 0.0)
            critique = writer.last_critique
            grounded = bool(getattr(critique, "grounded", False))
            grammar_notes = tuple(
                str(problem) for problem in (getattr(critique, "problems", None) or [])
            )
    if not text and taught:
        text = taught.strip()
        source = "taught"
        confidence = 1.0
        grounded = True
    if not text:
        text = answering.NOT_LEARNED
        source = "not_learned"
    return CoachingAnswer(
        text=text or "",
        confidence=confidence,
        grounded=grounded,
        source=source,
        grammar_notes=grammar_notes,
    )


def _instruction(message: str, age_band: str, correction: str) -> str:
    """The coaching request the writer answers from."""
    parts = [
        "You are the coach of a parenting chat. Answer the parent warmly and clearly.",
        f"The parent said: {message}.",
    ]
    if (age_band or "").strip():
        parts.append(f"The child is about {age_band}.")
    if (correction or "").strip():
        parts.append(f"The coach should have said this instead: {correction}.")
    return " ".join(parts)


def _card_for(topic: str, cards: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The taught passage the coach answers from."""
    key = (topic or "General").strip().lower()
    for card in cards or []:
        if key in str(card.get("subject") or "").lower():
            return card
        if key in f"{card.get('question') or ''} {card.get('answer') or ''}".lower():
            return card
    return None


def _context_of(card: dict[str, Any] | None) -> str:
    """The material the coach is given to answer from."""
    if card is None:
        return ""
    question = str(card.get("question") or "").strip()
    answer = str(card.get("answer") or "").strip()
    return f"{question}. {answer}" if question else answer


def _record_accepted(tenant_id: int | None, topic: str, answer: CoachingAnswer) -> None:
    """Keep the coaching answer the coach gave when it passed its own gate."""
    if answer.source != "writer" or not answer.text:
        return
    brain.record_feedback(
        tenant_id,
        brain.Feedback(
            question=f"parenting:{topic}",
            answer=answer.text,
            correct=True,
        ),
    )
