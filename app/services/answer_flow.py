"""The order the chat reads a question and looks for an answer.

Every source is tried in the same order here, and every source says what it
found. The answer route and the answer stream read the same events, so the
answer a reader receives cannot differ from the answer a page streams.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.schemas import (
    ConnectionOut,
    QueryRequest,
    QueryResponse,
    QuestionUnderstandingOut,
    WordReadingOut,
)
from app.services import apps, model_choice, nlp
from app.services.answer_sources import (
    _disabled_response,
    _disabled_sources,
    _docs_step,
    _faq_step,
    _loose_intent_step,
    _taught_step,
    _weather_step,
    _wiki_step,
)
from app.services.connections import Connection
from app.services.language import finishing
from app.services.model_choice import ModelChoice
from app.services.neural import grammar
from app.services.nlu import QuestionReading, read_question

DefaultResponse = Callable[..., QueryResponse]

# Sources whose answer is already live or already absent: a teacher never
# rewrites them, because there is no taught material to ground it on.
UNGROUNDED_SOURCES = frozenset({"pending", "weather", "wiki"})


@dataclass(frozen=True)
class Turn:
    """The question, the plan, and the model that will answer it."""

    reading: QuestionReading
    plan: list[str]
    disabled: list[str]
    tenant_id: int | None
    choice: ModelChoice
    available: bool


def _planned_sources(detected: list[str], choice: ModelChoice) -> list[str]:
    """The sources the reader will try, in the order it will try them."""
    plan = ["Saved answers (FAQ)", "Taught lessons and the offline dictionary"]
    if "weather" in detected:
        plan.append("Weather")
    if "wiki" in detected:
        plan.append("Wikipedia")
    plan.append("Your documents")
    if choice.kind == "huggingface":
        plan.append(f"Write the answer with {choice.name}")
    return plan


def _verification(response: QueryResponse, answer: str) -> tuple[bool, str]:
    if not answer:
        return False, "The answer is empty."
    if response.disabled_external_sources:
        return False, "A live source is disabled in Settings; no trained answer was found."
    if response.source == "pending":
        return False, "No source had an answer, so a trainer was notified."
    if response.source == "intent":
        return False, "The closest saved answer was not exact; the answer says so."
    return True, "The answer came from the matched material."


def build_understanding(
    reading: QuestionReading,
    plan: list[str],
    connections: list[Connection],
    answer: str,
    check: tuple[bool, str],
) -> QuestionUnderstandingOut:
    """The reading, the plan and the connected sources, as the page reads them."""
    verified, verification = check
    return QuestionUnderstandingOut(
        words=[WordReadingOut(**word.as_dict()) for word in reading.words],
        question_type=reading.question_type,
        key_words=list(reading.key_words),
        plan=list(plan),
        connections=[ConnectionOut(**item.as_dict()) for item in connections],
        grammar_notes=grammar.notes(answer)[:3],
        verified=verified,
        verification=verification,
    )


def _finalize(
    response: QueryResponse,
    turn: Turn,
    connections: list[Connection],
) -> QueryResponse:
    answer = finishing.finish(response.answer)
    verified, verification = _verification(response, answer)
    understanding = build_understanding(
        turn.reading, turn.plan, connections, answer, (verified, verification)
    )
    updates: dict = {
        "answer": answer,
        "understanding": understanding,
        "model": turn.choice.id,
        "model_name": turn.choice.name,
    }
    if turn.disabled:
        updates["disabled_external_sources"] = turn.disabled
    return response.model_copy(update=updates)


def _writes_answer(turn: Turn, response: QueryResponse) -> bool:
    """Whether the selected teacher is the one writing this answer."""
    return (
        turn.available
        and turn.choice.kind == "huggingface"
        and response.source not in UNGROUNDED_SOURCES
    )


def _finished_events(
    response: QueryResponse,
    turn: Turn,
    connections: list[Connection],
) -> Iterator[tuple[str, dict]]:
    if _writes_answer(turn, response):
        yield "writing", {"model": turn.choice.name}
        response, connection = model_choice.grounded(turn.tenant_id, turn.choice, response)
        if connection is not None:
            connections.append(connection)
            yield "connecting", connection.as_dict()
    final = _finalize(response, turn, connections)
    yield "answer", final.model_dump(mode="json")
    yield "done", {}


def _no_source_answer(
    req: QueryRequest,
    entities: list,
    turn: Turn,
    fallback: Callable[[], QueryResponse],
) -> tuple[QueryResponse, Connection]:
    if turn.disabled:
        names = " and ".join(
            "Weather" if item == "weather" else "Wikipedia" for item in turn.disabled
        )
        verb = "is" if len(turn.disabled) == 1 else "are"
        detail = f"{names} {verb} disabled in Settings, so no live answer was tried."
        disabled_response = _disabled_response(req, entities, turn.disabled)
        return disabled_response, Connection("pending", "skipped", detail)
    connection = Connection("pending", "miss", "No source had an answer; a trainer was notified.")
    return fallback(), connection


def iter_answer_events(
    req: QueryRequest,
    db: Session,
    tenant_id: int | None,
    default_response: DefaultResponse,
) -> Iterator[tuple[str, dict]]:
    """Read, think, connect the sources, and answer, yielding each stage."""
    reading = read_question(req.question)
    entities = nlp.extract_entities(req.question)
    app_settings = req.apps or {}
    location = req.location or {}
    detected = apps.detect_intents(req.question)
    disabled = _disabled_sources(detected, app_settings)
    choice, available = model_choice.model_of(tenant_id, req.model)
    plan = _planned_sources(detected, choice)
    turn = Turn(reading, plan, disabled, tenant_id, choice, available)
    yield "reading", reading.as_dict()
    yield (
        "thinking",
        {
            "question_type": reading.question_type,
            "key_words": list(reading.key_words),
            "plan": plan,
        },
    )
    connections: list[Connection] = []
    if not available:
        skipped = Connection(
            "model",
            "miss",
            f"The selected model is not installed, so {choice.name} answers instead.",
        )
        connections.append(skipped)
        yield "connecting", skipped.as_dict()
    finders = (
        lambda: _faq_step(req, entities, tenant_id),
        lambda: _loose_intent_step(req, entities, tenant_id),
        lambda: _taught_step(req, entities, tenant_id),
        lambda: _weather_step(req, entities, app_settings, location, detected),
        lambda: _wiki_step(req, entities, app_settings, detected),
        lambda: _docs_step(req, entities, tenant_id),
    )
    for find_step in finders:
        step = find_step()
        for connection in step.connections:
            connections.append(connection)
            yield "connecting", connection.as_dict()
        if step.response is not None:
            yield from _finished_events(step.response, turn, connections)
            return

    def fallback() -> QueryResponse:
        return default_response(db, req, entities, None, tenant_id)

    response, connection = _no_source_answer(req, entities, turn, fallback)
    connections.append(connection)
    yield "connecting", connection.as_dict()
    yield from _finished_events(response, turn, connections)


def answer_for_tenant(
    req: QueryRequest,
    db: Session,
    tenant_id: int | None,
    default_response: DefaultResponse,
) -> QueryResponse:
    """The answer to a question, read from the same events the stream yields."""
    for event, data in iter_answer_events(req, db, tenant_id, default_response):
        if event == "answer":
            return QueryResponse.model_validate(data)
    raise RuntimeError("The answer flow ended without an answer.")
