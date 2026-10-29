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
from app.services.answer_live import weather_step, wiki_step
from app.services.answer_sources import (
    _compose_step,
    _docs_step,
    _faq_step,
    _loose_intent_step,
    _taught_step,
    _web_step,
)
from app.services.connections import Connection
from app.services.foundation import answers as foundation_answers
from app.services.foundation.identity import FoundationError
from app.services.language import finishing
from app.services.model_choice import ModelChoice
from app.services.neural import grammar
from app.services.nlu import QuestionReading, read_question

DefaultResponse = Callable[..., QueryResponse]

# Sources whose answer is already live, already absent or already written:
# a teacher never rewrites them, because there is no taught material to ground
# it on, and a written answer is labelled as written.
UNGROUNDED_SOURCES = frozenset({"pending", "weather", "wiki", "web", "generated", "composed"})


@dataclass(frozen=True)
class Turn:
    """The question, the plan, and the model that will answer it."""

    reading: QuestionReading
    plan: list[str]
    disabled: list[str]
    tenant_id: int | None
    choice: ModelChoice
    available: bool


def _planned_sources(
    detected: list[str], choice: ModelChoice, compose: bool = False, web: bool = False
) -> list[str]:
    """The sources the reader will try, in the order it will try them."""
    plan = ["Saved answers (FAQ)", "Taught lessons and the offline dictionary"]
    if "weather" in detected:
        plan.append("Weather")
    if "wiki" in detected:
        plan.append("Wikipedia")
    if compose:
        plan.append("Write the answer from numbered passages")
    plan.append("Your documents")
    if web:
        plan.append("Web evidence (unverified)")
    if choice.kind == "huggingface":
        plan.append(f"Write the answer with {choice.name}")
    elif foundation_answers.installed():
        plan.append("Write the answer with Luke's foundation model")
    return plan


def _verification(response: QueryResponse, answer: str) -> tuple[bool, str]:
    written = {
        "generated": (
            False,
            "Luke wrote this answer from its base checkpoint; no taught material "
            "covered the question, so it is labelled written and unverified.",
        ),
        "hybrid": (
            True,
            "Luke wrote this answer from the numbered passages and cited them.",
        ),
    }
    if not answer:
        return False, "The answer is empty."
    if response.mode in written:
        return written[response.mode]
    if response.disabled_external_sources:
        return False, "A live source is disabled in Settings; no trained answer was found."
    as_of = str((response.app or {}).get("as_of") or "")
    source_notes = {
        "pending": (False, "No source had an answer, so a trainer was notified."),
        "intent": (False, "The closest saved answer was not exact; the answer says so."),
        "web": (
            True,
            f"The answer quotes a stored web page fetched on {as_of}; it is "
            "unverified and not taught material.",
        ),
    }
    return source_notes.get(response.source, (True, "The answer came from the matched material."))


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


def disabled_sources(detected: list[str], app_settings: dict) -> list[str]:
    """The live sources the question asked for that Settings has turned off."""
    disabled: list[str] = []
    if "weather" in detected and not app_settings.get("weather"):
        disabled.append("weather")
    if "wiki" in detected and not app_settings.get("wiki"):
        disabled.append("wiki")
    return disabled


def disabled_response(req: QueryRequest, entities: list, disabled: list[str]) -> QueryResponse:
    """The placeholder that names the live source the trainer could enable."""
    names = ["Weather" if item == "weather" else "Wikipedia" for item in disabled]
    joined = " and ".join(names)
    return QueryResponse(
        question=req.question,
        source="pending",
        answer=(
            "I don't have a trained answer for that yet. "
            f"Enable {joined} in Settings to get live answers from your location or the web."
        ),
        hits=[],
        entities=entities,
        disabled_external_sources=disabled,
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
        placeholder = disabled_response(req, entities, turn.disabled)
        return placeholder, Connection("pending", "skipped", detail)
    connection = Connection("pending", "miss", "No source had an answer; a trainer was notified.")
    return fallback(), connection


def _written_events(
    req: QueryRequest,
    entities: list,
    turn: Turn,
    connections: list[Connection],
    fallback: Callable[[], QueryResponse],
) -> Iterator[tuple[str, dict]]:
    """Luke writes the answer the taught material did not have, and it is labelled."""
    response: QueryResponse | None = None
    try:
        for event, payload in foundation_answers.stream(
            req, turn.tenant_id, entities, turn.choice.name
        ):
            if event == "answer":
                response = payload
                continue
            yield event, payload
    except FoundationError as error:
        connection = Connection("foundation", "fail", str(error))
        connections.append(connection)
        yield "connecting", connection.as_dict()
    if response is None:
        response, connection = _no_source_answer(req, entities, turn, fallback)
    else:
        connection = Connection(
            "foundation",
            "match",
            "Luke wrote this answer from its base checkpoint; no taught material covered "
            "the question. It is labelled written and is not stored as knowledge.",
        )
    connections.append(connection)
    yield "connecting", connection.as_dict()
    final = _finalize(response, turn, connections)
    yield "answer", final.model_dump(mode="json")
    yield "done", {}


def _read_turn(req: QueryRequest, tenant_id: int | None) -> tuple[Turn, list[str]]:
    """The question, the plan and the model, before any source is tried."""
    reading = read_question(req.question)
    detected = apps.detect_intents(req.question)
    disabled = disabled_sources(detected, req.apps or {})
    choice, available = model_choice.model_of(tenant_id, req.model)
    plan = _planned_sources(detected, choice, req.compose, req.web)
    turn = Turn(reading, plan, disabled, tenant_id, choice, available)
    return turn, detected


def _skipped_model(choice: ModelChoice, connections: list[Connection]) -> Connection:
    """The dot that says the chosen model is not installed on this machine."""
    connection = Connection(
        "model",
        "miss",
        f"The selected model is not installed, so {choice.name} answers instead.",
    )
    connections.append(connection)
    return connection


def _without_foundation_events(
    req: QueryRequest,
    entities: list,
    turn: Turn,
    connections: list[Connection],
    fallback: Callable[[], QueryResponse],
) -> Iterator[tuple[str, dict]]:
    """The path when no checkpoint is installed: say so, then notify a trainer."""
    missing = foundation_answers.missing_connection()
    connections.append(missing)
    yield "connecting", missing.as_dict()
    response, connection = _no_source_answer(req, entities, turn, fallback)
    connections.append(connection)
    yield "connecting", connection.as_dict()
    yield from _finished_events(response, turn, connections)


def iter_answer_events(
    req: QueryRequest,
    db: Session,
    tenant_id: int | None,
    default_response: DefaultResponse,
) -> Iterator[tuple[str, dict]]:
    """Read, think, connect the sources, and answer, yielding each stage."""
    turn, detected = _read_turn(req, tenant_id)
    reading = turn.reading
    entities = nlp.extract_entities(req.question)
    app_settings = req.apps or {}
    location = req.location or {}
    yield "reading", reading.as_dict()
    yield (
        "thinking",
        {
            "question_type": reading.question_type,
            "key_words": list(reading.key_words),
            "plan": turn.plan,
        },
    )
    connections: list[Connection] = []
    if not turn.available:
        skipped = _skipped_model(turn.choice, connections)
        yield "connecting", skipped.as_dict()
    finders = [
        lambda: _faq_step(req, entities, turn.tenant_id),
        lambda: _loose_intent_step(req, entities, turn.tenant_id),
        lambda: _taught_step(req, entities, turn.tenant_id),
        lambda: weather_step(req, entities, app_settings, location, detected),
        lambda: wiki_step(req, entities, app_settings, detected),
    ]
    if req.compose:
        finders.append(lambda: _compose_step(req, entities, turn.tenant_id, turn.choice.name))
    finders.append(lambda: _docs_step(req, entities, turn.tenant_id))
    if req.web:
        finders.append(lambda: _web_step(req, entities, turn.tenant_id, db))
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

    if foundation_answers.installed():
        yield from _written_events(req, entities, turn, connections, fallback)
        return
    yield from _without_foundation_events(req, entities, turn, connections, fallback)


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
