"""Tests for the answer streams: real stages, one safe ending, JSON parity."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.security import get_bearer_user
from app.db.database import User, get_session
from app.routers import parenting as parenting_router
from app.routers import query as query_router
from app.schemas import QueryRequest, QueryResponse
from app.services import answer_flow, answer_sources, answer_streams, streaming
from app.services.connections import Connection
from app.services.language import answering
from app.services.language import preview as preview_service
from app.services.parenting import CoachingAnswer, CoachingTurn
from app.services.parenting.sessions import CoachingSession


class _ClosingSession:
    """A session that only has to close, for streams that answer without it."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _EmptySession:
    """A dependency session the auth lookup never has to read."""


def _user() -> User:
    return User(
        id=1,
        email="parent@example.com",
        name="Parent",
        is_systemadmin=False,
        tenant_id=1,
    )


def _default_response(db, req, entities, disabled, tenant_id=None) -> QueryResponse:
    return QueryResponse(
        question=req.question,
        source="pending",
        answer="pending",
        hits=[],
        entities=entities,
    )


@pytest.fixture()
def no_index(monkeypatch) -> None:
    monkeypatch.setattr(answer_sources.vector_store, "search_faq", lambda *a, **k: [])
    monkeypatch.setattr(answer_sources.vector_store, "search_faq_loose", lambda *a, **k: [])
    monkeypatch.setattr(answer_sources.vector_store, "query", lambda *a, **k: [])
    monkeypatch.setattr(answer_flow.apps, "detect_intents", lambda question: [])


@pytest.fixture()
def taught(monkeypatch) -> answering.TaughtAnswer:
    card = answering.TaughtAnswer(
        answer="A drum is something you hit with a stick.",
        source="lesson_cards",
        matched_question="What is a drum?",
        evidence=(
            Connection(
                "lesson_cards",
                "match",
                "Lesson “What is a drum?” covers the question.",
                "What is a drum?",
            ),
        ),
    )
    monkeypatch.setattr(
        answer_sources.answering, "answer_from_curriculum", lambda question, tenant_id: card
    )
    return card


@pytest.fixture()
def client(monkeypatch) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(query_router.router)
    app.include_router(parenting_router.router)
    app.dependency_overrides[get_bearer_user] = _user
    app.dependency_overrides[get_session] = _EmptySession
    with TestClient(app) as test_client:
        yield test_client


def test_a_frame_carries_one_named_event():
    assert streaming.frame("reading", {"word": "drum"}) == (
        'event: reading\ndata: {"word": "drum"}\n\n'
    )


def test_a_failure_ends_the_stream_with_one_safe_error():
    def events():
        yield "reading", {"words": []}
        raise RuntimeError("the database went away")

    frames = list(streaming.frames(events()))

    assert [line for line in frames[0].splitlines() if line.startswith("event:")] == [
        "event: reading"
    ]
    assert frames[-1].startswith("event: error\n")
    assert streaming.SAFE_FAILURE in frames[-1]
    assert "database went away" not in "".join(frames)
    assert not any("event: done" in item for item in frames)


def test_the_chat_stream_reads_then_answers_then_done(no_index, taught, monkeypatch):
    session = _ClosingSession()
    monkeypatch.setattr(answer_streams.database, "SessionLocal", lambda: session)

    events = list(answer_streams.query_stream(QueryRequest(question="What is a drum?"), 1))

    assert [event for event, _ in events] == [
        "reading",
        "thinking",
        "connecting",
        "connecting",
        "connecting",
        "answer",
        "done",
    ]
    assert session.closed is True


def test_the_streamed_answer_is_the_json_answer(no_index, taught, monkeypatch):
    session = _ClosingSession()
    monkeypatch.setattr(answer_streams.database, "SessionLocal", lambda: session)
    req = QueryRequest(question="What is a drum?")

    online = answer_flow.answer_for_tenant(req, _ClosingSession(), 1, _default_response)
    events = list(answer_streams.query_stream(req, 1))
    streamed = next(data for event, data in events if event == "answer")

    assert streamed == online.model_dump(mode="json")
    sources = [item["source"] for item in streamed["understanding"]["connections"]]
    assert "lesson_cards" in sources
    assert streamed["understanding"]["verified"] is True


def test_the_route_returns_event_stream_headers(client, monkeypatch):
    monkeypatch.setattr(
        answer_streams,
        "query_stream",
        lambda req, tenant_id: iter([("answer", {"answer": "hi"}), ("done", {})]),
    )

    response = client.post("/query/stream", json={"question": "What is a drum?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert "event: answer" in response.text
    assert "event: done" in response.text


def test_a_stream_without_a_signed_in_user_is_a_401(monkeypatch):
    app = FastAPI()
    app.include_router(query_router.router)
    app.dependency_overrides[get_session] = _EmptySession
    with TestClient(app) as test_client:
        response = test_client.post("/query/stream", json={"question": "What is a drum?"})

    assert response.status_code == 401


def test_the_preview_stream_keeps_the_preview_fields(monkeypatch):
    session = _ClosingSession()
    monkeypatch.setattr(answer_streams.database, "SessionLocal", lambda: session)
    result = preview_service.PreviewResult(
        payload={
            "answer": "A drum is hit.",
            "source": "lesson_cards",
            "candidates": [{"question": "What is a drum?"}],
            "needs_clarification": True,
            "reasoning": {"steps": [], "confidence": 0.8},
        },
        connections=(Connection("lesson_cards", "match", "The lesson covers it."),),
        verified=True,
        verification="The answer came from the matched material.",
    )
    monkeypatch.setattr(
        answer_streams.preview_service, "preview_payload", lambda *args, **kwargs: result
    )

    events = list(answer_streams.preview_stream("what is a drum?", 1))
    answer = next(data for event, data in events if event == "answer")

    assert answer["candidates"] == [{"question": "What is a drum?"}]
    assert answer["needs_clarification"] is True
    assert answer["reasoning"]["confidence"] == 0.8
    assert answer["understanding"]["plan"] == list(answer_streams.PREVIEW_PLAN)
    assert session.closed is True


def test_the_preview_route_returns_the_shared_payload(monkeypatch):
    from app.routers import language_model

    app = FastAPI()
    app.include_router(language_model.router)
    app.dependency_overrides[get_bearer_user] = _user
    app.dependency_overrides[get_session] = _EmptySession
    monkeypatch.setattr(
        preview_service,
        "preview_payload",
        lambda *args, **kwargs: preview_service.PreviewResult(
            payload={"answer": "A drum is hit.", "source": "lesson_cards"},
            connections=(),
            verified=True,
            verification="The answer came from the matched material.",
        ),
    )
    with TestClient(app) as test_client:
        response = test_client.post("/language-model/preview", json={"question": "what is a drum?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "A drum is hit.", "source": "lesson_cards"}


def test_an_unknown_coaching_session_is_a_404_before_the_stream(client, monkeypatch):
    monkeypatch.setattr(answer_streams.parenting, "get_coaching", lambda session_id: None)
    monkeypatch.setattr(
        answer_streams,
        "coaching_stream",
        lambda *args: pytest.fail("the stream must not start"),
    )

    response = client.post(
        "/parenting/chat/session/missing/turn/stream",
        json={"message": "How do I stay calm?"},
    )

    assert response.status_code == 404


def test_the_coaching_stream_connects_the_taught_turn(monkeypatch):
    session = CoachingSession(id="parent-1", topic="Calm", child_age_band="3-5")
    card = Connection("taught", "match", "Lesson “Staying calm” covers the message.")
    turn = CoachingTurn(
        status="answered",
        answer=CoachingAnswer(
            text="Breathe slowly.",
            confidence=1.0,
            grounded=True,
            source="taught",
            evidence=(card,),
        ),
    )
    monkeypatch.setattr(answer_streams.lesson_service, "all_lesson_cards", lambda tenant_id: [])
    monkeypatch.setattr(answer_streams.parenting, "coaching_turn", lambda payload: turn)

    events = list(answer_streams.coaching_stream(session, "How do I stay calm?", 1))

    assert [event for event, _ in events] == [
        "reading",
        "thinking",
        "connecting",
        "answer",
        "done",
    ]
    answer = next(data for event, data in events if event == "answer")
    assert answer["status"] == "answered"
    assert answer["answer"]["text"] == "Breathe slowly."
    assert answer["understanding"]["connections"][0]["source"] == "taught"
    assert answer["understanding"]["verified"] is True
