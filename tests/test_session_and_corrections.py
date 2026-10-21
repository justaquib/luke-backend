"""Tests for the session, the system administrator and the correction flow.

The session has to last, the system administrator has to be able to use the
chat, and an answer that was wrong has to be corrected.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.security import get_bearer_user
from app.db.database import User, get_session
from app.routers import language_brain, language_model
from app.schemas import QueryRequest
from app.services import brain as brain_service
from app.services import query as query_service
from app.services import vector_store
from app.services.language import runtime


class _FakeResult:
    """A query result with no rows."""

    def scalars(self) -> _FakeResult:
        return self

    def all(self) -> list:
        return []

    def scalar_one_or_none(self) -> None:
        return None

    def first(self) -> None:
        return None

    def scalar_one(self) -> int:
        return 0


class _FakeSession:
    """A session that answers every query with nothing."""

    def __init__(self) -> None:
        self._next_id = 1

    def execute(self, statement) -> _FakeResult:
        return _FakeResult()

    def add(self, item) -> None:
        if getattr(item, "id", None) is None:
            item.id = self._next_id
            self._next_id += 1

    def commit(self) -> None:
        pass

    def refresh(self, item) -> None:
        pass


def _system_admin() -> User:
    return User(
        id=1,
        email="systemadmin@example.com",
        name="System Admin",
        is_systemadmin=True,
        tenant_id=None,
    )


def _default_response(db, req, entities, disabled, tenant_id=None):
    return query_service.QueryResponse(
        question=req.question,
        source="pending",
        answer="I don't have an answer for that yet. A trainer has been notified.",
        hits=[],
        entities=entities,
        pending_id=None,
    )


@pytest.fixture()
def no_index(monkeypatch) -> None:
    """An empty index, so the test does not read the real material."""
    monkeypatch.setattr(vector_store, "search_faq", lambda *args, **kwargs: [])
    monkeypatch.setattr(vector_store, "search_faq_loose", lambda *args, **kwargs: [])
    monkeypatch.setattr(vector_store, "query", lambda *args, **kwargs: [])


def test_the_system_admin_can_use_the_chat(no_index):
    """The system administrator has no tenant, but the chat still answers."""
    response = query_service.answer_question(
        QueryRequest(question="What are the rudiments?"),
        "Bearer token",
        _FakeSession(),
        lambda authorization, db: _system_admin(),
        _default_response,
    )

    assert response.source == "pending"
    assert response.answer


def test_a_user_without_a_tenant_is_turned_away(no_index):
    user = User(
        id=2,
        email="nobody@example.com",
        name="Nobody",
        is_systemadmin=False,
        tenant_id=None,
    )

    with pytest.raises(Exception) as error:
        query_service.answer_question(
            QueryRequest(question="What are the rudiments?"),
            "Bearer token",
            _FakeSession(),
            lambda authorization, db: user,
            _default_response,
        )

    assert "tenant" in str(error.value)


@pytest.fixture()
def client(monkeypatch, tmp_path) -> TestClient:
    """A client for the language model routes with an empty tenant."""
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    runtime._CACHE.clear()
    monkeypatch.setattr(brain_service, "brain_dir", lambda: tmp_path)

    app = FastAPI()
    app.include_router(language_model.router)
    app.include_router(language_brain.router)
    app.dependency_overrides[get_bearer_user] = _system_admin
    app.dependency_overrides[get_session] = _FakeSession
    with TestClient(app) as test_client:
        yield test_client
    runtime._CACHE.clear()


def test_the_session_can_be_renewed():
    """A session that is still good is renewed, so nobody signs in again."""
    from app.services import auth as auth_service

    token = auth_service.refresh_session(_system_admin())

    assert token.access_token
    assert "access_token" in token.model_dump()


def test_a_correction_is_taught_and_kept(client: TestClient, monkeypatch):
    """The right answer is stored, and the brain keeps it."""
    monkeypatch.setattr(
        brain_service,
        "verify_correction",
        lambda tenant_id, question, correction: {"learned": True, "answer": correction},
    )
    monkeypatch.setattr(vector_store, "upsert_faq", lambda *args, **kwargs: None)

    response = client.post(
        "/language-model/brain/correction",
        json={
            "question": "What are the rudiments?",
            "answer": "I do not know.",
            "correction": "The rudiments are the basic first principles of a subject.",
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["learned"] is True
    assert "rudiments are the basic first principles" in result["answer"]
    assert result["corrections"] == 1


def test_a_correction_without_an_answer_is_turned_away(client: TestClient):
    response = client.post(
        "/language-model/brain/correction",
        json={"question": "What are the rudiments?", "correction": "  "},
    )

    assert response.status_code == 400
