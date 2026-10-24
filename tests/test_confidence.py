"""Tests for how sure the assistant is about an answer it already stores."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import main as api
from app.db.database import Base, Faq
from app.services import language_logic
from app.services import query as query_service


def _use_temp_state(monkeypatch, tmp_path):
    monkeypatch.setattr(language_logic, "_DATA_DIR", tmp_path)
    language_logic._CACHE.clear()


@pytest.fixture()
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def test_a_taught_lesson_starts_with_a_plain_confidence(monkeypatch, tmp_path):
    _use_temp_state(monkeypatch, tmp_path)

    state = language_logic.teach(
        None,
        "Q: What is a kite?\nA: A kite is a light frame with a tail.",
        "nursery",
        "General",
    )

    assert state["lesson_cards"][0]["confidence"] == api.settings.base_answer_confidence


def test_a_confirmed_lesson_answer_becomes_more_trusted(monkeypatch, tmp_path):
    _use_temp_state(monkeypatch, tmp_path)
    language_logic.teach(
        None,
        "Q: What is a kite?\nA: A kite is a light frame with a tail.",
        "nursery",
        "General",
    )

    card = language_logic.bump_lesson_confidence(None, "A kite is a light frame with a tail.")

    assert card["confidence"] > api.settings.base_answer_confidence
    assert card["confirmations"] == 1


def test_the_confidence_never_passes_the_maximum(monkeypatch, tmp_path):
    _use_temp_state(monkeypatch, tmp_path)
    language_logic.teach(
        None,
        "Q: What is a kite?\nA: A kite is a light frame with a tail.",
        "nursery",
        "General",
    )

    for _ in range(10):
        card = language_logic.bump_lesson_confidence(None, "A kite is a light frame with a tail.")

    assert card["confidence"] == api.settings.max_answer_confidence


def test_a_confirmed_lesson_answer_is_not_copied_to_the_trained_answers(monkeypatch, tmp_path, db):
    _use_temp_state(monkeypatch, tmp_path)
    monkeypatch.setattr(query_service.vector_store, "upsert_faq", lambda *args, **kwargs: None)
    language_logic.teach(
        None,
        "Q: What is a kite?\nA: A kite is a light frame with a tail.",
        "nursery",
        "General",
    )

    result = query_service.confirm_existing_answer(
        None,
        "what is a kite?",
        "A kite is a light frame with a tail.",
        db,
    )

    assert result["target"] == "lesson"
    assert result["faq_id"] is None
    assert db.query(Faq).count() == 0
    assert result["confidence"] > api.settings.base_answer_confidence


def test_a_confirmed_new_answer_becomes_a_trained_answer(monkeypatch, tmp_path, db):
    _use_temp_state(monkeypatch, tmp_path)
    monkeypatch.setattr(query_service.vector_store, "upsert_faq", lambda *args, **kwargs: None)

    result = query_service.confirm_existing_answer(
        None,
        "what is the airspeed of a swallow?",
        "It flies at about 40 km/h.",
        db,
    )

    assert result["target"] == "trained"
    assert result["faq_id"] is not None
    assert db.query(Faq).count() == 1
    assert db.get(Faq, result["faq_id"]).confidence == api.settings.base_answer_confidence


def test_a_confirmed_trained_answer_becomes_more_trusted_without_a_copy(monkeypatch, tmp_path, db):
    _use_temp_state(monkeypatch, tmp_path)
    monkeypatch.setattr(query_service.vector_store, "upsert_faq", lambda *args, **kwargs: None)
    db.add(
        Faq(
            question="What is a kite?",
            answer="A kite is a light frame with a tail.",
            status="approved",
            tenant_id=None,
            confidence=api.settings.base_answer_confidence,
        )
    )
    db.commit()

    result = query_service.confirm_existing_answer(
        None,
        "what is a kite?",
        "A kite is a light frame with a tail.",
        db,
    )

    assert result["target"] == "trained"
    assert db.query(Faq).count() == 1
    assert result["confidence"] > api.settings.base_answer_confidence


def test_a_trusted_answer_wins_over_a_closer_unproven_one():
    close = {"distance": 0.30, "confidence": 0.5}
    trusted = {"distance": 0.34, "confidence": 1.0}

    assert query_service.faq_confidence_rank(trusted) < query_service.faq_confidence_rank(close)
