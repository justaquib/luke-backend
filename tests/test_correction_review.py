"""Tests for the confidence gate on a trainer's correction.

A correction of an answer the assistant was already sure of is kept for
review instead of being used at once; a correction of a guess settles the
answer straight away.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db.database import Base, Faq
from app.services import brain as brain_service
from app.services import vector_store
from app.services.language import corrections

QUESTION = "What is the colour of the ocean?"
OLD_ANSWER = "The ocean is dark."
ANSWER = "The ocean is deep blue."


@pytest.fixture()
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _use_quiet_brain(monkeypatch, tmp_path):
    path = tmp_path / "feedback-0.json"
    monkeypatch.setattr(brain_service, "feedback_path", lambda tenant_id: path)
    monkeypatch.setattr(vector_store, "upsert_faq", lambda *args, **kwargs: None)
    monkeypatch.setattr(vector_store, "search_faq", lambda *args, **kwargs: [])


def _taught_confidently(db: Session, confidence: float) -> None:
    db.add(
        Faq(
            question=QUESTION,
            answer=OLD_ANSWER,
            status="approved",
            tenant_id=None,
            confidence=confidence,
        )
    )
    db.commit()


def test_a_correction_of_a_guess_is_used_at_once(monkeypatch, tmp_path, db):
    _use_quiet_brain(monkeypatch, tmp_path)
    _taught_confidently(db, settings.base_answer_confidence)

    result = corrections.correct(None, QUESTION, OLD_ANSWER, ANSWER, db)

    assert result["needs_review"] is False
    assert result["status"] == "approved"
    assert result["answer"] == ANSWER
    faq = db.query(Faq).one()
    assert faq.confidence == settings.max_answer_confidence


def test_a_correction_of_a_trusted_answer_waits_for_review(monkeypatch, tmp_path, db):
    _use_quiet_brain(monkeypatch, tmp_path)
    _taught_confidently(db, settings.max_answer_confidence)

    result = corrections.correct(None, QUESTION, OLD_ANSWER, ANSWER, db)

    assert result["needs_review"] is True
    assert result["status"] == "pending"
    assert result["confidence"] == round(settings.max_answer_confidence, 3)


def test_a_correction_of_a_question_never_answered_is_used_at_once(monkeypatch, tmp_path, db):
    _use_quiet_brain(monkeypatch, tmp_path)

    result = corrections.correct(None, QUESTION, "", ANSWER, db)

    assert result["needs_review"] is False
    assert result["status"] == "approved"
