"""Tests for the coaching chat of the parenting module."""

import pytest

from app.services import parenting
from app.services.language import answering, runtime, teaching
from app.services.parenting.chat_engine import CoachingTurnInput
from app.services.parenting.sessions import CoachingSession

TANTRUM_CARD = {
    "id": "tantrum-1",
    "question": "How do I stay calm when my child throws a tantrum?",
    "answer": "Stay calm and name the feeling before you set the limit.",
    "subject": "Emotions",
    "subjects": ["Emotions"],
    "grade_id": "lkg",
}

SHAKESPEARE_CARD = {
    "id": "gk-1",
    "question": "Who wrote the play Romeo and Juliet?",
    "answer": "William Shakespeare wrote the play Romeo and Juliet.",
    "subject": "General Knowledge",
    "subjects": ["General Knowledge"],
    "grade_id": "lkg",
}


def _session(topic: str = "Emotions", child_age_band: str = "two") -> CoachingSession:
    return parenting.start_coaching(topic=topic, child_age_band=child_age_band)


def _turn(session: CoachingSession, cards) -> object:
    return parenting.coaching_turn(
        CoachingTurnInput(
            tenant_id=None,
            session=session,
            message="My child throws a tantrum every evening.",
            cards=cards,
        ),
    )


@pytest.fixture()
def temp_runtime(monkeypatch, tmp_path):
    """An empty learner, so the coaching turn reads no real lessons."""
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    runtime._CACHE.clear()
    yield tmp_path
    runtime._CACHE.clear()


def test_a_coaching_session_is_open_on_a_topic():
    session = _session()

    assert session.status == "open"
    assert session.topic == "Emotions"
    assert session.child_age_band == "two"
    assert session.turns == 0


def test_the_coach_answers_from_the_taught_card():
    session = _session()

    result = _turn(session, [TANTRUM_CARD])

    assert result.status == "answered"
    assert result.answer.source == "taught"
    assert "Stay calm and name the feeling" in result.answer.text
    assert session.turns == 1


def test_a_question_is_answered_from_the_shared_taught_lesson(temp_runtime):
    teaching.teach(
        None,
        "Q: What is a drum?\nA: A drum is something you hit with a stick.",
        stage="nursery",
        subject="Music",
    )
    session = _session()

    result = parenting.coaching_turn(
        CoachingTurnInput(
            tenant_id=None,
            session=session,
            message="What is a drum?",
            cards=[TANTRUM_CARD],
        ),
    )

    assert result.status == "answered"
    assert result.answer is not None
    assert result.answer.source == "lesson_cards"
    assert result.answer.text == "A drum is something you hit with a stick."
    assert result.answer.confidence == 1.0
    assert result.answer.grounded is True


def test_a_poem_about_a_star_is_answered_with_the_taught_poem(temp_runtime):
    teaching.teach(
        None,
        "Q: Poem: Twinkle Twinkle Little Star\n"
        "A: Twinkle, twinkle, little star,\nHow I wonder what you are.",
        stage="nursery",
        subject="Poems",
    )
    session = _session(topic="General")

    result = parenting.coaching_turn(
        CoachingTurnInput(
            tenant_id=None,
            session=session,
            message="tell me a poem on star",
            cards=[],
        ),
    )

    assert result.status == "answered"
    assert result.answer is not None
    assert result.answer.source == "taught"
    assert "Twinkle, twinkle" in result.answer.text


def test_a_request_that_was_never_taught_is_not_answered_from_another_card(temp_runtime):
    session = _session(topic="Emotions")

    result = parenting.coaching_turn(
        CoachingTurnInput(
            tenant_id=None,
            session=session,
            message="write me a poem about volcanoes",
            cards=[TANTRUM_CARD],
        ),
    )

    assert result.status == "answered"
    assert result.answer is not None
    assert result.answer.source == "not_learned"
    assert result.answer.text == answering.NOT_LEARNED


def test_a_card_from_another_subject_is_not_the_answer(temp_runtime):
    session = _session(topic="General")

    result = parenting.coaching_turn(
        CoachingTurnInput(
            tenant_id=None,
            session=session,
            message="My child keeps asking about the weather.",
            cards=[SHAKESPEARE_CARD],
        ),
    )

    assert result.status == "answered"
    assert result.answer is not None
    assert result.answer.source == "not_learned"
    assert result.answer.text == answering.NOT_LEARNED


def test_a_body_part_question_finds_the_eye_lesson(temp_runtime):
    teaching.teach(
        None,
        "Q: What is the eye?\nA: The eye is a body part. It helps us see.",
        stage="nursery",
        subject="Body",
    )
    session = _session(topic="General")

    result = parenting.coaching_turn(
        CoachingTurnInput(
            tenant_id=None,
            session=session,
            message="We see through which body part?",
            cards=[SHAKESPEARE_CARD],
        ),
    )

    assert result.status == "answered"
    assert result.answer is not None
    assert result.answer.text == "The eye is a body part. It helps us see."
    assert result.answer.source == "lesson_cards"
