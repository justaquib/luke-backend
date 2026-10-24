"""Tests for the coaching chat of the parenting module."""

from pathlib import Path

import pytest

from app.services import brain, parenting
from app.services.language import answering, runtime, teaching
from app.services.neural.critique import Critique
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


class _FakeWriter:
    """A writer that answers, and keeps the critique it made of that answer."""

    def __init__(self, answer: str, problems: list[str]) -> None:
        self.answer = answer
        self.problems = problems
        self.trained = True
        self.model = object()
        self.last_confidence = 0.5
        self.last_critique = Critique(
            problems=problems,
            notes=[],
            confidence=0.5,
            least_word="calm",
            least_chance=0.2,
            grounded=True,
        )

    def respond(self, instruction: str, context: str = "", max_words: int = 64, reading=None):
        return self.answer


def _session(topic: str = "Emotions", child_age_band: str = "two") -> CoachingSession:
    return parenting.start_coaching(topic=topic, child_age_band=child_age_band)


def _turn(session: CoachingSession, cards, writer) -> object:
    return parenting.coaching_turn(
        CoachingTurnInput(
            tenant_id=None,
            session=session,
            message="My child throws a tantrum every evening.",
            cards=cards,
            writer=writer,
        ),
    )


def _use_temp_feedback(monkeypatch, tmp_path) -> Path:
    path = tmp_path / "feedback-0.json"
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(brain, "feedback_path", lambda tenant_id: path)
    return path


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


def test_the_coach_falls_back_to_the_taught_lesson_without_a_writer():
    session = _session()

    result = _turn(session, [TANTRUM_CARD], _UntrainedWriter())

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
    writer = _FakeWriter("The coach said something else.", [])

    result = parenting.coaching_turn(
        CoachingTurnInput(
            tenant_id=None,
            session=session,
            message="What is a drum?",
            cards=[TANTRUM_CARD],
            writer=writer,
        ),
    )

    assert result.status == "answered"
    assert result.answer is not None
    assert result.answer.source == "lesson_cards"
    assert result.answer.text == "A drum is something you hit with a stick."
    assert result.answer.confidence == 1.0
    assert result.answer.grounded is True
    assert result.scaffold is None


def test_a_poem_about_a_star_is_answered_with_the_taught_poem(temp_runtime):
    teaching.teach(
        None,
        "Q: Poem: Twinkle Twinkle Little Star\n"
        "A: Twinkle, twinkle, little star,\nHow I wonder what you are.",
        stage="nursery",
        subject="Poems",
    )
    session = _session(topic="General")
    writer = _FakeWriter("the book telephone.", ["The sentence has no verb."])

    result = parenting.coaching_turn(
        CoachingTurnInput(
            tenant_id=None,
            session=session,
            message="tell me a poem on star",
            cards=[],
            writer=writer,
        ),
    )

    assert result.status == "answered"
    assert result.answer is not None
    assert result.answer.source == "taught"
    assert "Twinkle, twinkle" in result.answer.text
    assert result.scaffold is None


def test_a_request_that_was_never_taught_is_not_written_from_another_card(temp_runtime):
    session = _session(topic="Emotions")
    writer = _FakeWriter("the book telephone.", ["The sentence has no verb."])

    result = parenting.coaching_turn(
        CoachingTurnInput(
            tenant_id=None,
            session=session,
            message="write me a poem about volcanoes",
            cards=[TANTRUM_CARD],
            writer=writer,
        ),
    )

    assert result.status == "answered"
    assert result.answer is not None
    assert result.answer.source == "not_learned"
    assert result.answer.text == answering.NOT_LEARNED
    assert result.scaffold is None


def test_a_card_from_another_subject_is_not_the_answer(temp_runtime):
    session = _session(topic="General")

    result = parenting.coaching_turn(
        CoachingTurnInput(
            tenant_id=None,
            session=session,
            message="My child keeps asking about the weather.",
            cards=[SHAKESPEARE_CARD],
            writer=_UntrainedWriter(),
        ),
    )

    assert result.status == "answered"
    assert result.answer is not None
    assert result.answer.source == "not_learned"
    assert result.answer.text == answering.NOT_LEARNED
    assert result.scaffold is None


def test_a_body_part_question_finds_the_eye_lesson(temp_runtime):
    teaching.teach(
        None,
        "Q: What is the eye?\nA: The eye is a body part. It helps us see.",
        stage="nursery",
        subject="Body",
    )
    session = _session(topic="General")
    writer = _FakeWriter(
        "William Shakespeare wrote the play Romeo and Juliet.",
        ["The answer does not say what the material says."],
    )

    result = parenting.coaching_turn(
        CoachingTurnInput(
            tenant_id=None,
            session=session,
            message="We see through which body part?",
            cards=[SHAKESPEARE_CARD],
            writer=writer,
        ),
    )

    assert result.status == "answered"
    assert result.answer is not None
    assert result.answer.text == "The eye is a body part. It helps us see."
    assert result.answer.source == "lesson_cards"
    assert result.scaffold is None


def test_the_taught_card_is_used_when_the_writer_rejects_its_own_answer():
    session = _session(topic="Emotions")
    writer = _FakeWriter("the book telephone.", ["The sentence has no verb."])

    result = _turn(session, [TANTRUM_CARD], writer)

    assert result.status == "answered"
    assert result.answer is not None
    assert result.answer.source == "taught"
    assert result.answer.text == TANTRUM_CARD["answer"]
    assert result.answer.grammar_notes == ()
    assert result.scaffold is None


class _UntrainedWriter:
    """A writer that has not learned anything yet."""

    trained = False
    model = None

    def respond(self, instruction: str, context: str = "", max_words: int = 64, reading=None):
        return None


def test_the_coach_sends_back_a_scaffold_when_its_answer_needs_repair():
    session = _session()
    writer = _FakeWriter(
        "The coach said calm down now please.", ["The answer was written with little certainty."]
    )

    result = _turn(session, [], writer)

    assert result.status == "needs_correction"
    assert result.answer.source == "writer"
    assert result.answer.grammar_notes == ("The answer was written with little certainty.",)
    assert result.scaffold is not None
    assert len(result.scaffold.suggestions) == 2
    assert result.scaffold.retry_context
    assert session.scaffold_failures == 1


def test_the_coach_retries_with_the_correction_and_accepts_the_answer():
    session = _session()
    first = _FakeWriter(
        "The coach said calm down now please.", ["The answer was written with little certainty."]
    )
    second = _FakeWriter("The coach said stay calm and name the feeling.", [])
    result = _turn(session, [], first)
    assert result.status == "needs_correction"

    retry = parenting.coaching_turn(
        CoachingTurnInput(
            tenant_id=None,
            session=session,
            message="My child throws a tantrum every evening.",
            cards=[],
            writer=second,
        ),
        correction="Stay calm and name the feeling.",
    )

    assert retry.status == "answered"
    assert retry.answer.source == "writer"
    assert retry.answer.text == "The coach said stay calm and name the feeling."
    assert retry.scaffold is None
    assert session.corrections_used == 1
    assert session.last_correction == ""


def test_the_coach_escalates_after_repeated_scaffolds():
    session = _session()
    first = _FakeWriter(
        "The coach said calm down now please.", ["The answer was written with little certainty."]
    )
    result = _turn(session, [], first)
    assert result.status == "needs_correction"

    second = _FakeWriter(
        "The coach said stay calm and name the feeling.",
        ["The answer was written with little certainty.", "The answer repeats itself."],
    )
    escalated = parenting.coaching_turn(
        CoachingTurnInput(
            tenant_id=None,
            session=session,
            message="My child throws a tantrum every evening.",
            cards=[],
            writer=second,
        ),
        correction="Stay calm and name the feeling.",
    )

    assert escalated.status == "escalated"
    assert session.scaffold_failures == 2


def test_the_coach_keeps_the_accepted_answer_for_the_training(monkeypatch, tmp_path):
    _use_temp_feedback(monkeypatch, tmp_path)
    kept = []

    def keep(tenant_id, feedback):
        kept.append(feedback)
        return {"entries": 0, "correct": 0, "learned_corrections": 0}

    monkeypatch.setattr(brain, "record_feedback", keep)
    session = _session()
    writer = _FakeWriter("The coach said stay calm and name the feeling.", [])

    result = _turn(session, [], writer)

    assert result.status == "answered"
    assert len(kept) == 1
    assert kept[0].answer == "The coach said stay calm and name the feeling."
    assert kept[0].correct is True
