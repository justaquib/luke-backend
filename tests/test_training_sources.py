"""Tests for what the brain trains from, and for teaching a passage.

The brain learns from every source the trainer has given it, and the report
the trainer reads comes from the same material the training uses.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.security import get_bearer_user
from app.db.database import User, get_session
from app.routers import language_brain, language_model
from app.services.language import runtime, teaching
from app.services.language import training_materials as materials_service
from app.services.language.lessons import all_lesson_cards


class _FakeResult:
    """A query result with the rows the test needs."""

    def __init__(self, rows: list) -> None:
        self._rows = rows

    def scalars(self) -> _FakeResult:
        return self

    def all(self) -> list:
        return self._rows

    def scalar_one(self) -> int:
        return 0


class _FakeSession:
    """A session that answers the training queries with the given rows."""

    def __init__(self, faqs: list | None = None, documents: list | None = None) -> None:
        self._faqs = list(faqs or [])
        self._documents = list(documents or [])

    def execute(self, statement) -> _FakeResult:
        text = str(statement)
        if "faqs" in text:
            return _FakeResult(self._faqs)
        if "documents" in text:
            return _FakeResult(self._documents)
        return _FakeResult([])


def _client(db: _FakeSession) -> TestClient:
    """A client for the language model routes with the given database."""
    app = FastAPI()
    app.include_router(language_model.router)
    app.include_router(language_brain.router)
    app.dependency_overrides[get_bearer_user] = lambda: User(
        id=1,
        email="trainer@example.com",
        name="Trainer",
        is_systemadmin=False,
        tenant_id=None,
    )
    app.dependency_overrides[get_session] = lambda: db
    return TestClient(app)


@pytest.fixture()
def learner(tmp_path, monkeypatch) -> None:
    """A learner file of its own, so the tests do not share one."""
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    runtime._CACHE.clear()
    yield
    runtime._CACHE.clear()


def test_the_brain_says_where_it_trains_from(learner):
    teaching.teach(None, "Q: What is a drum?\nA: A drum is hit with a stick.", "nursery", "Music")

    client = _client(_FakeSession())
    response = client.get("/language-model/brain/sources")

    assert response.status_code == 200
    report = response.json()
    lessons = next(source for source in report["sources"] if source["key"] == "lesson")
    assert lessons["items"] == 1
    assert lessons["words"] > 0
    assert report["totals"]["items"] >= 1
    assert report["steps"], "the report says how the training works"


def test_the_approved_answers_are_part_of_the_training(learner):
    db = _FakeSession(faqs=[("What is a kite?", "A kite is a light frame with a tail.")])

    materials = materials_service.gather(None, db)
    training = materials_service.training_set(materials)

    requests_made = [instruction for instruction, _ in training.instructions]
    assert any("kite" in instruction for instruction in requests_made), (
        "the approved question is asked for, not only its answer"
    )
    assert any("kite" in sentence for sentence in training.sentences)


def test_the_documents_are_part_of_the_training(learner):
    db = _FakeSession(documents=[("notes.pdf", "The lighthouse stands on the cliff.")])

    materials = materials_service.gather(None, db)
    training = materials_service.training_set(materials)

    assert any("lighthouse" in sentence for sentence in training.sentences)
    assert any("notes.pdf" in instruction for instruction, _ in training.instructions)


def test_the_words_of_every_source_reach_the_vocabulary(learner):
    teaching.teach(None, "Q: What is a drum?\nA: A drum is hit with a stick.", "nursery", "Music")
    db = _FakeSession(
        faqs=[("What is a kite?", "A kite is a light frame with a tail.")],
        documents=[("notes.pdf", "The lighthouse stands on the cliff.")],
    )

    materials = materials_service.gather(None, db)
    training = materials_service.training_set(materials)
    words = " ".join(training.sentences).lower()

    assert "drum" in words
    assert "kite" in words
    assert "lighthouse" in words


def test_a_story_is_taught_as_one_passage(learner):
    client = _client(_FakeSession())

    taught = client.post(
        "/language-model/lessons",
        json={
            "lesson": "The cat sat on the mat. It was a warm afternoon.",
            "title": "The Cat on the Mat",
            "kind": "story",
            "stage": "nursery",
            "subject": "Stories",
        },
    )

    assert taught.status_code == 200
    state = taught.json()
    assert state["added_lessons"] == 1
    card = next(item for item in state["lesson_cards"] if item.get("entry") == "story")
    assert card["question"] == "Story: The Cat on the Mat"
    assert "The cat sat on the mat." in card["answer"]
    assert state["trainer_lessons"] == 2, "every sentence of the passage is taught"


def test_a_story_is_asked_for_by_its_title(learner):
    teaching.teach_story(
        None,
        teaching.StoryLesson(
            text="The cat sat on the mat. It was a warm afternoon.",
            title="The Cat on the Mat",
            stage="nursery",
            subject="Stories",
        ),
    )

    cards = all_lesson_cards(None)
    story = next(card for card in cards if card["question"].startswith("Story:"))

    assert story["answer"].startswith("The cat sat on the mat.")


def test_a_story_answers_the_question_about_it(learner):
    from app.services.language import preview

    teaching.teach_story(
        None,
        teaching.StoryLesson(
            text="The kite flew over the field. It was the highest kite of the day.",
            title="The Kite over the Field",
            stage="nursery",
            subject="Stories",
        ),
    )

    answer = preview.preview_answer("What is the story about the kite over the field?", None)

    assert "kite flew over the field" in answer["answer"]


def test_a_story_without_a_name_is_not_taught(learner):
    with pytest.raises(ValueError, match="name"):
        teaching.teach_story(
            None,
            teaching.StoryLesson(
                text="The cat sat on the mat.",
                title="  ",
                stage="nursery",
                subject="Stories",
            ),
        )


def test_the_same_story_is_not_stored_twice(learner):
    story = teaching.StoryLesson(
        text="The cat sat on the mat.",
        title="The Cat on the Mat",
        stage="nursery",
        subject="Stories",
    )

    teaching.teach_story(None, story)
    again = teaching.teach_story(None, story)

    assert again["added_lessons"] == 0
    assert again["skipped_duplicates"]
