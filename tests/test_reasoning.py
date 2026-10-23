"""Tests for the reasoning pass: the two hops, the trace and the intent head."""

from __future__ import annotations

import pytest

from app.services import reasoning, vector_store
from app.services.language import runtime
from app.services.reasoning import entities as entities_service
from app.services.reasoning import intent as intent_service
from app.services.reasoning import recall as recall_service


@pytest.fixture()
def isolated(monkeypatch, tmp_path):
    """The tests write to a folder of their own, never to the real brain."""
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    runtime._CACHE.clear()
    yield tmp_path
    runtime._CACHE.clear()


# The words the fake embedding of the test knows about.
_KEYWORDS = (
    "poem",
    "name some",
    "circle",
    "how many",
    "capital letter",
    "spell",
    "plus",
    "time",
)


def _lesson_hit(question: str, answer: str, distance: float, number: int) -> dict:
    """One lesson as the vector store gives it back."""
    return {
        "lesson_id": number,
        "question": question,
        "answer": answer,
        "distance": distance,
    }


def test_a_node_is_joined_to_the_question_by_its_own_words():
    """The edge says what the node and the question share."""
    nodes = [
        {"id": "question", "type": "question", "entities": ["circle"]},
        {"id": "lesson:1", "type": "lesson", "entities": ["square"]},
        {"id": "lesson:2", "type": "lesson", "entities": ["circle"]},
    ]
    edges: list[dict] = []

    entities_service.relate(nodes, edges, "What is a circle?")

    assert edges == [{"source": "lesson:2", "target": "question", "label": "circle"}]


def test_the_second_pass_asks_about_the_words_the_first_pass_found(monkeypatch):
    """A question that names two things is answered in two hops."""
    asked: list[str] = []

    def search_lesson(question, n_results=3, tenant_id=None):
        asked.append(question)
        if "Saturn" in question:
            return []
        return [_lesson_hit("Which planet is this?", "This is Saturn.", 0.1, 1)]

    monkeypatch.setattr(vector_store, "search_lessons", search_lesson)
    monkeypatch.setattr(vector_store, "search_faq", lambda *args, **kwargs: [])
    monkeypatch.setattr(vector_store, "search_intents", lambda *args, **kwargs: [])
    monkeypatch.setattr(vector_store, "query", lambda *args, **kwargs: [])

    recall_service.recall("Which planet is famous for its rings?", None)

    assert len(asked) > 1
    assert any("saturn" in question.lower() for question in asked[1:])


def test_the_trace_says_why_the_answer_was_chosen(monkeypatch):
    """The trace says what was understood, what was found, and why."""
    monkeypatch.setattr(
        vector_store,
        "search_lessons",
        lambda *args, **kwargs: [_lesson_hit("What is a circle?", "A circle is round.", 0.1, 1)],
    )
    monkeypatch.setattr(vector_store, "search_faq", lambda *args, **kwargs: [])
    monkeypatch.setattr(vector_store, "search_intents", lambda *args, **kwargs: [])
    monkeypatch.setattr(vector_store, "query", lambda *args, **kwargs: [])

    result = reasoning.explain("What is a circle?", None, "A circle is round.", "lesson_cards")

    labels = [step["label"] for step in result["steps"]]
    assert "Understand" in labels
    assert "Why" in labels
    assert "Decide" in labels
    why = next(step for step in result["steps"] if step["label"] == "Why")
    assert "circle" in why["detail"]


def test_a_candidate_that_is_an_exact_match_is_fully_close(monkeypatch):
    """A distance of zero means the same words, not no closeness at all."""
    monkeypatch.setattr(
        vector_store,
        "search_lessons",
        lambda *args, **kwargs: [_lesson_hit("What is a circle?", "A circle is round.", 0.0, 1)],
    )
    monkeypatch.setattr(vector_store, "search_faq", lambda *args, **kwargs: [])
    monkeypatch.setattr(vector_store, "search_intents", lambda *args, **kwargs: [])
    monkeypatch.setattr(vector_store, "query", lambda *args, **kwargs: [])

    result = reasoning.explain("What is a circle?", None, "A circle is round.", "lesson_cards")

    assert result["confidence"] == 1.0


def test_the_intent_head_reads_the_kind_of_the_question(isolated, monkeypatch):
    """The head learns the kind of a question and is asked about it."""
    cards = [
        {
            "question": "Write a poem about the rain.",
            "answer": "Rain, rain.",
            "subjects": ["Poems"],
        },
        {
            "question": "Name some fruits.",
            "answer": "Apple, banana, cherry.",
            "subjects": ["Fruits"],
        },
        {
            "question": "What is a circle?",
            "answer": "A circle is round.",
            "subjects": ["Shapes"],
        },
        {
            "question": "How many days are in a week?",
            "answer": "Seven days.",
            "subjects": ["Dates"],
        },
        {
            "question": "What is the capital letter of the word apple?",
            "answer": "A.",
            "subjects": ["Language"],
        },
        {
            "question": "Spell the word cat.",
            "answer": "C A T.",
            "subjects": ["Language"],
        },
        {
            "question": "What is three plus two?",
            "answer": "Five.",
            "subjects": ["Math"],
        },
        {
            "question": "What time is it now?",
            "answer": "It is noon.",
            "subjects": ["Dates"],
        },
    ]
    monkeypatch.setattr(
        "app.services.embeddings.embed_texts",
        lambda texts: [[float(keyword in text.lower()) for keyword in _KEYWORDS] for text in texts],
    )

    trained = intent_service.train(None, cards)

    assert trained == len(cards)
    kind, source = intent_service.classify(None, "Write a poem about the rain.")
    assert source == "trained"
    assert kind in intent_service.KINDS


def test_the_shape_of_the_question_answers_before_the_head_is_trained(isolated):
    """Without a trained head the shape of the question is still read."""
    kind, source = intent_service.classify(None, "How many days are in a week?")

    assert source == "shape"
    assert kind == "counting"
