"""Tests for finding the lesson that answers a question.

The learner has to find the lesson it was taught, even when the question is
asked in other words, and even when two lessons share the same words.
"""

from __future__ import annotations

from app.services.language import preview, runtime, teaching
from app.services.language.matching import _matched_learned_lesson

RUDIMENT_CARD = {
    "question": "GK: What are the rudiments?",
    "answer": "The rudiments are the basic first principles of a subject.",
    "confidence": 0.5,
}

HEAD_CARDS = [
    {
        "question": "What is the head?",
        "answer": "The head is a body part.",
        "confidence": 0.5,
    },
    {
        "question": "What does the head do?",
        "answer": "The head holds the brain and supports the eyes.",
        "confidence": 0.5,
    },
]

ROOT_CARDS = [
    {
        "question": "What is the square root of 16?",
        "answer": "The square root of 16 is 4.",
        "confidence": 0.5,
    },
    {
        "question": "What is the square root of 25?",
        "answer": "The square root of 25 is 5.",
        "confidence": 0.5,
    },
]


def test_the_mark_that_names_the_kind_is_not_a_word_of_the_question():
    """The "GK:" in front of a lesson is not part of the question."""
    matched = _matched_learned_lesson("What are the rudiment?", [RUDIMENT_CARD])

    assert matched is not None
    assert matched["answer"] == RUDIMENT_CARD["answer"]


def test_a_question_asked_in_other_words_still_finds_the_lesson():
    matched = _matched_learned_lesson("What are the rudiments?", [RUDIMENT_CARD])

    assert matched is not None
    assert matched["answer"] == RUDIMENT_CARD["answer"]


def test_two_lessons_that_share_their_words_are_told_apart():
    matched = _matched_learned_lesson("What does the head do?", HEAD_CARDS)

    assert matched is not None
    assert matched["answer"] == "The head holds the brain and supports the eyes."


def test_a_number_in_the_question_is_not_ignored():
    matched = _matched_learned_lesson("What is the square root of 25?", ROOT_CARDS)

    assert matched is not None
    assert matched["answer"] == "The square root of 25 is 5."


def test_a_question_about_something_never_taught_is_not_answered():
    cards = [{"question": "Poem: Rain Rain Go Away", "answer": "Rain, rain, go away."}]

    assert _matched_learned_lesson("Write a short poem about volcanoes.", cards) is None


def test_the_preview_answers_with_the_lesson_that_was_taught(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    runtime._CACHE.clear()
    teaching.teach(
        None,
        "Q: GK: What are the rudiments?\n"
        "A: The rudiments are the basic first principles of a subject.",
        "nursery",
        "General Knowledge",
    )

    answer = preview.preview_answer("What are the rudiment?", None)

    assert answer["source"] == "lesson_cards"
    assert "rudiments are the basic first principles" in answer["answer"]
    runtime._CACHE.clear()
