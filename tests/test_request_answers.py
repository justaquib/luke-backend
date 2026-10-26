"""Tests for answering a request from the taught material.

The request path serves "write a poem about the rain" from the lesson the
trainer taught. It never rewords the lesson: the taught answer is the answer.
"""

from app.services.language import requests

POEM_CARD = {
    "id": "poem-1",
    "question": "Poem: Rain Rain Go Away",
    "answer": "Rain, rain, go away,\nCome again another day.\nLittle children want to play.",
    "subject": "Language, Poems",
    "subjects": ["Language", "Poems"],
    "grade_id": "lkg",
}


def test_a_request_is_answered_from_the_taught_material():
    result = requests.answer_request("Write a short poem about the rain.", None, [POEM_CARD])

    assert result is not None
    assert result["answer"] == POEM_CARD["answer"]
    assert result["source"] == "taught"
    assert result["topic"] == "rain"
    assert result["kind"] == "poem"


def test_a_request_about_something_never_taught_is_refused():
    result = requests.answer_request("Write a short poem about volcanoes.", None, [POEM_CARD])

    assert result is None


def test_a_question_is_not_a_request():
    result = requests.answer_request("What is the rain?", None, [POEM_CARD])

    assert result is None


def test_a_lesson_with_no_answer_does_not_answer_a_request():
    empty = {**POEM_CARD, "answer": "   "}

    result = requests.answer_request("Write a short poem about the rain.", None, [empty])

    assert result is None
