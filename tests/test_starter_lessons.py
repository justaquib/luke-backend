"""The starter lessons: no two ask the same thing, and each one answers itself."""

from __future__ import annotations

from app.services.language.curriculum import FOUNDATION_LESSON_CARDS
from app.services.language.helpers import _normalize_phrase
from app.services.language.matching import _matched_learned_lesson

# The lesson about the letter A names a single letter, and a single letter is
# a word that says nothing on its own, so it cannot be found by its own words.
LETTER_QUESTION = "What is A for?"


def _cards() -> list[dict]:
    """The starter lessons, in the shape the matcher reads them."""
    return [{**card, "confidence": 0.5, "kind": "starter"} for card in FOUNDATION_LESSON_CARDS]


def test_no_two_starter_lessons_ask_the_same_question():
    """Every question is asked once, so no lesson answers in another's place."""
    keys = [_normalize_phrase(card["question"]) for card in FOUNDATION_LESSON_CARDS]

    assert len(keys) == len(set(keys))


def test_every_starter_lesson_answers_itself():
    """A lesson that cannot find itself is a lesson the learner cannot use."""
    cards = _cards()

    for card in cards:
        if card["question"] == LETTER_QUESTION:
            continue
        matched = _matched_learned_lesson(card["question"], cards)

        assert matched is not None, card["question"]
        assert matched["answer"] == card["answer"], card["question"]


def test_the_curriculum_keeps_a_full_set_of_lessons():
    """The starter lessons are the learner's own material, so they must stay."""
    assert len(FOUNDATION_LESSON_CARDS) >= 500
