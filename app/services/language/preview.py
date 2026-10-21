"""Answering from the local curriculum without the embedding index."""

from __future__ import annotations

import re

from app.services import dictionary
from app.services.language import runtime
from app.services.language.alphabet import _answer_alphabet_question
from app.services.language.basics import (
    _answer_counting_question,
    _answer_number_sequence_question,
    _answer_number_spelling_question,
    answer_assistant_age,
    answer_current_date_time,
)
from app.services.language.calculations import _answer_calculation_question
from app.services.language.curriculum import (
    FOUNDATION_LESSON_CARDS,
    STAGES,
    _allowed_grades,
    _foundation_sentences_for_stage,
    _foundation_words_for_stage,
    _tenant_words_for_stage,
)
from app.services.language.grammar import apply_requested_count
from app.services.language.lessons import all_lesson_cards
from app.services.language.matching import _answer_learned_lesson, _matched_learned_lesson
from app.services.language.patterns import _NEXT_WORD_REQUEST, _SENTENCE_REQUEST
from app.services.language.stages import _answer_letter_question, _asked_word
from app.services.language.writing import _answer_from_sentence_examples


def _current_stage(state: dict) -> str:
    return next(
        (item["id"] for item in STAGES if item["id"] not in state["passed_stages"]), "complete"
    )


def _curriculum_quick(question: str) -> dict | None:
    calculation = _answer_calculation_question(question)
    if calculation:
        return {"answer": calculation, "matched_question": ""}
    current_date_time = answer_current_date_time(question)
    if current_date_time:
        return {"answer": current_date_time, "matched_question": ""}
    for answer in (
        _answer_number_spelling_question(question),
        _answer_counting_question(question),
        _answer_number_sequence_question(question),
        _answer_alphabet_question(question),
    ):
        if answer:
            return {"answer": answer, "matched_question": ""}
    return None


def _curriculum_word(question: str, state: dict, current_stage: str) -> dict | None:
    word = _asked_word(question)
    if not word:
        return None
    for item in [
        *_tenant_words_for_stage(state, current_stage),
        *_foundation_words_for_stage(current_stage),
    ]:
        if item["word"].casefold() == word:
            return {
                "answer": f"{item['word'].capitalize()} means {item['meaning']}.",
                "matched_question": f"What does {item['word']} mean?",
            }
    return None


def answer_from_curriculum(question: str, tenant_id: int | None) -> str | None:
    """Recall letter, word, and lesson facts explicitly taught in the curriculum."""
    detail = answer_from_curriculum_detail(question, tenant_id)
    return detail["answer"] if detail else None


def answer_from_curriculum_detail(
    question: str, tenant_id: int | None, include_lessons: bool = True
) -> dict | None:
    """The curriculum answer plus the question it was taught as.

    ``include_lessons`` is off for the normal chat path, where the taught
    lessons are matched semantically from the embedding index instead.
    """
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        current_stage = _current_stage(state)
        quick = _curriculum_quick(question)
        if quick:
            return quick
        answer = _answer_letter_question(question, state["letters"])
        if answer:
            return {"answer": answer, "matched_question": ""}
        word = _curriculum_word(question, state, current_stage)
        if word:
            return word
        if include_lessons:
            allowed_grades = _allowed_grades(current_stage)
            lessons = [
                item
                for item in [*state["lesson_cards"], *FOUNDATION_LESSON_CARDS]
                if item.get("grade_id", "nursery") in allowed_grades
            ]
            matched = _matched_learned_lesson(question, lessons)
            if matched:
                return {
                    "answer": matched["answer"],
                    "matched_question": matched.get("question", ""),
                }
    return None


def _preview_quick(question: str, state: dict) -> dict[str, str] | None:
    calculation = _answer_calculation_question(question)
    if calculation:
        return {"answer": calculation, "source": "number_cards"}
    age_answer = answer_assistant_age(question)
    if age_answer:
        return {"answer": age_answer, "source": "assistant_profile"}
    current_date_time = answer_current_date_time(question)
    if current_date_time:
        return {"answer": current_date_time, "source": "lesson_cards"}
    for answer in (
        _answer_number_spelling_question(question),
        _answer_counting_question(question),
        _answer_number_sequence_question(question),
    ):
        if answer:
            return {"answer": answer, "source": "number_cards"}
    alphabet_answer = _answer_alphabet_question(question)
    if alphabet_answer:
        return {"answer": alphabet_answer, "source": "lesson_cards"}
    answer = _answer_letter_question(question, state["letters"])
    if answer:
        return {"answer": answer, "source": "letters"}
    return None


def _preview_word(question: str, state: dict, current_stage: str) -> dict[str, str] | None:
    word = _asked_word(question)
    if not word:
        return None
    for item in [
        *_tenant_words_for_stage(state, current_stage),
        *_foundation_words_for_stage(current_stage),
    ]:
        if item["word"].casefold() == word:
            return {
                "answer": f"{item['word'].capitalize()} means {item['meaning']}.",
                "source": "word_cards",
            }
    return None


def _preview_lesson(question: str, state: dict, current_stage: str) -> dict[str, str] | None:
    available_grades = _allowed_grades(current_stage)
    available_lessons = [
        item
        for item in [*state["lesson_cards"], *FOUNDATION_LESSON_CARDS]
        if item.get("grade_id", "nursery") in available_grades
    ]
    answer = _answer_learned_lesson(question, available_lessons)
    if not answer:
        return None
    matched = _matched_learned_lesson(question, available_lessons)
    return {
        "answer": apply_requested_count(question, answer),
        "source": "lesson_cards",
        "matched_question": matched.get("question", "") if matched else "",
    }


def _preview_generated(question: str, state: dict, current_stage: str) -> dict[str, str] | None:
    # A word the learner has not been taught still has a meaning in the
    # offline dictionary, so answer "what is boy here?" instead of giving up.
    dictionary_answer = dictionary.answer_dictionary_question(question)
    if dictionary_answer:
        return {"answer": dictionary_answer, "source": "dictionary"}
    next_match = _NEXT_WORD_REQUEST.search(question.strip())
    if next_match:
        prefix = (next_match.group(1) or next_match.group(2) or "").strip(" \t\r\n'\".,?!")
        options = runtime._model(state, current_stage).predict_next(prefix)
        if options:
            return {
                "answer": (
                    f"From my sentence lessons, after “{prefix}” I predict “{options[0]['word']}”."
                ),
                "source": "sentence_patterns",
            }
    sentence_match = _SENTENCE_REQUEST.search(question.strip())
    if sentence_match:
        requested = re.sub(r"[^A-Za-z'’-].*$", "", sentence_match.group(1)).strip(" '’-")
        answer = (
            runtime._model(state, current_stage).generate_sentence(seed=requested)
            if requested
            else None
        )
        if answer:
            return {"answer": answer, "source": "sentence_patterns"}
    examples = [
        *(item["text"] for item in state["lessons"]),
        *_foundation_sentences_for_stage(current_stage),
    ]
    answer = _answer_from_sentence_examples(question, examples)
    if answer:
        return {"answer": answer, "source": "sentence_examples"}
    return None


def _preview_request(question: str, tenant_id: int | None) -> dict[str, str] | None:
    """Answer a request for something to be written, from the taught material.

    A request like "write a short poem about the rain" is not a question, so
    it is not looked up. The lesson about the topic is found instead, and the
    trained model writes the answer from it.
    """
    from app.services import brain
    from app.services.language import requests as request_service

    try:
        cards = all_lesson_cards(tenant_id)
    except (OSError, ValueError):
        return None
    result = request_service.answer_request(
        question, tenant_id, cards, writer=brain.load_writer(tenant_id)
    )
    if result is None:
        return None
    return {"answer": result["answer"], "source": result["source"]}


def preview_answer(question: str, tenant_id: int | None) -> dict[str, str]:
    """Answer a Trainer preview using only this learner's local curriculum."""
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        current_stage = _current_stage(state)
        for result in (
            _preview_quick(question, state),
            _preview_word(question, state, current_stage),
            _preview_lesson(question, state, current_stage),
            _preview_request(question, tenant_id),
            _preview_generated(question, state, current_stage),
        ):
            if result:
                return result
        return {
            "answer": (
                "I haven't learned an answer to that from my lessons yet. Teach me the word, "
                "sentence, or question and answer, then ask again."
            ),
            "source": "not_learned",
        }
