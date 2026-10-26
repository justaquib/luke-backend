"""Tests for finding a taught lesson through a synonym of the question word."""

from app.services import dictionary
from app.services.language import answering, fallback, runtime, teaching

CLARINET_LESSON = {
    "id": "woodwind-1",
    "question": "What is a woodwind?",
    "answer": "A woodwind makes the sound of a clarinet.",
    "confidence": 0.5,
}


def _expansions(*synonyms: str):
    return lambda word, per_word=3, max_terms=6: list(synonyms)


def test_a_synonym_reaches_the_one_lesson(monkeypatch):
    monkeypatch.setattr(dictionary, "search_expansions", _expansions("woodwind"))

    attempt = fallback.find_fallback("What is a clarinet?", [CLARINET_LESSON])

    assert attempt is not None
    assert attempt.answer is not None
    assert attempt.answer.answer == CLARINET_LESSON["answer"]
    assert attempt.answer.matched_question == "What is a woodwind?"
    assert attempt.connection.outcome == "pass"
    assert "woodwind" in attempt.connection.detail
    assert "clarinet" in attempt.connection.detail


def test_two_synonyms_that_reach_different_lessons_are_not_guessed_between(monkeypatch):
    monkeypatch.setattr(dictionary, "search_expansions", _expansions("shore", "depository"))
    lessons = [
        {"id": "a", "question": "What is a shore?", "answer": "A shore is the land beside water."},
        {"id": "b", "question": "What is a depository?", "answer": "A depository is a store."},
    ]

    attempt = fallback.find_fallback("What is a bank?", lessons)

    assert attempt is not None
    assert attempt.answer is None
    assert attempt.connection.outcome == "fail"
    assert "2 different lessons" in attempt.connection.detail


def test_a_reached_lesson_must_share_a_word_with_the_question(monkeypatch):
    monkeypatch.setattr(dictionary, "search_expansions", _expansions("look"))
    lessons = [{"id": "c", "question": "What is a look?", "answer": "A look is a gaze."}]

    attempt = fallback.find_fallback("What is a glim?", lessons)

    assert attempt is not None
    assert attempt.answer is None
    assert attempt.connection.outcome == "fail"
    assert "shares no key word" in attempt.connection.detail


def test_no_synonym_is_reported_as_a_failure(monkeypatch):
    monkeypatch.setattr(dictionary, "search_expansions", _expansions())

    attempt = fallback.find_fallback("What is a glim?", [CLARINET_LESSON])

    assert attempt is not None
    assert attempt.answer is None
    assert "No known synonym" in attempt.connection.detail


def test_a_question_that_is_not_a_definition_is_left_alone(monkeypatch):
    monkeypatch.setattr(dictionary, "search_expansions", _expansions("rain"))

    assert fallback.find_fallback("Tell me a poem about rain.", [CLARINET_LESSON]) is None


def test_a_taught_lesson_answers_a_word_the_dictionary_lacks(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    runtime._CACHE.clear()
    teaching.teach(
        None,
        "Q: What is a woodwind?\nA: A woodwind makes the sound of a clarinet.",
        stage="nursery",
        subject="Music",
    )
    monkeypatch.setattr(
        dictionary,
        "answer_dictionary_question_with_status",
        lambda question: ('I couldn\'t find "clarinet" in the offline English dictionary.', False),
    )
    monkeypatch.setattr(dictionary, "search_expansions", _expansions("woodwind"))

    result = answering.answer_from_curriculum("What is a clarinet?", None)

    assert result is not None
    assert result.source == "lesson_cards"
    assert result.answer == "A woodwind makes the sound of a clarinet."
    assert result.matched_question == "What is a woodwind?"
    assert result.evidence[0].source == "fallback"
    assert result.evidence[0].outcome == "pass"
    runtime._CACHE.clear()


def test_a_dictionary_entry_still_answers_when_no_lesson_is_reached(monkeypatch):
    monkeypatch.setattr(
        dictionary,
        "answer_dictionary_question_with_status",
        lambda question: ("Clarinet (noun): a woodwind instrument.", True),
    )
    monkeypatch.setattr(dictionary, "search_expansions", _expansions())
    state = {"lesson_cards": [], "word_cards": [], "lessons": []}

    result = answering._generated_answer("What is a clarinet?", state, "nursery")

    assert result is not None
    assert result.source == "dictionary"
    assert result.answer == "Clarinet (noun): a woodwind instrument."
    assert result.evidence[0].source == "dictionary"
    assert result.evidence[0].outcome == "match"
    assert result.evidence[1].source == "fallback"
    assert result.evidence[1].outcome == "fail"


def test_a_lesson_reached_through_a_synonym_answers_over_the_dictionary(monkeypatch):
    monkeypatch.setattr(
        dictionary,
        "answer_dictionary_question_with_status",
        lambda question: ("Clarinet (noun): a woodwind instrument.", True),
    )
    monkeypatch.setattr(dictionary, "search_expansions", _expansions("woodwind"))
    state = {"lesson_cards": [CLARINET_LESSON], "word_cards": [], "lessons": []}

    result = answering._generated_answer("What is a clarinet?", state, "nursery")

    assert result is not None
    assert result.source == "lesson_cards"
    assert result.answer == CLARINET_LESSON["answer"]
    assert result.matched_question == "What is a woodwind?"
    assert result.evidence[0].source == "fallback"
    assert result.evidence[0].outcome == "pass"
