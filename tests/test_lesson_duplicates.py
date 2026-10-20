"""Tests for the word by word lesson duplicate check."""

from app.services import language_logic


def test_the_same_words_are_the_same_question():
    assert language_logic._same_words("What is your name?", "what is your name?") is True
    assert language_logic._same_words("Who is your mother?", "your mother who is") is True


def test_one_different_word_is_a_different_question():
    assert language_logic._same_words("What is the opposite of in?", "What is the opposite of loud?") is False
    assert language_logic._same_words("Who is Daraksha?", "Who is Aquib Shahbaz's wife?") is False


def test_a_question_with_another_word_is_not_reported_as_similar():
    score = language_logic._word_by_word_similarity("What is the opposite of in?", "What is the opposite of loud?")

    assert score == 0.0


def test_a_question_with_one_extra_word_is_similar():
    score = language_logic._word_by_word_similarity("What is a circle?", "What is a circle shape?")

    assert score == 0.9


def test_a_question_with_two_extra_words_is_not_similar():
    score = language_logic._word_by_word_similarity(
        "What is the capital of France?",
        "What is the capital of France and Germany?",
    )

    assert score == 0.0


def test_the_same_question_with_another_answer_is_found(monkeypatch, tmp_path):
    monkeypatch.setattr(language_logic, "_DATA_DIR", tmp_path)
    language_logic._CACHE.clear()
    language_logic.teach(
        None,
        "Q: What is the opposite of in?\nA: Out!",
        "nursery",
        "Opposites",
    )

    report = language_logic.check_duplicate_lessons(
        None,
        "Q: What is the opposite of in?\nA: Not up.",
        "nursery",
        "Opposites",
    )

    assert report["exact"] == []
    assert [item["question"] for item in report["same_question"]] == ["What is the opposite of in?"]
    assert report["similar"] == []


def test_a_different_opposite_is_not_a_duplicate(monkeypatch, tmp_path):
    monkeypatch.setattr(language_logic, "_DATA_DIR", tmp_path)
    language_logic._CACHE.clear()
    language_logic.teach(
        None,
        "Q: What is the opposite of in?\nA: Out!",
        "nursery",
        "Opposites",
    )

    report = language_logic.check_duplicate_lessons(
        None,
        "Q: What is the opposite of sideways?\nA: Not up.",
        "nursery",
        "Opposites",
    )

    assert report["exact"] == []
    assert report["same_question"] == []
    assert report["similar"] == []
