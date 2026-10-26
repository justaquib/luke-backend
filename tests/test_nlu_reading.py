"""Tests for reading a question before it is answered."""

from app.services import dictionary, nlp
from app.services.nlu import read_question


def test_a_question_is_read_word_by_word():
    reading = read_question("What does the eye do?")

    assert [word.word for word in reading.words] == ["What", "does", "the", "eye", "do"]
    assert reading.key_words == ("eye",)
    assert reading.question_type == "definition"
    assert all(isinstance(word.lemma, str) for word in reading.words)
    assert all(isinstance(word.part_of_speech, str) for word in reading.words)


def test_a_key_word_is_read_with_its_meaning_and_synonyms(monkeypatch):
    monkeypatch.setattr(
        dictionary,
        "lookup_word",
        lambda word: (
            {"definitions": ["the organ of sight"], "part_of_speech": "noun"}
            if word == "eye"
            else None
        ),
    )
    monkeypatch.setattr(
        dictionary,
        "search_expansions",
        lambda word, per_word=2, max_terms=2: ["oculus", "optic"] if word == "eye" else [],
    )

    reading = read_question("What does the eye do?")
    eye = next(word for word in reading.words if word.word == "eye")
    what = next(word for word in reading.words if word.word == "What")

    assert eye.meaning == "the organ of sight"
    assert eye.part_of_speech == "noun"
    assert eye.synonyms == ("oculus", "optic")
    assert what.meaning == ""


def test_a_dictionary_that_cannot_be_read_still_leaves_a_reading(monkeypatch):
    monkeypatch.setattr(
        dictionary, "lookup_word", lambda word: (_ for _ in ()).throw(ImportError())
    )
    monkeypatch.setattr(
        dictionary,
        "search_expansions",
        lambda *args, **kwargs: (_ for _ in ()).throw(ImportError()),
    )

    reading = read_question("What does the eye do?")

    assert reading.key_words == ("eye",)
    assert [word.word for word in reading.words] == ["What", "does", "the", "eye", "do"]
    assert all(word.meaning == "" for word in reading.words)


def test_the_kind_of_question_is_named_from_its_shape():
    assert read_question("What is 2 + 2?").question_type == "calculation"
    assert read_question("How many apples are there?").question_type == "counting"
    assert read_question("Tell me about the rain.").question_type == "lookup"


def test_a_reading_carries_key_words_even_without_a_parse(monkeypatch):
    monkeypatch.setattr(nlp, "get_nlp", lambda: (_ for _ in ()).throw(ImportError()))

    reading = read_question("What does the eye do?")

    assert reading.question_type == "definition"
    assert reading.key_words == ("eye",)
    assert [word.word for word in reading.words] == ["eye"]
