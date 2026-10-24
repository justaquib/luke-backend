"""The tokenizer both neural models read and write with."""

from app.services.neural import tokens


def test_numbers_are_kept_as_words():
    assert tokens.tokenize("A cat weighs 3.5 kg.") == [
        "a",
        "cat",
        "weighs",
        "3.5",
        "kg",
        ".",
    ]


def test_punctuation_and_line_breaks_have_marks_of_their_own():
    assert tokens.tokenize("Rain, rain!\nGo away.") == [
        "rain",
        ",",
        "rain",
        "!",
        tokens.LINE_BREAK,
        "go",
        "away",
        ".",
    ]


def test_the_written_text_has_no_space_before_its_mark():
    assert tokens.decode_text(["the", "cow", "is", "an", "animal", "."]) == (
        "the cow is an animal."
    )


def test_a_mark_is_not_a_word_that_can_be_written():
    for mark in tokens.SPECIAL_TOKENS:
        assert tokens.is_a_word(mark) is False
    assert tokens.is_a_word("cow") is True


def test_an_apostrophe_stays_one_word():
    assert tokens.tokenize("The bee's hive.") == ["the", "bee's", "hive", "."]
