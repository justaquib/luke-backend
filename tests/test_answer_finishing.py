"""Tests for the last touch on an answer: a capital and a full stop."""

from app.services.language.finishing import finish


def test_a_sentence_gets_its_capital_and_full_stop():
    assert finish("  the eye is a body part  ") == "The eye is a body part."


def test_a_question_keeps_its_question_mark():
    assert finish("what is the eye?") == "What is the eye?"


def test_an_exclamation_is_kept():
    assert finish("rain, rain, go away!") == "Rain, rain, go away!"


def test_a_number_ends_with_a_full_stop():
    assert finish("the answer is 42") == "The answer is 42."


def test_a_closing_bracket_is_preserved():
    assert finish("the eye (a body part)") == "The eye (a body part)"


def test_finishing_twice_changes_nothing():
    once = finish("the eye is a body part")

    assert finish(once) == once


def test_an_empty_answer_stays_empty():
    assert finish("   ") == ""
