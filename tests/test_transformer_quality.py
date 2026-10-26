"""Tests for the material the word model learns from, and its own grammar."""

from __future__ import annotations

from app.services import neural
from app.services.neural import difficulty, grammar, quality
from app.services.neural import examples as example_service
from app.services.neural.settings import TrainingSettings


def test_a_repeated_sentence_is_not_worth_the_training():
    """A sentence that says the same thing again and again is left out."""
    kept = quality.keep(
        ["The cat sat on the mat and the cat sat on the mat and the cat sat on the mat."]
    )

    assert kept == []


def test_a_sentence_that_is_too_short_or_too_long_is_left_out():
    """A sentence needs a few words, and not too many."""
    kept = quality.keep(["A cat.", " ".join(["word"] * 60)])

    assert kept == []


def test_a_sentence_with_words_the_model_never_sees_is_left_out():
    """A sentence of words that appear nowhere else teaches nothing."""
    sentences = ["The kite is a light frame with a tail."] * 20 + ["Zorblax quiffled."]

    assert "Zorblax quiffled." not in quality.keep(sentences)


def test_a_simple_a_compound_and_a_complex_sentence_are_told_apart():
    """The three stages of the training are told apart by the words."""
    assert difficulty.level_of("A kite is a light frame with a tail.") == difficulty.SIMPLE
    assert difficulty.level_of("A kite is a frame and it has a tail.") == difficulty.COMPOUND
    assert difficulty.level_of("A kite has a tail because it is light.") == difficulty.COMPLEX


def test_the_steps_the_training_asked_for_meet_the_material():
    """Steps of zero mean the passes over the material decide how long."""
    one = [example_service.Example([1, 2, 3], "one")]
    many = [example_service.Example([1, 2, 3], "one")] * 40000

    assert neural.training._steps_for(one, TrainingSettings()) == 400
    asked = neural.training._steps_for(many, TrainingSettings())
    assert asked == 2500
    assert neural.training._steps_for(many, TrainingSettings(steps=60)) == 60


def test_the_word_model_keeps_its_own_horizon():
    """The word model takes eight passes when the steps are not fixed."""
    assert neural.words._paced(TrainingSettings()).passes == neural.words.WORD_PASSES
    assert neural.words._paced(TrainingSettings(passes=2)).passes == 2
    assert neural.words._paced(TrainingSettings(steps=50)).passes is None


def test_a_verb_that_does_not_agree_with_its_subject_is_a_note():
    """The subject and the verb have to agree."""
    assert grammar.notes("The cat sit on the mat.")
    assert not grammar.notes("The cats sit on the mat.")
