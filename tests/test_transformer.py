"""Tests for the word model trained from scratch on the tenant's own words."""

import math

from app.services import neural
from app.services.neural.settings import TrainingSettings
from app.services.neural.words import WordTask

CORPUS = [
    "The cow is an animal.",
    "The cow is a farm animal that gives milk.",
    "A cow gives milk.",
    "The moon shines at night.",
    "The night sky is black.",
    "Water is the clear liquid people drink.",
    "A liquid takes the shape of its container.",
]

SMALL_MODEL = {"d_model": 32, "layers": 1, "heads": 2, "context": 32}


def test_the_words_of_a_text_are_tokenised():
    assert neural.tokenize("The cow is an animal.") == [
        "the",
        "cow",
        "is",
        "an",
        "animal",
        ".",
    ]


def test_the_model_trains_on_its_own():
    task = WordTask(config=SMALL_MODEL)

    report = task.fit(CORPUS, settings=TrainingSettings(steps=30))

    assert report["trained"] is True
    assert report["vocabulary"] > 10
    assert report["loss"] is not None
    assert task.trained is True


def test_too_little_to_train_on_is_reported():
    task = WordTask(config=SMALL_MODEL)

    report = task.fit(["Hi."], settings=TrainingSettings(steps=5))

    assert report["trained"] is False
    assert task.trained is False


def test_the_model_writes_words_it_knows():
    task = WordTask(config=SMALL_MODEL)
    task.fit(CORPUS, settings=TrainingSettings(steps=40))

    written = task.generate_sentence("The cow", max_words=12)

    # The model may end at once on a small corpus, but whatever it writes is
    # made of words it knows.
    if written:
        assert "<unk>" not in written
        assert "<pad>" not in written
        assert "<start>" not in written


def test_the_model_scores_the_word_order_it_has_seen():
    task = WordTask(config=SMALL_MODEL)
    task.fit(CORPUS, settings=TrainingSettings(steps=60))

    known = task.score_words("The cow is a farm animal.".rstrip(".").split())
    unseen = task.score_words(["Purple", "elephants", "fly", "sideways"])

    assert known < 0.0
    assert math.isfinite(known)
    assert unseen == 0.0, "a sentence of words it never saw is not scored at all"


def test_a_saved_model_can_be_loaded_again(tmp_path):
    task = WordTask(config=SMALL_MODEL)
    task.fit(CORPUS, settings=TrainingSettings(steps=40))
    path = tmp_path / "words.pt"
    task.save(path)

    loaded = WordTask.load(path)

    assert loaded is not None
    assert loaded.trained is True
    assert loaded.vocabulary == task.vocabulary
    assert round(loaded.score_words("The cow is a farm animal.".rstrip(".").split()), 4) == round(
        task.score_words("The cow is a farm animal.".rstrip(".").split()), 4
    )


def test_a_model_that_was_never_trained_says_nothing():
    task = WordTask(config=SMALL_MODEL)

    assert task.generate_sentence("The cow") is None
    assert task.score_words("The cow is an animal.".rstrip(".").split()) == 0.0
