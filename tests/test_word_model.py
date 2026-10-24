"""Tests for the neural word model that serves the learner surfaces.

The word model keeps the promises those surfaces read: predictions with a
chance and a real count, a written sentence, and numbers it measured rather
than invented.
"""

from app.services.neural.settings import TrainingSettings
from app.services.neural.tokens import tokenize
from app.services.neural.words import PUNCTUATION, WordTask

CORPUS = [
    "The cow is an animal.",
    "The cow is a farm animal that gives milk.",
    "A cow gives milk every day.",
    "The moon shines at night.",
    "The night sky is black and clear.",
    "Water is the clear liquid people drink.",
    "A liquid takes the shape of its container.",
    "The farm is quiet in the morning.",
    "The sky is bright at noon.",
    "People drink water every day.",
]

SMALL = {"d_model": 24, "layers": 1, "heads": 2, "context": 12}
TRAINING = TrainingSettings(
    steps=40,
    batch_size=4,
    check_every=4,
    min_steps=2,
    patience=6,
    validation_share=0.25,
    min_validation=2,
    min_validation_groups=2,
    max_seconds=60,
)


def _trained() -> WordTask:
    task = WordTask(SMALL)
    report = task.fit(CORPUS, settings=TRAINING)
    assert report["trained"] is True
    return task


def test_the_word_model_trains_and_reports_its_numbers():
    task = _trained()

    assert task.trained is True
    assert len(task.vocabulary) > 8
    assert task.sentence_count == len(CORPUS)
    assert task.token_count > task.sentence_count
    assert task.transition_count > 0
    assert task.word_counts["cow"] == 3
    assert task.report["validation_loss"] is not None
    assert task.report["validation_examples"] >= 2
    assert task.report["next_word_accuracy"] is not None
    assert task.report["perplexity"] is not None


def test_the_next_words_are_words_it_knows_with_a_chance():
    task = _trained()

    rows = task.predict_next("The cow", limit=3)

    assert rows
    for row in rows:
        assert row["source"] == "neural"
        assert row["word"] not in PUNCTUATION
        assert not str(row["word"]).startswith("<")
        assert 0.0 <= row["probability"] <= 1.0
        assert row["count"] >= 0


def test_a_known_prefix_only_predicts_words_of_the_stage():
    task = _trained()

    rows = task.predict_next("The cow", limit=3, allowed={"milk", "animal", "water"})

    assert all(row["word"] in {"milk", "animal", "water"} for row in rows)


def test_a_prefix_with_no_known_word_predicts_from_the_start():
    task = _trained()

    rows = task.predict_next("xyzzy", limit=2)

    assert all(row["source"] == "neural" for row in rows)


def test_a_saved_word_model_can_be_loaded_again(tmp_path):
    task = _trained()
    path = tmp_path / "words.pt"
    task.save(path, {"sentences": len(CORPUS)})

    loaded = WordTask.load(path)

    assert loaded is not None
    assert loaded.trained is True
    assert loaded.vocabulary == task.vocabulary
    assert loaded.sentence_count == task.sentence_count
    assert loaded.word_counts == task.word_counts
    assert [row["word"] for row in loaded.predict_next("The cow", limit=3)] == [
        row["word"] for row in task.predict_next("The cow", limit=3)
    ]


def test_an_untrained_word_model_says_nothing(tmp_path):
    task = WordTask()
    task.save(tmp_path / "nothing.pt")

    assert task.predict_next("The cow") == []
    assert task.generate_sentence() is None
    assert task.score_words(["the", "cow"]) == 0.0
    assert task.taught_sentence("the cow") is None
    assert not (tmp_path / "nothing.pt").exists()


def test_a_written_sentence_says_words_the_model_knows():
    task = _trained()

    written = task.generate_sentence(seed="the cow", max_words=8)

    if written is not None:
        assert "<unk>" not in written
        assert "<pad>" not in written
        assert "<start>" not in written
        assert all(token in task.vocabulary for token in tokenize(written))


def test_the_order_of_words_it_has_seen_is_scored():
    task = _trained()

    known = task.score_words(["the", "cow", "gives", "milk"])
    jumbled = task.score_words(["milk", "the", "cow", "gives"])

    assert isinstance(known, float)
    assert isinstance(jumbled, float)
    assert known != jumbled
