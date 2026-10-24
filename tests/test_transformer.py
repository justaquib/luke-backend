"""Tests for the transformer that is trained from scratch on the tenant's own words."""

from app.services import neural
from app.services.neural.settings import TrainingSettings, WritingSettings

CORPUS = [
    "The cow is an animal.",
    "The cow is a farm animal that gives milk.",
    "A cow gives milk.",
    "The moon shines at night.",
    "The night sky is black.",
    "Water is the clear liquid people drink.",
    "A liquid takes the shape of its container.",
]


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
    writer = neural.AnswerTask()
    writer.config = {**writer.config, "d_model": 32, "layers": 1, "heads": 2, "context": 32}

    report = writer.fit(neural.AnswerCorpus(CORPUS), settings=TrainingSettings(steps=30))

    assert report["trained"] is True
    assert report["vocabulary"] > 10
    assert report["loss"] is not None
    assert writer.trained is True


def test_too_little_to_train_on_is_reported():
    writer = neural.AnswerTask()

    report = writer.fit(neural.AnswerCorpus(["Hi."]), settings=TrainingSettings(steps=5))

    assert report["trained"] is False
    assert writer.trained is False


def test_the_model_writes_words_it_knows():
    writer = neural.AnswerTask()
    writer.config = {**writer.config, "d_model": 32, "layers": 1, "heads": 2, "context": 32}
    writer.fit(neural.AnswerCorpus(CORPUS), settings=TrainingSettings(steps=40))

    written = writer.generate("The cow", settings=WritingSettings(max_words=12, seed=3))

    # The model may end at once on a small corpus, but whatever it writes is
    # made of words it knows.
    if written:
        assert "<unk>" not in written
        assert "<pad>" not in written
        assert "<start>" not in written


def test_the_model_scores_what_it_has_seen():
    writer = neural.AnswerTask()
    writer.config = {**writer.config, "d_model": 32, "layers": 1, "heads": 2, "context": 32}
    writer.fit(neural.AnswerCorpus(CORPUS), settings=TrainingSettings(steps=60))

    known = writer.score("The cow is a farm animal.")
    unknown = writer.score("Purple elephants fly sideways at noon.")

    assert known != 0.0
    assert known > unknown


def test_a_saved_model_can_be_loaded_again(tmp_path):
    writer = neural.AnswerTask()
    writer.config = {**writer.config, "d_model": 32, "layers": 1, "heads": 2, "context": 32}
    writer.fit(neural.AnswerCorpus(CORPUS), settings=TrainingSettings(steps=40))
    path = tmp_path / "writer.pt"
    writer.save(path)

    loaded = neural.AnswerTask.load(path)

    assert loaded is not None
    assert loaded.trained is True
    assert loaded.vocabulary == writer.vocabulary
    assert round(loaded.score("The cow is a farm animal."), 4) == round(
        writer.score("The cow is a farm animal."), 4
    )


def test_a_model_that_was_never_trained_says_nothing():
    writer = neural.AnswerTask()

    assert writer.generate("The cow") is None
    assert writer.score("The cow is an animal.") == 0.0
