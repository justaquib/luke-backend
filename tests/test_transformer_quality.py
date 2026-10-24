"""Tests for the writer: the data it learns from, the way it learns, and the
way it reads its own answer before it is given away.
"""

from __future__ import annotations

from app.services import neural
from app.services.neural import critique as critique_service
from app.services.neural import difficulty, grammar, quality
from app.services.neural import examples as example_service
from app.services.neural import measurement as measurement_service
from app.services.neural.settings import TrainingSettings

SMALL_MODEL = {"d_model": 32, "layers": 1, "heads": 2, "context": 64}

SENTENCES = [
    "Rain, rain, go away.",
    "Come again another day.",
    "Little children want to play.",
    "A kite is a light frame with a tail.",
]


def _writer(steps: int = 60):
    """A small model trained on a handful of lessons."""
    writer = neural.AnswerTask(config=SMALL_MODEL)
    writer.fit(
        neural.AnswerCorpus(
            SENTENCES,
            [("Write a short poem about the rain.", "Rain, rain, go away.")],
            [],
        ),
        settings=TrainingSettings(steps=steps),
    )
    return writer


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


def test_the_examples_are_learned_from_the_easy_ones_to_the_hard_ones():
    """The training is a curriculum: the simple sentences come first."""
    writer = neural.AnswerTask(config=SMALL_MODEL)
    writer._build_vocabulary(
        [
            "a",
            "kite",
            "is",
            "light",
            "frame",
            "with",
            "tail",
            "cat",
            "sat",
            "because",
            "it",
            "was",
            ".",
        ]
    )
    built = example_service.build(
        writer,
        [
            "The cat sat because it was tired.",
            "A kite is a light frame with a tail.",
        ],
        [],
        [],
    )
    words = [writer.vocabulary[token] for token in built[0].sequence]

    assert words[1] == "a"
    assert len(built) == 2


def test_a_simple_a_compound_and_a_complex_sentence_are_told_apart():
    """The three stages of the training are told apart by the words."""
    assert difficulty.level_of("A kite is a light frame with a tail.") == difficulty.SIMPLE
    assert difficulty.level_of("A kite is a frame and it has a tail.") == difficulty.COMPOUND
    assert difficulty.level_of("A kite has a tail because it is light.") == difficulty.COMPLEX


def test_the_training_says_how_it_went():
    """The training reports the loss, the steps and the curve of the loss."""
    writer = _writer(steps=40)
    report = writer.fit(
        neural.AnswerCorpus(
            SENTENCES,
            [("Write a short poem about the rain.", "Rain, rain, go away.")],
            [],
        ),
        settings=TrainingSettings(steps=40, check_every=10),
    )

    assert report["trained"] is True
    assert report["steps"] == 40
    assert report["loss"] > 0
    assert report["curve"]
    assert report["best_loss"] <= report["loss"]


def test_the_training_stops_when_the_loss_stops_going_down():
    """The training ends early when it has stopped learning."""
    report = neural.AnswerTask(config=SMALL_MODEL).fit(
        neural.AnswerCorpus(
            SENTENCES,
            [("Write a short poem about the rain.", "Rain, rain, go away.")],
            [],
        ),
        settings=TrainingSettings(steps=2000, check_every=10, patience=2, min_steps=20),
    )

    assert report["stopped_early"] is True
    assert report["steps"] < 2000


def test_the_steps_the_training_asked_for_meet_the_material():
    """Steps of zero mean the passes over the material decide how long."""
    one = [example_service.Example([1, 2, 3], 1, "one")]
    many = [example_service.Example([1, 2, 3], 1, "one")] * 40000

    assert neural.training._steps_for(one, TrainingSettings()) == 400
    asked = neural.training._steps_for(many, TrainingSettings())
    assert asked == 2500
    assert neural.training._steps_for(many, TrainingSettings(steps=60)) == 60


def test_the_word_model_keeps_its_own_horizon():
    """The answer model anneals one pass; the smaller word model takes eight."""
    assert neural.words._paced(TrainingSettings()).passes == neural.words.WORD_PASSES
    assert neural.words._paced(TrainingSettings(passes=2)).passes == 2
    assert neural.words._paced(TrainingSettings(steps=50)).passes is None


def test_the_loss_shown_is_the_knowledge_of_the_model_not_one_batch():
    """The reported loss is the average the model is left knowing."""
    writer = neural.AnswerTask(config=SMALL_MODEL)
    instruction = ("Write a short poem about the rain.", "Rain, rain, go away.")
    words = example_service.all_words(SENTENCES, [instruction], ())
    writer._build_vocabulary(words)
    kept = quality.keep(SENTENCES)
    examples = example_service.build(writer, kept, [instruction], ())
    config = {**writer.config, "pad_id": writer.index.get("<pad>", 0)}
    model, report, _ = neural.training.train_model(
        len(writer.vocabulary), examples, config, settings=TrainingSettings(steps=2)
    )
    seed = TrainingSettings().seed + 1
    batches = measurement_service.fixed_batches(examples, config, TrainingSettings(), seed)
    read_again = measurement_service.read_loss(model, batches, config["pad_id"])

    assert report.loss > 0
    assert abs(report.loss - read_again) <= 1e-4


def test_the_model_says_how_sure_it_was_about_every_word():
    """The chance of every word is kept, so the answer can be read again."""
    writer = _writer()
    writer.respond("Write a short poem about the rain.", context="Rain, rain, go away.")

    assert writer.last_token_confidences
    assert len(writer.last_token_confidences) == len(writer.last_written_words)
    assert 0.0 <= writer.last_least_chance <= 1.0


def test_a_sentence_without_a_verb_is_a_problem():
    """A text that is not a sentence cannot be given away."""
    assert grammar.problems("A circle round shape.")
    assert not grammar.problems("A circle is a round shape.")


def test_a_verb_that_does_not_agree_with_its_subject_is_a_note():
    """The subject and the verb have to agree."""
    assert grammar.notes("The cat sit on the mat.")
    assert not grammar.notes("The cats sit on the mat.")


def test_the_critique_finds_the_word_the_model_was_not_sure_of():
    """The critique says what the model is not sure about."""
    result = critique_service.critique(
        "A kite is a light frame with a tail.",
        [0.9, 0.9, 0.1, 0.8, 0.8, 0.8, 0.8],
        ["A", "kite", "is", "a", "light", "frame", "tail"],
    )

    assert result.problems
    assert "a" in result.unsure_about


def test_an_answer_that_does_not_say_what_the_material_says_is_a_problem():
    """The answer has to come from the material it was given."""
    result = critique_service.critique(
        "The moon is made of cheese.",
        [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        ["The", "moon", "is", "made", "of", "cheese"],
        material="A kite is a light frame with a tail.",
    )

    assert result.problems
    assert result.grounded is False


def test_an_answer_that_says_a_pair_again_and_again_is_a_problem():
    """The model stuck in a loop is not worth giving away."""
    answer = "Rain, go away, go away, go away, go away."

    assert quality.repeats_itself(answer)
    result = critique_service.critique(
        answer, [0.9] * 9, ["Rain", "go", "away", "go", "away", "go", "away", "go", "away"]
    )

    assert any("same words" in problem for problem in result.problems)


def test_a_poem_that_says_its_line_again_is_not_a_loop():
    """A poem says the same pair twice on purpose, and never three times."""
    assert not quality.repeats_itself(
        "Rain, rain, go away,\nCome again another day.\nLittle children want to play,"
        "\nRain, rain, go away."
    )
