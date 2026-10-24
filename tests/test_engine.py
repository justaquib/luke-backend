"""Tests for the engine that owns the models of one tenant.

The engine is the only thing that loads checkpoints, trains and decides which
word backend serves. These tests keep its files apart, its cache honest and
its reports measured.
"""

import math
import os
from typing import Any

import pytest

from app.services.neural import checkpoint, store
from app.services.neural import engine as engine_module
from app.services.neural.engine import TrainingJob, engine_for
from app.services.neural.settings import TrainingSettings
from app.services.neural.tasks import AnswerCorpus

CORPUS = [
    "The cow is an animal.",
    "The cow is a farm animal that gives milk.",
    "A cow gives milk.",
    "The moon shines at night.",
    "The night sky is black and clear.",
    "Water is the clear liquid people drink.",
    "A liquid takes the shape of its container.",
    "The farm is quiet in the morning.",
    "The sky is bright at noon.",
    "People drink water every day.",
]

SMALL_ANSWER = {"d_model": 24, "layers": 1, "heads": 2, "context": 24}
SMALL_WORDS = {"d_model": 24, "layers": 1, "heads": 2, "context": 12}
SMALL_TRAINING = TrainingSettings(
    steps=6,
    batch_size=4,
    check_every=2,
    min_steps=1,
    patience=4,
    validation_share=0.25,
    min_validation=2,
    min_validation_groups=2,
    max_seconds=60,
)


@pytest.fixture
def brain(monkeypatch, tmp_path):
    """Keep every model this test writes in a folder of its own."""
    monkeypatch.setattr(store, "brain_dir", lambda: tmp_path)
    return tmp_path


def _job(config, training=SMALL_TRAINING) -> TrainingJob:
    return TrainingJob(corpus={"sentences": len(CORPUS)}, config=config, settings=training)


def test_a_tenant_without_models_reports_no_word_model(brain):
    state = engine_for(9101).state()

    assert state["schema"] == 0
    assert state["answer"]["trained"] is False
    assert state["words"]["trained"] is False
    assert state["words"]["backend"] == "untrained"
    assert state["words"]["backend_reason"]
    assert state["runs"] == []


def test_a_trained_answer_model_is_saved_and_recorded(brain):
    engine = engine_for(9102)

    report = engine.train_answer(AnswerCorpus(CORPUS), _job(SMALL_ANSWER))

    assert report["trained"] is True
    assert store.answer_path(9102).is_file()
    state = engine.state()
    assert state["answer"]["trained"] is True
    assert state["answer"]["validation_examples"] >= 2
    assert state["answer"]["chance_loss"] == round(math.log(state["answer"]["vocabulary"]), 4)
    assert state["runs"][0]["task"] == "answer"
    answer = engine.answer("The cow")
    assert answer is None or isinstance(answer, str)


def test_the_word_model_is_kept_in_its_own_file(brain):
    engine = engine_for(9103)

    report = engine.train_words(CORPUS, _job(SMALL_WORDS))

    assert report["trained"] is True
    assert store.words_path(9103).is_file()
    assert not store.answer_path(9103).exists()
    state = engine.state()
    assert state["words"]["trained"] is True
    assert state["words"]["backend"] == "neural"
    assert "held-out" in state["words"]["backend_reason"]


def test_the_serving_word_is_the_neural_model_as_soon_as_it_is_trained(brain):
    engine = engine_for(9104)
    engine.train_words(CORPUS, _job(SMALL_WORDS))

    serving = engine.word_serving()

    assert serving.backend == "neural"
    assert "held-out" in serving.backend_reason


def test_the_engine_is_reloaded_when_a_model_file_changes(brain):
    engine = engine_for(9105)
    assert engine_for(9105) is engine
    path = store.answer_path(9105)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"a newer file")
    later = os.path.getmtime(path) + 10
    os.utime(path, (later, later))

    reloaded = engine_for(9105)

    assert reloaded is not engine


def test_a_saved_answer_model_is_read_back_by_a_new_engine(brain):
    engine = engine_for(9106)
    engine.train_answer(AnswerCorpus(CORPUS), _job(SMALL_ANSWER))
    engine_module.forget(9106)

    state = engine_for(9106).state()

    assert state["answer"]["trained"] is True
    assert state["schema"] == checkpoint.SCHEMA


def test_reading_the_state_builds_no_model(brain):
    engine = engine_for(9107)
    engine.train_answer(AnswerCorpus(CORPUS), _job(SMALL_ANSWER))
    engine_module.forget(9107)

    state = engine_for(9107).state()

    assert state["answer"]["trained"] is True
    assert state["memory"]["loaded"] is False
    assert state["memory"]["answer_bytes"] > 0
    assert state["answer"]["weights_bytes"] == state["memory"]["answer_bytes"]


def test_only_the_model_that_is_asked_for_is_built(brain):
    engine = engine_for(9108)
    engine.train_answer(AnswerCorpus(CORPUS), _job(SMALL_ANSWER))
    engine.train_words(CORPUS, _job(SMALL_WORDS))
    engine_module.forget(9108)
    engine = engine_for(9108)
    assert engine.state()["memory"]["loaded"] is False

    assert engine.answer_task is not None

    memory = engine.state()["memory"]
    assert memory["answer_loaded"] is True
    assert memory["words_loaded"] is False


def test_only_one_engine_keeps_its_models(brain):
    first = engine_for(9109)
    engine_for(9110)

    assert engine_module.loaded_engines() == engine_module.MAX_ENGINES
    assert engine_for(9109) is not first


def test_a_checkpoint_is_mapped_instead_of_read_into_memory(brain, monkeypatch):
    engine = engine_for(9111)
    engine.train_answer(AnswerCorpus(CORPUS), _job(SMALL_ANSWER))
    seen: dict = {}
    real_load = checkpoint.torch.load

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(checkpoint.torch, "load", spy)

    stored = checkpoint.load(store.answer_path(9111))

    assert stored is not None
    assert seen["mmap"] is True
    assert seen["weights_only"] is True


def test_a_reloaded_model_can_write_an_answer(brain):
    engine = engine_for(9112)
    engine.train_answer(AnswerCorpus(CORPUS), _job(SMALL_ANSWER))
    engine_module.forget(9112)

    loaded = engine_for(9112)
    written = loaded.write("The cow is", max_tokens=4)

    assert written is None or isinstance(written, str)
    assert loaded.state()["memory"]["loaded"] is True


def test_a_reloaded_word_model_can_write_a_sentence(brain):
    engine = engine_for(9113)
    engine.train_words(CORPUS, _job(SMALL_WORDS))
    engine_module.forget(9113)

    loaded = engine_for(9113)
    task = loaded.word_task

    assert task is not None
    sentence = task.generate_sentence("The cow", max_words=6)
    assert sentence is None or isinstance(sentence, str)
