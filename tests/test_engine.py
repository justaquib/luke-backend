"""Tests for the engine that owns the word model of one tenant.

The engine is the only thing that loads checkpoints, trains and decides which
word backend serves. These tests keep its files apart, its cache honest and
its reports measured.
"""

import os
from typing import Any

import pytest

from app.config import settings
from app.services.neural import checkpoint, store
from app.services.neural import engine as engine_module
from app.services.neural.engine import TrainingJob, engine_for
from app.services.neural.settings import TrainingSettings

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


@pytest.fixture(autouse=True)
def _empty_engine_cache():
    """Hold each test to its own engines, as a fresh process would."""
    engine_module._CACHE.clear()
    yield
    engine_module._CACHE.clear()


def _job(config, training=SMALL_TRAINING) -> TrainingJob:
    return TrainingJob(corpus={"sentences": len(CORPUS)}, config=config, settings=training)


def test_a_tenant_without_a_model_reports_no_word_model(brain):
    state = engine_for(9101).state()

    assert state["schema"] == 0
    assert state["words"]["trained"] is False
    assert state["words"]["backend"] == "untrained"
    assert state["words"]["backend_reason"]
    assert state["runs"] == []


def test_the_word_model_is_kept_in_its_own_file(brain):
    engine = engine_for(9103)

    report = engine.train_words(CORPUS, _job(SMALL_WORDS))

    assert report["trained"] is True
    assert store.words_path(9103).is_file()
    state = engine.state()
    assert state["words"]["trained"] is True
    assert state["words"]["backend"] == "neural"
    assert "held-out" in state["words"]["backend_reason"]
    assert state["runs"][0]["task"] == "words"


def test_the_serving_word_is_the_neural_model_as_soon_as_it_is_trained(brain):
    engine = engine_for(9104)
    engine.train_words(CORPUS, _job(SMALL_WORDS))

    serving = engine.word_serving()

    assert serving.backend == "neural"
    assert "held-out" in serving.backend_reason


def test_the_engine_is_reloaded_when_a_model_file_changes(brain):
    engine = engine_for(9105)
    assert engine_for(9105) is engine
    path = store.words_path(9105)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"a newer file")
    later = os.path.getmtime(path) + 10
    os.utime(path, (later, later))

    reloaded = engine_for(9105)

    assert reloaded is not engine


def test_a_saved_word_model_is_read_back_by_a_new_engine(brain):
    engine = engine_for(9106)
    engine.train_words(CORPUS, _job(SMALL_WORDS))
    engine_module.forget(9106)

    state = engine_for(9106).state()

    assert state["words"]["trained"] is True
    assert state["schema"] == checkpoint.SCHEMA


def test_reading_the_state_builds_no_model(brain):
    engine = engine_for(9107)
    engine.train_words(CORPUS, _job(SMALL_WORDS))
    engine_module.forget(9107)

    state = engine_for(9107).state()

    assert state["words"]["trained"] is True
    assert state["memory"]["loaded"] is False
    assert state["memory"]["words_bytes"] > 0
    assert state["words"]["weights_bytes"] == state["memory"]["words_bytes"]


def test_only_the_model_that_is_asked_for_is_built(brain):
    engine = engine_for(9108)
    engine.train_words(CORPUS, _job(SMALL_WORDS))
    engine_module.forget(9108)
    engine = engine_for(9108)
    assert engine.state()["memory"]["loaded"] is False

    assert engine.word_task is not None

    memory = engine.state()["memory"]
    assert memory["words_loaded"] is True


def test_an_engine_stays_while_its_model_fits_the_memory_cap(brain):
    first = engine_for(9109)
    engine_for(9110)

    assert engine_for(9109) is first
    assert engine_module.loaded_engines() == 2


def test_the_oldest_engine_is_dropped_when_the_cap_is_crossed(brain, monkeypatch):
    older = store.words_path(9112)
    older.parent.mkdir(parents=True, exist_ok=True)
    older.write_bytes(b"x" * 10)
    engine_for(9112)
    monkeypatch.setattr(settings, "weights_cache_bytes", 5)

    newer = store.words_path(9113)
    newer.write_bytes(b"x" * 10)
    engine_for(9113)

    assert engine_module.loaded_engines() == 1
    assert engine_module._CACHE.find(9112, store.stamp(9112)) is None


def test_a_checkpoint_is_mapped_instead_of_read_into_memory(brain, monkeypatch):
    engine = engine_for(9111)
    engine.train_words(CORPUS, _job(SMALL_WORDS))
    seen: dict = {}
    real_load = checkpoint.torch.load

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(checkpoint.torch, "load", spy)

    stored = checkpoint.load(store.words_path(9111))

    assert stored is not None
    assert seen["mmap"] is True
    assert seen["weights_only"] is True


def test_a_reloaded_word_model_can_write_a_sentence(brain):
    engine = engine_for(9113)
    engine.train_words(CORPUS, _job(SMALL_WORDS))
    engine_module.forget(9113)

    loaded = engine_for(9113)
    task = loaded.word_task

    assert task is not None
    sentence = task.generate_sentence("The cow", max_words=6)
    assert sentence is None or isinstance(sentence, str)
    assert loaded.state()["memory"]["loaded"] is True
