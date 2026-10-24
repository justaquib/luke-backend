"""Tests for the checkpoint that carries a model, its words and its numbers.

A checkpoint has to be readable after the write that made it, readable after
an older version wrote it, and never mistaken for a model when the file is
broken. These tests pin all three.
"""

import torch

from app.services.neural import checkpoint
from app.services.neural.model import TinyTransformer

SMALL = {"d_model": 16, "layers": 1, "heads": 2, "context": 8}
VOCABULARY = [f"w{index}" for index in range(12)]


def _model() -> TinyTransformer:
    return TinyTransformer(len(VOCABULARY), SMALL)


def _checkpoint() -> checkpoint.Checkpoint:
    return checkpoint.Checkpoint(
        task="answer",
        model=_model(),
        vocabulary=VOCABULARY,
        config={**SMALL, "pad_id": 0},
        metrics={"loss": 1.5, "validation_loss": 1.9, "steps": 10},
        corpus={"sentences": 4},
    )


def test_a_checkpoint_keeps_everything_the_model_needs(tmp_path):
    path = tmp_path / "model.pt"
    checkpoint.save(path, _checkpoint())

    stored = checkpoint.load(path)

    assert stored is not None
    assert stored.schema == checkpoint.SCHEMA
    assert stored.task == "answer"
    assert stored.vocabulary[3] == "w3"
    assert stored.metrics["validation_loss"] == 1.9
    assert stored.corpus == {"sentences": 4}
    assert stored.extras == {}
    assert stored.model() is not None


def test_a_save_is_moved_into_place(tmp_path):
    path = tmp_path / "model.pt"

    checkpoint.save(path, _checkpoint())

    assert path.is_file()
    assert not path.with_suffix(".tmp").exists()


def test_a_missing_checkpoint_is_not_a_model(tmp_path):
    assert checkpoint.load(tmp_path / "missing.pt") is None


def test_a_file_that_is_not_a_checkpoint_is_not_a_model(tmp_path):
    path = tmp_path / "broken.pt"
    path.write_bytes(b"not a trained model")

    assert checkpoint.load(path) is None


def test_a_checkpoint_without_weights_is_not_a_model(tmp_path):
    path = tmp_path / "empty.pt"
    torch.save({"config": SMALL, "vocabulary": VOCABULARY}, path)

    assert checkpoint.load(path) is None


def test_a_v1_checkpoint_still_loads_and_keeps_its_numbers(tmp_path):
    path = tmp_path / "old.pt"
    torch.save(
        {
            "config": dict(SMALL),
            "vocabulary": VOCABULARY,
            "state": _model().state_dict(),
            "loss": 5.0,
            "steps": 12,
        },
        path,
    )

    stored = checkpoint.load(path)

    assert stored is not None
    assert stored.schema == 1
    assert stored.metrics["loss"] == 5.0
    assert stored.metrics["steps"] == 12
    assert stored.model() is not None


def test_weights_that_do_not_fit_the_vocabulary_are_not_a_model(tmp_path):
    path = tmp_path / "mismatch.pt"
    torch.save(
        {
            "config": dict(SMALL),
            "vocabulary": [f"w{index}" for index in range(20)],
            "state": _model().state_dict(),
        },
        path,
    )

    stored = checkpoint.load(path)

    assert stored is not None
    assert stored.model() is None
