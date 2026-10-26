"""The word model's training report: a train read and a held-out read, kept apart."""

import math
from typing import Any

from app.services.neural.settings import TrainingSettings
from app.services.neural.words import WordTask

CORPUS = [
    "The cow is a farm animal.",
    "A cow gives milk every morning.",
    "The moon shines at night.",
    "The night sky is full of stars.",
    "Water is a clear liquid.",
    "A liquid takes the shape of its container.",
    "The sun rises in the east.",
    "A kite flies on a windy day.",
    "The bee lives in a hive.",
    "A hive holds honey.",
    "The river runs to the sea.",
    "A mountain is very high.",
]


def _task() -> WordTask:
    return WordTask(config={"d_model": 24, "layers": 1, "heads": 2, "context": 24})


def _settings(**overrides: Any) -> TrainingSettings:
    values: dict[str, Any] = {
        "steps": 8,
        "batch_size": 4,
        "check_every": 2,
        "min_steps": 1,
        "patience": 4,
        "validation_share": 0.25,
        "min_validation": 2,
        "min_validation_groups": 2,
        "max_seconds": 60,
        **overrides,
    }
    return TrainingSettings(**values)


def test_the_report_keeps_the_train_and_held_out_reads_apart():
    report = _task().fit(CORPUS, settings=_settings())

    assert report["trained"] is True
    assert report["loss"] is not None
    assert report["validation_loss"] is not None
    assert report["validation_examples"] >= 2
    assert report["train_examples"] + report["validation_examples"] == len(CORPUS)
    assert report["stop_reason"] in {"steps", "early", "time"}


def test_the_chance_level_is_the_log_of_its_own_vocabulary():
    report = _task().fit(CORPUS, settings=_settings())

    assert report["chance_loss"] == round(math.log(report["vocabulary"]), 4)


def test_a_stale_held_out_read_stops_the_training_early():
    report = _task().fit(CORPUS, settings=_settings(steps=40, check_every=1, patience=0))

    assert report["stopped_early"] is True
    assert report["stop_reason"] == "early"
    assert report["steps"] == 1


def test_a_short_training_still_reports_finite_numbers():
    report = _task().fit(CORPUS, settings=_settings(steps=1))

    assert math.isfinite(report["loss"])
    assert report["steps"] == 1


def test_the_curve_is_kept_for_the_page():
    report = _task().fit(CORPUS, settings=_settings(steps=8))

    assert report["curve"]
    assert all(math.isfinite(value) for value in report["curve"])
