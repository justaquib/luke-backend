"""Tests for the checklist the training writes while it runs.

The checklist says which parts of the training are done, how many of the
model's steps have happened and what is left, and — when the training is
finished — how long the whole job took.
"""

from app.services import brain, training_progress
from app.services.language import runtime
from app.services.language import training_materials as materials_service
from app.services.neural.settings import TrainingSettings
from app.services.neural.words import WordTask

# A small model, so the tests that train one stay quick.
SMALL_MODEL = {"d_model": 32, "layers": 1, "heads": 2, "context": 64}


def test_the_checklist_says_which_parts_are_done(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    training_progress.start(None)
    training_progress.stage(None, "words", training_progress.DONE)
    training_progress.stage(None, "emotion", training_progress.WORKING)

    record = training_progress.read(None)
    statuses = {stage["key"]: stage["status"] for stage in record["stages"]}

    assert statuses["words"] == training_progress.DONE
    assert statuses["emotion"] == training_progress.WORKING
    assert statuses["intent"] == training_progress.WAITING
    assert statuses["feedback"] == training_progress.WAITING


def test_the_steps_say_what_is_left(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    training_progress.start(None)
    training_progress.steps(None, 1200, 3000, eta_seconds=60)

    record = training_progress.read(None)

    assert record["steps_done"] == 1200
    assert record["steps_total"] == 3000
    assert record["eta_seconds"] == 60
    assert record["finished"] is None


def test_the_finished_checklist_says_how_long_the_training_took(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    training_progress.start(None)
    training_progress.finish(None)

    record = training_progress.read(None)

    assert record["finished"] is not None
    assert record["seconds"] >= 0
    assert record["eta_seconds"] is None


def test_a_tenant_that_has_not_trained_reads_no_checklist(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)

    assert training_progress.read(None) == {}


def test_the_model_reports_its_steps_while_it_trains(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    calls: list[tuple[int, int]] = []
    task = WordTask(config=SMALL_MODEL)

    task.fit(
        [
            "The rain falls on the roof and the wind blows.",
            "The rain falls on the field and the sun shines.",
            "The kite flies high above the trees and the clouds.",
            "The kite flies high above the field and the rain.",
            "The children play beside the sea when the rain stops.",
        ],
        settings=TrainingSettings(steps=3, check_every=1),
        progress=lambda done, total, loss: calls.append((done, total)),
    )

    assert calls
    assert calls[0][1] == 3
    assert max(done for done, _ in calls) == 3


QUESTION = "What is {}?"
ANSWER = "The {} is a thing we know from a lesson."


def test_a_trained_brain_leaves_a_finished_checklist(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    things = ["kite", "rain", "sun", "sea"]
    materials = [
        materials_service.Material("lesson", QUESTION.format(thing), ANSWER.format(thing))
        for thing in things
    ]

    brain.train(None, materials_service.training_set(materials), config=SMALL_MODEL, steps=6)

    record = training_progress.read(None)
    assert all(stage["status"] == training_progress.DONE for stage in record["stages"]), (
        "every part of the training should be marked done"
    )
    assert record["finished"] is not None
    assert record["seconds"] >= 0


def test_the_brain_state_carries_the_checklist(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    training_progress.start(None)
    training_progress.stage(None, "words", training_progress.WORKING)
    training_progress.steps(None, 40, 3000, eta_seconds=120)

    progress = brain.state(None)["progress"]

    assert progress["steps_done"] == 40
    assert progress["steps_total"] == 3000
    assert progress["eta_seconds"] == 120
    assert progress["stages"][0]["status"] == training_progress.WORKING
