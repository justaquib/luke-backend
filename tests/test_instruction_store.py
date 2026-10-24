"""The instruction measurement is stored, not taken on every brain read.

A fresh measurement takes about a minute, so the brain page reads the last
one that was kept, and a new one is taken only when the trainer asks for it.
"""

import json
from typing import Any

import pytest

from app.services import brain, brain_paths
from app.services.language import evaluation


def _write_a_measurement(tmp_path, tenant_id=7, score=0.75):
    payload = {
        "score": score,
        "passed": 9,
        "total": 12,
        "probes": [
            {"question": "Write a small poem about a star.", "correct": True},
        ],
        "measured": True,
    }
    path = tmp_path / f"instruction-{tenant_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_the_brain_page_reads_the_stored_measurement(monkeypatch, tmp_path):
    monkeypatch.setattr(brain_paths, "brain_dir", lambda: tmp_path)
    _write_a_measurement(tmp_path)

    def no_reading(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the stored measurement must not re-run the test")

    monkeypatch.setattr(evaluation, "evaluate", no_reading)
    score = brain.instruction_score(7)
    assert score["score"] == 0.75
    assert score["measured"] is True


def test_before_the_first_test_the_score_is_a_missing_number(monkeypatch, tmp_path):
    monkeypatch.setattr(brain_paths, "brain_dir", lambda: tmp_path)
    score = brain.instruction_score(7)
    assert score["measured"] is False
    assert score["total"] == 0
    assert score["measured_at"] is None


def test_the_trainer_test_runs_it_again_and_keeps_it(monkeypatch, tmp_path):
    monkeypatch.setattr(brain_paths, "brain_dir", lambda: tmp_path)
    monkeypatch.setattr(
        evaluation,
        "evaluate",
        lambda tenant_id, cards, writer: {"score": 0.5, "passed": 1, "total": 2, "probes": []},
    )
    result = brain.remeasure_instruction(7)
    assert result["score"] == 0.5
    stored = brain.instruction_score(7)
    assert stored["score"] == 0.5
    assert stored["measured"] is True
    assert stored["measured_at"]


def test_a_broken_measurement_file_reads_as_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(brain_paths, "brain_dir", lambda: tmp_path)
    (tmp_path / "instruction-7.json").write_text("{broken", encoding="utf-8")
    score = brain.instruction_score(7)
    assert score["measured"] is False


@pytest.mark.parametrize(
    ("tenant_id", "name"),
    [(7, "instruction-7.json"), (None, "instruction-0.json")],
)
def test_the_measurement_is_kept_per_tenant(monkeypatch, tmp_path, tenant_id, name):
    monkeypatch.setattr(brain_paths, "brain_dir", lambda: tmp_path)
    brain.save_instruction(tenant_id, {"score": 0.5, "passed": 1, "total": 2, "probes": []})
    assert (tmp_path / name).exists()
