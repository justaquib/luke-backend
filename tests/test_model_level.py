"""The level ladder: hundredths, the approval gate and the maturity scale.

The model level is measured by the rungs, but it is the trainer who approves
a promotion, and the maturity scale keeps its own human stage names so the
amount of material never borrows the language of intelligence.
"""

import json

import pytest

from app.services import brain


def _patch_the_measurements(monkeypatch) -> None:
    """Make the level cheap to compute, without touching the real probes."""
    monkeypatch.setattr(brain, "state", lambda tenant_id: {})
    monkeypatch.setattr(brain, "model_capabilities", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        brain,
        "generality_score",
        lambda tenant_id: {"score": 0.0, "cap": None, "probes": []},
    )
    monkeypatch.setattr(
        brain,
        "instruction_score",
        lambda tenant_id: {"score": 0.0, "probes": 0, "total": 0},
    )
    monkeypatch.setattr(brain, "_coverage", lambda tenant_id: {})
    monkeypatch.setattr(brain, "next_steps", lambda *args, **kwargs: [])


def _rungs(passed_ids: set[str]) -> list[dict[str, object]]:
    rungs = []
    for minor in range(1, 10):
        rung_id = f"0.0.{minor}"
        passed = rung_id in passed_ids
        rungs.append(
            {
                "rung": rung_id,
                "probes": 1,
                "passed": 1 if passed else 0,
                "rate": 1.0 if passed else 0.0,
            }
        )
    return rungs


def test_the_ladder_climbs_in_hundredths_to_a_tenth() -> None:
    ids = [str(entry["level"]) for entry in brain.LUKE_LEVELS]
    assert ids[0] == "0.0.0"
    assert ids[1:-1] == [f"0.0.{minor}" for minor in range(1, 10)]
    assert ids[-1] == "0.1.0"
    assert ids == sorted(ids, key=brain.level_order)


def test_the_displayed_level_waits_for_the_trainer(monkeypatch, tmp_path) -> None:
    _patch_the_measurements(monkeypatch)
    monkeypatch.setattr(brain, "brain_dir", lambda: tmp_path)
    monkeypatch.setattr(
        brain,
        "rung_results",
        lambda *args, **kwargs: _rungs({"0.0.1", "0.0.2"}),
    )
    level = brain.model_level(7)
    assert level["level"] == "0.0.0"
    assert level["measured_level"] == "0.0.2"
    assert level["approval_ready"] is True


def test_the_trainer_promotes_to_the_measured_level(monkeypatch, tmp_path) -> None:
    _patch_the_measurements(monkeypatch)
    monkeypatch.setattr(brain, "brain_dir", lambda: tmp_path)
    monkeypatch.setattr(
        brain,
        "rung_results",
        lambda *args, **kwargs: _rungs({"0.0.1", "0.0.2"}),
    )
    promoted = brain.approve_level(7)
    assert promoted["approved_level"] == "0.0.2"
    assert promoted["previous_level"] == "0.0.0"
    level = brain.model_level(7)
    assert level["level"] == "0.0.2"
    assert level["approval_ready"] is False
    history = json.loads((tmp_path / "level-7.json").read_text(encoding="utf-8"))["history"]
    assert history[-1]["level"] == "0.0.2"


def test_the_trainer_cannot_promote_without_a_higher_measurement(monkeypatch, tmp_path) -> None:
    _patch_the_measurements(monkeypatch)
    monkeypatch.setattr(brain, "brain_dir", lambda: tmp_path)
    monkeypatch.setattr(brain, "rung_results", lambda *args, **kwargs: _rungs(set()))
    with pytest.raises(ValueError):
        brain.approve_level(7)


def test_a_gap_in_the_rungs_stops_the_measured_climb(monkeypatch, tmp_path) -> None:
    _patch_the_measurements(monkeypatch)
    monkeypatch.setattr(brain, "brain_dir", lambda: tmp_path)
    monkeypatch.setattr(
        brain,
        "rung_results",
        lambda *args, **kwargs: _rungs({"0.0.2"}),
    )
    level = brain.model_level(7)
    assert level["measured_level"] == "0.0.0"


def test_the_stretch_reaches_the_tenth(monkeypatch, tmp_path) -> None:
    _patch_the_measurements(monkeypatch)

    def with_a_stretch(tenant_id):
        return {"score": 0.5, "cap": None, "probes": []}

    monkeypatch.setattr(brain, "generality_score", with_a_stretch)
    monkeypatch.setattr(brain, "rung_results", lambda *args, **kwargs: _rungs(set()))
    level = brain.model_level(7)
    assert level["measured_level"] == "0.1.0"
    assert level["approval_ready"] is True


def test_the_maturity_scale_uses_human_stages() -> None:
    volume = brain.training_volume_level(120)
    assert volume["title"] == "Schoolchild"
    assert "level" not in volume
    stages = [stage["title"] for stage in volume["stages"]]
    assert stages == ["Newborn", "First words", "Toddler", "Schoolchild", "Scholar"]
    current = [stage["current"] for stage in volume["stages"]]
    assert current == [False, False, False, True, False]
    assert volume["stages"][0]["reached"] is True
    assert volume["next_stage"]["title"] == "Scholar"
