"""The level ladder: hundredths, the approval gate and the maturity scale.

The model level is measured by the rungs, but it is the trainer who approves
a promotion, and the maturity scale keeps its own human stage names so the
amount of material never borrows the language of intelligence.
"""

import json

import pytest

from app.services import (
    brain,
    brain_instruction,
    brain_material,
    brain_paths,
    brain_roadmap,
    brain_rung_results,
    brain_state,
)


def _patch_the_measurements(monkeypatch) -> None:
    """Make the level cheap to compute, without touching the real probes."""
    monkeypatch.setattr(brain_state, "state", lambda tenant_id: {})
    monkeypatch.setattr(brain_material, "model_capabilities", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        brain_instruction,
        "generality_score",
        lambda tenant_id: {"score": 0.0, "cap": None, "probes": []},
    )
    monkeypatch.setattr(
        brain_instruction,
        "instruction_score",
        lambda tenant_id: {"score": 0.0, "probes": 0, "total": 0},
    )
    monkeypatch.setattr(brain_material, "coverage", lambda tenant_id: {})
    monkeypatch.setattr(brain_material, "next_steps", lambda *args, **kwargs: [])


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


def test_the_whole_road_is_shown_to_the_generalist() -> None:
    roadmap = brain_roadmap.ROADMAP
    ids = [str(entry["level"]) for entry in roadmap]

    assert len(ids) == 101
    assert ids[:10] == [f"0.0.{index}" for index in range(10)]
    assert ids[10] == "0.1.0"
    assert ids[-1] == "1.0.0"
    assert ids == sorted(ids, key=brain.level_order)
    assert len(set(ids)) == len(ids)
    assert all(entry["band"] and entry["band_name"] for entry in roadmap)
    assert len({str(entry["name"]) for entry in roadmap}) == len(roadmap)


def test_only_the_tested_levels_are_open() -> None:
    roadmap = brain_roadmap.ROADMAP
    open_ids = [str(entry["level"]) for entry in roadmap if entry["open"]]

    assert open_ids == [str(entry["level"]) for entry in brain.LUKE_LEVELS]
    assert open_ids == ["0.0.0", *[f"0.0.{minor}" for minor in range(1, 10)], "0.1.0"]
    assert next(entry for entry in roadmap if entry["level"] == "1.0.0")["open"] is False


def test_the_page_reads_the_whole_road_and_locks_the_next_level(monkeypatch, tmp_path) -> None:
    _patch_the_measurements(monkeypatch)
    monkeypatch.setattr(brain_paths, "brain_dir", lambda: tmp_path)
    monkeypatch.setattr(brain_rung_results, "rung_results", lambda *args, **kwargs: _rungs(set()))
    (tmp_path / "level-7.json").write_text(
        json.dumps({"approved_level": "0.1.0"}), encoding="utf-8"
    )

    level = brain.model_level(7)

    assert level["level"] == "0.1.0"
    assert level["max_level"] == "1.0.0"
    assert level["ladder_size"] == 101
    assert len(level["levels"]) == 101
    assert level["ladder_share"] == 0.1
    assert level["next_level"] == "Luke 0.1.1 · Recalled"
    assert "locked" in level["next_requirement"]
    assert "no test" in level["next_requirement"].lower()


def test_the_displayed_level_waits_for_the_trainer(monkeypatch, tmp_path) -> None:
    _patch_the_measurements(monkeypatch)
    monkeypatch.setattr(brain_paths, "brain_dir", lambda: tmp_path)
    monkeypatch.setattr(
        brain_rung_results,
        "rung_results",
        lambda *args, **kwargs: _rungs({"0.0.1", "0.0.2"}),
    )
    level = brain.model_level(7)
    assert level["level"] == "0.0.0"
    assert level["measured_level"] == "0.0.2"
    assert level["approval_ready"] is True


def test_the_trainer_promotes_to_the_measured_level(monkeypatch, tmp_path) -> None:
    _patch_the_measurements(monkeypatch)
    monkeypatch.setattr(brain_paths, "brain_dir", lambda: tmp_path)
    monkeypatch.setattr(
        brain_rung_results,
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
    monkeypatch.setattr(brain_paths, "brain_dir", lambda: tmp_path)
    monkeypatch.setattr(brain_rung_results, "rung_results", lambda *args, **kwargs: _rungs(set()))
    with pytest.raises(ValueError):
        brain.approve_level(7)


def test_a_gap_in_the_rungs_stops_the_measured_climb(monkeypatch, tmp_path) -> None:
    _patch_the_measurements(monkeypatch)
    monkeypatch.setattr(brain_paths, "brain_dir", lambda: tmp_path)
    monkeypatch.setattr(
        brain_rung_results,
        "rung_results",
        lambda *args, **kwargs: _rungs({"0.0.2"}),
    )
    level = brain.model_level(7)
    assert level["measured_level"] == "0.0.0"


def test_the_stretch_reaches_the_tenth(monkeypatch, tmp_path) -> None:
    _patch_the_measurements(monkeypatch)

    def with_a_stretch(tenant_id):
        return {"score": 0.5, "cap": None, "probes": []}

    monkeypatch.setattr(brain_instruction, "generality_score", with_a_stretch)
    monkeypatch.setattr(brain_rung_results, "rung_results", lambda *args, **kwargs: _rungs(set()))
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
