"""The writing rung reads the held-out writing measurement, not the training state."""

from typing import Any

from app.services import brain_instruction, brain_rung_results, brain_state


def _writing_rung(monkeypatch, measurement: dict[str, Any]) -> dict[str, Any]:
    monkeypatch.setattr(brain_rung_results, "rung_probes", lambda tenant_id: {})
    monkeypatch.setattr(brain_rung_results, "subject_coverage", lambda tenant_id: {})
    monkeypatch.setattr(brain_instruction, "stored_instruction", lambda tenant_id: measurement)
    monkeypatch.setattr(brain_state, "state", lambda tenant_id: {"trained": True})
    rungs = brain_rung_results.rung_results(None, force=True)
    return next(rung for rung in rungs if rung["rung"] == "0.0.4")


def test_the_writing_rung_is_unmeasured_without_a_measurement(monkeypatch) -> None:
    rung = _writing_rung(monkeypatch, {})

    assert rung["passed"] == 0
    assert rung["failures"][0]["form"] == "not measured yet"
    assert rung["detail"] == "no held-out writing measurement yet"


def test_the_writing_rung_fails_when_the_writer_wrote_nothing(monkeypatch) -> None:
    rung = _writing_rung(monkeypatch, {"measured": True, "written": 0, "total": 2})

    assert rung["passed"] == 0
    assert rung["failures"][0]["form"] == "looked up instead"
    assert rung["detail"] == "the writer wrote 0 of 2 held-out answers"


def test_the_writing_rung_passes_when_the_writer_wrote_one_itself(monkeypatch) -> None:
    rung = _writing_rung(monkeypatch, {"measured": True, "written": 1, "total": 2})

    assert rung["passed"] == 1
    assert rung["rate"] == 1.0
    assert rung["failures"] == []
