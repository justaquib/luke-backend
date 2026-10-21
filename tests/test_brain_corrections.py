"""Tests for learning from a trainer's correction."""

import json

from app.services import brain
from app.services.language import runtime


def _use_temp_feedback(monkeypatch, tmp_path):
    path = tmp_path / "feedback-0.json"
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(brain, "feedback_path", lambda tenant_id: path)
    return path


def test_a_correction_is_kept_when_luke_answers_it_back(monkeypatch, tmp_path):
    _use_temp_feedback(monkeypatch, tmp_path)
    monkeypatch.setattr(
        brain.vector_store,
        "search_faq",
        lambda question, n_results=1, tenant_id=None: [
            {"question": question, "answer": "The ocean is deep blue.", "distance": 0.02, "faq_id": 1}
        ],
    )

    result = brain.verify_correction(None, "what is the color of the ocean?", "The ocean is deep blue.")

    assert result["learned"] is True
    assert result["answer"] == "The ocean is deep blue."


def test_a_correction_is_not_kept_when_the_answer_stays_wrong(monkeypatch, tmp_path):
    _use_temp_feedback(monkeypatch, tmp_path)
    monkeypatch.setattr(
        brain.vector_store,
        "search_faq",
        lambda question, n_results=1, tenant_id=None: [
            {"question": question, "answer": "I do not know.", "distance": 0.9, "faq_id": 1}
        ],
    )

    result = brain.verify_correction(None, "what is the color of the ocean?", "The ocean is deep blue.")

    assert result["learned"] is False


def test_the_feedback_log_counts_only_the_kept_corrections(monkeypatch, tmp_path):
    path = _use_temp_feedback(monkeypatch, tmp_path)

    brain.record_feedback(None, "q1", "a1", False, "right one", learned=True)
    brain.record_feedback(None, "q2", "a2", False, "right two", learned=False)
    brain.record_feedback(None, "q3", "a3", True, "", learned=None)

    entries = json.loads(path.read_text(encoding="utf-8"))

    assert len(entries) == 3
    assert brain.learned_corrections(None) == 1
    assert brain.state(None)["learned_corrections"] == 1


def test_the_corrected_rung_needs_five_kept_corrections(monkeypatch, tmp_path):
    _use_temp_feedback(monkeypatch, tmp_path)
    monkeypatch.setattr(brain, "rung_probes", lambda tenant_id: {})
    monkeypatch.setattr(brain, "state", lambda tenant_id: {"trained": True, "corrections": 5, "learned_corrections": 4})
    monkeypatch.setattr(brain, "_subject_coverage", lambda tenant_id: {})

    rungs = brain.rung_results(None, force=True)
    corrected = next(rung for rung in rungs if rung["rung"] == "0.6")

    assert corrected["probes"] == 5
    assert corrected["passed"] == 4
    assert corrected["rate"] == 0.8
    assert corrected["failures"], "it should still show what is missing"


def test_the_corrected_rung_passes_with_five_kept_corrections(monkeypatch, tmp_path):
    _use_temp_feedback(monkeypatch, tmp_path)
    monkeypatch.setattr(brain, "rung_probes", lambda tenant_id: {})
    monkeypatch.setattr(brain, "state", lambda tenant_id: {"trained": True, "corrections": 5, "learned_corrections": 5})
    monkeypatch.setattr(brain, "_subject_coverage", lambda tenant_id: {})

    rungs = brain.rung_results(None, force=True)
    corrected = next(rung for rung in rungs if rung["rung"] == "0.6")

    assert corrected["passed"] == 5
    assert corrected["rate"] == 1.0
    assert corrected["failures"] == []
