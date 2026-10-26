"""Tests for the Hugging Face teachers, the model choice and predictions.

No test downloads or trains a real model: the run loop is replaced with a
small stand-in that writes what it measured, so the service, the store and the
routes' behaviour are tested without waiting on a fine-tune.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.schemas import QueryResponse
from app.services import model_choice
from app.services.huggingface import catalog, service, serving, store, text, tuning
from app.services.huggingface.service import (
    BaseModelBusyError,
    BaseModelError,
    BaseModelMissingError,
)
from app.services.language import runtime

MODEL = "HuggingFaceTB/SmolLM2-135M"


@pytest.fixture()
def brain(monkeypatch, tmp_path: Path):
    """A temporary brain folder, and no run left behind."""
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    runtime._CACHE.clear()
    service._RUNNING.clear()
    yield tmp_path
    service._RUNNING.clear()
    runtime._CACHE.clear()


def _weights(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.safetensors").write_bytes(b"weights")


def _teacher(tenant_id: int | None, model_id: str = MODEL) -> str:
    folder = store.teacher_folder(model_id)
    path = store.teacher_path(tenant_id, folder)
    _weights(path)
    store.write_json(
        path / "training.json",
        {"model_id": model_id, "name": catalog.name_of(model_id), "steps": 5, "loss": 1.5},
    )
    return folder


def _wait_for_end(tenant_id: int | None) -> None:
    deadline = time.time() + 5
    while service.running(tenant_id) and time.time() < deadline:
        time.sleep(0.01)
    assert not service.running(tenant_id)


def test_teacher_folder_stays_one_safe_path_part() -> None:
    folder = store.teacher_folder("HuggingFaceTB/SmolLM2-135M")
    assert folder == "teacher-HuggingFaceTB-SmolLM2-135M"
    assert "/" not in folder and "." not in folder


def test_training_text_carries_the_taught_answer() -> None:
    written = text.training_text("  What is a drum?  ", "A drum is hit with a stick.")
    assert written == (
        "Question: What is a drum?\n"
        "Reference: A drum is hit with a stick.\n"
        "Answer: A drum is hit with a stick."
    )


def test_training_texts_skip_empty_pairs_and_repeats() -> None:
    pairs = [
        ("What is a drum?", "A drum is hit with a stick."),
        ("", "An answer without a question."),
        ("What is a drum?", "A drum is hit with a stick."),
        ("What is a flute?", "A flute is blown."),
    ]
    assert len(text.training_texts(pairs)) == 2


def test_grounding_keeps_the_answer_then_the_passages() -> None:
    reference = text.grounding("A drum is hit.", ["A drum is hit.", "It has a skin."])
    assert reference == "A drum is hit. It has a skin."


def test_prompt_has_one_shape_with_and_without_material() -> None:
    assert text.prompt("What is a drum?") == "Question: What is a drum?\nAnswer:"
    assert text.prompt("What is a drum?", "A drum is hit.") == (
        "Question: What is a drum?\nReference: A drum is hit.\nAnswer:"
    )


def test_view_lists_trained_base_models(brain: Path, monkeypatch) -> None:
    monkeypatch.setattr(service, "_machine", lambda: {"device": "cpu"})
    _teacher(1)
    state = service.view(1)
    trained = [row for row in state["base_models"] if row["trained"]]
    assert [row["id"] for row in trained] == [MODEL]
    assert state["teachers"][0]["folder"] == store.teacher_folder(MODEL)
    assert state["running"] is False


def test_start_records_the_run_when_it_finishes(brain: Path, monkeypatch) -> None:
    monkeypatch.setattr(service, "_machine", lambda: {"device": "cpu"})

    def fake_tune(model_id, folder, pairs, steps, progress):
        progress(0, "download", None)
        progress(1, "tune", 2.0)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.safetensors").write_bytes(b"weights")
        return {
            "model_id": model_id,
            "name": catalog.name_of(model_id),
            "steps": steps,
            "loss": 2.0,
            "seconds": 0.1,
            "examples": len(pairs),
        }

    monkeypatch.setattr(tuning, "tune", fake_tune)
    state = service.start(1, MODEL, 2, [("What is a drum?", "A drum is hit.")])
    assert state["running"] is True
    _wait_for_end(1)
    last = store.read_last_run(1)
    assert last["steps"] == 2
    assert last["loss"] == 2.0
    assert last["stages"] == {"download": "done", "tune": "done"}
    assert store.teacher_path(1, store.teacher_folder(MODEL)).is_dir()


def test_start_refuses_a_model_it_does_not_offer(brain: Path) -> None:
    with pytest.raises(BaseModelError):
        service.start(1, "not/offered", 1, [("q", "a")])


def test_start_refuses_a_second_run(brain: Path) -> None:
    service._RUNNING[1] = MODEL
    with pytest.raises(BaseModelBusyError):
        service.start(1, MODEL, 1, [("q", "a")])


def test_failed_run_records_its_error(brain: Path, monkeypatch) -> None:
    monkeypatch.setattr(service, "_machine", lambda: {"device": "cpu"})

    def explode(*_args: object, **_kwargs: object) -> None:
        raise ValueError("the base model could not be read")

    monkeypatch.setattr(tuning, "tune", explode)
    service.start(1, MODEL, 1, [("q", "a")])
    _wait_for_end(1)
    last = store.read_last_run(1)
    assert last["stage"] == "failed"
    assert "could not be read" in last["error"]


def test_remove_teacher_deletes_the_weights(brain: Path) -> None:
    folder = _teacher(1)
    service.remove_teacher(1, folder)
    assert not store.teacher_path(1, folder).is_dir()


def test_remove_teacher_rejects_a_missing_folder(brain: Path) -> None:
    with pytest.raises(BaseModelMissingError):
        service.remove_teacher(1, "teacher-missing")


def test_store_refuses_path_traversal(brain: Path) -> None:
    assert store.remove_teacher(1, "teacher-../secret") is False


def test_catalog_lists_luke_and_the_installed_teachers(brain: Path) -> None:
    _teacher(1)
    state = model_choice.catalog(1)
    ids = [row["id"] for row in state["models"]]
    assert ids[0] == "luke"
    assert ids[1] == f"hf:{store.teacher_folder(MODEL)}"
    assert state["default"] == "luke"


def test_model_of_falls_back_to_luke_when_the_teacher_is_gone(brain: Path) -> None:
    choice, available = model_choice.model_of(1, "hf:teacher-gone")
    assert choice.id == "luke"
    assert available is False


def test_resolve_returns_none_for_an_unknown_model() -> None:
    assert model_choice.resolve(1, "gpt") is None


def _grounded_response() -> QueryResponse:
    return QueryResponse(
        question="What is a drum?",
        source="faq",
        answer="A drum is something you hit.",
        hits=[],
        entities=[],
    )


def test_grounded_uses_the_written_answer_when_it_uses_the_material(
    brain: Path, monkeypatch
) -> None:
    _teacher(1)
    choice = model_choice.ModelChoice("hf:teacher", "SmolLM2 135M", "huggingface", "teacher")
    monkeypatch.setattr(
        model_choice.hf_service,
        "answer",
        lambda *args: "A drum is something you hit with a stick.",
    )
    response, connection = model_choice.grounded(1, choice, _grounded_response())
    assert response.answer == "A drum is something you hit with a stick."
    assert connection is not None and connection.outcome == "match"


def test_grounded_keeps_the_matched_answer_when_writing_is_ungrounded(
    brain: Path, monkeypatch
) -> None:
    _teacher(1)
    choice = model_choice.ModelChoice("hf:teacher", "SmolLM2 135M", "huggingface", "teacher")
    monkeypatch.setattr(model_choice.hf_service, "answer", lambda *args: "Zebras gallop.")
    response, connection = model_choice.grounded(1, choice, _grounded_response())
    assert response.answer == "A drum is something you hit."
    assert connection is not None and connection.outcome == "fail"


def test_material_guard_requires_a_real_rewrite() -> None:
    reference = "The sky appears blue because sunlight scatters off the air."
    wrong = (
        "Sunlight contains all colors of light, but only blue and violet are visible to humans. "
        "Blue light has shorter wavelength compared with other colors. "
        "This makes it easier for us to see in the dark."
    )
    assert not model_choice._uses_material(wrong, reference)
    assert model_choice._uses_material(
        "The air scatters sunlight, so the sky looks blue.", reference
    )


def test_grounded_leaves_a_live_source_alone(brain: Path, monkeypatch) -> None:
    choice = model_choice.ModelChoice("hf:teacher", "SmolLM2 135M", "huggingface", "teacher")
    monkeypatch.setattr(
        model_choice.hf_service, "answer", lambda *args: pytest.fail("must not be called")
    )
    response = _grounded_response().model_copy(update={"source": "wiki"})
    written, connection = model_choice.grounded(1, choice, response)
    assert written.answer == response.answer
    assert connection is None


def test_predict_reads_the_teacher_when_it_is_selected(brain: Path, monkeypatch) -> None:
    _teacher(1)
    monkeypatch.setattr(
        model_choice.hf_service,
        "predict",
        lambda *args: {"tokens": ["what", "is"], "attention": None, "next": []},
    )
    result = model_choice.predict(1, f"hf:{store.teacher_folder(MODEL)}", "what is", 3)
    assert result is not None
    assert result["kind"] == "huggingface"
    assert result["model_name"] == "SmolLM2 135M"


def test_predict_reads_lukes_word_model_when_luke_is_selected(brain: Path, monkeypatch) -> None:
    task = type("Task", (), {"config": {"heads": 4, "layers": 2}})()
    engine = type("Engine", (), {"word_task": task})()
    monkeypatch.setattr(model_choice, "engine_for", lambda tenant: engine)
    monkeypatch.setattr(
        model_choice.word_prediction,
        "predict_with_attention",
        lambda *_args: {"tokens": ["what"], "attention": None, "next": []},
    )
    result = model_choice.predict(1, "luke", "what", 3)
    assert result is not None
    assert result["kind"] == "luke"
    assert result["heads"] == 4
    assert result["layers"] == 2


def test_serving_has_no_weights_to_predict_with(brain: Path) -> None:
    assert service.predict(1, "teacher-missing", "what", 3) is None
    assert serving.loaded() == 0
