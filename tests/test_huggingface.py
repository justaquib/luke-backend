"""Tests for the Hugging Face teachers, the model choice and predictions.

No test downloads or trains a real model: the run loop is replaced with a
small stand-in that writes what it measured, so the service, the store and the
routes' behaviour are tested without waiting on a fine-tune.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from app.schemas import QueryResponse
from app.services import model_choice
from app.services.huggingface import catalog, service, serving, store, text
from app.services.huggingface.service import (
    BaseModelBusyError,
    BaseModelError,
    BaseModelMissingError,
)
from app.services.language import runtime
from app.services.language.training_materials import LESSON
from app.services.training import operations, splits
from app.services.training.dataset import SFT, Dataset, Example

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


def _dataset() -> Dataset:
    """Two taught sources, the second held out, so a run is allowed to start."""
    return Dataset(
        (
            Example(SFT, "lesson:a", LESSON, "What is a drum?", "A drum is hit.", splits.TRAIN),
            Example(
                SFT,
                "lesson:b",
                LESSON,
                "What is a flute?",
                "A flute is blown.",
                splits.HELD_OUT,
            ),
        ),
        {"lesson:a": splits.TRAIN, "lesson:b": splits.HELD_OUT},
    )


def _fake_run(plan: operations.Plan, progress) -> dict:
    """Stand in for the loop: report its stages, write what the operation writes."""
    progress(0, "load", None)
    progress(1, "tune", 2.0)
    progress(1, "save", 2.0)
    plan.folder.mkdir(parents=True, exist_ok=True)
    weights = "model.safetensors"
    if plan.kind == operations.ADAPTER:
        weights = "adapter_model.safetensors"
    (plan.folder / weights).write_bytes(b"weights")
    return {
        "operation": plan.operation,
        "kind": plan.kind,
        "model_id": plan.model_id,
        "name": catalog.name_of(plan.model_id),
        "folder": plan.folder.name,
        "steps": 1,
        "stop": "steps",
        "loss": 2.0,
        "seconds": 0.1,
        "device": "cpu",
        "rows": 2,
        "sources": 2,
        "trainable": 1,
        "parameters": 1,
        "trainable_percent": 100.0,
        "target_modules": [],
        "settings": {},
        "dataset": {"sources": 2},
        "written": True,
    }


def test_teacher_folder_stays_one_safe_path_part() -> None:
    folder = store.teacher_folder("HuggingFaceTB/SmolLM2-135M")
    assert folder == "teacher-HuggingFaceTB-SmolLM2-135M"
    assert "/" not in folder and "." not in folder


def test_grounding_keeps_the_answer_then_the_passages() -> None:
    reference = text.grounding("A drum is hit.", ["A drum is hit.", "It has a skin."])
    assert reference == "A drum is hit. It has a skin."


class _AskingTokenizer:
    """A tokenizer whose own template is visible in the string it hands the model."""

    chat_template = "{{ messages }}"
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.asked: list[str] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return f"<|user|>{messages[0]['content']}<|assistant|>"

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None) -> dict:
        self.asked.append(text)
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, ids, skip_special_tokens=True):
        return "A drum is hit."


class _WritingModel(torch.nn.Module):
    """A model that answers with one id after the three it was given."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def generate(self, **kwargs: object) -> torch.Tensor:
        return torch.tensor([[1, 2, 3, 4]])


def test_a_teacher_is_asked_in_the_template_it_was_trained_in(tmp_path: Path) -> None:
    tokenizer, model = _AskingTokenizer(), _WritingModel()
    teacher = serving.Teacher(model, tokenizer, tmp_path)
    assert teacher.generate("What is a drum?", "A drum is hit.") == "A drum is hit."
    assert tokenizer.asked == ["<|user|>What is a drum?\nReference: A drum is hit.<|assistant|>"]


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
    monkeypatch.setattr(operations, "run", _fake_run)
    state = service.start(1, MODEL, operations.SFT_OPERATION, 2, _dataset())
    assert state["running"] is True
    _wait_for_end(1)
    last = store.read_last_run(1)
    assert last["operation"] == "sft"
    assert last["kind"] == operations.FULL_WEIGHTS
    assert last["steps"] == 1
    assert last["steps_total"] == 2
    assert last["loss"] == 2.0
    assert last["stages"] == {"load": "done", "tune": "done", "save": "done"}
    assert store.teacher_path(1, store.teacher_folder(MODEL)).is_dir()


def test_an_adapter_run_writes_its_own_folder(brain: Path, monkeypatch) -> None:
    monkeypatch.setattr(service, "_machine", lambda: {"device": "cpu"})
    monkeypatch.setattr(operations, "run", _fake_run)
    service.start(1, MODEL, operations.LORA_OPERATION, 2, _dataset())
    _wait_for_end(1)
    last = store.read_last_run(1)
    assert last["kind"] == operations.ADAPTER
    adapter = store.teacher_path(1, f"{store.teacher_folder(MODEL)}-lora")
    assert (adapter / "adapter_model.safetensors").is_file()
    assert (adapter / "training.json").is_file()
    assert not store.teacher_path(1, store.teacher_folder(MODEL)).exists()
    assert store.list_teachers(1) == []


def test_start_refuses_a_model_it_does_not_offer(brain: Path) -> None:
    with pytest.raises(BaseModelError):
        service.start(1, "not/offered", operations.SFT_OPERATION, 1, _dataset())


def test_start_refuses_an_operation_it_does_not_offer(brain: Path) -> None:
    with pytest.raises(BaseModelError, match="not one of the operations"):
        service.start(1, MODEL, "distill", 1, _dataset())


def test_start_refuses_a_dataset_with_nothing_held_out(brain: Path) -> None:
    single = Dataset(
        (Example(SFT, "lesson:a", LESSON, "One?", "One answer.", splits.TRAIN),),
        {"lesson:a": splits.TRAIN},
    )
    with pytest.raises(BaseModelError, match="held-out side is empty"):
        service.start(1, MODEL, operations.SFT_OPERATION, 1, single)


def test_start_refuses_a_second_run(brain: Path) -> None:
    service._RUNNING[1] = MODEL
    with pytest.raises(BaseModelBusyError):
        service.start(1, MODEL, operations.SFT_OPERATION, 1, _dataset())


def test_failed_run_records_its_error(brain: Path, monkeypatch) -> None:
    monkeypatch.setattr(service, "_machine", lambda: {"device": "cpu"})

    def explode(*_args: object, **_kwargs: object) -> None:
        raise ValueError("the base model could not be read")

    monkeypatch.setattr(operations, "run", explode)
    service.start(1, MODEL, operations.SFT_OPERATION, 1, _dataset())
    _wait_for_end(1)
    last = store.read_last_run(1)
    assert last["stage"] == "failed"
    assert last["stages"]["load"] == "failed"
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
