"""The Phase 7 operations: three separate asks, each refusing what it cannot do."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from app.services.language.training_materials import DOCUMENT, LESSON
from app.services.training import formats, loop, operations, splits
from app.services.training.dataset import PRETRAIN, SFT, Dataset, Example

VOCAB = 48
SIZE = 4
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class Teacher(torch.nn.Module):
    """A model the operations can train on, wrap and write without a download."""

    def __init__(self, fail: str = "") -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(VOCAB, SIZE)
        self.head = torch.nn.Linear(SIZE, VOCAB)
        self.saved: list[Path] = []
        self.fail = fail

    def forward(self, input_ids: object, attention_mask: object, labels: object) -> object:
        logits = self.head(self.embedding(input_ids))
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB), labels.reshape(-1), ignore_index=formats.IGNORE
        )
        return SimpleNamespace(loss=loss)

    def save_pretrained(self, folder: Path | str) -> None:
        target = Path(folder)
        if self.fail:
            raise OSError(self.fail)
        self.saved.append(target)
        (target / "model.safetensors").write_text("weights", encoding="utf-8")


class Tokenizer:
    """Word ids that stay inside the tiny model's vocabulary."""

    chat_template = None
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.saved: list[Path] = []

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        ids = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) + 2
            ids.append(self.vocab[word] % VOCAB)
        return {"input_ids": ids}

    def save_pretrained(self, folder: Path | str) -> None:
        target = Path(folder)
        self.saved.append(target)
        (target / "tokenizer.json").write_text("tokenizer", encoding="utf-8")


@pytest.fixture
def loaded(monkeypatch: pytest.MonkeyPatch) -> tuple[Teacher, Tokenizer]:
    """One teacher and tokenizer standing in for the download, installed once per test."""
    teacher, tokenizer = Teacher(), Tokenizer()
    monkeypatch.setattr(operations.tuning, "load_base", lambda model_id: (teacher, tokenizer))
    return teacher, tokenizer


def answers(count: int = 6) -> Dataset:
    """A corpus of taught questions, every third source held out."""
    examples: list[Example] = []
    assignment: dict[str, str] = {}
    for index in range(count):
        source = f"lesson:item {index}"
        split = splits.HELD_OUT if index % 3 == 0 else splits.TRAIN
        assignment[source] = split
        examples.append(
            Example(
                SFT, source, LESSON, f"What is item {index}?", f"Item {index} is taught.", split
            )
        )
    return Dataset(tuple(examples), assignment)


def passages() -> Dataset:
    """A corpus of two documents to read, one of them held out for the measurement."""
    assignment = {"document:a.pdf": splits.TRAIN, "document:b.pdf": splits.HELD_OUT}
    return Dataset(
        (
            Example(
                PRETRAIN,
                "document:a.pdf",
                DOCUMENT,
                "",
                "Water the soil twice a week and keep it out of the sun.",
                splits.TRAIN,
            ),
            Example(
                PRETRAIN,
                "document:b.pdf",
                DOCUMENT,
                "",
                "Volcanoes open where the crust is thin and lava cools into dark rock.",
                splits.HELD_OUT,
            ),
        ),
        assignment,
    )


def plan_of(
    operation: str,
    folder: Path,
    dataset: Dataset | None = None,
) -> operations.Plan:
    return operations.Plan(
        operation,
        MODEL_ID,
        answers() if dataset is None else dataset,
        folder,
        loop.Settings(steps=2),
    )


def stages(seen: list[tuple[int, str, float | None]]) -> list[str]:
    return [entry[1] for entry in seen]


def test_a_fine_tune_writes_the_weights_where_the_plan_says(
    tmp_path: Path, loaded: tuple[Teacher, Tokenizer]
) -> None:
    seen: list[tuple[int, str, float | None]] = []
    teacher, tokenizer = loaded
    folder = tmp_path / "teacher-qwen"
    record = operations.run(
        operations.Plan(
            operations.SFT_OPERATION, MODEL_ID, answers(), folder, loop.Settings(steps=2)
        ),
        lambda step, stage, loss: seen.append((step, stage, loss)),
    )
    assert record["written"] is True
    assert (folder / "model.safetensors").read_text(encoding="utf-8") == "weights"
    assert (folder / "tokenizer.json").read_text(encoding="utf-8") == "tokenizer"
    assert tokenizer.saved == [tmp_path / ".teacher-qwen.writing"]
    assert not list(tmp_path.glob("*.writing"))
    assert stages(seen)[:2] == ["load", "tune"]
    assert stages(seen)[-1] == "save"
    assert teacher.saved[0].name == ".teacher-qwen.writing"


def test_a_fine_tune_records_every_parameter_it_moved(
    tmp_path: Path, loaded: tuple[Teacher, Tokenizer]
) -> None:
    record = operations.run(
        plan_of(operations.SFT_OPERATION, tmp_path / "teacher"), lambda *args: None
    )
    assert record["kind"] == operations.FULL_WEIGHTS
    assert record["trainable"] == record["parameters"]
    assert record["trainable_percent"] == 100.0
    assert record["target_modules"] == []
    assert "rank" not in record["settings"]
    assert record["settings"]["steps"] == 2


def test_an_adapter_run_records_the_layers_it_was_given(
    tmp_path: Path, monkeypatch, loaded: tuple[Teacher, Tokenizer]
) -> None:
    def wrap(model: object, adapter: operations.Adapter) -> tuple[object, tuple[str, ...]]:
        for parameter in model.embedding.parameters():
            parameter.requires_grad_(False)
        return model, ("down_proj", "gate_proj", "q_proj")

    monkeypatch.setattr(operations, "_wrapped", wrap)
    seen: list[tuple[int, str, float | None]] = []
    record = operations.run(
        plan_of(operations.LORA_OPERATION, tmp_path / "adapter-qwen"),
        lambda step, stage, loss: seen.append((step, stage, loss)),
    )
    assert record["kind"] == operations.ADAPTER
    assert record["target_modules"] == ["down_proj", "gate_proj", "q_proj"]
    assert record["trainable"] < record["parameters"]
    assert record["trainable_percent"] < 100.0
    assert record["settings"]["rank"] == operations.Adapter().rank
    assert "wrap" not in stages(seen)


def test_an_adapter_run_leaves_the_teacher_it_did_not_change_alone(
    tmp_path: Path, monkeypatch, loaded: tuple[Teacher, Tokenizer]
) -> None:
    teacher, tokenizer = loaded

    def wrap(model: object, adapter: operations.Adapter) -> tuple[object, tuple[str, ...]]:
        return model, ("q_proj",)

    monkeypatch.setattr(operations, "_wrapped", wrap)
    operations.run(plan_of(operations.LORA_OPERATION, tmp_path / "adapter"), lambda *args: None)
    assert len(teacher.saved) == 1
    assert tokenizer.saved == []


def test_a_stopped_run_writes_nothing_and_says_so(
    tmp_path: Path, loaded: tuple[Teacher, Tokenizer]
) -> None:
    folder = tmp_path / "teacher"
    folder.mkdir()
    (folder / "model.safetensors").write_text("earlier run", encoding="utf-8")
    stopped = lambda: True  # noqa: E731 - the shortest way to say "ask it to stop"
    record = operations.run(plan_of(operations.SFT_OPERATION, folder), lambda *args: None, stopped)
    assert record["written"] is False
    assert record["stop"] == loop.CANCELLED
    assert record["loss"] is None
    assert (folder / "model.safetensors").read_text(encoding="utf-8") == "earlier run"


def test_a_run_refuses_a_corpus_with_nothing_held_out(
    tmp_path: Path, loaded: tuple[Teacher, Tokenizer]
) -> None:
    single = Dataset(
        (Example(SFT, "lesson:one", LESSON, "One?", "The first answer.", splits.TRAIN),),
        {"lesson:one": splits.TRAIN},
    )
    with pytest.raises(operations.TrainingError, match="held-out side is empty"):
        operations.run(
            plan_of(operations.SFT_OPERATION, tmp_path / "teacher", single), lambda *args: None
        )


def test_a_run_refuses_a_corpus_that_trains_on_what_it_measures(
    tmp_path: Path, loaded: tuple[Teacher, Tokenizer]
) -> None:
    shared = "The same words on both sides."
    leaky = Dataset(
        (
            Example(SFT, "lesson:a", LESSON, "A?", shared, splits.TRAIN),
            Example(SFT, "lesson:b", LESSON, "B?", shared, splits.HELD_OUT),
        ),
        {"lesson:a": splits.TRAIN, "lesson:b": splits.HELD_OUT},
    )
    with pytest.raises(operations.TrainingError, match="training side"):
        operations.run(
            plan_of(operations.SFT_OPERATION, tmp_path / "teacher", leaky), lambda *args: None
        )


def test_an_operation_that_does_not_exist_is_refused_before_a_model_is_read(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        operations.tuning, "load_base", lambda model_id: calls.append(model_id) or (None, None)
    )
    with pytest.raises(operations.TrainingError, match="not one of the operations"):
        operations.run(plan_of("distill", tmp_path / "teacher"), lambda *args: None)
    assert calls == []


def test_a_pretrain_run_refuses_a_corpus_of_only_answers(
    tmp_path: Path, loaded: tuple[Teacher, Tokenizer]
) -> None:
    with pytest.raises(operations.TrainingError, match="no passage"):
        operations.run(
            plan_of(operations.PRETRAIN_OPERATION, tmp_path / "teacher"), lambda *args: None
        )


def test_a_pretrain_run_reads_passages_and_writes_weights(
    tmp_path: Path, loaded: tuple[Teacher, Tokenizer]
) -> None:
    record = operations.run(
        plan_of(operations.PRETRAIN_OPERATION, tmp_path / "teacher", dataset=passages()),
        lambda *args: None,
    )
    assert record["operation"] == operations.PRETRAIN_OPERATION
    assert record["rows"] == 1
    assert record["settings"]["max_tokens"] == formats.DEFAULT_MAX_TOKENS


def test_a_failed_write_leaves_the_previous_weights_alone(
    tmp_path: Path, loaded: tuple[Teacher, Tokenizer]
) -> None:
    teacher, _ = loaded
    teacher.fail = "disk full"
    folder = tmp_path / "teacher"
    folder.mkdir()
    (folder / "model.safetensors").write_text("earlier run", encoding="utf-8")
    asked = operations.Plan(
        operations.SFT_OPERATION, MODEL_ID, answers(), folder, loop.Settings(steps=1)
    )
    with pytest.raises(OSError, match="disk full"):
        operations.run(asked, lambda *args: None)
    assert (folder / "model.safetensors").read_text(encoding="utf-8") == "earlier run"
    assert not (folder.parent / ".teacher.writing").exists()


def test_the_record_carries_the_split_the_run_was_trained_from(
    tmp_path: Path, loaded: tuple[Teacher, Tokenizer]
) -> None:
    record = operations.run(
        plan_of(operations.SFT_OPERATION, tmp_path / "teacher"), lambda *args: None
    )
    held = record["dataset"]
    assert held["sources"] == 6
    assert held["split_taken"] is True
    assert held["counts"]["sft_held_out"] == 2
    assert held["salt"] == splits.DEFAULT_SALT
    assert record["name"]


def test_a_run_is_seeded_before_the_model_or_its_adapter_is_drawn(
    tmp_path: Path, monkeypatch, loaded: tuple[Teacher, Tokenizer]
) -> None:
    events: list[str] = []
    original = torch.manual_seed

    def record(seed: int) -> torch.Generator:
        events.append(f"seed {seed}")
        return original(seed)

    teacher, tokenizer = loaded
    monkeypatch.setattr(torch, "manual_seed", record)

    def read(model_id: str) -> tuple[Teacher, Tokenizer]:
        events.append("load")
        return teacher, tokenizer

    monkeypatch.setattr(operations.tuning, "load_base", read)
    operations.run(plan_of(operations.SFT_OPERATION, tmp_path / "teacher"), lambda *args: None)
    assert events.index("seed 0") < events.index("load")
