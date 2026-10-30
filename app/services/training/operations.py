"""The three training operations Phase 7 offers, each named and each separate.

A supervised fine-tune moves every weight of a teacher; an adapter moves a small
set of added matrices and leaves the teacher alone; a continued pretrain reads
passages rather than questions and answers. They are three operations, not one
with a flag, because what each may be given, what each writes and what each has
to prove before it ships are different. All three go through the same dataset,
the same split checks and the same step loop, so their measured numbers compare.

Nothing here starts on a corpus that measures itself: the held-out side is
checked before a weight moves, and a corpus too thin to fill the batch it asks
for is refused by name instead of trained on air.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.services.huggingface import tuning
from app.services.huggingface.catalog import name_of
from app.services.training import formats, loop, splits, targets
from app.services.training.dataset import PRETRAIN, SFT, Dataset, leakage

SFT_OPERATION = "sft"
LORA_OPERATION = "lora"
PRETRAIN_OPERATION = "pretrain"
OPERATIONS = (SFT_OPERATION, LORA_OPERATION, PRETRAIN_OPERATION)

FULL_WEIGHTS = "weights"
ADAPTER = "adapter"

NO_ROWS = {
    SFT: "There is no taught question and answer to train on yet.",
    PRETRAIN: "There is no passage to continue-pretrain on: the document store is empty.",
}

Progress = loop.Progress
ShouldStop = loop.ShouldStop
Write = Callable[[Path], None]


class TrainingError(Exception):
    """The run cannot start, and the message names what is missing."""


@dataclass(frozen=True)
class Adapter:
    """The size of the adapter an adapter run adds, and nothing else."""

    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05


@dataclass(frozen=True)
class Plan:
    """One operation, asked for: what to train, on what, and where to write it."""

    operation: str
    model_id: str
    dataset: Dataset
    folder: Path
    settings: loop.Settings = field(default_factory=loop.Settings)
    adapter: Adapter = field(default_factory=Adapter)

    @property
    def task(self) -> str:
        """Which kind of example this operation is given."""
        if self.operation == PRETRAIN_OPERATION:
            return PRETRAIN
        if self.operation in (SFT_OPERATION, LORA_OPERATION):
            return SFT
        raise TrainingError(
            f"{self.operation} is not one of the operations: {', '.join(OPERATIONS)}."
        )

    @property
    def kind(self) -> str:
        """Whether this run writes an adapter or whole weights."""
        return ADAPTER if self.operation == LORA_OPERATION else FULL_WEIGHTS


@dataclass(frozen=True)
class Loaded:
    """A plan with its model read in, its rows shaped and its layers named."""

    model: Any
    tokenizer: Any
    rows: Sequence[formats.Row]
    names: tuple[str, ...]
    free: int
    total: int


def _refuse_unchosen(dataset: Dataset) -> None:
    """A corpus whose held-out side is empty cannot measure anything, so it is not trained."""
    if not dataset.taken():
        raise TrainingError(
            "The held-out side is empty, so nothing this run learns could be measured. "
            "Teach more material, from more than one source, first."
        )


def _refuse_leaks(dataset: Dataset) -> None:
    leaks = leakage(dataset)
    if leaks:
        raise TrainingError(
            f"{len(leaks)} held-out source(s) also sit on the training side: "
            f"{', '.join(leaks[:3])}."
        )


def _rows_of(tokenizer: Any, plan: Plan, task: str) -> Sequence[formats.Row]:
    """The trainable rows of one side, with the ones that teach nothing left out."""
    budget = plan.settings.max_tokens
    rows = formats.rows(tokenizer, plan.dataset.side(task, splits.TRAIN), budget)
    if not rows:
        raise TrainingError(NO_ROWS[task])
    return rows


def config_of(model: Any, adapter: Adapter) -> Any:
    """The adapter config for this model, with the layers it really has.

    The names come from the loaded model and not from a recipe, because the
    recipe names are refused before a step runs - measured in
    ``docs/07-model-measurements.md``.
    """
    from peft import LoraConfig, TaskType

    names = targets.target_modules(model)
    if not names:
        raise TrainingError("The loaded model has no linear layer an adapter could wrap.")
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=adapter.rank,
        lora_alpha=adapter.alpha,
        lora_dropout=adapter.dropout,
        target_modules=list(names),
    )


def _wrapped(model: Any, adapter: Adapter) -> tuple[Any, tuple[str, ...]]:
    """The same model with an adapter round the linear layers it really has."""
    from peft import get_peft_model

    config = config_of(model, adapter)
    return get_peft_model(model, config), tuple(sorted(config.target_modules))


def _count(model: Any) -> tuple[int, int]:
    """The parameters this run may move, and every parameter the model has."""
    free = sum(int(parameter.numel()) for parameter in loop.trainable_parameters(model))
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    return free, total


def _seed(seed: int) -> None:
    """Put the run's seed into torch before anything random happens.

    An adapter's added matrices are drawn when it is wrapped, not when the loop
    starts, so seeding inside the loop alone would leave two runs of the same
    plan with different starting adapters and nothing to compare.
    """
    import torch

    torch.manual_seed(seed)


def review(plan: Plan) -> str:
    """The checks that can be made before a model is read into memory, and the task."""
    task = plan.task
    _refuse_unchosen(plan.dataset)
    _refuse_leaks(plan.dataset)
    return task


def load(plan: Plan, task: str) -> Loaded:
    """Read the base model, shape the rows and wrap an adapter if this run adds one."""
    model, tokenizer = tuning.load_base(plan.model_id)
    rows = _rows_of(tokenizer, plan, task)
    names: tuple[str, ...] = ()
    if plan.operation == LORA_OPERATION:
        model, names = _wrapped(model, plan.adapter)
    free, total = _count(model)
    return Loaded(model, tokenizer, rows, names, free, total)


def _write_atomically(folder: Path, write: Write) -> None:
    """Fill a staging folder, then move it into place in one step.

    A run that fails or is stopped leaves the weights it already had readable,
    because a reader only ever sees a folder that was completed.
    """
    staging = folder.parent / f".{folder.name}.writing"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        write(staging)
        if folder.exists():
            shutil.rmtree(folder)
        staging.replace(folder)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _save(plan: Plan, loaded: Loaded) -> None:
    """Write what this operation produces, and only what it produces."""
    model, tokenizer = loaded.model, loaded.tokenizer
    if plan.operation == LORA_OPERATION:

        def write_adapter(staging: Path) -> None:
            model.save_pretrained(staging)

        _write_atomically(plan.folder, write_adapter)
        return

    def write_weights(staging: Path) -> None:
        model.save_pretrained(staging)
        tokenizer.save_pretrained(staging)

    _write_atomically(plan.folder, write_weights)


def _settings_of(plan: Plan) -> dict[str, Any]:
    """The ask of this run, without the knobs an operation does not have."""
    asked = asdict(plan.settings) | asdict(plan.adapter)
    if plan.operation != LORA_OPERATION:
        for name in asdict(plan.adapter):
            asked.pop(name, None)
    return asked


def _record(plan: Plan, loaded: Loaded, result: loop.Result) -> dict[str, Any]:
    """Everything a finished run knows about itself, for the page and the next run."""
    dataset = plan.dataset
    return {
        "operation": plan.operation,
        "kind": plan.kind,
        "model_id": plan.model_id,
        "name": name_of(plan.model_id),
        "folder": plan.folder.name,
        "steps": result.steps,
        "stop": result.stop,
        "loss": result.loss,
        "seconds": result.seconds,
        "device": result.device,
        "rows": result.rows,
        "sources": result.sources,
        "trainable": loaded.free,
        "parameters": loaded.total,
        "trainable_percent": round(100 * loaded.free / loaded.total, 4) if loaded.total else None,
        "target_modules": list(loaded.names),
        "settings": _settings_of(plan),
        "dataset": {
            "salt": dataset.salt,
            "share": dataset.share,
            "sources": len(dataset.assignment),
            "examples": len(dataset.examples),
            "counts": dataset.counts(),
            "split_taken": dataset.taken(),
        },
    }


def run(plan: Plan, progress: Progress, should_stop: ShouldStop | None = None) -> dict[str, Any]:
    """Carry out one named operation over one dataset and write what it made.

    A run that was stopped before its last step writes nothing: half-trained
    weights are not a teacher, and the record says which of the two happened.
    """
    task = review(plan)
    _seed(plan.settings.seed)
    progress(0, "load", None)
    loaded = load(plan, task)
    progress(0, "tune", None)
    result = loop.train(
        loop.Run(loaded.model, loaded.tokenizer, loaded.rows, plan.settings, progress, should_stop)
    )
    written = result.stop == loop.FINISHED
    if written:
        progress(result.steps, "save", result.loss)
        loaded.model.to("cpu")
        _save(plan, loaded)
    return _record(plan, loaded, result) | {"written": written}
