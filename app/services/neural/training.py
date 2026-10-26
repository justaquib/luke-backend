"""The training loop of the neural engine.

The model learns from the tenant's sentences and scores every word, so it
learns to predict the next one. Sentences go simplest first; a held-out number
is read from unseen groups after the best state is restored.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from app.services.neural.examples import Example
from app.services.neural.measurement import (
    Batch,
    build_batch,
    chance_loss,
    fixed_batches,
    group_split,
    loss_tensor,
    read_accuracy,
    read_loss,
)
from app.services.neural.model import TinyTransformer, device_name
from app.services.neural.settings import TrainingSettings

# The GPU is used for the training. The answers are written on the processor,
# so the training in the background never meets another use of the GPU.
_GPU_LOCK = threading.Lock()

# The share of the examples the training starts with.
MIN_POOL_SHARE = 0.25

# The horizon when a caller does not ask for one. The word model asks for its
# own eight passes, because its material is small enough that one pass would
# stop at the minimum step count.
TRAINING_PASSES = 1

# The steps the training never goes past, so a huge material cannot keep
# the background job running all day.
MAX_STEPS = 20_000


@dataclass(frozen=True)
class TrainingReport:
    """What the training did, measured honestly."""

    loss: float | None = None
    best_loss: float | None = None
    steps: int = 0
    stopped_early: bool = False
    curve: list[float] = field(default_factory=list)
    validation_loss: float | None = None
    best_validation_loss: float | None = None
    best_step: int = 0
    stop_reason: str = "steps"
    passes: int = 0
    train_examples: int = 0
    validation_examples: int = 0
    validation_curve: list[float] = field(default_factory=list)
    seconds: float = 0.0
    chance_loss: float | None = None
    train_accuracy: float | None = None
    validation_accuracy: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """The report as the checkpoint and the page read it."""
        return {
            "loss": self.loss,
            "best_loss": self.best_loss,
            "steps": self.steps,
            "stopped_early": self.stopped_early,
            "curve": list(self.curve),
            "validation_loss": self.validation_loss,
            "best_validation_loss": self.best_validation_loss,
            "best_step": self.best_step,
            "stop_reason": self.stop_reason,
            "passes": self.passes,
            "train_examples": self.train_examples,
            "validation_examples": self.validation_examples,
            "validation_curve": list(self.validation_curve),
            "seconds": self.seconds,
            "chance_loss": self.chance_loss,
            "train_accuracy": self.train_accuracy,
            "validation_accuracy": self.validation_accuracy,
        }


@dataclass
class Session:
    """Everything one training run works with, so its helpers stay small."""

    model: TinyTransformer
    optimiser: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    generator: torch.Generator
    examples: list[Example]
    held: list[Example]
    config: dict[str, Any]
    settings: TrainingSettings
    progress: Callable[[int, int, float | None], None] | None
    loop: dict[str, Any]
    started: float


def _steps_for(examples: list[Example], settings: TrainingSettings) -> int:
    """The steps the training asks for: the passes over the training material."""
    if settings.steps > 0:
        return settings.steps
    passes = settings.passes if settings.passes is not None else TRAINING_PASSES
    wanted = _per_pass(examples, settings) * passes
    return max(settings.min_steps, min(wanted, settings.max_steps, MAX_STEPS))


def _pool(examples: list[Example], step: int, steps: int) -> list[Example]:
    """The examples the model may learn from at this step, easiest first."""
    share = min(1.0, (step + 1) / max(steps, 1))
    floor = math.ceil(len(examples) * MIN_POOL_SHARE)
    size = max(floor, int(len(examples) * share))
    return examples[: max(1, size)]


def _stale(curve: list[float], settings: TrainingSettings) -> bool:
    """Whether the checked loss has not gone down for ``patience`` checks."""
    if len(curve) <= settings.patience:
        return False
    return not curve[-1] < curve[-settings.patience - 1] - settings.improvement


def _draw(
    pool: list[Example],
    config: dict[str, Any],
    settings: TrainingSettings,
    generator: torch.Generator,
) -> Batch | None:
    """A batch drawn from the pool the curriculum allows at this step."""
    choices = torch.randint(0, len(pool), (settings.batch_size,), generator=generator)
    rows = [pool[int(position)] for position in choices]
    return build_batch(rows, int(config.get("pad_id", 0)), int(config["context"]))


def _rate(step: int, steps: int, warmup: int) -> float:
    """The learning-rate factor: a linear warmup, then a cosine decay."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(steps - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _setup(
    vocabulary_size: int,
    config: dict[str, Any],
    settings: TrainingSettings,
    steps: int,
) -> tuple[TinyTransformer, torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR, str]:
    """The model, its optimiser and its schedule, on the device they train on."""
    device = device_name()
    model = TinyTransformer(vocabulary_size, config).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: _rate(step, steps, settings.warmup_steps)
    )
    return model, optimiser, scheduler, device


def _learn(
    model: TinyTransformer,
    batch: Batch,
    optimiser: torch.optim.Optimizer,
    settings: TrainingSettings,
) -> float:
    """One step of learning, and the train loss it bore."""
    loss = loss_tensor(model, batch, int(model.config.get("pad_id", 0)), model.vocabulary_size)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
    optimiser.step()
    return float(loss.item())


def _snapshot(model: TinyTransformer) -> dict[str, torch.Tensor]:
    """The weights as they are now, so the best ones can be put back."""
    return copy.deepcopy(model.state_dict())


@dataclass(frozen=True)
class Prepared:
    """The material split, frozen and counted before the first step."""

    train: list[Example]
    held: list[Example]
    train_batches: list[Batch]
    held_batches: list[Batch]
    steps: int
    config: dict[str, Any]


@dataclass(frozen=True)
class Reads:
    """The final losses and accuracies of a finished training run."""

    loss: float | None
    validation_loss: float | None
    accuracy: float | None
    validation_accuracy: float | None


def _prepare(
    examples: list[Example],
    config: dict[str, Any],
    settings: TrainingSettings,
) -> Prepared:
    """Split the material, freeze the batches and find the step count."""
    train_examples, held = group_split(examples, settings)
    pad_id = int(config.get("pad_id", 0))
    config = {**config, "pad_id": pad_id}
    return Prepared(
        train=train_examples,
        held=held,
        train_batches=fixed_batches(train_examples, config, settings, settings.seed + 1),
        held_batches=fixed_batches(held, config, settings, settings.seed + 2) if held else [],
        steps=_steps_for(train_examples, settings),
        config=config,
    )


def _loop(prepared: Prepared) -> dict[str, Any]:
    """The running state of the training, read by the checks and the report."""
    return {
        "train": prepared.train_batches,
        "held": prepared.held_batches,
        "best": None,
        "best_loss": float("inf"),
        "best_step": 0,
        "curve": [],
        "held_curve": [],
        "steps": prepared.steps,
        "done": 0,
        "stopped": False,
        "reason": "steps",
    }


def _report(
    session: Session,
    reads: Reads,
    started: float,
    vocabulary_size: int,
) -> TrainingReport:
    """The measured record of the run, for the checkpoint and the page."""
    loop = session.loop
    return TrainingReport(
        loss=reads.loss,
        best_loss=round(float(loop["best_loss"]), 4) if loop["best_loss"] < float("inf") else None,
        steps=int(loop["done"]),
        stopped_early=bool(loop["stopped"]),
        curve=list(loop["curve"]),
        validation_loss=reads.validation_loss,
        best_validation_loss=_best_held(loop["held_curve"]),
        best_step=int(loop["best_step"]),
        stop_reason=str(loop["reason"]),
        passes=round(int(loop["done"]) / _per_pass(session.examples, session.settings)),
        train_examples=len(session.examples),
        validation_examples=len(session.held),
        validation_curve=list(loop["held_curve"]),
        seconds=round(time.time() - started, 1),
        chance_loss=chance_loss(vocabulary_size),
        train_accuracy=reads.accuracy,
        validation_accuracy=reads.validation_accuracy,
    )


def train_model(
    vocabulary_size: int,
    examples: list[Example],
    config: dict[str, Any],
    settings: TrainingSettings | None = None,
    progress: Callable[[int, int, float | None], None] | None = None,
) -> tuple[TinyTransformer, TrainingReport, float]:
    """Train a fresh transformer on the training split and measure the held-out one.

    ``progress`` is told how many steps have happened, how many were asked for
    and the loss of the step, so a training that runs in its own process can
    be watched from the page.
    """
    settings = settings or TrainingSettings()
    prepared = _prepare(examples, config, settings)
    generator = torch.Generator().manual_seed(settings.seed)
    started = time.time()
    with _GPU_LOCK:
        torch.manual_seed(settings.seed)
        model, optimiser, scheduler, _ = _setup(
            vocabulary_size, prepared.config, settings, prepared.steps
        )
        session = Session(
            model=model,
            optimiser=optimiser,
            scheduler=scheduler,
            generator=generator,
            examples=prepared.train,
            held=prepared.held,
            config=prepared.config,
            settings=settings,
            progress=progress,
            loop=_loop(prepared),
            started=time.time(),
        )
        _run(session)
        reads = _finish(session, int(prepared.config["pad_id"]))
        model.eval()
    return model, _report(session, reads, started, vocabulary_size), time.time() - started


def _per_pass(examples: list[Example], settings: TrainingSettings) -> int:
    """How many steps one pass over the training material takes."""
    return max(math.ceil(len(examples) / max(settings.batch_size, 1)), 1)


def _run(session: Session) -> None:
    """The steps of the training, with the checks and the stopping rules."""
    loop = session.loop
    session.model.train()
    for step in range(int(loop["steps"])):
        pool = _pool(session.examples, step, int(loop["steps"]))
        batch = _draw(pool, session.config, session.settings, session.generator)
        if batch is None:
            loop["reason"] = "no batches"
            break
        loss = _learn(session.model, batch, session.optimiser, session.settings)
        session.scheduler.step()
        loop["done"] = step + 1
        if session.progress is not None:
            session.progress(loop["done"], int(loop["steps"]), round(loss, 4))
        if loop["done"] % session.settings.check_every == 0:
            _check(session.model, loop, session.settings)
            settled = _stale(_metric(loop), session.settings)
            if settled and loop["done"] >= session.settings.min_steps:
                loop["stopped"] = True
                loop["reason"] = "early"
                break
        if time.time() - session.started > session.settings.max_seconds:
            loop["reason"] = "time"
            break


def _check(model: TinyTransformer, loop: dict[str, Any], settings: TrainingSettings) -> None:
    """Read the fixed train and held-out batches and keep the better weights."""
    pad_id = int(model.config.get("pad_id", 0))
    train = read_loss(model, loop["train"], pad_id)
    held = read_loss(model, loop["held"], pad_id) if loop["held"] else None
    if train is not None:
        loop["curve"].append(train)
    if held is not None:
        loop["held_curve"].append(held)
    metric = held if held is not None else train
    if metric is not None and metric < float(loop["best_loss"]):
        loop["best_loss"] = metric
        loop["best_step"] = int(loop["done"])
        loop["best"] = _snapshot(model)


def _metric(loop: dict[str, Any]) -> list[float]:
    """The curve the stopping rule reads: held-out when there is one."""
    return list(loop["held_curve"] or loop["curve"])


def _best_held(curve: list[float]) -> float | None:
    return round(min(curve), 4) if curve else None


def _finish(session: Session, pad_id: int) -> Reads:
    """Restore the best weights and read the final honest losses and accuracies."""
    model, loop = session.model, session.loop
    if loop["best"] is not None:
        model.load_state_dict(loop["best"])
    loss = read_loss(model, loop["train"], pad_id)
    held = read_loss(model, loop["held"], pad_id) if loop["held"] else None
    accuracy = read_accuracy(model, loop["train"])
    held_accuracy = read_accuracy(model, loop["held"]) if loop["held"] else None
    if loss is not None and (not loop["curve"] or loop["curve"][-1] != loss):
        loop["curve"].append(loss)
    return Reads(loss, held, accuracy, held_accuracy)
