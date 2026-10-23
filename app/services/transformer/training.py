"""The training loop of the tenant's transformer.

The model learns from three kinds of example, and the loss is taken over the
answer alone in every one of them:

* a plain sentence, so the model keeps writing in the trainer's style,
* a request and its answer, so the model answers what it is asked,
* a request, the material it may use, and the answer, so the model answers
  from what the trainer taught rather than from what it half-remembers.

Taking the loss over the answer only is what makes the model answer a request:
the request and the material are read, never scored.

The training is a curriculum and it is watched. The simple sentences are
learned first, the compound ones next and the complex ones last, and the
training ends early when the loss stops going down.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn import functional

from app.services.transformer.model import TinyTransformer, device_name
from app.services.transformer.settings import TrainingSettings

# The GPU is used for the training. The answers are written on the processor,
# so the training in the background never meets another use of the GPU.
_GPU_LOCK = threading.Lock()

# The share of the examples the training starts with.
MIN_POOL_SHARE = 0.25

# How many passes over the material the training asks for by default.
# One pass learns to guess; the model needs several before it knows its
# words, and the measured training only started telling its lesson apart
# from noise when the third pass began.
TRAINING_PASSES = 4

# The steps the training never goes past, so a huge material cannot keep
# the background job running all day.
MAX_STEPS = 12000


@dataclass(frozen=True)
class TrainingReport:
    """What the training did."""

    loss: float = 0.0
    best_loss: float = 0.0
    steps: int = 0
    stopped_early: bool = False
    curve: list[float] = field(default_factory=list)


def _batch(
    examples: list[tuple[list[int], int]],
    pad_id: int,
    settings: TrainingSettings,
    context: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """A batch of examples, with the mark of which words are the answer."""
    rows: list[tuple[list[int], int]] = []
    for _ in range(settings.batch_size):
        index = int(torch.randint(0, len(examples), (1,), generator=generator).item())
        sequence, answer_start = examples[index]
        if len(sequence) < 2:
            continue
        rows.append((sequence[:context], answer_start))
    if not rows:
        return None
    width = max(len(sequence) for sequence, _ in rows)
    padded = torch.full((len(rows), width), pad_id, dtype=torch.long)
    scored = torch.zeros((len(rows), width), dtype=torch.bool)
    for row, (sequence, answer_start) in enumerate(rows):
        padded[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        # The answer starts at answer_start, so the word before it is the last
        # word of the request, and the loss begins there.
        scored[row, max(answer_start - 1, 0) : len(sequence)] = True
    return padded, scored


def _loss(
    model: TinyTransformer,
    inputs: torch.Tensor,
    scored: torch.Tensor,
    pad_id: int,
    vocabulary_size: int,
) -> torch.Tensor:
    """The loss over the answer alone."""
    logits = model(inputs[:, :-1])
    targets = inputs[:, 1:]
    losses = functional.cross_entropy(
        logits.reshape(-1, vocabulary_size),
        targets.reshape(-1),
        ignore_index=pad_id,
        reduction="none",
    )
    mask = scored[:, 1:].reshape(-1)
    return (losses * mask).sum() / mask.sum().clamp_min(1)


def _pool(
    examples: list[tuple[list[int], int]],
    step: int,
    steps: int,
) -> list[tuple[list[int], int]]:
    """The examples the model may learn from at this step.

    The examples are ordered by how hard they are, so the training is a
    curriculum: the easy ones are learned first and the hard ones join later.
    """
    share = min(1.0, (step + 1) / max(steps, 1))
    floor = math.ceil(len(examples) * MIN_POOL_SHARE)
    size = max(floor, int(len(examples) * share))
    return examples[: max(1, size)]


def _stale(curve: list[float], settings: TrainingSettings) -> int:
    """How many checks in a row the loss has not gone down."""
    if len(curve) <= settings.patience:
        return 0
    recent = curve[-1]
    earlier = curve[-settings.patience - 1]
    if recent < earlier - settings.improvement:
        return 0
    return 1


def _steps_for(examples: list[tuple[list[int], int]], settings: TrainingSettings) -> int:
    """The steps the training asks for.

    An explicit number of steps is honoured as it was given. A step count of
    zero means the training is measured against the material: the passes it
    has to make decide, so a bigger material trains for longer and a small
    one still reaches its floor.
    """
    if settings.steps > 0:
        return settings.steps
    length = math.ceil(len(examples) / max(settings.batch_size, 1)) * TRAINING_PASSES
    return max(settings.min_steps, min(length, MAX_STEPS))


def _check(
    model: TinyTransformer,
    examples: list[tuple[list[int], int]],
    config: dict[str, Any],
    settings: TrainingSettings,
    reader: torch.Generator,
) -> float:
    """The loss the model knows, read over a few fixed batches.

    One batch of training tells little — the batch that bore 2.41 on the
    panel sat next to batches that bore 0.60. The model is read in its
    resting state over the same batches every time, so the curve tells how
    well it knows the material and not how one draw of the lottery went.
    """
    device = next(model.parameters()).device
    pad_id = int(config.get("pad_id", 0))
    context = int(config["context"])
    totals: list[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(max(settings.check_batches, 1)):
            batch = _batch(examples, pad_id, settings, context, reader)
            if batch is None:
                continue
            inputs, scored = batch
            losses = _loss(
                model, inputs.to(device), scored.to(device), pad_id, model.vocabulary_size
            )
            totals.append(float(losses.item()))
    model.train()
    return sum(totals) / max(len(totals), 1)


def train_model(
    vocabulary_size: int,
    examples: list[tuple[list[int], int]],
    config: dict[str, Any],
    settings: TrainingSettings | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[TinyTransformer, TrainingReport, float]:
    """Train a fresh transformer and return it with its report and duration.

    The padding mark comes in the ``config`` as ``pad_id``. ``progress`` is
    told how many steps have happened and how many were asked for, at every
    check the loss is read at, so a training that runs in its own process
    can be watched from the page.
    """
    settings = settings or TrainingSettings()
    pad_id = int(config.get("pad_id", 0))
    context = int(config["context"])
    generator = torch.Generator().manual_seed(settings.seed)
    reader = torch.Generator().manual_seed(settings.seed)
    device = torch.device(device_name())
    curve: list[float] = []
    best_loss = float("inf")
    stale = 0
    stopped_early = False
    steps_done = 0
    steps = _steps_for(examples, settings)
    with _GPU_LOCK:
        model = TinyTransformer(vocabulary_size, config).to(device)
        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(steps, 1))
        model.train()
        started = time.time()
        for step in range(steps):
            pool = _pool(examples, step, steps)
            batch = _batch(pool, pad_id, settings, context, generator)
            if batch is None:
                break
            inputs, scored = batch
            loss = _loss(model, inputs.to(device), scored.to(device), pad_id, vocabulary_size)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
            optimiser.step()
            scheduler.step()
            steps_done = step + 1
            if progress is not None:
                progress(steps_done, steps)
            if steps_done % settings.check_every == 0:
                read = round(_check(model, examples, config, settings, reader), 4)
                curve.append(read)
                best_loss = min(best_loss, read)
                stale += _stale(curve, settings)
                if stale >= settings.patience and steps_done >= settings.min_steps:
                    stopped_early = True
                    break
        final = curve[-1] if curve else _check(model, examples, config, settings, reader)
        # The model writes now, so the dropout that helped it learn is off.
        model.eval()
    report = TrainingReport(
        loss=round(final, 4),
        best_loss=round(min(best_loss, final), 4),
        steps=steps_done,
        stopped_early=stopped_early,
        curve=curve,
    )
    return model, report, time.time() - started
