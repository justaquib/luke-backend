"""The held-out split and the fixed reads that make a loss honest.

A loss read from the batches the model is learning from says how well it
fits what it has already seen. The number on the panel has to say how well it
knows what it was *not* trained on, so the examples are split by group, the
groups are never mixed, and the validation batches are fixed once and read
the same way every time.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional

from app.services.neural.examples import Example
from app.services.neural.model import TinyTransformer
from app.services.neural.settings import TrainingSettings


@dataclass(frozen=True)
class Batch:
    """A fixed read: the padded words and the mark of what is scored."""

    inputs: torch.Tensor
    scored: torch.Tensor


def group_split(
    examples: list[Example],
    settings: TrainingSettings,
) -> tuple[list[Example], list[Example]]:
    """Split the examples by group, so one answer is never on both sides.

    A held-out number is only worth showing when there is enough of it: with
    fewer than ``min_validation`` examples or ``min_validation_groups``
    groups the validation side is empty and the report says so.
    """
    groups: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        groups[example.group].append(example)
    names = sorted(groups)
    if len(names) < settings.min_validation_groups:
        return list(examples), []
    wanted = max(1, round(len(examples) * settings.validation_share))
    generator = torch.Generator().manual_seed(settings.seed)
    order = [names[int(position)] for position in torch.randperm(len(names), generator=generator)]
    held: list[Example] = []
    chosen: set[str] = set()
    for name in order:
        if len(held) >= wanted:
            break
        held.extend(groups[name])
        chosen.add(name)
    if len(held) < settings.min_validation or len(held) >= len(examples):
        return list(examples), []
    train = [example for example in examples if example.group not in chosen]
    return train, held


def _trim(example: Example, context: int) -> list[int] | None:
    """An example that fits the context, keeping the words that are scored."""
    sequence = example.sequence
    if len(sequence) > context:
        sequence = sequence[-context:]
    if len(sequence) < 2:
        return None
    return sequence


def build_batch(rows: list[Example], pad_id: int, context: int) -> Batch | None:
    """Pad a handful of examples into one batch, never scoring the padding."""
    trimmed = [row for example in rows if (row := _trim(example, context)) is not None]
    if not trimmed:
        return None
    width = max(len(sequence) for sequence in trimmed)
    inputs = torch.full((len(trimmed), width), pad_id, dtype=torch.long)
    scored = torch.zeros((len(trimmed), width), dtype=torch.bool)
    for index, sequence in enumerate(trimmed):
        inputs[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        scored[index, : len(sequence)] = True
    return Batch(inputs, scored)


def fixed_batches(
    examples: list[Example],
    config: dict[str, Any],
    settings: TrainingSettings,
    seed: int,
) -> list[Batch]:
    """The same batches every time the loss is read.

    The examples are drawn once, with a generator that only lives for this
    call, so a check cannot borrow from the training's stream and the number
    cannot change from one read to the next.
    """
    if not examples:
        return []
    generator = torch.Generator().manual_seed(seed)
    pad_id = int(config.get("pad_id", 0))
    context = int(config["context"])
    wanted = settings.batch_size * max(settings.check_batches, 1)
    positions = torch.randint(0, len(examples), (wanted,), generator=generator)
    rows = [examples[int(position)] for position in positions]
    batches: list[Batch] = []
    for start in range(0, len(rows), settings.batch_size):
        batch = build_batch(rows[start : start + settings.batch_size], pad_id, context)
        if batch is not None:
            batches.append(batch)
    return batches


def read_loss(model: TinyTransformer, batches: list[Batch], pad_id: int) -> float | None:
    """The loss over fixed batches, read with the model at rest."""
    if not batches:
        return None
    vocabulary_size = model.vocabulary_size
    totals: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            totals.append(batch_loss(model, batch, pad_id, vocabulary_size))
    model.train()
    return round(sum(totals) / len(totals), 4)


def batch_loss(
    model: TinyTransformer,
    batch: Batch,
    pad_id: int,
    vocabulary_size: int,
) -> float:
    """The loss over the scored words of one batch."""
    return float(loss_tensor(model, batch, pad_id, vocabulary_size).item())


def loss_tensor(
    model: TinyTransformer,
    batch: Batch,
    pad_id: int,
    vocabulary_size: int,
) -> torch.Tensor:
    """The loss over the scored words of one batch, still attached to the graph."""
    device = next(model.parameters()).device
    inputs = batch.inputs.to(device)
    scored = batch.scored.to(device)
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


def read_accuracy(model: TinyTransformer, batches: list[Batch]) -> float | None:
    """The share of the scored words the model would guess right, at rest."""
    if not batches:
        return None
    device = next(model.parameters()).device
    hits = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in batches:
            inputs = batch.inputs.to(device)
            scored = batch.scored.to(device)
            predicted = model(inputs[:, :-1]).argmax(dim=-1)
            targets = inputs[:, 1:]
            mask = scored[:, 1:]
            hits += int(((predicted == targets) & mask).sum().item())
            total += int(mask.sum().item())
    model.train()
    return round(hits / max(total, 1), 4)


def chance_loss(vocabulary_size: int) -> float:
    """The loss of a model that guesses evenly over every word."""
    return round(math.log(max(vocabulary_size, 1)), 4)
