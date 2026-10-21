"""The training loop of the tenant's transformer.

The model learns from three kinds of example, and the loss is taken over the
answer alone in every one of them:

* a plain sentence, so the model keeps writing in the trainer's style,
* a request and its answer, so the model answers what it is asked,
* a request, the material it may use, and the answer, so the model answers
  from what the trainer taught rather than from what it remembers.

Taking the loss over the answer only is what makes the model answer a request:
the request and the material are read, never scored.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import torch
from torch.nn import functional

from app.services.transformer.model import TinyTransformer, device_name
from app.services.transformer.settings import TrainingSettings

# The GPU is used for the training. The answers are written on the processor,
# so the training in the background never meets another use of the GPU.
_GPU_LOCK = threading.Lock()


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


def train_model(
    vocabulary_size: int,
    examples: list[tuple[list[int], int]],
    config: dict[str, Any],
    settings: TrainingSettings | None = None,
    pad_id: int = 0,
) -> tuple[TinyTransformer, float, float]:
    """Train a fresh transformer and return it with its last loss and duration."""
    settings = settings or TrainingSettings()
    context = int(config["context"])
    generator = torch.Generator().manual_seed(settings.seed)
    device = torch.device(device_name())
    with _GPU_LOCK:
        model = TinyTransformer(vocabulary_size, config).to(device)
        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=max(settings.steps, 1)
        )
        model.train()
        started = time.time()
        last_loss = 0.0
        for _ in range(settings.steps):
            batch = _batch(examples, pad_id, settings, context, generator)
            if batch is None:
                break
            inputs, scored = batch
            inputs = inputs.to(device)
            scored = scored.to(device)
            loss = _loss(model, inputs, scored, pad_id, vocabulary_size)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
            optimiser.step()
            scheduler.step()
            last_loss = float(loss.item())
    # The model writes now, so the dropout that helped it learn is turned off.
    model.eval()
    return model, last_loss, time.time() - started
