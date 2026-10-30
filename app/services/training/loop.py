"""The step machinery the three Phase 7 operations share.

One loop, three uses. What an operation decides is which rows it hands over and
which parameters may move; what is the same is the batching, the loss, the
clipping, the reporting and the way a stop is honoured. Keeping that in one
place is what lets a number measured on a supervised run be compared with one
measured on an adapter run at all.

The order the rows are drawn in comes from hashing, not from a random number
generator, for the same reason the held-out split does: a run that cannot be
repeated exactly cannot be measured against the run before it.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from app.services.training.formats import DEFAULT_MAX_TOKENS, Row, pad

Progress = Callable[[int, str, float | None], None]
ShouldStop = Callable[[], bool]

BATCH_SIZE = 2
LEARNING_RATE = 5e-4
REPORT_EVERY = 5
GRADIENT_CLIP = 1.0
DEFAULT_SEED = 0

FINISHED = "steps"
CANCELLED = "cancelled"


@dataclass(frozen=True)
class Settings:
    """What a run is asked for, kept whole so its record can repeat it."""

    steps: int = 40
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    max_tokens: int = DEFAULT_MAX_TOKENS
    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class Result:
    """What one run measured about itself while it was going."""

    steps: int
    loss: float | None
    seconds: float
    stop: str
    rows: int
    sources: int
    device: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "loss": self.loss,
            "seconds": self.seconds,
            "stop": self.stop,
            "rows": self.rows,
            "sources": self.sources,
            "device": self.device,
        }


@dataclass(frozen=True)
class Run:
    """Everything one pass of the loop needs: a model, rows and a way to report."""

    model: Any
    tokenizer: Any
    rows: Sequence[Row]
    settings: Settings
    progress: Progress
    should_stop: ShouldStop | None = None

    @property
    def pad_id(self) -> int | None:
        """The id that fills the shortfall of a batch, or none when there is no such id."""
        value = getattr(self.tokenizer, "pad_token_id", None)
        return int(value) if value is not None else None


def _device(torch: Any) -> Any:
    """The device a run trains on: the GPU this machine has, or its CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _rank(seed: int, position: int) -> str:
    return hashlib.sha256(f"luke-training-{seed}:{position}".encode()).hexdigest()


def order(count: int, seed: int) -> list[int]:
    """Every row once, in an order the same seed always draws again."""
    return sorted(range(count), key=lambda position: _rank(seed, position))


def batches(count: int, steps: int, batch_size: int, seed: int) -> tuple[tuple[int, ...], ...]:
    """The row positions of each step, starting a fresh pass when one runs out.

    A step gets a whole batch unless the corpus is smaller than one, in which
    case it takes what there is rather than padding with repeats.
    """
    plan: list[tuple[int, ...]] = []
    queue: list[int] = []
    draw = 0
    for _ in range(steps):
        if len(queue) < batch_size:
            queue.extend(order(count, seed + draw))
            draw += 1
        take = min(batch_size, len(queue)) if count < batch_size else batch_size
        plan.append(tuple(queue[:take]))
        del queue[:take]
    return tuple(plan)


def trainable_parameters(model: Any) -> list[Any]:
    """The parameters this run may move: all of them, or only an adapter's."""
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _tensors(batch: dict[str, list[list[int]]], torch: Any, device: Any) -> dict[str, Any]:
    return {key: torch.tensor(value, device=device) for key, value in batch.items()}


def _step(run: Run, positions: tuple[int, ...], optimizer: Any, torch: Any, device: Any) -> float:
    """One step: the batch, its loss, and the update that follows it."""
    batch = _tensors(pad([run.rows[index] for index in positions], int(run.pad_id)), torch, device)
    loss = run.model(**batch).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_parameters(run.model), GRADIENT_CLIP)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.item())


def _opens(run: Run) -> None:
    """The checks a run makes before it moves a single weight."""
    if not run.rows:
        raise ValueError("There are no examples to train on.")
    if run.settings.steps < 1:
        raise ValueError(f"A run needs at least one step, got {run.settings.steps}.")
    if run.pad_id is None:
        raise ValueError("The tokenizer has no padding token to hide a batch with.")


def train(run: Run) -> Result:
    """Run the steps and report them, stopping when asked or when they are done."""
    import torch

    _opens(run)
    torch.manual_seed(run.settings.seed)
    device = _device(torch)
    run.model.to(device)
    run.model.train()
    optimizer = torch.optim.AdamW(trainable_parameters(run.model), lr=run.settings.learning_rate)
    started = time.time()
    losses: list[float] = []
    stop = FINISHED
    done = 0
    plan = batches(len(run.rows), run.settings.steps, run.settings.batch_size, run.settings.seed)
    for position, positions in enumerate(plan, start=1):
        if run.should_stop is not None and run.should_stop():
            stop = CANCELLED
            break
        losses.append(_step(run, positions, optimizer, torch, device))
        done = position
        if position % REPORT_EVERY == 0 or position == len(plan):
            run.progress(position, "tune", _mean(losses[-REPORT_EVERY:]))
    return Result(
        steps=done,
        loss=_mean(losses[-REPORT_EVERY:]),
        seconds=round(time.time() - started, 1),
        stop=stop,
        rows=len(run.rows),
        sources=len({row.source for row in run.rows}),
        device=str(device),
    )
