"""How a trained model is written down and read back.

A checkpoint holds everything needed to answer with the model again: its
config, the words it knows, the weights, and the numbers it was measured at.
It is written to a temporary file and moved into place, so a training that
is interrupted never leaves half a model behind. A checkpoint from before
this schema can still be read; it simply has no held-out numbers, which the
serving gate reads as "not measured".
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from app.services.neural.model import TinyTransformer

logger = logging.getLogger(__name__)

SCHEMA = 3

# The keys a checkpoint owns; everything else is a task's own extra.
RESERVED = frozenset(
    {
        "schema",
        "task",
        "created",
        "config",
        "pad_id",
        "vocabulary",
        "tokenizer",
        "state",
        "corpus",
        "torch",
        "metrics",
    }
)

METRIC_KEYS = frozenset(
    {
        "loss",
        "best_loss",
        "steps",
        "stopped_early",
        "curve",
        "validation_loss",
        "best_validation_loss",
        "best_step",
        "stop_reason",
        "passes",
        "train_examples",
        "validation_examples",
        "validation_curve",
        "seconds",
        "chance_loss",
        "validation_accuracy",
        "train_accuracy",
        "next_word_accuracy",
        "perplexity",
        "dropped_sentences",
        "sentence_count",
        "token_count",
        "transition_count",
        "vocabulary",
        "parameters",
        "device",
    }
)


@dataclass(frozen=True)
class Checkpoint:
    """A model and everything that explains its numbers."""

    task: str
    model: torch.nn.Module
    vocabulary: list[str]
    config: dict[str, Any]
    metrics: dict[str, Any]
    corpus: dict[str, Any] = field(default_factory=dict)
    tokenizer: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        """The whole checkpoint, as it is written to disk."""
        metrics = dict(self.metrics)
        return {
            "schema": SCHEMA,
            "task": self.task,
            "created": time.time(),
            "config": self.config,
            "pad_id": self.config.get("pad_id"),
            "vocabulary": list(self.vocabulary),
            "tokenizer": dict(self.tokenizer),
            "state": self.model.state_dict(),
            "corpus": dict(self.corpus),
            "torch": str(torch.__version__),
            "metrics": metrics,
            "loss": metrics.get("loss"),
            "steps": metrics.get("steps"),
            **self.extras,
        }


@dataclass(frozen=True)
class Stored:
    """A checkpoint that was read back, before its model is built."""

    schema: int
    task: str
    config: dict[str, Any]
    vocabulary: list[str]
    tokenizer: dict[str, Any]
    state: dict[str, Any]
    metrics: dict[str, Any]
    corpus: dict[str, Any]
    extras: dict[str, Any]
    created: float

    def model(self) -> TinyTransformer | None:
        """The model the checkpoint holds, or None when the weights do not fit.

        The weights are assigned, not copied, so the model keeps the mapped
        file as its storage: the operating system pages in only the weights a
        step touches and can take them back when memory is short.
        """
        try:
            model = TinyTransformer(max(len(self.vocabulary), 1), self.config)
            model.load_state_dict(self.state, assign=True)
            model.eval()
            return model
        except (RuntimeError, ValueError, KeyError) as error:
            logger.warning("The checkpoint's weights could not be read: %s", error)
            return None


def save(path: Path, checkpoint: Checkpoint) -> None:
    """Write a checkpoint in one move, so an interrupted write leaves no model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint.payload(), temporary)
    temporary.replace(path)


def load(path: Path) -> Stored | None:
    """Read a checkpoint, or None when the file is missing or unreadable.

    The file is mapped rather than read into memory, so the numbers of a
    tenant's model can be reported without reserving room for its weights.
    """
    if not path.is_file():
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as error:  # noqa: BLE001 -- the checkpoint read boundary
        logger.warning("The checkpoint %s could not be read: %s", path, error)
        return None
    if not isinstance(data, dict) or "state" not in data:
        logger.warning("The checkpoint %s does not hold a model", path)
        return None
    metrics = dict(data.get("metrics") or {})
    # A v1 checkpoint kept its loss and steps at the top, and nothing else.
    for key in ("loss", "steps", "best_loss", "stopped_early", "curve"):
        if key not in metrics and key in data:
            metrics[key] = data[key]
    extras = {
        key: value for key, value in data.items() if key not in RESERVED and key not in METRIC_KEYS
    }
    return Stored(
        schema=int(data.get("schema", 1)),
        task=str(data.get("task", "answer")),
        config=dict(data.get("config") or {}),
        vocabulary=[str(token) for token in data.get("vocabulary", [])],
        tokenizer=dict(data.get("tokenizer") or {}),
        state=data["state"],
        metrics=metrics,
        corpus=dict(data.get("corpus") or {}),
        extras=extras,
        created=float(data.get("created") or 0.0),
    )
