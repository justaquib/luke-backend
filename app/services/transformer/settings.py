"""The settings the writer and its training loop use.

They are gathered in one place so that the functions that take them do not
need a long list of numbers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingSettings:
    """How long the model trains and how big the batches are.

    ``steps`` of 0 means the training is measured against the material: one
    pass count decides, and the steps the training asks for grow with the
    material it has to learn. An explicit number of steps is honoured as it
    was given, so a caller that needs a short training still gets one.
    """

    steps: int = 0
    batch_size: int = 16
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 7
    # How often the loss is looked at to see whether the training still learns.
    check_every: int = 100
    # How many checks without progress end the training.
    patience: int = 8
    # The training never ends before this many steps.
    min_steps: int = 400
    # How much the loss has to improve for it to count as progress.
    improvement: float = 0.01
    # How many fixed batches the check reads the loss over.
    check_batches: int = 10


@dataclass(frozen=True)
class WritingSettings:
    """How the model writes when it is answering or writing freely."""

    max_words: int = 64
    temperature: float = 0.3
    top_k: int = 5
    seed: int | None = None
    include_prompt: bool = False


@dataclass(frozen=True)
class ReadingSettings:
    """How much of the request and the material the model reads."""

    instruction_words: int = 24
    material_words: int = 96


DEFAULT_READING = ReadingSettings()
DEFAULT_TRAINING = TrainingSettings()
DEFAULT_WRITING = WritingSettings()
