"""The examples the word model learns from.

A word example scores every token, so the model learns to predict the next
word from the words before it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.services.neural import difficulty
from app.services.neural.tokens import (
    FINISH,
    START,
    UNKNOWN,
    tokenize,
)


@dataclass(frozen=True)
class Example:
    """One training example: the words, its group and its level."""

    sequence: list[int]
    group: str
    level: int = difficulty.SIMPLE


def _ids(task: object, words: Sequence[str]) -> list[int]:
    """The ids of the words, with the unknown mark for the ones not known."""
    return [task.index.get(token, task.index[UNKNOWN]) for token in words]


def _group(text: str) -> str:
    """The group an example belongs to, so the same answer never splits."""
    return re.sub(r"\s+", " ", text.strip().casefold())


def word_example(task: object, sentence: str) -> Example:
    """A sentence scored from its first word to its last, for the word model."""
    sequence = [task.index[START], *_ids(task, tokenize(sentence)), task.index[FINISH]]
    return Example(sequence, _group(sentence), difficulty.level_of(sentence))
