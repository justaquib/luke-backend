"""The examples the neural models learn from.

Every example is a sequence of words and the place where the answer starts.
The model reads the whole sequence and is scored only on the answer, which is
what makes it answer a request rather than continue a text. A word example
scores every token, so the same loop learns to predict the next word.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import zip_longest

from app.services.neural import difficulty
from app.services.neural.tokens import (
    CONTEXT,
    FINISH,
    INSTRUCTION,
    RESPONSE,
    START,
    UNKNOWN,
    tokenize,
)

# The answer the model is taught for a topic it never learned about.
REFUSAL_START = "i do not know anything about"


@dataclass(frozen=True)
class Example:
    """One training example: the words, where the answer starts, its group and level."""

    sequence: list[int]
    answer_start: int
    group: str
    level: int = difficulty.SIMPLE


def _ids(writer: object, words: Sequence[str]) -> list[int]:
    """The ids of the words, with the unknown mark for the ones not known."""
    return [writer.index.get(token, writer.index[UNKNOWN]) for token in words]


def _group(text: str) -> str:
    """The group an example belongs to, so the same answer never splits."""
    return re.sub(r"\s+", " ", text.strip().casefold())


def sentence_example(writer: object, sentence: str) -> Example:
    """A plain sentence, so the model keeps writing in the taught style."""
    head = [writer.index[START]]
    body = [*_ids(writer, tokenize(sentence)), writer.index[FINISH]]
    return Example([*head, *body], len(head), _group(sentence), difficulty.level_of(sentence))


def instruction_example(
    writer: object,
    instruction: str,
    response: str,
    context: str = "",
) -> Example:
    """A request, the material it may use, and the answer to it."""
    head = [writer.index[START], writer.index[INSTRUCTION]]
    head += _ids(writer, tokenize(instruction))
    if context:
        head += [writer.index[CONTEXT]]
        head += _ids(writer, tokenize(context))
    head += [writer.index[RESPONSE]]
    body = [*_ids(writer, tokenize(response)), writer.index[FINISH]]
    group = _group(instruction) if _group(response).startswith(REFUSAL_START) else _group(response)
    return Example([*head, *body], len(head), group, difficulty.level_of(response))


def word_example(writer: object, sentence: str) -> Example:
    """A sentence scored from its first word to its last, for the word model."""
    sequence = [writer.index[START], *_ids(writer, tokenize(sentence)), writer.index[FINISH]]
    return Example(sequence, 0, _group(sentence), difficulty.level_of(sentence))


def all_words(
    sentences: Sequence[str],
    instructions: Sequence[tuple[str, str]],
    contexts: Sequence[tuple[str, str, str]],
) -> list[str]:
    """Every word the model has to know to read what it is taught."""
    words: list[str] = []
    for sentence in sentences:
        words.extend(tokenize(sentence))
    for instruction, response in instructions:
        words.extend(tokenize(instruction))
        words.extend(tokenize(response))
    for instruction, context, response in contexts:
        words.extend(tokenize(instruction))
        words.extend(tokenize(context))
        words.extend(tokenize(response))
    return words


def _ordered(items: Sequence[tuple[Example, int]]) -> list[Example]:
    """The examples mixed by kind and ordered from the easy ones to the hard."""
    groups = [
        sorted((item for item in items if item[1] == kind), key=lambda item: item[0].answer_start)
        for kind in (0, 1, 2)
    ]
    mixed: list[Example] = []
    for row in zip_longest(*groups):
        mixed.extend(item[0] for item in row if item is not None)
    mixed.sort(key=lambda item: item.level)
    return mixed


def build(
    writer: object,
    sentences: Sequence[str],
    instructions: Sequence[tuple[str, str]],
    contexts: Sequence[tuple[str, str, str]],
) -> list[Example]:
    """Every example the model learns from, the easy ones first."""
    plain = [(sentence_example(writer, sentence), 0) for sentence in sentences if sentence.strip()]
    asked = [
        (instruction_example(writer, instruction, response), 1)
        for instruction, response in instructions
    ]
    grounded = [
        (instruction_example(writer, instruction, response, context), 2)
        for instruction, context, response in contexts
    ]
    return _ordered([*plain, *asked, *grounded])
