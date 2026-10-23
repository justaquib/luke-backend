"""The examples the transformer learns from.

Every example is a sequence of words and the place where the answer starts.
The model reads the whole sequence and is scored only on the answer, which is
what makes it answer a request rather than continue a text.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import zip_longest

from app.services.transformer import difficulty
from app.services.transformer.tokens import (
    CONTEXT,
    FINISH,
    INSTRUCTION,
    RESPONSE,
    START,
    UNKNOWN,
    tokenize,
)


def _ids(writer: object, words: Sequence[str]) -> list[int]:
    """The ids of the words, with the unknown mark for the ones not known."""
    return [writer.index.get(token, writer.index[UNKNOWN]) for token in words]


def sentence_example(writer: object, sentence: str) -> tuple[list[int], int]:
    """A plain sentence, so the model keeps writing in the taught style."""
    head = [writer.index[START]]
    body = [*_ids(writer, tokenize(sentence)), writer.index[FINISH]]
    return [*head, *body], len(head)


def instruction_example(
    writer: object,
    instruction: str,
    response: str,
    context: str = "",
) -> tuple[list[int], int]:
    """A request, the material it may use, and the answer to it."""
    head = [writer.index[START], writer.index[INSTRUCTION]]
    head += _ids(writer, tokenize(instruction))
    if context:
        head += [writer.index[CONTEXT]]
        head += _ids(writer, tokenize(context))
    head += [writer.index[RESPONSE]]
    body = [*_ids(writer, tokenize(response)), writer.index[FINISH]]
    return [*head, *body], len(head)


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


def _ordered(
    sentences: Sequence[tuple[list[int], int, int]],
    instructions: Sequence[tuple[list[int], int, int]],
    contexts: Sequence[tuple[list[int], int, int]],
) -> list[tuple[list[int], int]]:
    """The examples in the order the model should learn them.

    The examples of every kind are mixed, so that a part of the training is
    always made of plain sentences, a part of requests and their answers, and
    a part of requests with the material they are answered from. They are then
    ordered by how hard they are, so the easy ones are learned first.
    """
    groups = [sorted(sentences, key=_level), sorted(instructions, key=_level)]
    groups.append(sorted(contexts, key=_level))
    mixed: list[tuple[list[int], int, int]] = []
    for row in zip_longest(*groups):
        for item in row:
            if item is not None:
                mixed.append(item)
    mixed.sort(key=_level)
    return [example for example, _, _ in mixed]


def _level(item: tuple[list[int], int, int]) -> int:
    """How hard an example is."""
    return item[2]


def build(
    writer: object,
    sentences: Sequence[str],
    instructions: Sequence[tuple[str, str]],
    contexts: Sequence[tuple[str, str, str]],
) -> list[tuple[list[int], int]]:
    """Every training example the model learns from, the easy ones first.

    The examples are ordered by how hard they are, so the training can be a
    curriculum: the simple sentences are learned before the compound ones.
    """
    plain = [
        (sentence_example(writer, sentence), 0, difficulty.level_of(sentence))
        for sentence in sentences
    ]
    asked = [
        (instruction_example(writer, instruction, response), 0, difficulty.level_of(response))
        for instruction, response in instructions
    ]
    grounded = [
        (
            instruction_example(writer, instruction, response, context),
            0,
            difficulty.level_of(response),
        )
        for instruction, context, response in contexts
    ]
    return _ordered(plain, asked, grounded)
