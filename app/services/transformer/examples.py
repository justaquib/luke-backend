"""The examples the transformer learns from.

Every example is a sequence of words and the place where the answer starts.
The model reads the whole sequence and is scored only on the answer, which is
what makes it answer a request rather than continue a text.
"""

from __future__ import annotations

from collections.abc import Sequence

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


def build(
    writer: object,
    sentences: Sequence[str],
    instructions: Sequence[tuple[str, str]],
    contexts: Sequence[tuple[str, str, str]],
) -> list[tuple[list[int], int]]:
    """Every training example the model learns from."""
    examples = [sentence_example(writer, sentence) for sentence in sentences]
    examples += [
        instruction_example(writer, instruction, response) for instruction, response in instructions
    ]
    examples += [
        instruction_example(writer, instruction, response, context)
        for instruction, context, response in contexts
    ]
    return examples
