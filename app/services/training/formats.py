"""The tokens a training run sees and the shape the finished model is asked in.

One renderer makes both halves of an example: the asked half the server sends
and the whole half the loss is measured on. Kept in one place because the two
must agree - a model prompted in a shape it never trained in answers badly, and
the mismatch is invisible in the numbers. A chat-template model is rendered
through its own template, with the generation prompt standing in for the asked
half; a model without one uses the plain question shape the app already uses.

Only the answer carries a loss. The asked prefix is labelled `IGNORE`, because
predicting the question the user just typed is not the skill being trained.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.services.training.dataset import PRETRAIN, SFT, Example

IGNORE = -100
DEFAULT_MAX_TOKENS = 512
PLAIN_ASK = "Question: {prompt}\nAnswer:"


@dataclass(frozen=True)
class Row:
    """One example as token ids, with the loss-bearing part marked."""

    ids: tuple[int, ...]
    labels: tuple[int, ...]
    source: str
    task: str
    split: str

    @property
    def supervised(self) -> int:
        """The tokens this row puts into the loss."""
        return sum(1 for label in self.labels if label != IGNORE)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "task": self.task,
            "split": self.split,
            "tokens": len(self.ids),
            "supervised": self.supervised,
        }


def messages(prompt: str, target: str) -> list[dict[str, str]]:
    """The example as a turn: the asking and the answer it was taught."""
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    ]


def render(tokenizer: Any, prompt: str, target: str, with_answer: bool) -> str:
    """One half of the example, in the shape this tokenizer answers in.

    The asked half is the exact string prefix of the whole half, either through
    the template's generation prompt or through the plain shape.
    """
    if getattr(tokenizer, "chat_template", None):
        turn = messages(prompt, target)
        if not with_answer:
            turn = turn[:1]
        return tokenizer.apply_chat_template(
            turn, tokenize=False, add_generation_prompt=not with_answer
        )
    asked = PLAIN_ASK.format(prompt=" ".join(prompt.split()))
    if not with_answer:
        return asked
    return f"{asked} {' '.join(target.split())}"


def ask_text(tokenizer: Any, prompt: str) -> str:
    """What the server sends when it wants this shape's answer."""
    return render(tokenizer, prompt, "", False)


def _encode(tokenizer: Any, text: str) -> list[int]:
    """The ids of text already rendered.

    No special tokens are added here: a chat template carries its own markers,
    and adding another set to one half of the example and not the other is how
    the prefix stops being a prefix. A real tokenizer answers with a
    `BatchEncoding`, which is a mapping rather than the list of ids, so the ids
    are taken by name when the answer is a mapping and read directly when it is
    already a sequence.
    """
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    return [int(token) for token in ids]


def _eos(tokenizer: Any) -> int | None:
    value = getattr(tokenizer, "eos_token_id", None)
    return int(value) if value is not None else None


def _shared_prefix_length(prefix: Sequence[int], whole: Sequence[int]) -> int:
    """How many leading ids the asked half really is inside the whole half."""
    for position, token in enumerate(prefix):
        if position >= len(whole) or whole[position] != token:
            return position
    return len(prefix)


def _mask(prefix_ids: Sequence[int], whole_ids: Sequence[int]) -> tuple[list[int], int]:
    """The labels of a whole row: `IGNORE` up to the answer, ids after it."""
    asked = _shared_prefix_length(prefix_ids, whole_ids)
    return [IGNORE] * asked + list(whole_ids[asked:]), asked


def _fit(ids: Sequence[int], labels: Sequence[int], max_tokens: int) -> tuple[list[int], list[int]]:
    return list(ids[:max_tokens]), list(labels[:max_tokens])


def sft_row(tokenizer: Any, example: Example, max_tokens: int = DEFAULT_MAX_TOKENS) -> Row | None:
    """A question and its taught answer, with only the answer supervised.

    A row whose answer is cut off by `max_tokens` supervises nothing, so it is
    dropped rather than kept as padding the loss would learn to ignore. The one
    token that carries the answer marker and the first answer word cannot be
    masked without cutting the answer open, so it stays supervised - the same
    boundary every chat-template fine-tune has, and worth knowing about when a
    row looks shorter than the question that was asked.
    """
    if not example.prompt.strip() or not example.target.strip():
        return None
    asked = _encode(tokenizer, render(tokenizer, example.prompt, example.target, False))
    ids = _encode(tokenizer, render(tokenizer, example.prompt, example.target, True))
    if not ids:
        return None
    eos = _eos(tokenizer)
    if eos is not None and ids[-1] != eos:
        ids.append(eos)
    labels, _ = _mask(asked, ids)
    ids, labels = _fit(ids, labels, max_tokens)
    row = Row(tuple(ids), tuple(labels), example.source, example.task, example.split)
    return row if row.supervised else None


def pretrain_row(
    tokenizer: Any, example: Example, max_tokens: int = DEFAULT_MAX_TOKENS
) -> Row | None:
    """A passage to read, with every token supervised."""
    if not example.target.strip():
        return None
    ids = _encode(tokenizer, example.target)
    if not ids:
        return None
    eos = _eos(tokenizer)
    if eos is not None and ids[-1] != eos:
        ids.append(eos)
    ids, labels = _fit(ids, list(ids), max_tokens)
    return Row(tuple(ids), tuple(labels), example.source, example.task, example.split)


def row(tokenizer: Any, example: Example, max_tokens: int = DEFAULT_MAX_TOKENS) -> Row | None:
    """The row for one example, in the shape its task asks for."""
    if example.task == SFT:
        return sft_row(tokenizer, example, max_tokens)
    if example.task == PRETRAIN:
        return pretrain_row(tokenizer, example, max_tokens)
    raise ValueError(f"Unknown training task: {example.task}")


def rows(
    tokenizer: Any, examples: Iterable[Example], max_tokens: int = DEFAULT_MAX_TOKENS
) -> tuple[Row, ...]:
    """The rows of a side of the dataset, with the unsupervisable ones left out."""
    built = (row(tokenizer, example, max_tokens) for example in examples)
    return tuple(item for item in built if item is not None)


def pad(rows: Sequence[Row], pad_token_id: int) -> dict[str, list[list[int]]]:
    """One padded batch of rows, with the padding left out of the loss.

    `IGNORE` is used for padded label positions as well as for the asked prefix,
    so a padded position can never contribute to the loss whichever task it
    came from.
    """
    width = max(len(item.ids) for item in rows)
    return {
        "input_ids": [list(item.ids) + [pad_token_id] * (width - len(item.ids)) for item in rows],
        "attention_mask": [[1] * len(item.ids) + [0] * (width - len(item.ids)) for item in rows],
        "labels": [list(item.labels) + [IGNORE] * (width - len(item.labels)) for item in rows],
    }
