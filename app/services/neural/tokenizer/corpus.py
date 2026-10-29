"""The passages a scratch tokenizer is trained from, with their provenance.

The tokenizer is trained on the same material the word model trains on, cut
into the same passages, and the held-out groups are chosen by the same seed
before a single merge is learned. An evaluation sentence therefore cannot
reach the tokenizer's vocabulary by accident.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.services.language.passages import MAX_WORDS, MIN_WORDS, split_passages
from app.services.neural.examples import Example
from app.services.neural.measurement import held_groups
from app.services.neural.settings import TrainingSettings


@dataclass(frozen=True)
class Passage:
    """One passage, and where it came from."""

    text: str
    group: str
    kind: str
    sha256: str
    chars: int


def group_of(sentence: str) -> str:
    """The group a sentence belongs to, as the training split reads it."""
    return " ".join(sentence.split()).casefold()


def split_training(
    sentences: Sequence[str],
    settings: TrainingSettings,
) -> tuple[list[str], list[str]]:
    """The sentences the tokenizer may learn from, and the held-out ones."""
    grouped: dict[str, list[Example]] = defaultdict(list)
    for sentence in sentences:
        grouped[group_of(sentence)].append(Example([], group_of(sentence)))
    chosen = held_groups(grouped, settings)
    if not chosen:
        return list(sentences), []
    held = set(chosen)
    train = [sentence for sentence in sentences if group_of(sentence) not in held]
    held_sentences = [sentence for name in chosen for sentence in _texts_named(sentences, name)]
    return train, held_sentences


def _texts_named(sentences: Sequence[str], name: str) -> list[str]:
    return [sentence for sentence in sentences if group_of(sentence) == name]


def passages_of(sentences: Sequence[str], kind: str = "training") -> list[Passage]:
    """The passages of the sentences, deduplicated, with a content hash."""
    seen: set[str] = set()
    passages: list[Passage] = []
    for sentence in sentences:
        group = group_of(sentence)
        for text in split_passages(sentence, MIN_WORDS, MAX_WORDS):
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            passages.append(
                Passage(
                    text=text,
                    group=group,
                    kind=kind,
                    sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    chars=len(text),
                )
            )
    return passages


def summary(passages: Sequence[Passage]) -> dict[str, Any]:
    """What the corpus holds, measured before any training is asked for."""
    characters = sum(passage.chars for passage in passages)
    words = {word for passage in passages for word in passage.text.split()}
    kinds: dict[str, int] = {}
    for passage in passages:
        kinds[passage.kind] = kinds.get(passage.kind, 0) + 1
    return {
        "passages": len(passages),
        "characters": characters,
        "distinct_words": len(words),
        "groups": len({passage.group for passage in passages}),
        "by_kind": kinds,
        "sha256": hashlib.sha256(
            "".join(passage.sha256 for passage in passages).encode("ascii")
        ).hexdigest(),
    }
