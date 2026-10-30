"""The dataset the training operations read.

Material becomes examples of two tasks: a passage to read, which is what
continued pretraining consumes, and a question with its answer, which is what
supervised fine-tuning consumes. Every example carries the source it came from
and the tag that source was split on, so a run can name what it learned from
and what it was measured against, and so a held-out source has no example on
the training side.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from sqlalchemy.orm import Session

from app.services.assessment.integrity import normalise
from app.services.language import training_materials
from app.services.language.passages import split_passages
from app.services.training import splits

# The two tasks a training operation is given.
PRETRAIN = "pretrain"
SFT = "sft"
TASKS = (PRETRAIN, SFT)

# The source of the taught voice. It is one source on purpose: a greeting on
# the held-out side and the same greeting trained on would measure nothing.
CONVERSATION_SOURCE = "conversation"

_DIGEST_LENGTH = 16


@dataclass(frozen=True)
class Example:
    """One thing the model can be given: a passage to read or a pair to answer."""

    task: str
    source: str
    kind: str
    prompt: str
    target: str
    split: str

    def as_dict(self) -> dict[str, str]:
        return {
            "task": self.task,
            "source": self.source,
            "kind": self.kind,
            "prompt": self.prompt,
            "target": self.target,
            "split": self.split,
        }


@dataclass(frozen=True)
class Dataset:
    """The examples, the split they were tagged with, and how they were drawn."""

    examples: tuple[Example, ...] = ()
    assignment: dict[str, str] = field(default_factory=dict)
    salt: str = splits.DEFAULT_SALT
    share: float = splits.DEFAULT_SHARE

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted(self.assignment))

    def taken(self) -> bool:
        """Whether one side was held out from the other."""
        return splits.is_split_taken(self.assignment)

    def side(self, task: str, split: str) -> tuple[Example, ...]:
        return tuple(
            example for example in self.examples if example.task == task and example.split == split
        )

    def counts(self) -> dict[str, int]:
        """How many examples of each task sit on each side."""
        return {
            f"{task}_{split}": len(self.side(task, split))
            for task in TASKS
            for split in splits.SPLITS
        }

    def report(self) -> dict[str, Any]:
        """The dataset as the trainer reads it: what is in it and how it is split."""
        return {
            "sources": len(self.assignment),
            "examples": len(self.examples),
            "split_taken": self.taken(),
            "salt": self.salt,
            "share": self.share,
            "splits": splits.summary(self.assignment),
            "counts": self.counts(),
            "passage_sources": len({e.source for e in self.examples if e.task == PRETRAIN}),
            "answer_sources": len({e.source for e in self.examples if e.task == SFT}),
        }


def source_of(material: training_materials.Material) -> str:
    """The unit this material belongs to.

    A document is one unit, so every chunk and every passage cut out of it
    shares the unit. A lesson, an approved answer or a correction is a unit in
    itself. The taught voice is one unit. The labels match the ones the leak
    scan uses, so the split and the scan cannot disagree about what a source is.
    """
    kind = material.kind
    if kind == training_materials.CONVERSATION:
        return CONVERSATION_SOURCE
    label = material.question.strip()
    if kind == training_materials.SENTENCE or not label:
        return f"{kind}:{_digest(material.answer)}"
    return f"{kind}:{label}"


def _digest(text: str) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()[:_DIGEST_LENGTH]


def _pretrain(material: training_materials.Material, split: str) -> list[Example]:
    """The passages of one document chunk, each an example to read."""
    return [
        Example(PRETRAIN, source_of(material), material.kind, "", passage, split)
        for passage in split_passages(material.answer)
    ]


def _sft(material: training_materials.Material, split: str) -> list[Example]:
    """The question and its answer, as one example to be asked for."""
    return [
        Example(
            SFT,
            source_of(material),
            material.kind,
            material.question.strip(),
            material.answer.strip(),
            split,
        )
    ]


def _examples_of(material: training_materials.Material, split: str) -> list[Example]:
    if material.kind == training_materials.DOCUMENT:
        return _pretrain(material, split)
    if material.kind in training_materials.ASKABLE_KINDS:
        return _sft(material, split)
    return []


def _units_of(examples: Sequence[Example]) -> dict[str, str]:
    """One split unit per group of sources that say the same words.

    The corpus repeats itself: two starter cards can give the same answer, and a
    document can quote a taught lesson. Splitting those by source would train
    one copy and measure the other, which reads as memory. Sources sharing a
    text are therefore joined, and the smallest label in a group names it, so
    the same corpus always gives the same units.
    """
    parent = {example.source: example.source for example in examples}

    def find(source: str) -> str:
        while parent[source] != source:
            parent[source] = parent[parent[source]]
            source = parent[source]
        return source

    def join(first: str, second: str) -> None:
        left, right = find(first), find(second)
        if left != right:
            parent[max(left, right)] = min(left, right)

    by_text: dict[str, list[str]] = {}
    for example in examples:
        by_text.setdefault(normalise(example.target), []).append(example.source)
    for group in by_text.values():
        for other in group[1:]:
            join(group[0], other)
    return {source: find(source) for source in sorted(parent)}


def build(
    materials: Iterable[training_materials.Material],
    salt: str = splits.DEFAULT_SALT,
    share: float = splits.DEFAULT_SHARE,
) -> Dataset:
    """The dataset: every source tagged, and no text tagged onto both sides.

    The tag is taken per unit, so a repeated answer cannot reach the training
    side while its copy is held out for the measurement.
    """
    raw = [example for material in materials for example in _examples_of(material, splits.TRAIN)]
    units = _units_of(raw)
    tags = splits.assign(set(units.values()), salt, share)
    assignment = {source: tags[unit] for source, unit in units.items()}
    examples = tuple(replace(example, split=assignment[example.source]) for example in raw)
    return Dataset(examples, assignment, salt=salt, share=share)


def corpus(tenant_id: int | None, db: Session) -> list[training_materials.Material]:
    """Every piece of material the workspace holds, with held-out items removed.

    This is the same reading the word model trains from, so the two cannot end
    up learning from different corpora without someone noticing.
    """
    return training_materials.gather(tenant_id, db)


def prepared(tenant_id: int | None, db: Session) -> Dataset:
    """This workspace's material, built into the dataset every operation reads."""
    return build(corpus(tenant_id, db))


def leakage(dataset: Dataset) -> tuple[str, ...]:
    """The held-out texts that also sit on the training side.

    Grouping makes this empty by construction, so it is the check a run makes
    before it starts: a non-empty answer means the dataset it was handed was not
    built here, and the measurement it would produce would be memory, not
    ability.
    """
    trained = {
        normalise(example.target) for example in dataset.examples if example.split == splits.TRAIN
    }
    held = {
        normalise(example.target): example.source
        for example in dataset.examples
        if example.split == splits.HELD_OUT
    }
    return tuple(sorted({held[text] for text in held if text in trained}))
