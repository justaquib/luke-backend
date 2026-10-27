"""The questions a knowledge transfer can ask a fine-tuned teacher.

Every kind of material the tenant gave Luke becomes a question: a lesson is
asked as it was written, a taught word is asked for its meaning, a document is
asked about by name, a subject is asked what was learned about it. Questions
Luke already has are left out, and the remaining ones are interleaved by kind so
a short transfer hears from every source.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import Faq
from app.services.language import training_materials

# How much of a document the teacher reads as its reference.
MAX_REFERENCE = 1200

# How many subjects get a topic question, and how many pairs one names.
MAX_TOPICS = 12
TOPIC_PAIRS = 6

# The kind of a question built from a lesson's subject.
TOPIC = "topic"

# The order the kinds are asked in, so a short run hears from each source.
_KIND_ORDER = (
    training_materials.LESSON,
    training_materials.STARTER,
    training_materials.CORRECTION,
    training_materials.WORD,
    training_materials.DOCUMENT,
    TOPIC,
)

# The kinds that already name a question, and so are asked as they are.
_ASKABLE = (
    training_materials.LESSON,
    training_materials.STARTER,
    training_materials.CORRECTION,
)


@dataclass(frozen=True)
class Seed:
    """One question to ask the teacher, and the material it should answer from."""

    kind: str
    question: str
    reference: str


def seeds_for(tenant_id: int | None, db: Session) -> list[Seed]:
    """Every question the material can ask, without the ones Luke answers already."""
    materials = training_materials.gather(tenant_id, db)
    rows = [
        *_askable_seeds(materials),
        *_word_seeds(materials),
        *_document_seeds(materials),
        *_topic_seeds(materials),
    ]
    return _interleaved(_unique(rows, _asked_before(tenant_id, db)))


def count_kinds(seeds: Sequence[Seed]) -> dict[str, int]:
    """How many questions each source of material would ask."""
    counts: dict[str, int] = {}
    for seed in seeds:
        counts[seed.kind] = counts.get(seed.kind, 0) + 1
    return counts


def _askable_seeds(materials: Iterable[training_materials.Material]) -> list[Seed]:
    """The lessons, starter lessons and corrections, asked as they were written."""
    return [
        Seed(material.kind, material.question.strip(), material.answer.strip())
        for material in materials
        if material.kind in _ASKABLE and material.question.strip() and material.answer.strip()
    ]


def _word_seeds(materials: Iterable[training_materials.Material]) -> list[Seed]:
    """The taught words, asked for their meaning."""
    seeds: list[Seed] = []
    for material in materials:
        word = material.question.strip()
        meaning = material.answer.strip()
        if material.kind != training_materials.WORD or not word or not meaning:
            continue
        seeds.append(
            Seed(material.kind, f"What does the word {word} mean?", f"{word} means {meaning}.")
        )
    return seeds


def _document_seeds(materials: Iterable[training_materials.Material]) -> list[Seed]:
    """One question per uploaded document, answered from all of its chunks."""
    grouped: dict[str, list[str]] = {}
    for material in materials:
        if material.kind == training_materials.DOCUMENT:
            grouped.setdefault(material.question, []).append(material.answer)
    seeds: list[Seed] = []
    for source, chunks in grouped.items():
        title = _readable(source)
        reference = " ".join(chunks)[:MAX_REFERENCE].strip()
        if title and reference:
            seeds.append(Seed(training_materials.DOCUMENT, f"Tell me about {title}.", reference))
    return seeds


def _topic_seeds(materials: Iterable[training_materials.Material]) -> list[Seed]:
    """One question per subject, answered from the lessons that teach it."""
    grouped: dict[str, list[tuple[str, str]]] = {}
    for material in materials:
        if material.kind not in (training_materials.LESSON, training_materials.STARTER):
            continue
        for subject in _subjects(material.subject):
            grouped.setdefault(subject, []).append((material.question, material.answer))
    return [
        Seed(
            TOPIC,
            f"What have you learned about {subject}?",
            " ".join(f"{question} {answer}" for question, answer in rows[:TOPIC_PAIRS]),
        )
        for subject, rows in list(grouped.items())[:MAX_TOPICS]
    ]


def _subjects(value: str) -> list[str]:
    """The subjects of a lesson, one by one."""
    return [part.strip() for part in value.split(",") if part.strip()]


def _readable(source: str) -> str:
    """A document's file name as a question can name it."""
    name = source.replace("\\", "/").rsplit("/", 1)[-1].strip()
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return stem.replace("_", " ").replace("-", " ").strip()


def _unique(seeds: Sequence[Seed], asked: Sequence[str]) -> list[Seed]:
    """The seeds, in their order, without a question asked or stored before."""
    seen = {_key(question) for question in asked}
    kept: list[Seed] = []
    for seed in seeds:
        key = _key(seed.question)
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(seed)
    return kept


def _interleaved(seeds: Sequence[Seed]) -> list[Seed]:
    """The seeds round-robin by kind, so a limited run hears from every source."""
    groups: dict[str, list[Seed]] = {}
    for seed in seeds:
        groups.setdefault(seed.kind, []).append(seed)
    queues = [iter(groups.get(kind, [])) for kind in _KIND_ORDER]
    rows: list[Seed] = []
    while queues:
        alive: list[Iterator[Seed]] = []
        for queue in queues:
            seed = next(queue, None)
            if seed is not None:
                rows.append(seed)
                alive.append(queue)
        queues = alive
    return rows


def _asked_before(tenant_id: int | None, db: Session) -> list[str]:
    """Every question this tenant already has, whatever its status."""
    statement = select(Faq.question)
    if tenant_id is not None:
        statement = statement.where(Faq.tenant_id == tenant_id)
    return [question for (question,) in db.execute(statement).all()]


def _key(question: str) -> str:
    """A question as it is compared, without case or spacing differences."""
    return " ".join(re.findall(r"[a-z0-9]+", question.casefold()))
