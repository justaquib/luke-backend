"""What the brain has trained from, and how the training works.

The trainer opens this from the brain page. It says where the material came
from, how much of it there is, and what the training does with it, so the
training is never a mystery.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy.orm import Session

from app.services.language import training_materials as materials_service
from app.services.language.training_materials import Material, TrainingSet

# The sources the brain learns from, in the order they are read.
SOURCES: tuple[tuple[str, str, str], ...] = (
    (
        materials_service.LESSON,
        "Lessons you taught",
        "The question and the answer of every lesson in the language learner.",
    ),
    (
        materials_service.STARTER,
        "Starter lessons",
        "The lessons of the school grades the learner has already reached.",
    ),
    (
        materials_service.CONVERSATION,
        "Basic conversation",
        "The greetings, the name and the small talk the assistant answers itself.",
    ),
    (
        materials_service.APPROVED,
        "Approved answers",
        "The questions and the answers you approved in the trainer.",
    ),
    (
        materials_service.CORRECTION,
        "Your corrections",
        "The answers you corrected, so the model stops making the same mistake.",
    ),
    (
        materials_service.DOCUMENT,
        "Your documents",
        "The chunks of every document you uploaded.",
    ),
    (
        materials_service.WORD,
        "Words you taught",
        "The words and the meanings you taught with the dictionary.",
    ),
    (
        materials_service.SENTENCE,
        "Sentences you wrote",
        "The sentences you taught on their own.",
    ),
)

# The steps of the training, in the order they happen.
STEPS: tuple[tuple[str, str], ...] = (
    ("Collect", "Every source above is gathered into one list of material."),
    ("Clean", "The wording is repaired, so the model does not learn a slip of the pen."),
    (
        "Build the vocabulary",
        "Every word of every source becomes a token the model can write with.",
    ),
    ("Train", "The word model reads every sentence and learns to predict the next word."),
    (
        "Measure",
        "The word model is measured on held-out sentences, so the score is honest.",
    ),
)


def source_report(
    materials: Sequence[Material],
    training: TrainingSet | None = None,
    words: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """What the brain has trained from, and how the training works."""
    counts = _counts(materials)
    sources = [
        {
            "key": key,
            "label": label,
            "detail": detail,
            "items": counts.get(key, {}).get("items", 0),
            "words": counts.get(key, {}).get("words", 0),
        }
        for key, label, detail in SOURCES
    ]
    return {
        "sources": sources,
        "steps": [{"title": title, "detail": detail} for title, detail in STEPS],
        "totals": {
            "items": sum(entry["items"] for entry in sources),
            "words": sum(entry["words"] for entry in sources),
            "sentences": len(training.sentences) if training else 0,
        },
        "documents": training.documents.as_dict() if training else None,
        "model": words or {},
    }


def tenant_report(
    tenant_id: int | None,
    db: Session,
    words: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """What this tenant's brain has trained from, and how the training works.

    The same material the training uses is counted here, so the report can
    never say something the training did not do.
    """
    materials = materials_service.gather(tenant_id, db)
    training = materials_service.training_set(materials, tenant_id)
    return source_report(materials, training, words)


def _counts(materials: Sequence[Material]) -> dict[str, dict[str, int]]:
    """How much material each source holds, and how many words it has."""
    counts: dict[str, dict[str, int]] = {}
    for material in materials:
        entry = counts.setdefault(material.kind, {"items": 0, "words": 0})
        entry["items"] += 1
        entry["words"] += len(material.question.split()) + len(material.answer.split())
    return counts
