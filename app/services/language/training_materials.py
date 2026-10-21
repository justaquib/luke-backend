"""Every piece of material the tenant has given the brain to learn from.

The brain learns from more than the lessons in the language learner. The
questions and answers the trainer approved, the corrections they gave, the
documents they uploaded, the words they taught and the sentences they wrote are
material as well.

All of it is collected here, so the training and the report the trainer reads
come from the same place and cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import Document, Faq
from app.services.language import instructions as instruction_service
from app.services.language import runtime
from app.services.language.lessons import all_lesson_cards
from app.services.language.runtime import subject_of
from app.services.language.topics import card_key, topic_index

# The kinds of material the tenant can give.
LESSON = "lesson"
STARTER = "starter_lesson"
APPROVED = "approved_answer"
CORRECTION = "correction"
DOCUMENT = "document"
WORD = "word"
SENTENCE = "sentence"

# The kinds that hold a question and an answer, so they can be asked for.
ASKABLE_KINDS = (LESSON, STARTER, APPROVED, CORRECTION)

# How many chunks of an uploaded document are used. A long document is still
# read, but the training does not grow without end.
MAX_DOCUMENT_CHUNKS = 500

# How many sentences the model is given at most.
MAX_SENTENCES = 20000


@dataclass(frozen=True)
class Material:
    """One piece of material the brain can learn from."""

    kind: str
    question: str
    answer: str
    subject: str = ""

    def as_card(self) -> dict[str, Any]:
        """The lesson shape the instruction builder works with."""
        return {
            "id": f"{self.kind}:{self.question}",
            "question": self.question,
            "answer": self.answer,
            "subject": self.subject,
            "subjects": [self.subject] if self.subject else [],
        }


@dataclass(frozen=True)
class TrainingSet:
    """The examples the model trains on, and the material they came from."""

    sentences: list[str] = field(default_factory=list)
    instructions: list[tuple[str, str]] = field(default_factory=list)
    contexts: list[tuple[str, str, str]] = field(default_factory=list)
    materials: list[Material] = field(default_factory=list)


def collect(
    tenant_id: int | None,
    approved: Sequence[tuple[str, str]] = (),
    documents: Sequence[tuple[str, str]] = (),
    corrections: Sequence[tuple[str, str]] = (),
) -> list[Material]:
    """Every piece of material this tenant has given the brain.

    The material that lives on disk is read here; the material that lives in
    the database is handed in, because the training runs in its own process.
    """
    materials: list[Material] = []
    materials.extend(_lesson_materials(tenant_id))
    materials.extend(_approved_materials(approved))
    materials.extend(_correction_materials(corrections))
    materials.extend(_document_materials(documents))
    materials.extend(_word_materials(tenant_id))
    materials.extend(_sentence_materials(tenant_id))
    return materials


def gather(tenant_id: int | None, db: Session) -> list[Material]:
    """Every piece of material, from the tenant's files and from the database.

    The approved answers and the uploaded documents live in the database, and
    the corrections live in the brain's own file, so all three are read here.
    """
    from app.services import brain as brain_service

    feedback = brain_service.load_feedback(tenant_id)
    corrections = [
        (str(entry.get("question") or ""), str(entry.get("correction") or ""))
        for entry in feedback
        if entry.get("correction")
    ]
    return collect(
        tenant_id,
        approved=_approved_from_database(tenant_id, db),
        documents=_documents_from_database(tenant_id, db),
        corrections=corrections,
    )


def training_set(materials: Sequence[Material]) -> TrainingSet:
    """The examples the model learns from, built from every source.

    A question and an answer become a request and its answer, in more than one
    wording, so the model answers a request it was never taught instead of
    only repeating the lesson. The material itself is also a sentence, so the
    words of a document or a correction reach the vocabulary.
    """
    cards = [material.as_card() for material in materials if material.kind in ASKABLE_KINDS]
    index = topic_index(cards)
    sentences: list[str] = []
    spoken: list[str] = []
    instructions: list[tuple[str, str]] = []
    contexts: list[tuple[str, str, str]] = []
    for material in materials:
        answer = material.answer.strip()
        if not answer:
            continue
        if material.kind in ASKABLE_KINDS:
            spoken.append(answer)
            spoken.append(f"{material.question} {answer}")
        else:
            sentences.append(answer)
        for instruction, response in _instructions_for(material, index):
            instructions.append((instruction, response))
            contexts.append((instruction, f"{material.question}. {answer}", response))
    sentences.extend(_repaired(spoken))
    return TrainingSet(
        sentences=_unique(sentences)[:MAX_SENTENCES],
        instructions=instructions,
        contexts=contexts,
        materials=list(materials),
    )


def _repaired(sentences: list[str]) -> list[str]:
    """The wording of the answers, with a missing article put back."""
    from app.services.language.grammar import _repair_sentences

    return _repair_sentences(sentences)


def _unique(sentences: list[str]) -> list[str]:
    """The sentences, without the ones that say the same thing twice."""
    seen: set[str] = set()
    result: list[str] = []
    for sentence in sentences:
        key = sentence.strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(sentence)
    return result


def _instructions_for(material: Material, index: dict[str, list[str]]) -> list[tuple[str, str]]:
    """The requests this material can be asked for, and the answer to each."""
    if material.kind == WORD:
        meaning = f"{material.question} means {material.answer}."
        return [(f"What does the word {material.question} mean?", meaning)]
    if material.kind == DOCUMENT:
        return [(f"Tell me about {material.question}.", material.answer)]
    card = material.as_card()
    return [
        (item.instruction, item.response)
        for item in instruction_service.instructions_for(card, index.get(card_key(card), []))
    ]


def _approved_from_database(tenant_id: int | None, db: Session) -> list[tuple[str, str]]:
    """The questions and answers the trainer approved."""
    statement = select(Faq.question, Faq.answer).where(Faq.status == "approved")
    if tenant_id is not None:
        statement = statement.where(Faq.tenant_id == tenant_id)
    return [(question, answer) for question, answer in db.execute(statement).all()]


def _documents_from_database(tenant_id: int | None, db: Session) -> list[tuple[str, str]]:
    """The chunks of the documents that were uploaded."""
    statement = select(Document.source, Document.text)
    if tenant_id is not None:
        statement = statement.where(Document.tenant_id == tenant_id)
    return [(source, text) for source, text in db.execute(statement).all()]


def _lesson_materials(tenant_id: int | None) -> list[Material]:
    """The lessons taught in the language learner, and the starter ones."""
    materials: list[Material] = []
    for card in all_lesson_cards(tenant_id):
        question = str(card.get("question") or "").strip()
        answer = str(card.get("answer") or "").strip()
        if not question or not answer:
            continue
        kind = STARTER if card.get("kind") == "starter" else LESSON
        materials.append(Material(kind, question, answer, subject_of(card)))
    return materials


def _approved_materials(approved: Sequence[tuple[str, str]]) -> list[Material]:
    """The questions and answers the trainer approved."""
    materials: list[Material] = []
    for question, answer in approved:
        clean_question = str(question or "").strip()
        clean_answer = str(answer or "").strip()
        if not clean_question or not clean_answer:
            continue
        materials.append(Material(APPROVED, clean_question, clean_answer))
    return materials


def _correction_materials(corrections: Sequence[tuple[str, str]]) -> list[Material]:
    """The answers the trainer corrected."""
    materials: list[Material] = []
    for question, correction in corrections:
        clean_question = str(question or "").strip()
        clean_correction = str(correction or "").strip()
        if not clean_correction:
            continue
        materials.append(Material(CORRECTION, clean_question, clean_correction))
    return materials


def _document_materials(documents: Sequence[tuple[str, str]]) -> list[Material]:
    """The chunks of the documents that were uploaded."""
    materials: list[Material] = []
    for source, text in list(documents)[:MAX_DOCUMENT_CHUNKS]:
        clean_text = str(text or "").strip()
        if not clean_text:
            continue
        materials.append(Material(DOCUMENT, str(source or "document").strip(), clean_text))
    return materials


def _word_materials(tenant_id: int | None) -> list[Material]:
    """The words that were taught, with the meaning they should remember."""
    materials: list[Material] = []
    for card in _state_of(tenant_id).get("word_cards", []):
        word = str(card.get("word") or "").strip()
        meaning = str(card.get("meaning") or "").strip()
        if not word or not meaning:
            continue
        materials.append(Material(WORD, word, meaning))
    return materials


def _sentence_materials(tenant_id: int | None) -> list[Material]:
    """The sentences that were taught on their own."""
    materials: list[Material] = []
    for lesson in _state_of(tenant_id).get("lessons", []):
        text = str(lesson.get("text") or "").strip()
        if text:
            materials.append(Material(SENTENCE, "", text))
    return materials


def _state_of(tenant_id: int | None) -> dict:
    """The saved learner state of this tenant."""
    with runtime._LOCK:
        return runtime._read(tenant_id)
