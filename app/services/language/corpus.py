"""The cleaned sentences the word model learns from."""

from __future__ import annotations

from app.services.language import runtime
from app.services.language.curriculum import FOUNDATION_LESSON_CARDS
from app.services.language.grammar import _repair_sentences
from app.services.language.lessons import all_lesson_cards


def training_corpus(tenant_id: int | None) -> list[str]:
    """Every sentence the model can learn from.

    It is the trainer's own material: the lessons that were taught, the
    answers that were approved, the sentences that were written and the
    corrections that were kept. The wording is cleaned first, so the model does
    not learn a slip of the pen.
    """
    with runtime._LOCK:
        state = runtime._read(tenant_id)
    raw: list[str] = []
    for card in all_lesson_cards(tenant_id):
        question = (card.get("question") or "").strip()
        answer = (card.get("answer") or "").strip()
        if answer:
            raw.append(answer)
        if question and answer:
            raw.append(f"{question} {answer}")
    for lesson in state.get("lessons", []):
        raw.append((lesson.get("text") or "").strip())
    for lesson in FOUNDATION_LESSON_CARDS:
        raw.append((lesson.get("answer") or "").strip())
    raw = [text for text in raw if text]
    sentences: list[str] = []
    seen: set[str] = set()
    for cleaned in _repair_sentences(raw):
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        sentences.append(cleaned)
    return sentences
