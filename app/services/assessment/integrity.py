"""Keeping the held-out assessment items out of everything the brain learns.

The bank is a measuring stick, not material. If an evaluation prompt is stored
as a document, a lesson or a sentence, the score starts measuring memory
instead of ability. Two defences live here: ``scan`` finds held-out text that
is already stored, and ``is_held_out`` lets every training loader drop a
sentence before it is kept. The ban is unconditional; there is no override.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

HELD_OUT_SPLIT = "eval"

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Leak:
    """One held-out item found in material the brain can read."""

    item_id: str
    skill: str
    where: str
    how: str


def normalise(text: str) -> str:
    """The form two texts are compared in: case, width and spacing folded."""
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    return _WHITESPACE.sub(" ", folded).strip()


@lru_cache(maxsize=1)
def _held_out() -> tuple[re.Pattern[str], tuple[tuple[str, str, str], ...]]:
    """Every held-out id and prompt, as entries and as one search pattern."""
    from app.services.assessment import bank

    entries: list[tuple[str, str, str]] = []
    needles: list[str] = []
    for item in bank.items(split=HELD_OUT_SPLIT):
        item_id = str(item.get("id") or "")
        prompt = normalise(str(item.get("prompt") or ""))
        if not item_id or not prompt:
            continue
        entries.append((item_id, str(item.get("skill") or ""), prompt))
        needles.extend((normalise(item_id), prompt))
    ordered = sorted(needles, key=len, reverse=True)
    return re.compile("|".join(re.escape(needle) for needle in ordered) or "(?!)"), tuple(entries)


def leaks_in(text: str, where: str = "") -> tuple[Leak, ...]:
    """The held-out items whose id or prompt appears in this text."""
    haystack = normalise(text)
    pattern, entries = _held_out()
    if not haystack or pattern.search(haystack) is None:
        return ()
    found: list[Leak] = []
    for item_id, skill, prompt in entries:
        if normalise(item_id) in haystack:
            found.append(Leak(item_id, skill, where, "id"))
        elif prompt in haystack:
            found.append(Leak(item_id, skill, where, "prompt"))
    return tuple(found)


def is_held_out(text: str) -> bool:
    """Whether this text carries any held-out item, so a loader must drop it."""
    haystack = normalise(text)
    if not haystack:
        return False
    return _held_out()[0].search(haystack) is not None


def leaks(sources: Iterable[tuple[str, str]]) -> list[Leak]:
    """Every leak in named texts, in the order the texts were given."""
    found: list[Leak] = []
    for where, text in sources:
        found.extend(leaks_in(text, where))
    return found


def stored_sources(session: Any, tenant_id: int | None = None) -> list[tuple[str, str]]:
    """The passages and approved answers as they sit in the database."""
    from sqlalchemy import select

    from app.db.database import Document, Faq

    documents = select(Document.id, Document.source, Document.text).order_by(Document.id)
    if tenant_id is not None:
        documents = documents.where(Document.tenant_id == tenant_id)
    sources = [
        (f"document:{document_id} {source}", text or "")
        for document_id, source, text in session.execute(documents).all()
    ]
    faqs = select(Faq.id, Faq.question, Faq.answer).order_by(Faq.id)
    if tenant_id is not None:
        faqs = faqs.where(Faq.tenant_id == tenant_id)
    for faq_id, question, answer in session.execute(faqs).all():
        sources.append((f"faq:{faq_id}", f"{question or ''}\n{answer or ''}"))
    return sources


def lesson_sources(tenant_id: int | None = None) -> list[tuple[str, str]]:
    """The lessons, sentences and words the tenant has taught, as stored."""
    from app.services.language import runtime
    from app.services.language.curriculum import (
        FOUNDATION_LESSON_CARDS,
        FOUNDATION_SENTENCES_BY_GRADE,
    )
    from app.services.language.lessons import all_lesson_cards

    sources: list[tuple[str, str]] = []
    for card in all_lesson_cards(tenant_id):
        label = card.get("id") or card.get("question") or ""
        text = f"{card.get('question') or ''}\n{card.get('answer') or ''}"
        sources.append((f"lesson:{label}", text))
    for card in FOUNDATION_LESSON_CARDS:
        question = card.get("question") or ""
        answer = card.get("answer") or ""
        sources.append(("foundation_lesson", f"{question}\n{answer}"))
    for grade, sentences in FOUNDATION_SENTENCES_BY_GRADE.items():
        sources.extend((f"foundation_sentence:{grade}", sentence) for sentence in sentences)
    with runtime._LOCK:
        state = runtime._read(tenant_id)
    for index, lesson in enumerate(state.get("lessons", [])):
        sources.append((f"sentence:{index}", str(lesson.get("text") or "")))
    for index, card in enumerate(state.get("word_cards", [])):
        text = f"{card.get('word') or ''} {card.get('meaning') or ''}"
        sources.append((f"word:{index}", text.strip()))
    return sources


def scan(session: Any, tenant_id: int | None = None) -> list[Leak]:
    """Every held-out item found in material this tenant can learn from."""
    return leaks([*stored_sources(session, tenant_id), *lesson_sources(tenant_id)])
