"""Turning a lesson question into a sentence that holds its answer."""

from __future__ import annotations

import logging
import re

from app.services.language.helpers import _normalize_phrase
from app.services.language.lessons import all_lesson_cards

logger = logging.getLogger(__name__)

_LESSON_QUESTION_STATEMENT = re.compile(
    r"^(?:what|which|who)\s+is\s+(.+?)\s*\??$",
    re.IGNORECASE,
)
_LESSON_WHICH_SUBJECT = re.compile(
    r"^which\s+(.+?)\s+(has|have|is|are|was|were|does|do|did|can|could|will|would|should)\s+"
    r"(.+?)\s*\??$",
    re.IGNORECASE,
)
_LESSON_NAME_REQUEST = re.compile(
    r"^name\s+(the|a|an)\s+(.+?)\s*\??$",
    re.IGNORECASE,
)
_LESSON_NAME_LIST = re.compile(
    r"^name\s+(?:some|all|the\s+list\s+of)\s+(.+?)\s*\??$",
    re.IGNORECASE,
)


def _clean_phrase(text: str) -> str:
    """A phrase without its end mark."""
    return (text or "").strip().rstrip(".!?").strip()


def _statement_from_lesson(question: str, answer: str) -> str | None:
    """Turn a lesson question into a statement that holds the answer."""
    answer = re.sub(r"\s*\([^)]*\)", "", answer or "").strip().rstrip(".!?").strip()
    if not answer or len(answer.split()) > 5:
        return None
    text = (question or "").strip()
    match = _LESSON_QUESTION_STATEMENT.match(text)
    if match:
        rest = _clean_phrase(match.group(1))
        return f"{answer} is {rest}." if rest else None
    match = _LESSON_WHICH_SUBJECT.match(text)
    if match:
        verb = match.group(2).strip()
        rest = _clean_phrase(match.group(3))
        return f"{answer} {verb} {rest}." if rest else None
    match = _LESSON_NAME_REQUEST.match(text)
    if match:
        article = match.group(1).strip()
        rest = _clean_phrase(match.group(2))
        return f"{answer} is {article} {rest}." if rest else None
    return None


def _list_statement_for_word(tenant_id: int | None, word: str) -> str | None:
    """A word taken from a list lesson becomes a sentence of its own."""
    target = _normalize_phrase(word)
    if not target:
        return None
    try:
        cards = all_lesson_cards(tenant_id)
    except Exception as error:  # noqa: BLE001 -- the taught lesson-card boundary
        logger.debug("the lesson sentences could not be read: %s", error)
        return None
    for card in cards:
        match = _LESSON_NAME_LIST.match((card.get("question") or "").strip())
        if not match:
            continue
        category = _clean_phrase(match.group(1))
        answer = card.get("answer") or ""
        parts = re.split(r"\bare\b", answer, maxsplit=1)
        items_text = parts[1] if len(parts) > 1 else answer
        items = [item.strip(" .!") for item in items_text.split(",")]
        if not any(_normalize_phrase(item) == target for item in items):
            continue
        singular = (
            category[:-1] if category.endswith("s") and not category.endswith("ss") else category
        )
        return f"{word.strip().capitalize()} is a {singular}."
    return None
