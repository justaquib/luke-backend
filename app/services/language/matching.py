"""Finding the lesson that answers a question.

A lesson is taught as a question and an answer, so the question that was
asked has to find the lesson again. The question is read in the words of the
lesson: the mark that names the kind of lesson ("GK:", "Poem:") is not a word
of the question, the endings of the words are left off, and the numbers are
kept, so "the square root of 25" is not answered with the lesson about 16.
"""

from __future__ import annotations

from app.services.language import runtime
from app.services.language.helpers import (
    _content_terms,
    _normalize_phrase,
    _phrase_similarity,
    _stem,
)
from app.services.language.topics import lesson_title

# The words that say nothing about what is being asked.
STOP_WORDS = frozenset(
    {
        "what",
        "who",
        "where",
        "when",
        "which",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "the",
        "a",
        "an",
        "of",
        "for",
        "please",
        "tell",
        "me",
        "about",
    }
)


def _stemmed_terms(text: str) -> set[str]:
    """The meaningful words of a text, without their endings."""
    return {_stem(term) for term in _content_terms(text, STOP_WORDS)}


def _matched_learned_lesson(question: str, lesson_cards: list[dict]) -> dict | None:
    """The lesson that best covers the question, with the question it was taught as.

    Two lessons can hold the same words and still be asked differently: "What
    is the head?" and "What does the head do?" are two lessons. The wording of
    the question tells them apart.
    """
    terms = _stemmed_terms(question)
    asked = _normalize_phrase(question)
    best: tuple[float, float, float, dict] | None = None
    for lesson in lesson_cards:
        title = lesson_title(str(lesson.get("question") or ""))
        candidate = _stemmed_terms(title)
        if not terms or not candidate:
            continue
        score = len(terms & candidate) / max(len(terms | candidate), 1)
        wording = _phrase_similarity(asked, _normalize_phrase(title))
        # A trusted answer wins over a weaker match of the same score.
        trust = float(lesson.get("confidence", runtime.settings.base_answer_confidence))
        if best is None or (score, wording, trust) > (best[0], best[1], best[2]):
            best = (score, wording, trust, lesson)
    return best[3] if best and best[0] >= 0.5 else None


def _answer_learned_lesson(question: str, lesson_cards: list[dict]) -> str | None:
    """The answer of the lesson that covers the question."""
    lesson = _matched_learned_lesson(question, lesson_cards)
    return lesson["answer"] if lesson else None
