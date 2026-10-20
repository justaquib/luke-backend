"""Short answers in the assistant's own words, with the long version on request.

A live source such as Wikipedia returns a full article extract. That is the
wrong size for a chat reply, so the assistant answers with the most relevant
part of it and keeps the full text for when the user asks to elaborate.
"""

from __future__ import annotations

import re


_ELABORATION_HINTS = re.compile(
    r"\b(explain|elaborate|more\s+detail|in\s+detail|tell\s+me\s+more|go\s+deeper|"
    r"expand|longer\s+answer|full\s+answer|walk\s+me\s+through|why\s+(?:is|are|do|does|did)|"
    r"how\s+come|what\s+happens\s+if)\b",
    re.IGNORECASE,
)

_QUESTION_WORDS = {
    "what", "who", "where", "when", "which", "why", "how", "is", "are", "was", "were", "do",
    "does", "did", "the", "a", "an", "of", "for", "in", "on", "at", "to", "with", "and", "or",
    "please", "tell", "me", "it", "its", "this", "that", "these", "those", "can", "could", "would",
    "should", "will", "color", "colour", "mean", "means", "called", "known", "there", "here",
}


def wants_elaboration(question: str) -> bool:
    """True when the user is asking for the longer answer."""
    return bool(_ELABORATION_HINTS.search(question or ""))


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [part.strip() for part in parts if part.strip()]


def _terms(question: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z]+", (question or "").lower())
        if len(word) > 2 and word not in _QUESTION_WORDS
    }


def shorten(text: str, question: str, limit: int = 180) -> str:
    """The most relevant part of a long answer, in a couple of sentences."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    if len(cleaned) <= limit:
        return cleaned

    terms = _terms(question)
    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(_sentences(cleaned)):
        words = set(re.findall(r"[a-z]+", sentence.lower()))
        overlap = len(words & terms)
        # Prefer sentences that answer the question, then the shorter ones.
        score = overlap * 2 - (len(sentence) / 400)
        scored.append((score, index, sentence))

    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored or scored[0][0] <= 0:
        # Nothing matched the question, so use the opening of the source.
        best = _sentences(cleaned)[0]
    else:
        best = scored[0][2]

    picked = [best]
    if len(best) < 110:
        for _, _, sentence in scored[1:]:
            if len(" ".join(picked)) + len(sentence) + 1 > limit:
                continue
            picked.append(sentence)
            break

    # Keep the sentences in the order they appear in the source.
    ordered = sorted(_sentences(cleaned), key=lambda sentence: picked.index(sentence) if sentence in picked else 99)
    summary = " ".join(ordered).strip()
    if len(summary) > limit:
        summary = summary[:limit].rsplit(" ", 1)[0] + "…"
    return summary
