"""Ranking passages: what the embedding found and what the question named.

The embedding answers questions the way the lesson was worded; the terms say
what the asker actually mentioned. A passage that shares a real term with the
question outranks a merely nearby one, and a shared stopword counts for
neither.
"""

from __future__ import annotations

from typing import Any

# The words that say nothing about what a passage is about.
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "was",
        "were",
        "with",
        "what",
        "which",
        "who",
        "how",
        "when",
        "where",
        "why",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "this",
        "that",
        "these",
        "those",
        "from",
        "into",
        "about",
        "there",
        "their",
        "your",
        "you",
        "not",
        "but",
        "has",
        "have",
        "had",
        "its",
    }
)

_SEMANTIC_WEIGHT = 0.4
_LEXICAL_WEIGHT = 0.6


def terms(text: str) -> set[str]:
    """The words of a text that can name something, lower-cased."""
    return {
        token
        for token in "".join(
            character if character.isalnum() else " " for character in text.lower()
        ).split()
        if len(token) > 2
    }


def rerank_hits(question: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order candidate chunks by meaning and by the terms the question names."""
    wanted = terms(question) - _STOPWORDS
    distances = [float(hit.get("distance") or 0.0) for hit in hits]
    lowest = min(distances, default=0.0)
    spread = (max(distances, default=0.0) - lowest) or 1.0
    scored = []
    for hit, distance in zip(hits, distances, strict=True):
        named = terms(str(hit.get("text", ""))) - _STOPWORDS
        lexical = len(wanted & named) / len(wanted) if wanted else 0.0
        semantic = 1.0 - (distance - lowest) / spread
        score = _SEMANTIC_WEIGHT * semantic + _LEXICAL_WEIGHT * lexical
        scored.append((score, -distance, hit))
    return [hit for _, _, hit in sorted(scored, key=lambda item: (-item[0], item[1]))]
