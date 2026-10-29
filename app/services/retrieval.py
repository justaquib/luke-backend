"""Ranking passages: what the embedding found and which words the question used.

The embedding answers questions the way the passage was worded; BM25 answers
which passage actually uses the words the asker typed. Neither score is on the
other's scale, so the two rankings are fused by reciprocal rank instead of
being averaged: a passage found by either ranking is kept, and a passage both
rankings agree on comes first. No candidate is rescaled against its neighbours,
so adding a passage cannot change the score of the passages beside it.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence

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

# BM25 counts a few more grammar words than the lesson matcher ignores.
_BM25_STOPWORDS = _STOPWORDS | frozenset(
    {
        "a",
        "an",
        "as",
        "at",
        "be",
        "been",
        "by",
        "do",
        "he",
        "her",
        "him",
        "his",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "just",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "than",
        "then",
        "they",
        "to",
        "us",
        "very",
        "we",
        "also",
    }
)

# Standard BM25 settings: a term stops helping after a few uses, and a long
# passage is not penalised for being long.
_BM25_K1 = 1.5
_BM25_B = 0.75

# The constant from the reciprocal rank fusion paper. At 60 a rank-1 hit from
# either search is worth the same, and a hit both searches like climbs above it.
RRF_K = 60


def terms(text: str) -> set[str]:
    """The words of a text that can name something, lower-cased."""
    return {
        token
        for token in "".join(
            character if character.isalnum() else " " for character in text.lower()
        ).split()
        if len(token) > 2
    }


def tokens(text: str) -> list[str]:
    """The words BM25 counts: lower-cased, punctuation split off, stopwords gone."""
    return [
        token
        for token in "".join(
            character if character.isalnum() else " " for character in text.lower()
        ).split()
        if token not in _BM25_STOPWORDS
    ]


class LexicalIndex:
    """BM25 over one corpus, with its word counts kept between questions.

    The words of a passage do not change while the passage is stored, so a
    corpus indexed once can score every question that follows without
    reading the passages or counting their words again.
    """

    def __init__(self, documents: Sequence[str]) -> None:
        self._counts = [Counter(tokens(document)) for document in documents]
        self._lengths = [sum(count.values()) for count in self._counts]
        self._average = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        self._seen: Counter[str] = Counter()
        for count in self._counts:
            self._seen.update(count.keys())
        self._total = len(self._counts)

    def scores(self, question: str) -> list[float]:
        """How much each stored passage is about the words of the question."""
        wanted = tokens(question)
        if not wanted or not self._counts:
            return [0.0] * len(self._counts)
        scores: list[float] = []
        for count, length in zip(self._counts, self._lengths, strict=True):
            score = 0.0
            for term in set(wanted):
                frequency = count.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(
                    1 + (self._total - self._seen[term] + 0.5) / (self._seen[term] + 0.5)
                )
                norm = frequency + _BM25_K1 * (
                    1 - _BM25_B + _BM25_B * length / max(self._average, 1e-9)
                )
                score += idf * frequency * (_BM25_K1 + 1) / norm
            scores.append(score)
        return scores

    def ranking(self, question: str) -> list[int]:
        """The positions of the passages that share a word with the question."""
        scores = self.scores(question)
        ranked = [index for index, score in enumerate(scores) if score > 0]
        return sorted(ranked, key=lambda index: (-scores[index], index))


def bm25_scores(question: str, documents: Sequence[str]) -> list[float]:
    """How much each passage is about the words of the question, by BM25."""
    return LexicalIndex(documents).scores(question)


def bm25_ranking(question: str, documents: Sequence[str]) -> list[int]:
    """The positions of the passages that share a word with the question."""
    return LexicalIndex(documents).ranking(question)


def fuse_rankings(rankings: Sequence[Sequence[Hashable]], k: int = RRF_K) -> list[Hashable]:
    """Fuse rankings by reciprocal rank, keeping every item either one found."""
    fused: dict[Hashable, float] = {}
    first_seen: dict[Hashable, int] = {}
    for order, ranking in enumerate(rankings):
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(item, order)
    return sorted(fused, key=lambda item: (-fused[item], first_seen[item], str(item)))


def similarity_from_distance(distance: float | None, space: str) -> float | None:
    """Read a Chroma distance as a similarity in the collection's real space.

    Chroma reports squared L2 by default, where a unit-vector cosine is
    ``1 - d / 2``; a cosine collection reports ``1 - d``; an inner-product
    collection reports ``1 - dot``. Mixing the three silently deflates a match,
    which made a near-exact hit read as "not learned".
    """
    if distance is None:
        return None
    value = float(distance)
    score = 1.0 - (value / 2.0 if space == "l2" else value)
    return round(max(0.0, min(1.0, score)), 4)
