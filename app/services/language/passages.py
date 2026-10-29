"""The one way a long text becomes training passages.

A document chunk is too long to be one example and too coarse to teach a
tokenizer, so it is cut into passages of 3 to 50 words. The training bridge
and the tokenizer corpus both call this splitter, so the two cannot learn from
different shapes of the same text.
"""

from __future__ import annotations

import re

MIN_WORDS = 3
MAX_WORDS = 50

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _units(text: str) -> list[str]:
    """The sentences of every non-empty line, with their spacing normalised."""
    units: list[str] = []
    for line in (text or "").splitlines():
        clean = " ".join(line.split())
        if not clean:
            continue
        units.extend(part.strip() for part in _SENTENCE_END.split(clean) if part.strip())
    return units


def _flush(current: list[str], min_words: int, passages: list[str]) -> None:
    if sum(len(part.split()) for part in current) >= min_words:
        passages.append(" ".join(current))
    current.clear()


def split_passages(
    text: str,
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
) -> list[str]:
    """The passages of a text, each within the word budget."""
    passages: list[str] = []
    current: list[str] = []
    for unit in _units(text):
        words = unit.split()
        if len(words) > max_words:
            _flush(current, min_words, passages)
            for start in range(0, len(words), max_words):
                chunk = " ".join(words[start : start + max_words])
                if len(chunk.split()) >= min_words:
                    passages.append(chunk)
            continue
        if sum(len(part.split()) for part in current) + len(words) > max_words:
            _flush(current, min_words, passages)
        current.append(unit)
    _flush(current, min_words, passages)
    return passages
