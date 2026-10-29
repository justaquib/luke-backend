"""Which language a passage is written in, as far as a word list can say.

Luke is served in English, so this is a coarse check, not a language
classifier: it reports ``en`` only when the letters are Latin and the passage
uses common English words, and ``unknown`` otherwise. Nothing is guessed from
the name of a file or the country of a trainer.
"""

from __future__ import annotations

import re

_PATTERN = re.compile(r"[A-Za-z']+")
_MIN_LETTERS = 8
_MIN_LATIN_SHARE = 0.9
_COMMON = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)


def guess(text: str) -> str:
    """The language of a passage: ``en`` when it reads as English, else ``unknown``."""
    letters = [character for character in text if character.isalpha()]
    if len(letters) < _MIN_LETTERS:
        return "unknown"
    latin_share = sum(1 for character in letters if character.isascii()) / len(letters)
    if latin_share < _MIN_LATIN_SHARE:
        return "unknown"
    words = {word.casefold() for word in _PATTERN.findall(text)}
    return "en" if words & _COMMON else "unknown"
