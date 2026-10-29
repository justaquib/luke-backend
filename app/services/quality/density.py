"""How much of a passage is words rather than symbols, numbers or marks."""

from __future__ import annotations


def text_density(text: str) -> float:
    """The share of the whitespace tokens that start with a letter, from 0 to 1.

    A page of prose is close to 1. A table row, a line of page numbers and a
    column of punctuation read as 0, so they sort below the prose.
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    words = sum(1 for token in tokens if token[:1].isalpha())
    return round(words / len(tokens), 4)
