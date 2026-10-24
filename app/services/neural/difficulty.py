"""How hard a sentence is, so the model can learn the easy ones first.

The model is trained in three stages: the simple sentences first, then the
compound ones, and the complex ones last. Each stage builds on the one
before it, the way a child learns a sentence before it learns a paragraph.
"""

from __future__ import annotations

from app.services.neural.tokens import tokenize

# The three stages of the training.
SIMPLE = 0
COMPOUND = 1
COMPLEX = 2

# The words that join two parts of a sentence of the same weight.
_COORDINATORS = frozenset({"and", "but", "or", "so", "yet"})

# The words that put one part of a sentence inside another.
_SUBORDINATORS = frozenset(
    {
        "because",
        "although",
        "though",
        "while",
        "when",
        "whenever",
        "where",
        "wherever",
        "which",
        "who",
        "whom",
        "whose",
        "that",
        "if",
        "unless",
        "until",
        "since",
        "as",
    }
)


def level_of(text: str) -> int:
    """How hard the sentence is: simple, compound or complex."""
    words = [word.lower() for word in tokenize(text)]
    if any(word in _SUBORDINATORS for word in words):
        return COMPLEX
    if any(word in _COORDINATORS for word in words):
        return COMPOUND
    return SIMPLE


def levels(sentences: list[str]) -> list[int]:
    """The stage every sentence belongs to."""
    return [level_of(sentence) for sentence in sentences]
