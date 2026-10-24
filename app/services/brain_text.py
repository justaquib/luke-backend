"""The words the brain measures and labels with.

The rung tests compare an answer with the one that was taught, and the
emotion head labels the words of a message. Both read text the same way, so
the reading lives here rather than in each of them.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*|[.,!?]")


def tokenize(text: str) -> list[str]:
    """The words and marks of a text, in lower case."""
    return [token.lower() for token in _WORD_RE.findall(text or "")]


def sentences(text: str) -> list[str]:
    """The sentences of a text."""
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text or "") if part.strip()]
