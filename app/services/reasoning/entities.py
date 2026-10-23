"""The words of a question, and the words of a node.

The entities of a question are the words that carry its meaning. Every node
the reasoning visits carries its own entities, so a node is joined to the
question by the words the two of them share, and not by the words of the
question alone.
"""

from __future__ import annotations

import re
from typing import Any

# The words that say nothing about what is being asked.
STOP_WORDS = frozenset(
    {
        "what",
        "who",
        "where",
        "when",
        "which",
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
        "in",
        "on",
        "at",
        "to",
        "with",
        "and",
        "or",
        "but",
        "if",
        "then",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "my",
        "your",
        "his",
        "her",
        "our",
        "their",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "here",
        "there",
        "from",
        "about",
        "how",
        "why",
        "s",
        "t",
    }
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")

# How many words of a text are kept as its entities.
MAX_ENTITIES = 8


def words_of(text: str) -> list[str]:
    """The words of a text that carry its meaning."""
    return [
        word
        for word in _WORD_RE.findall((text or "").lower())
        if word not in STOP_WORDS and len(word) > 1
    ]


def entities_of(text: str, limit: int = MAX_ENTITIES) -> list[str]:
    """The meaningful words of a text, each of them once."""
    result: list[str] = []
    seen: set[str] = set()
    for word in words_of(text):
        if word in seen:
            continue
        seen.add(word)
        result.append(word)
        if len(result) >= limit:
            break
    return result


def shared_words(left: list[str], right: list[str]) -> list[str]:
    """The words two texts have in common."""
    return sorted(set(left) & set(right))


def relate(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], question: str) -> None:
    """Join every node to the question by the words they share.

    A node carries the words of its own label, so the edge says what the node
    and the question have in common, and not what the question said in general.
    """
    question_words = entities_of(question, limit=MAX_ENTITIES * 2)
    for node in nodes:
        if node.get("type") == "question":
            continue
        shared = shared_words(question_words, list(node.get("entities") or []))
        if shared:
            edges.append(
                {
                    "source": node["id"],
                    "target": "question",
                    "label": ", ".join(shared[:3]),
                }
            )
