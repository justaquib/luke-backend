"""The subject of a lesson, and whether that subject names one lesson.

A lesson is a question and an answer, but the subject of the lesson is what
lets it be asked for in words the trainer never used: the lesson "What do we
use to hear?" is about ears, and "Tell me everything about ears" asks for it.

A subject has to name one lesson and not several, or a request about it
cannot pick the right answer. The words that many lessons share are left out
for that reason.
"""

from __future__ import annotations

import re
from typing import Any

# The words of an answer that are never its subject.
_ANSWER_NOISE = frozenset(
    {
        "a",
        "an",
        "the",
        "our",
        "my",
        "his",
        "her",
        "their",
        "its",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "can",
        "will",
        "does",
        "do",
        "says",
        "say",
        "lives",
        "live",
        "gives",
        "give",
        "sounds",
        "sound",
        "means",
        "mean",
        "called",
        "known",
        "found",
        "made",
        "used",
        "jumps",
        "jump",
        "one",
        "two",
    }
)

# The words that carry no meaning of their own.
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "to",
        "of",
        "in",
        "on",
        "up",
        "down",
        "go",
        "away",
        "again",
        "another",
        "day",
        "my",
        "it",
        "its",
        "i",
        "you",
        "we",
        "they",
        "he",
        "she",
        "him",
        "her",
        "them",
        "us",
        "his",
        "their",
        "our",
        "your",
        "me",
        "with",
        "for",
        "from",
        "at",
        "by",
        "as",
        "so",
        "not",
        "no",
        "yes",
        "if",
        "that",
        "this",
        "sir",
        "man",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "shall",
        "can",
        "could",
        "would",
        "should",
        "little",
        "big",
        "old",
        "new",
        "come",
        "let",
        "get",
        "put",
        "ran",
        "run",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "be",
        "been",
        "being",
        "there",
        "here",
        "about",
        "please",
        "tell",
        "give",
        "name",
        "write",
        "make",
        "say",
    }
)

# The words of a lesson title that say what kind of lesson it is.
LEADING_PREFIXES = (
    "poem:",
    "gk:",
    "science:",
    "grammar:",
    "math:",
    "reading:",
    "tiny:",
    "riddle:",
    "story:",
    "joke:",
    "general knowledge:",
)

# How many lessons may share a word before it stops naming a single one.
MAX_LESSONS_PER_WORD = 4

# How many topics one lesson is asked by.
MAX_TOPICS_PER_LESSON = 2

_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")


def words_of(text: str) -> list[str]:
    """The words of a text, in lower case."""
    return [match.group(0).lower() for match in _WORD_RE.finditer(text or "")]


def topic_words(text: str) -> list[str]:
    """The words of a text that carry a meaning of their own."""
    seen: list[str] = []
    for word in words_of(text):
        if len(word) <= 2 or word in STOP_WORDS:
            continue
        if word not in seen:
            seen.append(word)
    return seen


def lesson_title(question: str) -> str:
    """The subject of a lesson, without the kind in front of it."""
    text = (question or "").strip()
    lowered = text.lower()
    for prefix in LEADING_PREFIXES:
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def answer_subject(answer: str) -> str:
    """The thing the answer names, when the answer names a thing.

    "Ears!" names ears. "A kite is a light frame with a tail." defines a kite
    rather than naming one, so it names nothing.
    """
    text = (answer or "").strip()
    if not text or len(text.split()) > 8:
        return ""
    if re.match(r"^\w+.*?\b(?:is|are|means|refers to)\b\s+(?:a|an|the|one|any|some)\b", text):
        return ""
    text = text.split("(")[0].split("/")[0].split(",")[0].strip()
    text = re.sub(r"^(?:the|a|an|our|my|his|her|their|its)\s+", "", text, flags=re.IGNORECASE)
    text = text.strip(" .!?")
    kept: list[str] = []
    for word in text.split():
        if word.lower() in _ANSWER_NOISE:
            break
        kept.append(word.lower())
        if len(kept) == 3:
            break
    return " ".join(kept)


def card_key(card: dict[str, Any]) -> str:
    """The key a lesson is filed under in the topic index."""
    return str(card.get("id") or card.get("question") or "")


def candidate_topics(card: dict[str, Any]) -> list[str]:
    """The subjects a lesson could be asked by, best first.

    A subject with more than one word names its lesson better than a single
    word does, so it comes first: "polar bear" is one lesson, while "polar"
    could be many.
    """
    candidates: list[str] = []
    subject = answer_subject(str(card.get("answer") or ""))
    if subject:
        candidates.append(subject)
    for word in topic_words(lesson_title(str(card.get("question") or ""))):
        if word not in candidates:
            candidates.append(word)
    return sorted(candidates, key=lambda topic: (len(topic.split()) == 1, topic))


def asks_about(card: dict[str, Any], topic: str) -> tuple[int, int]:
    """How directly the lesson asks about the subject."""
    asked = set(topic_words(str(card.get("question") or "")))
    shared = len(set(topic.split()) & asked)
    return (shared, -len(str(card.get("answer") or "")))


def topic_index(cards: list[dict[str, Any]]) -> dict[str, list[str]]:
    """The subjects each lesson can be asked by, and only the ones that name it.

    A word that many lessons share does not name any one of them, so it is
    left out. A request about such a word cannot pick a lesson, and the
    assistant says so rather than choosing one at random.

    A subject belongs to one lesson only. When two lessons claim the same
    subject, the one that asks about it the most directly keeps it, so the
    model is never taught two different answers to the same request.
    """
    counts: dict[str, int] = {}
    for card in cards:
        for word in topic_words(f"{card.get('question', '')} {card.get('answer', '')}"):
            counts[word] = counts.get(word, 0) + 1
    claims: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        for topic in candidate_topics(card):
            if all(counts.get(word, 0) <= MAX_LESSONS_PER_WORD for word in topic.split()):
                claims.setdefault(topic, []).append(card)
    index: dict[str, list[str]] = {}
    for topic, holders in claims.items():
        owner = max(holders, key=lambda card: asks_about(card, topic))
        topics = index.setdefault(card_key(owner), [])
        if topic not in topics:
            topics.append(topic)
    for key, topics in index.items():
        index[key] = topics[:MAX_TOPICS_PER_LESSON]
    return index
