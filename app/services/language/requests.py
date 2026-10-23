"""The brain answering a request instead of looking one up.

Retrieval answers a question that was asked the way it was taught. A request
like "write a short poem about the rain" is not that: the words of the request
are mostly the words of the request, not the words of the lesson.

So the answer comes in two steps:

1. The request is read as a *shape* — a poem about a topic, a list of a topic,
   some facts about a topic — and the topic is taken out of it.
2. The lesson about that topic is found, and the trained model writes the
   answer from it.

The shapes here are the same ones the model is trained on, so a request is
answered the way the model was taught to answer it. The material it is given
is the lesson the trainer taught. The result is grounded: every word of the
answer comes from the tenant's own material, and a request about something
never taught is refused rather than invented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.services.language import instructions as instruction_service
from app.services.language.shapes import REQUEST_SHAPES
from app.services.language.topics import topic_words
from app.services.transformer import quality as quality_service

# The words that are never the topic of a request.
_TOPIC_NOISE = frozenset(
    {
        "please",
        "can",
        "could",
        "would",
        "me",
        "a",
        "an",
        "the",
        "short",
        "little",
        "new",
        "quick",
        "really",
        "very",
        "just",
        "some",
        "write",
        "compose",
        "make",
        "create",
        "give",
        "recite",
        "say",
        "tell",
        "name",
        "list",
        "about",
        "for",
        "with",
        "in",
        "on",
        "of",
        "to",
        "and",
        "or",
        "my",
        "our",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "i",
        "it",
        "that",
        "this",
        "know",
        "information",
        "info",
        "poem",
        "rhyme",
        "song",
        "verse",
        "story",
        "letter",
        "sentence",
    }
)

_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")


@dataclass(frozen=True)
class Request:
    """A request the model can be asked to answer."""

    text: str
    topic: str
    kind: str
    words: tuple[str, ...]


def _words(text: str) -> list[str]:
    """The words of a text, in lower case."""
    return [match.group(0).lower() for match in _WORD_RE.finditer(text or "")]


def _kind_of(shape: str, match: re.Match[str]) -> str:
    """What kind of writing the request asks for."""
    if shape != "art":
        return shape
    art = (match.group("art") or "").lower()
    return "poem" if art in {"poem", "rhyme", "song", "verse"} else art


def _topic_words(text: str) -> tuple[str, ...]:
    """The words the request is about."""
    words = [word for word in _words(text) if word not in _TOPIC_NOISE and len(word) > 2]
    seen: list[str] = []
    for word in words:
        if word not in seen:
            seen.append(word)
    return tuple(seen)


def parse_request(question: str) -> Request | None:
    """Read a request as a topic to answer, or None when it is not one."""
    text = (question or "").strip()
    if not text:
        return None
    for shape, kind in REQUEST_SHAPES:
        match = shape.search(text)
        if match is None:
            continue
        topic = _topic_words(match.group("topic"))
        if not topic:
            continue
        return Request(
            text=text,
            topic=" ".join(topic),
            kind=_kind_of(kind, match),
            words=topic,
        )
    return None


def _overlaps(request: Request, card: dict[str, Any]) -> int:
    """How many of the words of the request the lesson is about."""
    words = set(topic_words(str(card.get("question") or "")))
    words.update(topic_words(str(card.get("answer") or "")))
    return len(set(request.words) & words)


def find_material(request: Request, cards: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The lesson the request is about, and nothing else.

    The lessons are looked up through the same topic index the model was
    trained on, but the word counts are taken over the lessons of the kind
    the request asks for: a request for a poem already chose its kind, so a
    word that eight fact lessons also say ("star") still names the one poem
    that speaks it. A word that many lessons of the same kind share names
    none of them, and the request is refused rather than answered at random.
    """
    pool = [card for card in cards if instruction_service.lesson_kind(card) == request.kind]
    lookup = _topic_lookup(pool)
    candidates: list[dict[str, Any]] = []
    for word in request.words:
        for card in lookup.get(word, []):
            if card not in candidates:
                candidates.append(card)
    if not candidates:
        return None
    return max(candidates, key=lambda card: _asked_about(request, card))


def _asked_about(request: Request, card: dict[str, Any]) -> tuple[int, int]:
    """How directly the lesson asks about the topic of the request."""
    asked = set(topic_words(str(card.get("question") or "")))
    shared = len(set(request.words) & asked)
    return (shared, -len(str(card.get("answer") or "")))


def _topic_lookup(cards: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """The lessons that answer for each topic word."""
    index = instruction_service.topic_index(cards)
    lookup: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        key = str(card.get("id") or card.get("question") or "")
        for topic in index.get(key, []):
            for word in topic.split():
                lookup.setdefault(word, []).append(card)
    return lookup


def context_of(card: dict[str, Any]) -> str:
    """The material the model is given to answer from."""
    answer = str(card.get("answer") or "").strip()
    question = str(card.get("question") or "").strip()
    return f"{question}. {answer}" if question else answer


def _grounded(answer: str, context: str) -> bool:
    """Whether every word of the answer comes from the material.

    This is what keeps the model honest: an answer that uses a word the
    trainer never taught is refused and the taught answer is used instead.
    """
    for word in _words(answer):
        if len(word) <= 2:
            continue
        if word not in context.lower():
            return False
    return True


def _with_uncertainty(answer: str, writer: Any) -> str:
    """The answer, with a word of it the model was not sure about.

    The model says what it is not sure of, so the reader knows that this word
    is a guess and not something the trainer taught.
    """
    from app.services.transformer import critique as critique_service

    critique = getattr(writer, "last_critique", None)
    if critique is None:
        return answer
    if critique.least_chance >= critique_service.MIN_WORD_CONFIDENCE:
        return answer
    return f"{answer} {critique.unsure_about}"


def answer_request(
    question: str,
    tenant_id: int | None,
    cards: list[dict[str, Any]],
    writer: Any = None,
) -> dict[str, Any] | None:
    """Answer a request from the material the tenant taught.

    ``None`` means the request is not one the model can answer: it is not a
    request for something to be written, or nothing was taught about its
    topic.
    """
    request = parse_request(question)
    if request is None:
        return None
    card = find_material(request, cards)
    if card is None:
        return None
    context = context_of(card)
    written = writer.respond(request.text, context=context) if writer is not None else None
    taught = str(card.get("answer") or "").strip()
    if written and _grounded(written, context) and not quality_service.repeats_itself(written):
        return {
            "answer": _with_uncertainty(written, writer),
            "source": "written",
            "topic": request.topic,
            "kind": request.kind,
            "confidence": getattr(writer, "last_confidence", 0.0),
            "uncertain_words": list(getattr(writer, "last_uncertain_words", []) or []),
        }
    if taught:
        return {
            "answer": taught,
            "source": "taught",
            "topic": request.topic,
            "kind": request.kind,
            "confidence": 0.0,
            "uncertain_words": [],
        }
    return None
