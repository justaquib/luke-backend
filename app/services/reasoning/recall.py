"""Pulling the candidates for a question out of every memory store.

The first pass asks the question as it was asked. The second pass asks again
with the words the first pass found, so a question that names two things is
answered by the lesson about the first one and the lesson about the second.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.services import vector_store
from app.services.reasoning.entities import entities_of

# How many candidates each store is asked for.
CANDIDATES = 3

# How many candidates the second pass asks for.
FOLLOW_UP_CANDIDATES = 2

# How many words of the first pass are followed up on.
FOLLOW_UP_WORDS = 3


def _lessons(question: str, tenant_id: int | None, count: int) -> list[dict[str, Any]]:
    """The lessons that are closest to the question."""
    hits = vector_store.search_lessons(question, n_results=count, tenant_id=tenant_id)
    return [
        {
            "id": f"lesson:{hit.get('lesson_id')}",
            "label": hit["question"],
            "answer": hit.get("answer", ""),
            "score": hit.get("distance"),
            "type": "lesson",
        }
        for hit in hits
    ]


def _answers(question: str, tenant_id: int | None, count: int) -> list[dict[str, Any]]:
    """The trained answers that are closest to the question."""
    hits = vector_store.search_faq(question, n_results=count, tenant_id=tenant_id)
    return [
        {
            "id": f"answer:{hit.get('faq_id')}",
            "label": hit["question"],
            "answer": hit.get("answer", ""),
            "score": hit.get("distance"),
            "type": "trained_answer",
        }
        for hit in hits
    ]


def _intents(question: str, tenant_id: int | None, count: int) -> list[dict[str, Any]]:
    """The saved intents that are closest to the question."""
    hits = vector_store.search_intents(question, n_results=count, tenant_id=tenant_id)
    return [
        {
            "id": f"intent:{index}",
            "label": hit.get("name", ""),
            "answer": hit.get("response_template", ""),
            "score": hit.get("distance"),
            "type": "intent",
        }
        for index, hit in enumerate(hits)
    ]


def _documents(question: str, tenant_id: int | None, count: int) -> list[dict[str, Any]]:
    """The passages of the documents that are closest to the question."""
    hits = vector_store.query(question, n_results=count, tenant_id=tenant_id)
    return [
        {
            "id": f"document:{hit.get('metadata', {}).get('source', index)}:{index}",
            "label": _document_label(hit.get("metadata", {}), index),
            "answer": hit.get("text", "")[:160],
            "score": hit.get("distance"),
            "type": "document",
        }
        for index, hit in enumerate(hits)
    ]


def _document_label(metadata: dict, index: int) -> str:
    """The name of the passage a hit came from."""
    source = metadata.get("source") or "Passage"
    page = metadata.get("page_number")
    return f"{source} · page {page}" if page else f"{source}"


# The memory stores, in the order they are read.
STORES: tuple[tuple[str, str, Callable[[str, int | None, int], list[dict[str, Any]]]], ...] = (
    ("Learner lessons", "language_model", _lessons),
    ("Trained answers", "faq", _answers),
    ("Saved intents", "intent", _intents),
    ("Documents", "docs", _documents),
)


def first_pass(question: str, tenant_id: int | None) -> list[dict[str, Any]]:
    """The closest candidates from every memory store."""
    recalls: list[dict[str, Any]] = []
    for store, source, search in STORES:
        recalls.append(
            {
                "store": store,
                "source": source,
                "hits": search(question, tenant_id, CANDIDATES),
                "hop": 1,
            }
        )
    return recalls


def _known_ids(recalls: list[dict[str, Any]]) -> set[str]:
    """The candidates that were already found."""
    return {hit["id"] for recall in recalls for hit in recall["hits"]}


def _found_words(question: str, recalls: list[dict[str, Any]]) -> list[str]:
    """The new words the first pass found in its closest candidates.

    The answer is read as well as the question of the candidate: the answer
    names the thing the question is about ("Saturn"), and that is the word
    the second pass asks about.
    """
    asked = set(entities_of(question, limit=16))
    found: list[str] = []
    for recall in recalls:
        for hit in recall["hits"][:1]:
            for text in (hit.get("label") or "", hit.get("answer") or ""):
                for word in entities_of(text, limit=6):
                    if word not in asked and word not in found:
                        found.append(word)
    return found[:FOLLOW_UP_WORDS]


def second_pass(
    question: str, tenant_id: int | None, recalls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Ask the stores again with the words the first pass found.

    A question that names two things ("which planet has the most moons and is
    famous for its rings") is answered by a lesson about the first thing and a
    lesson about the second. The first pass finds the first one, and this pass
    asks about the words it found, so the second one is found too.
    """
    words = _found_words(question, recalls)
    if not words:
        return []
    query = f"{question} {' '.join(words)}"
    known = _known_ids(recalls)
    found: list[dict[str, Any]] = []
    for store, source, search in STORES:
        hits = [
            hit for hit in search(query, tenant_id, FOLLOW_UP_CANDIDATES) if hit["id"] not in known
        ]
        if hits:
            found.append(
                {
                    "store": store,
                    "source": source,
                    "hits": hits[:1],
                    "hop": 2,
                    "query": query,
                }
            )
    return found


def recall(question: str, tenant_id: int | None) -> list[dict[str, Any]]:
    """Every candidate for the question, from both passes."""
    recalls = first_pass(question, tenant_id)
    recalls.extend(second_pass(question, tenant_id, recalls))
    return recalls
