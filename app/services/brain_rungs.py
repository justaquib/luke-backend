"""The question tests each rung of the ladder asks.

Each rung asks for the same taught material in a harder way, so teaching
more, and teaching it in more ways, is what moves the level up. The probes
are built from what the tenant has taught; none of them is written by hand
for a particular lesson.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from app.services.brain_paths import tenant_key as _tenant_key
from app.services.brain_text import tokenize

LOGGER = logging.getLogger(__name__)

GENERALITY_PROBES = [
    "Who wrote the play Romeo and Juliet?",
    "Write a short poem about the rain.",
    "How many continents are there?",
    "What is the boiling point of water?",
    # Most AI models cannot answer this one, so it is a fair test of what this
    # model really knows for itself.
    "What are the rudiment?",
]

PROBE_LIMIT = 6
CORRECTION_LIMIT = 5
RUNG_CACHE_TTL = 1800.0

# The same probe questions are asked every time, so their answers are kept.
RUNG_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
PROBE_CACHE: dict[str, list[dict[str, Any]]] = {}

RUNG_TESTS = {
    "0.0.1": "Can it find the answer when asked exactly as it was taught?",
    "0.0.2": "Can it find the answer when the question is worded differently?",
    "0.0.3": "Can it answer about the topic without the exact question?",
    "0.0.4": "Can it write new sentences in the style you taught?",
    "0.0.5": "Can it explain an answer when it is asked to?",
    "0.0.6": "Has it learned from a correction and kept it?",
    "0.0.7": "Has it passed the final exam of a grade?",
    "0.0.8": "Does it say it does not know instead of guessing?",
    "0.0.9": "Does it hold one whole subject in depth?",
    "0.1.0": "Can it answer a question it was never taught, in its own words?",
}


def cache_key(tenant_id: int | None) -> str:
    """The name a tenant's rung and probe results are kept under."""
    return f"{_tenant_key(tenant_id)}"


def reframe(question: str) -> str:
    """Ask the same question in a different way."""
    text = question.strip().rstrip("?.! ")
    if not text:
        return question
    return f"Could you tell me {text[0].lower() + text[1:]}?"


def swap_synonym(question: str) -> str | None:
    """Ask the question with a word replaced by a synonym."""
    for word in re.findall(r"[A-Za-z]{4,}", question):
        try:
            from app.services.dictionary import search_expansions

            synonyms = [item for item in search_expansions(word) if item != word.lower()]
        except (ImportError, OSError, ValueError) as error:
            LOGGER.warning("The dictionary could not expand %r: %s", word, error)
            synonyms = []
        if not synonyms:
            continue
        replacement = synonyms[0]
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        swapped = pattern.sub(replacement, question, count=1)
        if swapped.lower() != question.lower():
            return swapped
    return None


def answer_matches(produced: str, expected: str) -> bool:
    """Whether the answer the model gave says the same thing."""
    if not produced or not expected:
        return False
    expected_tokens = set(tokenize(expected))
    produced_tokens = set(tokenize(produced))
    if not expected_tokens:
        return False
    overlap = len(expected_tokens & produced_tokens)
    return overlap / len(expected_tokens) >= 0.6


def rung_probes(tenant_id: int | None) -> dict[str, list[dict[str, Any]]]:
    """The question tests for each rung, built from what the tenant has taught."""
    from app.services import language_logic

    try:
        cards = language_logic.all_lesson_cards(tenant_id)
    except (OSError, ValueError):
        cards = []
    taught = [
        {"question": card["question"], "answer": card["answer"]}
        for card in cards
        if card.get("question") and card.get("answer")
    ][:PROBE_LIMIT]
    exact = [
        {"question": item["question"], "answer": item["answer"], "form": "asked as taught"}
        for item in taught
    ]
    reworded = [
        {
            "question": reframe(item["question"]),
            "answer": item["answer"],
            "form": "asked in other words",
        }
        for item in taught
    ]
    swapped: list[dict[str, Any]] = []
    for item in taught:
        alternative = swap_synonym(item["question"])
        if alternative:
            swapped.append(
                {
                    "question": alternative,
                    "answer": item["answer"],
                    "form": "asked with another word",
                }
            )
    explained = [
        {
            "question": f"Explain: {item['question']}",
            "answer": item["answer"],
            "form": "asked for an explanation",
        }
        for item in taught
    ]
    return {
        "0.0.1": exact,
        "0.0.2": reworded,
        "0.0.3": swapped,
        "0.0.5": explained,
    }


def subject_coverage(tenant_id: int | None) -> dict[str, int]:
    """How many questions the tenant holds for each subject."""
    from app.services import language_logic

    try:
        cards = language_logic.all_lesson_cards(tenant_id)
    except (OSError, ValueError) as error:
        LOGGER.warning("The subjects could not be read: %s", error)
        return {}
    counts: dict[str, int] = {}
    for card in cards:
        subject = (card.get("subject") or "General").strip() or "General"
        counts[subject] = counts.get(subject, 0) + 1
    return counts


def cached_rungs(tenant_id: int | None) -> list[dict[str, Any]] | None:
    """The last rung results, while they are still fresh."""
    cached = RUNG_CACHE.get(cache_key(tenant_id))
    if cached and (time.time() - cached[0]) < RUNG_CACHE_TTL:
        return cached[1]
    return None
