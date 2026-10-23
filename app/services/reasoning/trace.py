"""The steps the assistant takes to answer a question.

The trace says what was understood, what was looked for, what was found, why
the answer was taken from where it was, and where the answer came from.
"""

from __future__ import annotations

from typing import Any

from app.services.reasoning.entities import entities_of, shared_words

# The label of the step that says why the answer was taken where it was.
WHY_LABEL = "Why"


def understood_step(kind: str, entities: list[str], source: str) -> dict[str, Any]:
    """The step that says how the question was read."""
    words = ", ".join(entities) if entities else "none"
    return {
        "label": "Understand",
        "detail": f"Treated as a {kind} question ({source}). Key words: {words}.",
        "outcome": "planned",
    }


def search_step(recall: dict[str, Any]) -> dict[str, Any]:
    """The step that says what a memory store was asked."""
    hits = recall.get("hits") or []
    if not hits:
        return {
            "label": recall.get("store", "Memory"),
            "detail": "Nothing found in this memory.",
            "outcome": "empty",
        }
    closest = hits[0]
    distance = closest.get("score")
    closeness = f" ({float(distance):.2f})" if distance is not None else ""
    detail = f"{len(hits)} candidate(s). Closest: {closest['label']}{closeness}."
    if recall.get("hop") == 2:
        detail = f"Asked again with the words the first pass found. {detail}"
    return {
        "label": recall.get("store", "Memory"),
        "detail": detail,
        "outcome": "searched",
    }


def why_step(question: str, best: dict[str, Any] | None) -> dict[str, Any]:
    """The step that says why the answer was taken from where it was."""
    if best is None:
        return {
            "label": WHY_LABEL,
            "detail": "Nothing was close enough to the question, so no candidate was used.",
            "outcome": "nothing",
        }
    shared = shared_words(entities_of(question, limit=16), list(best.get("entities") or []))
    reason = (
        f"It says the same words as the question ({', '.join(shared[:4])})"
        if shared
        else "It was the closest candidate"
    )
    score = float(best.get("score") or 0.0)
    return {
        "label": WHY_LABEL,
        "detail": f"{reason}, and it was {score:.0%} close, so the answer was taken from it.",
        "outcome": "justified",
    }


def decide_step(source: str, matched_question: str | None) -> dict[str, Any]:
    """The step that says where the answer came from."""
    if matched_question:
        detail = (
            "The reply came from "
            f'{source.replace("_", " ")} using the taught question "{matched_question}".'
        )
    else:
        detail = f"The reply came from {source.replace('_', ' ')}."
    return {"label": "Decide", "detail": detail, "outcome": "selected"}
