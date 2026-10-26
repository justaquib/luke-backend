"""The learner preview: the same taught answer the rest of the app gives.

The preview used to keep its own answer order, which is how the same question
could be answered from one lesson here and another there. It now reads the
shared order in `answering.py`, so the trainer preview, the chat, the grade
test and the coaching chat all recall the same answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.services import reasoning
from app.services.connections import Connection
from app.services.language import answering, reporting

NOT_TAUGHT_DETAIL = (
    "No lesson, word card, dictionary entry or sentence example covers the question."
)


@dataclass(frozen=True)
class PreviewResult:
    """The preview the page reads, and the dots connected to reach it."""

    payload: dict[str, Any]
    connections: tuple[Connection, ...]
    verified: bool
    verification: str


def _taught_preview(question: str, tenant_id: int | None) -> PreviewResult:
    taught = answering.answer_from_curriculum(question, tenant_id)
    if taught is None:
        return PreviewResult(
            payload={"answer": answering.NOT_LEARNED, "source": "not_learned"},
            connections=(Connection("taught", "miss", NOT_TAUGHT_DETAIL),),
            verified=False,
            verification="No taught material had an answer.",
        )
    evidence = taught.evidence or (
        Connection(
            taught.source,
            "match",
            f"{taught.detail} answers this.",
            taught.matched_question,
        ),
    )
    return PreviewResult(
        payload=taught.as_dict(),
        connections=evidence,
        verified=True,
        verification="The answer came from the matched material.",
    )


def _reasoning(question: str, tenant_id: int | None, payload: dict[str, Any]) -> dict[str, Any]:
    explained = reasoning.explain(
        question,
        tenant_id,
        payload.get("answer", ""),
        payload.get("source", "not_learned"),
        payload.get("matched_question"),
    )
    return {
        "steps": [
            {"label": step.get("label", ""), "detail": step.get("detail", "")}
            for step in explained.get("steps", [])
        ],
        "confidence": explained.get("confidence", 0),
    }


def preview_payload(
    question: str, tenant_id: int | None, db: Session, skip_trainer: bool = False
) -> PreviewResult:
    """The learner preview, with the sources the backend actually read."""
    candidates = [] if skip_trainer else reporting.candidate_answers(question, db, tenant_id)
    exact = reporting.exact_candidate(question, candidates)
    if exact is None:
        result = _taught_preview(question, tenant_id)
    else:
        matched = str(exact["question"])
        detail = "A trainer-approved answer matched the question exactly."
        result = PreviewResult(
            payload={
                "answer": str(exact["answer"]),
                "source": "faq",
                "matched_question": matched,
            },
            connections=(Connection("faq", "match", detail, matched),),
            verified=True,
            verification=detail,
        )
    payload = dict(result.payload)
    payload["candidates"] = candidates
    payload["needs_clarification"] = bool(candidates) and exact is None
    payload["reasoning"] = _reasoning(question, tenant_id, payload)
    connections = list(result.connections)
    if candidates and exact is None:
        connections.append(
            Connection(
                "faq",
                "miss",
                f"{len(candidates)} trainer-approved answer(s) are close; none matched exactly.",
            )
        )
    return PreviewResult(payload, tuple(connections), result.verified, result.verification)


def answer_from_curriculum(question: str, tenant_id: int | None) -> str | None:
    """Recall the taught answer to a question, or None when it is not known."""
    detail = answer_from_curriculum_detail(question, tenant_id)
    return detail["answer"] if detail else None


def answer_from_curriculum_detail(
    question: str, tenant_id: int | None, include_lessons: bool = True
) -> dict[str, str] | None:
    """The taught answer plus the question it was taught as.

    ``include_lessons`` is off for callers that only want the facts, words and
    sentences, not the taught question and answer cards.
    """
    result = answering.answer_from_curriculum(question, tenant_id, include_lessons=include_lessons)
    if result is None:
        return None
    return {
        "answer": result.answer,
        "matched_question": result.matched_question,
        "source": result.source,
    }


def preview_answer(question: str, tenant_id: int | None) -> dict[str, str]:
    """Answer a Trainer preview from the learner's own taught material."""
    result = answering.answer_from_curriculum(question, tenant_id)
    if result is None:
        return {"answer": answering.NOT_LEARNED, "source": "not_learned"}
    return result.as_dict()
