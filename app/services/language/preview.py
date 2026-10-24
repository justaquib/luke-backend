"""The learner preview: the same taught answer the rest of the app gives.

The preview used to keep its own answer order, which is how the same question
could be answered from one lesson here and another there. It now reads the
shared order in `answering.py`, so the trainer preview, the chat, the grade
test and the coaching chat all recall the same answer.
"""

from __future__ import annotations

from app.services.language import answering


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
