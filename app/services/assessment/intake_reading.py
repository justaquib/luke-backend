"""Reading items: extractive spans from SQuAD and yes/no passages from BoolQ."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.services.assessment import intake, skills

SPAN_PROMPT = "Read the passage and answer with a phrase taken from it."
YES_NO_PROMPT = "Read the passage and answer the question with the letter for yes or no."
YES_NO_CHOICES = ("Yes", "No")


def _squad_texts(record: Mapping[str, Any]) -> tuple[str, ...]:
    answers = record.get("answers")
    texts = answers.get("text") if isinstance(answers, Mapping) else None
    cleaned = [str(text).strip() for text in texts or () if str(text).strip()]
    return tuple(dict.fromkeys(cleaned))


def squad_draft(record: Mapping[str, Any], index: int) -> intake.Draft | None:
    """A question whose answer is a phrase inside the passage it came from."""
    context = str(record.get("context") or "").strip()
    question = str(record.get("question") or "").strip()
    texts = _squad_texts(record)
    if not context or not question or not texts:
        return None
    canonical = min(texts, key=len)
    return intake.Draft(
        skill="reading",
        source="squad",
        record=str(record.get("id") or index),
        topic=intake.question_lead(question),
        stage=intake.stage_for(intake.word_count(canonical), 2, 6),
        prompt=f"{SPAN_PROMPT}\n\nPassage: {context}\n\nQuestion: {question}",
        kind=skills.SPAN,
        answer=canonical,
        acceptable=texts,
        extra={"title": str(record.get("title") or ""), "accepted": len(texts)},
    )


def boolq_draft(record: Mapping[str, Any], index: int) -> intake.Draft | None:
    """A closed question over a Wikipedia passage, scored as a letter."""
    passage = str(record.get("passage") or "").strip()
    question = str(record.get("question") or "").strip()
    answer = record.get("answer")
    if not passage or not question or not isinstance(answer, bool):
        return None
    record_id = str(index)
    flipped = intake.coin(f"order:reading:boolq:{record_id}")
    choices = (YES_NO_CHOICES[1], YES_NO_CHOICES[0]) if flipped else YES_NO_CHOICES
    correct = YES_NO_CHOICES[0] if answer is True else YES_NO_CHOICES[1]
    prompt = "\n\n".join(
        (
            YES_NO_PROMPT,
            f"Passage: {passage}",
            f"Question: {question}",
            intake.choice_lines(choices),
        )
    )
    return intake.Draft(
        skill="reading",
        source="boolq",
        record=record_id,
        topic=intake.question_lead(question),
        stage=intake.stage_for(intake.word_count(passage), 80, 160),
        prompt=prompt,
        kind=skills.OPTION,
        answer=intake.CHOICE_LABELS[choices.index(correct)],
        choices=choices,
    )
