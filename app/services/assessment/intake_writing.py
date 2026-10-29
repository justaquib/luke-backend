"""Writing and summarising items from Dolly, scored by a trainer, not by a model.

These are the only rubric items in the bank. The reference answer is the
dataset's own human response, kept so the trainer can compare, and a skill built
from them stays unassessed until a review has been recorded.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.services.assessment import intake, skills

CREATIVE_LEAD = "Write what the request asks for."
SUMMARY_LEAD = "Summarise the passage in your own words."
CREATIVE_CATEGORY = "creative_writing"
SUMMARY_CATEGORY = "summarization"

CREATIVE_SHAPES = (("poem", "poem"), ("haiku", "poem"), ("song", "poem"), ("story", "story"))

CREATIVE_CRITERIA = (
    "It is the form the request asked for, not a factual answer.",
    "It follows every constraint stated in the request.",
    "It has a beginning, a middle and an end, or a clear verse shape.",
    "It uses words a reader at this stage would know.",
    "It does not invent things a real person said or did.",
)
SUMMARY_CRITERIA = (
    "It keeps only points the passage states.",
    "It is shorter than the passage it summarises.",
    "It adds no fact that is not in the passage.",
    "It covers the main point rather than one detail.",
)


def _dolly_parts(record: Mapping[str, Any]) -> tuple[str, str, str]:
    instruction = str(record.get("instruction") or "").strip()
    context = str(record.get("context") or "").strip()
    response = str(record.get("response") or "").strip()
    return instruction, context, response


def _creative_shape(instruction: str) -> str:
    lowered = instruction.casefold()
    for needle, shape in CREATIVE_SHAPES:
        if needle in lowered:
            return shape
    return "scene or paragraph"


def dolly_creative_draft(record: Mapping[str, Any], index: int) -> intake.Draft | None:
    """A writing request with the human answer kept as a reference."""
    if str(record.get("category") or "").strip().casefold() != CREATIVE_CATEGORY:
        return None
    instruction, context, response = _dolly_parts(record)
    if not instruction or not response:
        return None
    parts = [CREATIVE_LEAD, instruction]
    if context:
        parts.append(f"Use only these details: {context}")
    return intake.Draft(
        skill="creative_writing",
        source="dolly_creative",
        record=str(index),
        topic=_creative_shape(instruction),
        stage=intake.stage_for(intake.word_count(instruction), 12, 25),
        prompt="\n\n".join(parts),
        kind=skills.RUBRIC,
        criteria=CREATIVE_CRITERIA,
        reference=response,
    )


def dolly_summary_draft(record: Mapping[str, Any], index: int) -> intake.Draft | None:
    """A passage to summarise, with the human summary kept as a reference."""
    if str(record.get("category") or "").strip().casefold() != SUMMARY_CATEGORY:
        return None
    instruction, context, response = _dolly_parts(record)
    if not context or not response:
        return None
    parts = [SUMMARY_LEAD]
    if instruction:
        parts.append(instruction)
    parts.append(f"Passage: {context}")
    return intake.Draft(
        skill="reading",
        source="dolly_summary",
        record=str(index),
        topic="summary of a passage",
        stage=intake.stage_for(intake.word_count(context), 80, 160),
        prompt="\n\n".join(parts),
        kind=skills.RUBRIC,
        criteria=SUMMARY_CRITERIA,
        reference=response,
    )
