"""Maths from GSM8K, science and general knowledge from ARC, code from CRUXEval."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from typing import Any

from app.services.assessment import intake, skills

MATH_PROMPT = "Solve the problem and give the final number."
ARC_PROMPT = "Answer the question with the letter of the correct option."
CODE_PROMPT = "What value does the call return? Give the exact value, not an explanation."
ANSWER_LINE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
CALC_STEPS = re.compile(r"<<")


def _final_number(text: str) -> int | float | None:
    """The number the dataset states after its answer marker."""
    match = ANSWER_LINE.search(text)
    if not match:
        return None
    raw = match.group(1).replace(",", "")
    value = float(raw)
    return int(value) if value.is_integer() else value


def gsm8k_draft(record: Mapping[str, Any], index: int) -> intake.Draft | None:
    """A word problem whose answer is the number the dataset states after ####."""
    question = str(record.get("question") or "").strip()
    working = str(record.get("answer") or "").strip()
    value = _final_number(working)
    if not question or value is None:
        return None
    steps = len(CALC_STEPS.findall(working))
    return intake.Draft(
        skill="mathematics",
        source="gsm8k",
        record=str(index),
        topic="word problem",
        stage=intake.stage_for(steps, 1, 2),
        prompt=f"{MATH_PROMPT}\n\n{question}",
        kind=skills.NUMBER,
        answer=value,
        extra={"steps": steps},
    )


def _arc_position(record: Mapping[str, Any], labels: list[str], count: int) -> int | None:
    key = str(record.get("answerKey") or "").strip().upper()
    if key in labels:
        return labels.index(key)
    if key.isdigit() and 1 <= int(key) <= count:
        return int(key) - 1
    return None


def _arc_draft(
    skill: str, source: str, record: Mapping[str, Any], index: int
) -> intake.Draft | None:
    question = str(record.get("question") or "").strip()
    choices = record.get("choices")
    texts = [str(text).strip() for text in (choices or {}).get("text", ()) if str(text).strip()]
    labels = [str(label).strip().upper() for label in (choices or {}).get("label", ())]
    position = _arc_position(record, labels, len(texts))
    if not question or position is None or len(texts) < 2:
        return None
    return intake.Draft(
        skill=skill,
        source=source,
        record=str(record.get("id") or index),
        topic=intake.question_lead(question),
        stage=intake.stage_for(intake.word_count(question + " " + " ".join(texts)), 25, 45),
        prompt="\n\n".join(
            (ARC_PROMPT, question, intake.choice_lines(texts)),
        ),
        kind=skills.OPTION,
        answer=intake.CHOICE_LABELS[position],
        choices=tuple(texts),
        extra={"dataset_labels": labels},
    )


def arc_easy_draft(record: Mapping[str, Any], index: int) -> intake.Draft | None:
    """An ARC-Easy question, held as the general-knowledge profile."""
    return _arc_draft("general_knowledge", "arc_easy", record, index)


def arc_challenge_draft(record: Mapping[str, Any], index: int) -> intake.Draft | None:
    """An ARC-Challenge question, held as the science profile."""
    return _arc_draft("science", "arc_challenge", record, index)


def _parse(code: str) -> ast.Module | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _program_topic(tree: ast.Module) -> str:
    for node in ast.walk(tree):
        if isinstance(node, ast.For | ast.While):
            return "iteration"
        if isinstance(node, ast.If):
            return "branching"
        if isinstance(node, ast.ListComp | ast.GeneratorExp | ast.SetComp | ast.DictComp):
            return "comprehension"
    return "sequence of statements"


def cruxeval_draft(record: Mapping[str, Any], index: int) -> intake.Draft | None:
    """A short program, an input and the value it returns, read without running it."""
    code = str(record.get("code") or "").strip()
    call_input = str(record.get("input") or "").strip()
    output = str(record.get("output") or "").strip()
    tree = _parse(code)
    if tree is None or not call_input or not output:
        return None
    return intake.Draft(
        skill="programming",
        source="cruxeval",
        record=str(record.get("id") or index),
        topic=_program_topic(tree),
        stage=intake.stage_for(len(list(ast.walk(tree))), 20, 45),
        prompt="\n\n".join((CODE_PROMPT, code, f"f({call_input}) = ?")),
        kind=skills.LITERAL,
        answer=output,
    )
