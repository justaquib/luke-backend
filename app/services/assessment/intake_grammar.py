"""Grammar items from BLiMP and instruction-following items from IFEval."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.services.assessment import intake, skills

GRAMMAR_PROMPT = "Which sentence is grammatically correct? Answer with the letter only."
# Phenomena that turn on structure rather than a single word, so they are the
# hardest band the grammar skill can be measured on.
STRUCTURAL_TOPICS = ("island", "ellipsis", "raising", "scope", "reconstruction", "npi")

IFEVAL_LEAD = "Follow every instruction in this request exactly."


def _blimp_topic(record: Mapping[str, Any]) -> str:
    return str(record.get("linguistics_term") or record.get("UID") or "").strip()


def _blimp_stage(paradigm: str, sentence: str) -> int:
    stage = intake.stage_for(intake.word_count(sentence), 7, 12)
    if any(hint in paradigm for hint in STRUCTURAL_TOPICS):
        return skills.MAX_STAGE
    return stage


def blimp_draft(record: Mapping[str, Any], index: int) -> intake.Draft | None:
    """A minimal pair, asked as a two-option choice in a stable position."""
    good = str(record.get("sentence_good") or "").strip()
    bad = str(record.get("sentence_bad") or "").strip()
    if not good or not bad or good.casefold() == bad.casefold():
        return None
    paradigm = str(record.get("UID") or "")
    topic = _blimp_topic(record) or paradigm
    record_id = f"{paradigm.lower()}-{record.get('pair_id', index)}"
    item_id = f"grammar:blimp:{record_id}"
    first_bad = intake.coin(f"order:{item_id}")
    choices = (bad, good) if first_bad else (good, bad)
    answer = intake.CHOICE_LABELS[first_bad] if first_bad else "A"
    return intake.Draft(
        skill="grammar",
        source="blimp",
        record=record_id,
        topic=topic,
        stage=_blimp_stage(paradigm or topic, good),
        prompt=f"{GRAMMAR_PROMPT}\n\n{intake.choice_lines(choices)}",
        kind=skills.OPTION,
        answer=answer,
        choices=choices,
        extra={"paradigm": paradigm},
    )


def _clean_kwargs(row: object) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        return {}
    return {str(key): value for key, value in row.items() if value is not None}


def ifeval_draft(record: Mapping[str, Any], index: int) -> intake.Draft | None:
    """A prompt whose constraints are checked by rules rather than by a judge."""
    prompt = str(record.get("prompt") or "").strip()
    names = [str(name) for name in record.get("instruction_id_list") or ()]
    rows = list(record.get("kwargs") or ())
    if not prompt or not names or len(rows) != len(names):
        return None
    rules = tuple(
        {"id": name, "kwargs": _clean_kwargs(row)} for name, row in zip(names, rows, strict=True)
    )
    return intake.Draft(
        skill="conversation",
        source="ifeval",
        record=str(record.get("key") or index),
        topic=names[0].split(":")[0],
        stage=intake.stage_for(len(names), 1, 2),
        prompt=f"{IFEVAL_LEAD}\n\n{prompt}",
        kind=skills.RULES,
        rules=rules,
        extra={"constraints": len(names)},
    )
