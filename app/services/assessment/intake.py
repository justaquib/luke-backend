"""The item every assessment source is normalised into.

A draft is what one source record becomes; an item is what the bank file
stores. The held-out split is assigned last, because it depends on the whole
pool of a skill rather than on a single record.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.services.assessment import skills

WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
CHOICE_LABELS = "ABCDEF"
SPLITS = ("eval", "train")
ITEM_FIELDS = ("id", "skill", "stage", "prompt", "topic", "expected", "split", "source")
SOURCE_FIELDS = ("name", "dataset", "record", "licence", "url")


class ItemError(ValueError):
    """An item the bank cannot be trusted to hold."""


@dataclass(frozen=True)
class SourceMeta:
    """Where a source came from, so every item carries its provenance."""

    key: str
    skill: str
    dataset: str
    licence: str
    url: str
    records: str = "row index"


@dataclass(frozen=True)
class Draft:
    """One source record, shaped into a question with a known answer."""

    skill: str
    source: str
    record: str
    topic: str
    stage: int
    prompt: str
    kind: str
    answer: object = None
    choices: tuple[str, ...] = ()
    acceptable: tuple[str, ...] = ()
    rules: tuple[Mapping[str, Any], ...] = ()
    criteria: tuple[str, ...] = ()
    reference: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def item_id(self) -> str:
        return f"{self.skill}:{self.source}:{self.record}"


def word_count(text: str) -> int:
    """The words in a passage, counted the same way in every stage band."""
    return len(WORD.findall(text))


def stage_for(value: int, *upper_bounds: int) -> int:
    """The first stage whose upper bound reaches the value."""
    for stage, bound in enumerate(upper_bounds, start=skills.MIN_STAGE):
        if value <= bound:
            return stage
    return skills.MAX_STAGE


def coin(key: str) -> int:
    """A stable 0 or 1 for a record, so an answer letter is not always the same."""
    return hashlib.sha256(key.encode("utf-8")).digest()[0] % 2


def choice_lines(choices: Sequence[str]) -> str:
    """The lettered options a multiple-choice prompt shows."""
    return "\n".join(f"{CHOICE_LABELS[position]}. {text}" for position, text in enumerate(choices))


def question_lead(text: str) -> str:
    """The opening word of a question, which is what the item asks about.

    A topic the profile can group by without a model judging the answer.
    """
    match = WORD.search(text.strip())
    return match.group(0).casefold() if match else "question"


def expected_of(draft: Draft) -> dict[str, Any]:
    """The scoring contract for one draft, by item kind."""
    if draft.kind == skills.OPTION:
        return {"kind": skills.OPTION, "answer": draft.answer}
    if draft.kind == skills.SPAN:
        return {
            "kind": skills.SPAN,
            "answer": draft.answer,
            "acceptable": list(draft.acceptable or (str(draft.answer),)),
        }
    if draft.kind == skills.NUMBER:
        return {"kind": skills.NUMBER, "answer": draft.answer}
    if draft.kind == skills.LITERAL:
        return {"kind": skills.LITERAL, "answer": draft.answer}
    if draft.kind == skills.RULES:
        return {"kind": skills.RULES, "rules": [dict(rule) for rule in draft.rules]}
    if draft.kind == skills.RUBRIC:
        return {
            "kind": skills.RUBRIC,
            "criteria": list(draft.criteria),
            "reference": draft.reference,
        }
    raise ItemError(f"{draft.item_id} has unknown item kind '{draft.kind}'")


def build_item(draft: Draft, meta: SourceMeta) -> dict[str, Any]:
    """The bank record for one draft, without a split yet."""
    item: dict[str, Any] = {
        "id": draft.item_id,
        "skill": draft.skill,
        "stage": draft.stage,
        "prompt": draft.prompt,
        "topic": draft.topic,
        "expected": expected_of(draft),
        "source": {
            "name": meta.key,
            "dataset": meta.dataset,
            "record": draft.record,
            "licence": meta.licence,
            "url": meta.url,
        },
    }
    if draft.choices:
        item["choices"] = list(draft.choices)
    if draft.extra:
        item["meta"] = dict(draft.extra)
    return item


def _rank(salt: str, item_id: str) -> str:
    return hashlib.sha256(f"{salt}:{item_id}".encode()).hexdigest()


def assign_splits(pool: Sequence[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    """The held-out and training slices, chosen by a stable digest order.

    The same salt always picks the same items, so an evaluation item can never
    move into the training pool by accident when the bank is rebuilt.
    """
    by_id = {item["id"]: item for item in pool}
    ordered = sorted(by_id, key=lambda item_id: _rank(salt, item_id))
    held_end = skills.EVAL_ITEMS_PER_SKILL
    taught_end = held_end + skills.TRAIN_ITEMS_PER_SKILL
    held = ordered[:held_end]
    taught = ordered[held_end:taught_end]
    chosen = [{**by_id[item_id], "split": "eval"} for item_id in held]
    chosen += [{**by_id[item_id], "split": "train"} for item_id in taught]
    return sorted(chosen, key=lambda item: item["id"])


def _check_present(item: Mapping[str, Any]) -> None:
    for name in ITEM_FIELDS:
        if name not in item:
            raise ItemError(f"{item.get('id')} has no {name}")
    for name in SOURCE_FIELDS:
        if not str(item["source"].get(name) or "").strip():
            raise ItemError(f"{item.get('id')} has no source {name}")


def _check_head(item: Mapping[str, Any]) -> None:
    if item["skill"] not in skills.SKILL_IDS:
        raise ItemError(f"{item['id']} names unknown skill '{item['skill']}'")
    if not str(item["prompt"]).strip():
        raise ItemError(f"{item['id']} has an empty prompt")
    if not str(item["topic"]).strip():
        raise ItemError(f"{item['id']} has no topic")
    if item["split"] not in SPLITS:
        raise ItemError(f"{item['id']} has unknown split '{item['split']}'")
    stage = item["stage"]
    if not isinstance(stage, int) or not skills.MIN_STAGE <= stage <= skills.MAX_STAGE:
        raise ItemError(f"{item['id']} has stage {stage}, outside the bank's range")


def _check_expected(item: Mapping[str, Any]) -> None:
    expected = item["expected"]
    kind = expected.get("kind")
    if kind not in skills.EXPECTED_KINDS:
        raise ItemError(f"{item['id']} has unknown item kind '{kind}'")
    if kind == skills.OPTION:
        _check_option(item, expected)
    elif kind == skills.RULES:
        _check_rules(item, expected)
    elif kind == skills.RUBRIC:
        _check_rubric(item, expected)
    elif not str(expected.get("answer") or "").strip() and expected.get("answer") != 0:
        raise ItemError(f"{item['id']} has no answer to score")


def _check_option(item: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    choices = item.get("choices") or []
    letters = CHOICE_LABELS[: len(choices)]
    if not choices:
        raise ItemError(f"{item['id']} is an option item with no choices")
    if expected.get("answer") not in letters:
        raise ItemError(f"{item['id']} answers an option item with an unknown letter")


def _check_rules(item: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    rules = expected.get("rules") or []
    if not rules or any(not rule.get("id") for rule in rules):
        raise ItemError(f"{item['id']} has instruction rules with no identifier")


def _check_rubric(item: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if not expected.get("criteria"):
        raise ItemError(f"{item['id']} is a rubric with no criteria for the trainer")
    if not str(expected.get("reference") or "").strip():
        raise ItemError(f"{item['id']} is a rubric with no reference answer")


def validate_item(item: Mapping[str, Any]) -> None:
    """Raise unless the item is complete enough to score and to cite."""
    _check_present(item)
    _check_head(item)
    _check_expected(item)
