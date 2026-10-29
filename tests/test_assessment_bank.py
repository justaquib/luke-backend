"""The vendored bank is complete, valid and split the same way every time."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.assessment import intake, skills
from app.services.assessment.sources import SPLIT_SALT
from scripts.build_item_bank import _pick_indices, page_offsets

BANK = Path(__file__).resolve().parents[1] / "app" / "data" / "assessment"


def _bank() -> dict[str, dict]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(BANK.glob("*.json"))
        if path.name != "manifest.json"
    }


def _items(count: int) -> list[dict]:
    return [
        {
            "id": f"grammar:blimp:pair-{position}",
            "skill": "grammar",
            "stage": 1,
            "prompt": "Which sentence is grammatically correct?",
            "topic": "anaphor_agreement",
            "expected": {"kind": "option", "answer": "A"},
            "choices": ["Susan revealed herself.", "Susan revealed themselves."],
            "source": {
                "name": "blimp",
                "dataset": "nyu-mll/blimp",
                "record": f"pair-{position}",
                "licence": "CC BY 4.0",
                "url": "https://huggingface.co/datasets/nyu-mll/blimp",
            },
        }
        for position in range(count)
    ]


def test_the_bank_holds_every_skill_with_both_splits() -> None:
    bank = _bank()
    assert set(bank) == set(skills.SKILL_IDS)
    for skill, payload in bank.items():
        items = payload["items"]
        held = [item for item in items if item["split"] == "eval"]
        taught = [item for item in items if item["split"] == "train"]
        assert len(held) == skills.EVAL_ITEMS_PER_SKILL, skill
        assert len(taught) == skills.TRAIN_ITEMS_PER_SKILL, skill
        assert payload["title"] == skills.SKILL_TITLES[skill]


def test_every_stored_item_validates_and_is_unique() -> None:
    seen: set[str] = set()
    for payload in _bank().values():
        for item in payload["items"]:
            intake.validate_item(item)
            assert item["id"] not in seen, item["id"]
            seen.add(item["id"])
            assert item["source"]["licence"]
            assert item["expected"]["kind"] in skills.EXPECTED_KINDS


def test_each_skill_is_scored_the_way_it_was_planned() -> None:
    kinds = {
        skill: {item["expected"]["kind"] for item in payload["items"]}
        for skill, payload in _bank().items()
    }
    assert kinds["grammar"] == {skills.OPTION}
    assert kinds["mathematics"] == {skills.NUMBER}
    assert kinds["programming"] == {skills.LITERAL}
    assert kinds["conversation"] == {skills.RULES}
    assert kinds["creative_writing"] == {skills.RUBRIC}
    assert skills.RUBRIC in kinds["reading"]
    assert skills.SPAN in kinds["reading"]


def test_the_manifest_records_a_licence_for_every_source() -> None:
    manifest = json.loads((BANK / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["salt"] == SPLIT_SALT
    assert {row["skill"] for row in manifest["files"]} == set(skills.SKILL_IDS)
    for source in manifest["sources"]:
        assert source["licence"], source["key"]
        assert source["dataset"] and source["url"].startswith("https://")
        assert source["shaped"] >= skills.EVAL_ITEMS_PER_SKILL + skills.TRAIN_ITEMS_PER_SKILL


def test_splits_are_sized_disjoint_and_stable() -> None:
    pool = _items(120)
    first = intake.assign_splits(pool, SPLIT_SALT)
    second = intake.assign_splits(list(reversed(pool)), SPLIT_SALT)
    assert [item["id"] for item in first] == [item["id"] for item in second]
    held = [item for item in first if item["split"] == "eval"]
    taught = [item for item in first if item["split"] == "train"]
    assert len(held) == skills.EVAL_ITEMS_PER_SKILL
    assert len(taught) == skills.TRAIN_ITEMS_PER_SKILL
    assert not {item["id"] for item in held} & {item["id"] for item in taught}


def test_a_different_salt_holds_out_different_items() -> None:
    pool = _items(120)
    here = {
        item["id"] for item in intake.assign_splits(pool, SPLIT_SALT) if item["split"] == "eval"
    }
    there = {
        item["id"] for item in intake.assign_splits(pool, "another-salt") if item["split"] == "eval"
    }
    assert here != there


@pytest.mark.parametrize(
    "change",
    [
        {"skill": "music"},
        {"prompt": "   "},
        {"stage": 4},
        {"split": "holdout"},
        {"expected": {"kind": "option", "answer": "C"}},
        {"expected": {"kind": "vibes", "answer": "A"}},
        {"choices": []},
        {"source": {"name": "blimp", "dataset": "", "record": "1", "licence": "", "url": ""}},
    ],
)
def test_validate_item_refuses_what_could_not_be_scored(change: dict) -> None:
    item = {**_items(1)[0], "split": "eval"}
    broken = {**item, **change}
    with pytest.raises(intake.ItemError):
        intake.validate_item(broken)


def test_a_rubric_without_criteria_is_refused() -> None:
    item = {
        **_items(1)[0],
        "skill": "creative_writing",
        "split": "eval",
        "expected": {"kind": "rubric", "criteria": [], "reference": ""},
    }
    with pytest.raises(intake.ItemError):
        intake.validate_item(item)


def test_page_offsets_spread_across_the_split() -> None:
    assert page_offsets(1000, 300, 100, True) == [0, 450, 900]
    assert page_offsets(50, 300, 100, True) == [0]
    assert page_offsets(1000, 300, 100, False) == [0]
    assert page_offsets(0, 300, 100, True) == []


def test_file_indices_cover_the_whole_file_without_repeats() -> None:
    chosen = _pick_indices(800, 400, True)
    assert len(chosen) == 400
    assert chosen[0] == 0 and chosen[-1] == 799
    assert chosen == sorted(set(chosen))
    assert _pick_indices(50, 400, True) == list(range(50))
