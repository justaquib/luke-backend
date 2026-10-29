"""Reading the vendored item bank, validated once and never from the network.

The bank under ``app/data/assessment/`` is the assessment engine's input. One
file per skill plus a manifest that records the version, the split salt, the
source fetches and each file's digest. Everything here is a read: an item that
fails validation raises instead of being scored, because a malformed item would
make the skill's number mean nothing.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.services.assessment import intake, skills

BANK_DIR = Path(__file__).resolve().parents[2] / "data" / "assessment"
MANIFEST = "manifest.json"
SPLITS = ("eval", "train")


class BankError(ValueError):
    """The shipped bank is missing, unreadable or inconsistent."""


@lru_cache(maxsize=1)
def manifest() -> dict[str, Any]:
    """The manifest as written by ``scripts/build_item_bank.py``."""
    path = BANK_DIR / MANIFEST
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BankError(f"the assessment manifest cannot be read: {error}") from error
    if not isinstance(payload, dict):
        raise BankError("the assessment manifest is not an object")
    return payload


def version() -> int:
    return int(manifest().get("version") or 0)


def salt() -> str:
    return str(manifest().get("salt") or "")


@lru_cache(maxsize=len(skills.SKILL_IDS))
def _skill_file(skill: str) -> tuple[dict[str, Any], ...]:
    if skill not in skills.SKILL_TITLES:
        raise BankError(f"unknown skill '{skill}'")
    path = BANK_DIR / f"{skill}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BankError(f"{skill} cannot be read: {error}") from error
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        raise BankError(f"{skill} has no items")
    for item in items:
        if not isinstance(item, dict):
            raise BankError(f"{skill} has an item that is not an object")
        if item.get("skill") != skill:
            raise BankError(f"{item.get('id')} is filed under {skill}")
        try:
            intake.validate_item(item)
        except intake.ItemError as error:
            raise BankError(f"{skill}: {error}") from error
    return tuple(items)


def items(skill: str | None = None, split: str | None = None) -> list[dict[str, Any]]:
    """Every item, optionally for one skill and one split, in bank order."""
    if split is not None and split not in SPLITS:
        raise BankError(f"unknown split '{split}'")
    chosen = [skill] if skill is not None else list(skills.SKILL_IDS)
    found = [item for name in chosen for item in _skill_file(name)]
    if split is not None:
        found = [item for item in found if item.get("split") == split]
    return found


@lru_cache(maxsize=1)
def _index() -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in items()}


def item(item_id: str) -> dict[str, Any] | None:
    """One item by id, or ``None`` when this bank version never had it."""
    return _index().get(item_id)


def counts() -> dict[str, dict[str, int]]:
    """Per skill, how many items each split holds."""
    return {
        skill: {
            name: sum(1 for item in _skill_file(skill) if item.get("split") == name)
            for name in SPLITS
        }
        for skill in skills.SKILL_IDS
    }


def check() -> list[str]:
    """Every problem found offline; an empty list means the bank is sound."""
    problems: list[str] = []
    if version() <= 0:
        problems.append("the manifest has no version")
    if not salt():
        problems.append("the manifest has no split salt")
    seen: list[str] = []
    failed = False
    for skill in skills.SKILL_IDS:
        try:
            found = _skill_file(skill)
        except BankError as error:
            problems.append(str(error))
            failed = True
            continue
        seen.extend(str(item["id"]) for item in found)
        held = sum(1 for item in found if item.get("split") == "eval")
        taught = sum(1 for item in found if item.get("split") == "train")
        if held != skills.EVAL_ITEMS_PER_SKILL:
            problems.append(f"{skill} holds {held} eval items")
        if taught != skills.TRAIN_ITEMS_PER_SKILL:
            problems.append(f"{skill} holds {taught} train items")
    if not failed and len(set(seen)) != len(seen):
        problems.append("two items share an id")
    return problems
