"""Merge the saved learner files onto workspace 0.

A lesson keeps its identity by the words of its question and answer, not by
the id it was saved under. Measurement files belong to the model that was
measured, so they are kept beside the state and never merged into it.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

STATE_FILE = "tenant-0.json"
CARD_LISTS = ("lessons", "lesson_cards", "word_cards")


def _read(path: Path) -> dict[str, Any]:
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return stored if isinstance(stored, dict) else {}


def _card_key(card: dict[str, Any]) -> str:
    body = (
        f"{str(card.get('question', '')).strip().casefold()}\n"
        f"{str(card.get('answer', '')).strip().casefold()}"
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _merge_cards(state: dict[str, Any], other: dict[str, Any], source: str) -> list[dict]:
    notes = []
    for name in CARD_LISTS:
        existing = state.setdefault(name, [])
        keys = {_card_key(card): card for card in existing if isinstance(card, dict)}
        ids = {str(card.get("id")) for card in existing if isinstance(card, dict)}
        for stored in other.get(name, []):
            if not isinstance(stored, dict):
                continue
            key = _card_key(stored)
            if key in keys:
                notes.append({"file": source, "list": name, "card": key, "action": "duplicate"})
                continue
            card = stored
            if str(stored.get("id")) in ids:
                card = {**stored, "id": uuid.uuid4().hex}
                notes.append({"file": source, "list": name, "card": key, "action": "reassigned"})
            existing.append(card)
            keys[key] = card
            ids.add(str(card.get("id")))
    return notes


def plan_state(directory: Path) -> dict[str, Any]:
    """What merging the learner files would add to workspace 0."""
    target = directory / STATE_FILE
    state = _read(target)
    sources = sorted(
        path.name for path in directory.glob("tenant-*.json") if path.name != STATE_FILE
    )
    notes: list[dict] = []
    for name in sources:
        notes.extend(_merge_cards(state, _read(directory / name), name))
    measurements = sorted(
        path.name
        for path in directory.glob("instruction-*.json")
        if path.name != "instruction-0.json"
    )
    return {
        "target": str(target),
        "sources": sources,
        "notes": notes,
        "measurements": [
            {"file": name, "action": "kept as history", "reason": "measured another model"}
            for name in measurements
        ],
        "merged_state": state,
        "changed": any(note["action"] != "duplicate" for note in notes),
    }


def apply_state(plan: dict[str, Any]) -> dict[str, Any]:
    """Write the merged learner file, and only when it gained a lesson."""
    if not plan["changed"]:
        return {"written": False}
    target = Path(plan["target"])
    target.write_text(json.dumps(plan["merged_state"], indent=2), encoding="utf-8")
    return {"written": True, "target": str(target)}
