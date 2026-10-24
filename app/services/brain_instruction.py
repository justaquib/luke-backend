"""The measurements the model takes of itself beyond the level ladder.

Two numbers are stored here: how well the writer answers requests it was
never worded for, and how far it can answer without being taught. Both are
measured on demand and kept, because a measurement that is taken on every
page load is a cost with no extra truth in it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.services import brain_paths
from app.services.brain_rungs import GENERALITY_PROBES


def instruction_path(tenant_id: int | None) -> Path:
    """Where the last instruction measurement is kept."""
    return brain_paths.brain_dir() / f"instruction-{brain_paths.tenant_key(tenant_id)}.json"


def unmeasured_instruction() -> dict[str, Any]:
    """The instruction score before the trainer has run the first test."""
    return {
        "score": 0.0,
        "passed": 0,
        "total": 0,
        "probes": [],
        "measured": False,
        "measured_at": None,
    }


def stored_instruction(tenant_id: int | None) -> dict[str, Any]:
    """The last instruction measurement that was stored, if there is one."""
    try:
        data = json.loads(instruction_path(tenant_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return unmeasured_instruction()
    return data if isinstance(data, dict) else unmeasured_instruction()


def save_instruction(tenant_id: int | None, result: dict[str, Any]) -> None:
    """Keep the instruction measurement beside the tenant's other numbers."""
    path = instruction_path(tenant_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **result,
        "measured": True,
        "measured_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def instruction_score(tenant_id: int | None) -> dict[str, Any]:
    """How well the model answers the requests it was never trained on.

    The measurement takes about a minute, so the last one is stored and the
    brain page reads the stored number instead of measuring on every load.
    A fresh measurement is taken when the trainer runs the level test, or
    when a training has just finished.
    """
    return stored_instruction(tenant_id)


def remeasure_instruction(tenant_id: int | None) -> dict[str, Any]:
    """Run the instruction test again and keep the measurement."""
    from app.services.brain_training import load_writer
    from app.services.language import evaluation as evaluation_service
    from app.services.language.lessons import all_lesson_cards

    try:
        cards = all_lesson_cards(tenant_id)
    except (OSError, ValueError):
        cards = []
    result = evaluation_service.evaluate(tenant_id, cards, load_writer(tenant_id))
    save_instruction(tenant_id, result)
    return result


def generality_score(tenant_id: int | None) -> dict[str, Any]:
    """How far the brain can answer beyond what it was taught.

    This is what separates a small local model from a frontier one: a frontier
    model answers questions it has never seen, this one cannot.
    """
    from app.services import language_logic

    answered = 0
    probes: list[dict[str, Any]] = []
    for probe in GENERALITY_PROBES:
        try:
            result = language_logic.preview_answer(probe, tenant_id)
        except (OSError, ValueError, RuntimeError):
            result = {"answer": "", "source": "not_learned"}
        source = result.get("source", "not_learned")
        # A dictionary lookup is a lookup, not general reasoning.
        general = source not in {"not_learned", "dictionary"}
        answered += 1 if general else 0
        probes.append({"question": probe, "answered": general, "source": source})
    score = answered / max(len(GENERALITY_PROBES), 1)
    return {"score": round(score, 2), "cap": _cap(score), "probes": probes}


def _cap(score: float) -> int:
    """The level the generality score allows."""
    if score >= 0.75:
        return 6
    if score >= 0.5:
        return 5
    if score >= 0.25:
        return 4
    return 3
