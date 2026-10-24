"""The ladder of model levels and the level the trainer approved.

The ladder is measured, not given: every rung is a capability the brain has
to show. What it measures as and what the page displays are two numbers, and
the displayed one only moves when the trainer approves the promotion.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.services import brain_paths

# The level the brain starts on. Further rungs are added above it as the
# brain learns to do more, so the ids keep the same three-segment shape.
SEED_LEVEL = "0.0.0"

BENCHMARK_TITLE = "Luna 6 · Frontier"

BENCHMARK_NOTE = (
    "Luna 6 is the class of a frontier chat model such as GPT or Claude at the "
    "top of its range. This model is not one of those: it is built from scratch "
    "and trained only on what you teach it, so it is measured against that bar "
    "honestly."
)

LUKE_LEVELS: list[dict[str, Any]] = [
    {
        "level": "0.0.0",
        "name": "Seed",
        "description": (
            "It holds what it was taught and can look it up, but it cannot "
            "answer anything on its own."
        ),
        "reference": "The floor of the scale: material without general answering.",
    },
    {
        "level": "0.0.1",
        "name": "Cued",
        "description": "It finds the answer when asked exactly as it was taught.",
        "reference": "Teach it something and it can give that back.",
    },
    {
        "level": "0.0.2",
        "name": "Paraphrased",
        "description": "It finds the answer when the question is worded differently.",
        "reference": "It survives a rewording of the question.",
    },
    {
        "level": "0.0.3",
        "name": "Generalised",
        "description": "It answers about the topic without the exact question.",
        "reference": "It knows the topic, not only the sentence.",
    },
    {
        "level": "0.0.4",
        "name": "Written",
        "description": "It writes new sentences in the style it was taught.",
        "reference": "It composes instead of quoting.",
    },
    {
        "level": "0.0.5",
        "name": "Explained",
        "description": "It explains an answer when it is asked to.",
        "reference": "It can say why, not only what.",
    },
    {
        "level": "0.0.6",
        "name": "Corrected",
        "description": "It learns from a correction and keeps it.",
        "reference": "A wrong answer turns into a right one, and stays right.",
    },
    {
        "level": "0.0.7",
        "name": "Examined",
        "description": "It passes the final exam of a school grade.",
        "reference": "The school says it is ready for the next grade.",
    },
    {
        "level": "0.0.8",
        "name": "Grounded",
        "description": "It says it does not know instead of guessing.",
        "reference": "Knowing the edge of its knowledge is a kind of intelligence.",
    },
    {
        "level": "0.0.9",
        "name": "Teacher",
        "description": "It holds a whole subject: many questions, one topic.",
        "reference": "Depth, not a list of facts.",
    },
    {
        "level": "0.1.0",
        "name": "Independent",
        "description": "It answers a question it was never taught, in its own words.",
        "reference": (
            "The long road: every lesson, every correction and every exam widens what it can do."
        ),
    },
]

# Stages of growth on the maturity side: how much material the brain has
# taken in. They sit on their own human scale, like the stages of a growing
# child, and never borrow the names of the model levels, because volume is
# not intelligence.
_MATURITY_STAGES: tuple[tuple[int, str, str], ...] = (
    (0, "Newborn", "Nothing is indexed yet; the brain has not started to take in the world."),
    (1, "First words", "The first lesson or answer is in; recognition begins."),
    (25, "Toddler", "A handful of words and lessons; it repeats what it was taught."),
    (100, "Schoolchild", "Enough material to answer across several subjects."),
    (500, "Scholar", "Deep material with documents, many topics and whole subjects."),
)


def training_volume_level(items: int) -> dict[str, Any]:
    """Where the amount of learned material sits on the maturity scale.

    Volume is not intelligence, so it never touches the model level: a
    hundred thousand documents still do not make a model reason. The stages
    follow the human brain instead, from newborn to scholar.
    """
    index = 0
    for position, stage in enumerate(_MATURITY_STAGES):
        if items >= stage[0]:
            index = position
    stages = [
        {
            "title": title,
            "items": threshold,
            "detail": detail,
            "reached": position <= index,
            "current": position == index,
        }
        for position, (threshold, title, detail) in enumerate(_MATURITY_STAGES)
    ]
    _, title, detail = _MATURITY_STAGES[index]
    next_stage: dict[str, Any] | None = None
    if index + 1 < len(_MATURITY_STAGES):
        threshold, next_title, _next_detail = _MATURITY_STAGES[index + 1]
        next_stage = {"title": next_title, "items": threshold}
    return {
        "title": title,
        "detail": detail,
        "items": items,
        "stages": stages,
        "next_stage": next_stage,
    }


def level_order(level: str) -> tuple[int, int, int]:
    """The ladder id as three numbers, so ids sort as 0.0.1, 0.0.9, 0.1.0."""
    values: list[int] = []
    for part in str(level).split("."):
        try:
            values.append(int(part))
        except ValueError:
            values.append(0)
    while len(values) < 3:
        values.append(0)
    return (values[0], values[1], values[2])


def level_value(level: str) -> float:
    """The same ladder id as one number, for bars and progress shares."""
    major, minor, patch = level_order(level)
    return major + minor / 10 + patch / 100


def level_title(level: str, model_name: str = "Luke") -> str:
    return f"{model_name} {level}"


def level_entry(level: str, model_name: str = "Luke") -> dict[str, Any]:
    """The ladder entry for a level id."""
    for entry in LUKE_LEVELS:
        if str(entry["level"]) == level:
            return {**entry, "title": level_title(level, model_name)}
    return {**LUKE_LEVELS[0], "title": level_title(SEED_LEVEL, model_name)}


def next_level_entry(
    level: str,
    model_name: str = "Luke",
    entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """The next entry above a level, from the open ladder or a given road."""
    order = level_order(level)
    for entry in entries or LUKE_LEVELS:
        if level_order(str(entry["level"])) > order:
            return {
                **entry,
                "title": f"{level_title(str(entry['level']), model_name)} · {entry['name']}",
            }
    return None


def level_state_path(tenant_id: int | None) -> Path:
    """Where the trainer's approved model level is kept."""
    return brain_paths.brain_dir() / f"level-{brain_paths.tenant_key(tenant_id)}.json"


def approved_level(tenant_id: int | None) -> str:
    """The level the trainer has approved, never below the starting version."""
    from app.services import model_settings

    path = level_state_path(tenant_id)
    level = SEED_LEVEL
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = None
    if isinstance(data, dict):
        stored = str(data.get("approved_level", SEED_LEVEL))
        if stored in {str(entry["level"]) for entry in LUKE_LEVELS}:
            level = stored
    return model_settings.floor_level(tenant_id, level)


def approve_level(tenant_id: int | None) -> dict[str, Any]:
    """Promote the brain to the level its measurements have reached.

    Nothing is promoted on its own: the measured level can run ahead of the
    displayed one, and this call from the trainer closes the gap. The
    approval is kept with a small history, so the climb is on record.
    """
    from app.services import model_settings
    from app.services.brain_state import model_level

    level = model_level(tenant_id)
    measured = str(level["measured_level"])
    previous = approved_level(tenant_id)
    if level_order(measured) <= level_order(previous):
        name = model_settings.name_for(tenant_id)
        raise ValueError(f"{name} has not measured a higher level than is approved.")
    path = level_state_path(tenant_id)
    history: list[Any] = []
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        history = list(stored.get("history", [])) if isinstance(stored, dict) else []
    except (OSError, json.JSONDecodeError):
        history = []
    history.append({"level": measured, "approved_at": datetime.now(UTC).isoformat()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"approved_level": measured, "history": history[-20:]}, indent=2),
        encoding="utf-8",
    )
    return {"approved_level": measured, "previous_level": previous, "history": history}
