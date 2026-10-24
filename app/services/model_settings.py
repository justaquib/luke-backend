"""The name of the model and the level it starts from.

A tenant can name its own model and choose the version it begins on. The name
is used wherever the model speaks about itself, and the starting version is
the floor the displayed level never drops below, so a model can begin life at
a version the trainer has already earned elsewhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services import brain_paths
from app.services.brain_levels import (
    LUKE_LEVELS,
    SEED_LEVEL,
    approved_level,
    level_order,
    level_title,
)

DEFAULT_NAME = "Luke"
DEFAULT_START_LEVEL = SEED_LEVEL
MAX_NAME_LENGTH = 40


@dataclass(frozen=True)
class ModelSettings:
    """The name the model answers to, and the version it starts on."""

    name: str = DEFAULT_NAME
    start_level: str = DEFAULT_START_LEVEL


def settings_path(tenant_id: int | None) -> Path:
    """Where a tenant's model name and starting version are kept."""
    return brain_paths.brain_dir() / f"model-{brain_paths.tenant_key(tenant_id)}.json"


def level_ids() -> set[str]:
    """Every version the ladder offers."""
    return {str(entry["level"]) for entry in LUKE_LEVELS}


def read(tenant_id: int | None) -> ModelSettings:
    """The tenant's settings, or the defaults when none were saved."""
    try:
        stored = json.loads(settings_path(tenant_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ModelSettings()
    if not isinstance(stored, dict):
        return ModelSettings()
    try:
        name = _valid_name(stored.get("name")) or DEFAULT_NAME
    except ValueError:
        name = DEFAULT_NAME
    level = str(stored.get("start_level") or DEFAULT_START_LEVEL)
    start_level = level if level in level_ids() else DEFAULT_START_LEVEL
    return ModelSettings(name=name, start_level=start_level)


def name_for(tenant_id: int | None) -> str:
    """The configured model name, ready to be used in a sentence."""
    return read(tenant_id).name


def save(tenant_id: int | None, name: str, start_level: str) -> ModelSettings:
    """Keep the model name and the version it starts from."""
    cleaned = _valid_name(name)
    if not cleaned:
        raise ValueError("The model needs a name.")
    level = str(start_level or "").strip()
    if level not in level_ids():
        raise ValueError("Choose a starting version from the model ladder.")
    settings = ModelSettings(name=cleaned, start_level=level)
    path = settings_path(tenant_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"name": settings.name, "start_level": settings.start_level}, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return settings


def starting_level(tenant_id: int | None) -> str:
    """The version the displayed level may not drop below."""
    return read(tenant_id).start_level


def floor_level(tenant_id: int | None, level: str) -> str:
    """The approved level, raised to the configured starting version."""
    start = starting_level(tenant_id)
    return start if level_order(start) > level_order(level) else level


def view(tenant_id: int | None) -> dict[str, Any]:
    """The settings and the ladder the settings page reads."""
    settings = read(tenant_id)
    level = floor_level(tenant_id, approved_level(tenant_id))
    return {
        "name": settings.name,
        "start_level": settings.start_level,
        "level": level,
        "defaults": {"name": DEFAULT_NAME, "start_level": DEFAULT_START_LEVEL},
        "levels": [
            {
                "level": entry["level"],
                "name": entry["name"],
                "title": level_title(str(entry["level"]), settings.name),
                "description": entry["description"],
            }
            for entry in LUKE_LEVELS
        ],
    }


def _valid_name(value: Any) -> str:
    """The name with tidy spacing, or nothing when it is unusable."""
    if value is None:
        return ""
    name = " ".join(str(value).split())
    if len(name) > MAX_NAME_LENGTH:
        raise ValueError(f"The model name can be at most {MAX_NAME_LENGTH} characters.")
    return name
