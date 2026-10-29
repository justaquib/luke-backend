"""Small typed checks shared by the read-only package validators."""

from __future__ import annotations

import json
from pathlib import Path

from app.services.language.package.manifest import PackageError


def require(condition: bool, message: str) -> None:
    """Refuse a package that fails one compatibility check, with a next step."""
    if not condition:
        raise PackageError(f"{message}. Export a complete, compatible package again.")


def as_object(value: object, label: str) -> dict:
    """A value that must be a JSON object."""
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def as_list(value: object, item_type: type, label: str) -> list:
    """A value that must be a list of one JSON type."""
    require(isinstance(value, list), f"{label} must be a list")
    require(all(isinstance(item, item_type) for item in value), f"{label} has invalid entries")
    return value


def read_json(path: Path) -> object:
    """A JSON file, refusing the non-finite numbers Python would otherwise accept."""
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_non_finite)


def _non_finite(value: str) -> None:
    raise ValueError(f"Non-finite JSON number {value}")
