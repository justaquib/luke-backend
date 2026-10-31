"""Per-tenant tool settings: the app allowlist.

The settings are stored in the tenant's tools JSON file alongside the tool
permissions. The file is read on every app-launch request so a change takes
effect immediately.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services import brain_paths


def settings_path(tenant_id: int | None) -> Path:
    """Where a tenant's tool settings are kept."""
    return brain_paths.brain_dir() / f"tools-{brain_paths.tenant_key(tenant_id)}.json"


def read(tenant_id: int | None) -> dict[str, Any]:
    """The tenant's tool settings, or empty defaults when none were saved."""
    try:
        stored = json.loads(settings_path(tenant_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(stored, dict):
        return {}
    return stored


def _normalise_entry(raw: str | dict[str, Any]) -> dict[str, Any]:
    """Accept a plain string or a full entry, always return a dict."""
    if isinstance(raw, str):
        return {"name": raw, "short_name": "", "aliases": []}
    return {
        "name": raw.get("name", ""),
        "short_name": raw.get("short_name", ""),
        "aliases": list(raw.get("aliases", [])),
    }


def app_allowlist(tenant_id: int | None) -> list[dict[str, Any]]:
    """The apps the tenant has allowed Luke to launch.

    Each entry is a dict with ``name``, ``short_name`` and ``aliases``.
    Old string-only entries are upgraded automatically.
    """
    raw = read(tenant_id).get("apps", {}).get("allowlist", [])
    return [_normalise_entry(entry) for entry in raw]


def set_app_allowlist(apps: list[dict[str, Any]], tenant_id: int | None) -> None:
    """Remember which apps the tenant allows Luke to launch."""
    path = settings_path(tenant_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = read(tenant_id)
    stored.setdefault("apps", {})["allowlist"] = apps
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    temporary.replace(path)


def resolve_app_name(query: str, tenant_id: int | None) -> str | None:
    """Match a query against the allowlist and return the real app name.

    Checks the full name first, then the short name, then aliases.
    All comparisons are case-insensitive.  Returns ``None`` when nothing
    matches.
    """
    lowered = query.strip().lower()
    if not lowered:
        return None
    for entry in app_allowlist(tenant_id):
        name = entry["name"]
        short = entry.get("short_name", "")
        aliases = entry.get("aliases", [])
        if name.lower() == lowered:
            return name
        if short and short.lower() == lowered:
            return name
        for alias in aliases:
            if alias.lower() == lowered:
                return name
    return None
