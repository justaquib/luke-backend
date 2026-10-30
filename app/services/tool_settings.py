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


def app_allowlist(tenant_id: int | None) -> list[str]:
    """The apps the tenant has allowed Luke to launch."""
    return read(tenant_id).get("apps", {}).get("allowlist", [])


def set_app_allowlist(apps: list[str], tenant_id: int | None) -> None:
    """Remember which apps the tenant allows Luke to launch."""
    path = settings_path(tenant_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = read(tenant_id)
    stored.setdefault("apps", {})["allowlist"] = apps
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    temporary.replace(path)
