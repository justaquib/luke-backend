"""Per-tenant tool permissions.

The permissions are stored in the tenant's settings JSON file alongside the
model name and voice module choice. The file is read on every tool execution
so a permission change takes effect immediately.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services import brain_paths


def settings_path(tenant_id: int | None) -> Path:
    """Where a tenant's tool permissions are kept."""
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


def can_use(tool_name: str, tenant_id: int | None) -> bool:
    """Whether the tenant has permission to use the named tool.

    A tool is allowed when its permission is not explicitly set to false, so
    new tools are available by default until the admin disables them.
    """
    permissions = read(tenant_id).get("permissions", {})
    return permissions.get(tool_name, True) is not False


def set_permission(tool_name: str, enabled: bool, tenant_id: int | None) -> None:
    """Remember whether the tenant may use the named tool."""
    path = settings_path(tenant_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = read(tenant_id)
    permissions = stored.setdefault("permissions", {})
    permissions[tool_name] = enabled
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    temporary.replace(path)


def all_permissions(tenant_id: int | None) -> dict[str, bool]:
    """Every tool permission the tenant has set, with the default for the rest."""
    from app.services.tools import list_tools

    stored = read(tenant_id).get("permissions", {})
    return {tool.name: stored.get(tool.name, True) for tool in list_tools()}
