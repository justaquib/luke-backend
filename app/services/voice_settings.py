"""Per-tenant voice settings: which module is active.

The settings are stored in the tenant's JSON file alongside the model name and
starting version. The file is read on every voice request so a module switch
takes effect immediately without a restart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services import brain_paths
from app.services.voice import VoiceModuleUnavailable, set_active_module


def settings_path(tenant_id: int | None) -> Path:
    """Where a tenant's voice module choice is kept."""
    return brain_paths.brain_dir() / f"voice-{brain_paths.tenant_key(tenant_id)}.json"


def read(tenant_id: int | None) -> dict[str, Any]:
    """The tenant's voice settings, or empty defaults when none were saved."""
    try:
        stored = json.loads(settings_path(tenant_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(stored, dict):
        return {}
    return stored


def active_module_name(tenant_id: int | None) -> str | None:
    """The module the tenant has chosen, or None if they haven't."""
    return read(tenant_id).get("active_module")


def set_active_module_name(tenant_id: int | None, name: str) -> None:
    """Remember which module the tenant wants to use."""
    path = settings_path(tenant_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = read(tenant_id)
    stored["active_module"] = name
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    temporary.replace(path)


def ensure_module_for(tenant_id: int | None) -> None:
    """Activate the tenant's chosen module, or the default if none was chosen.

    Called at the start of every TTS/STT request so the module registry is in
    sync with the tenant's settings file.
    """
    name = active_module_name(tenant_id)
    if name is not None:
        try:
            set_active_module(name)
        except VoiceModuleUnavailable:
            pass
