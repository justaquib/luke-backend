"""Launch-app tool: open an installed application on the OS.

On macOS, the tool uses `open -a` to launch the app by name. Requires
confirmation before executing, and the app must be in the tenant's allowlist.
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any

from app.services.tools.registry import Tool, register


def launch_app(app_name: str, tenant_id: int | None = None) -> dict[str, Any]:
    """Open an installed application by name.

    The app must be in the tenant's allowlist. On macOS, uses `open -a`.
    """
    if not _is_allowed(app_name, tenant_id):
        raise PermissionError(f"App not in allowlist: {app_name}")
    system = platform.system()
    if system == "Darwin":
        result = subprocess.run(  # noqa: S603
            ["open", "-a", app_name],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to launch {app_name}: {result.stderr}")
        return {"launched": True, "app": app_name, "platform": system}
    if system == "Linux":
        result = subprocess.run(  # noqa: S603
            ["xdg-open", app_name],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to launch {app_name}: {result.stderr}")
        return {"launched": True, "app": app_name, "platform": system}
    raise RuntimeError(f"Unsupported platform: {system}")


def _is_allowed(app_name: str, tenant_id: int | None) -> bool:
    """Check whether the app is in the tenant's allowlist."""
    from app.services import tool_settings

    allowlist = tool_settings.app_allowlist(tenant_id)
    return not allowlist or app_name in allowlist


def list_installed_apps() -> list[dict[str, str]]:
    """List applications installed on the OS.

    On macOS, scans /Applications and ~/Applications. Returns a list of
    {name, path} dicts.
    """
    system = platform.system()
    if system == "Darwin":
        apps = []
        for directory in [Path("/Applications"), Path.home() / "Applications"]:
            if directory.exists():
                for app_path in directory.glob("*.app"):
                    apps.append({"name": app_path.stem, "path": str(app_path)})
        return sorted(apps, key=lambda a: a["name"].lower())
    return []


register(
    Tool(
        name="launch_app",
        description="Open an installed application. The app must be in the allowlist.",
        parameters={
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "The name of the application to launch.",
                }
            },
            "required": ["app_name"],
        },
        requires_confirmation=True,
        handler=launch_app,
    )
)
