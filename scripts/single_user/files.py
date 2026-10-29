"""Move the uploads that live at the root into the workspace folder.

The upload route now writes to ``uploads/0``. The files that were saved
before it must follow, or the file routes cannot find them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

WORKSPACE_DIR = "0"


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def plan(root: Path) -> dict[str, Any]:
    """Where each loose upload goes, and what happens if the name is taken."""
    destination = root / WORKSPACE_DIR
    moves = []
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        if path.is_dir():
            continue
        target = destination / path.name
        action = "move"
        if target.exists():
            action = "same" if _digest(path) == _digest(target) else "conflict"
        moves.append({"source": path.name, "action": action})
    return {"root": str(root), "workspace": str(destination), "moves": moves}


def apply(plan_payload: dict[str, Any], root: Path) -> dict[str, Any]:
    """Move the loose uploads, leaving a conflicting file where it is."""
    destination = root / WORKSPACE_DIR
    destination.mkdir(parents=True, exist_ok=True)
    moved = []
    for entry in plan_payload["moves"]:
        source = root / entry["source"]
        if entry["action"] == "move":
            source.replace(destination / entry["source"])
            moved.append(entry["source"])
        elif entry["action"] == "same":
            source.unlink()
    return {"moved": moved, "workspace": str(destination)}
