"""Write the migration's report, conflicts, and rollback instructions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _write(path: Path, payload: Any) -> str:
    path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    return str(path)


def write_report(directory: Path, report: dict[str, Any]) -> str:
    """The complete record of what the run inspected and changed."""
    return _write(directory / "report.json", report)


def write_conflicts(directory: Path, conflicts: list[dict[str, Any]]) -> str:
    """Every row that could not be carried over unchanged."""
    return _write(directory / "conflicts.json", {"conflicts": conflicts})


def write_rollback(directory: Path, undo: dict[str, Any], backup: dict[str, Any]) -> str:
    """The data needed to undo the change, and how to use it."""
    _write(directory / "undo.json", undo)
    dump = backup["database"]
    steps = [
        "# Rollback",
        "",
        f"Database backup: `{Path(dump['target']).name}`",
        f"SHA-256: `{dump['sha256']}`",
        "",
        "Stop the backend, then restore the rows and files:",
        "",
        "```",
        f"cd {directory.parents[1]}",
        f".venv/bin/python scripts/restore_single_user.py {directory.name} --yes",
        "```",
        "",
        "The restore replaces every table with the rows recorded in",
        "`database.json` and copies `.chroma`, `.language_model` and `uploads`",
        "back from this folder.",
    ]
    path = directory / "rollback.md"
    path.write_text("\n".join(steps) + "\n", encoding="utf-8")
    return str(path)


def summary(report: dict[str, Any]) -> str:
    """The few lines a person reads before approving the change."""
    lines = ["Luke Lab single-user migration", ""]
    for entry in report.get("database", {}).get("tables", []):
        lines.append(
            f"  {entry['table']:<20} {entry['total']:>5} rows, {entry['outside']:>5} to remap"
        )
    faqs = report.get("database", {}).get("faqs", [])
    users = report.get("database", {}).get("users", {})
    lines.append(f"  FAQ groups merged: {len(faqs)}")
    lines.append(f"  Users removed: {len(users.get('drop', []))}")
    for entry in report.get("vectors", {}).get("collections", []):
        lines.append(
            f"  {entry['collection']:<20} {entry['total']:>5} vectors, {entry['stale']:>5} to label"
        )
    for entry in report.get("uploads", {}).get("moves", []):
        lines.append(f"  upload {entry['source']}: {entry['action']}")
    lines.append("")
    lines.append(f"  backup: {report.get('backup', {}).get('root', '')}")
    return "\n".join(lines)
