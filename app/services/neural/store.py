"""Where the neural word model of a tenant is kept, and the runs that trained it."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.services.brain_paths import brain_dir, tenant_key
from app.services.model_cache import file_size

WORDS_FILE = "words-{key}.pt"
RUNS_FILE = "runs-{key}.json"

# How many runs the builder can read back.
MAX_RUNS = 5


def words_path(tenant_id: int | None) -> Path:
    """Where the word model is kept."""
    return brain_dir() / WORDS_FILE.format(key=tenant_key(tenant_id))


def runs_path(tenant_id: int | None) -> Path:
    """Where the summaries of the last trainings are kept."""
    return brain_dir() / RUNS_FILE.format(key=tenant_key(tenant_id))


def mtime(path: Path) -> float | None:
    """When a file last changed, or None when it is not there."""
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def stamp(tenant_id: int | None) -> float | None:
    """The word model's change time, so the engine cache can follow it."""
    return mtime(words_path(tenant_id))


def words_size(tenant_id: int | None) -> int:
    """How many bytes the word model takes, so the cache can hold its cap."""
    return file_size(words_path(tenant_id))


def load_runs(tenant_id: int | None) -> list[dict[str, Any]]:
    """The summaries of the last trainings, newest first."""
    path = runs_path(tenant_id)
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    if not isinstance(stored, list):
        return []
    return [run for run in stored if isinstance(run, dict)][:MAX_RUNS]


def append_run(tenant_id: int | None, run: dict[str, Any]) -> None:
    """Keep the newest run at the top and never more than ``MAX_RUNS``."""
    entry = {"at": time.time(), **run}
    runs = [entry, *load_runs(tenant_id)][:MAX_RUNS]
    path = runs_path(tenant_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        return
