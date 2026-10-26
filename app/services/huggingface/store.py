"""Where the fine-tuned teachers and their run records are kept.

A teacher lives beside the tenant's own learner files, so the weights and the
material they were trained from stay together. The last run and its progress
are plain JSON, which the Base model tab reads while the run is still going.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.services.brain_paths import brain_dir, tenant_key


def teacher_folder(model_id: str) -> str:
    """The folder name of a fine-tuned teacher, safe on one path part."""
    return "teacher-" + model_id.replace("/", "-").replace(".", "-")


def teachers_root(tenant_id: int | None) -> Path:
    """The folder every teacher of this tenant is written under."""
    return brain_dir() / "huggingface" / f"tenant-{tenant_key(tenant_id)}"


def teacher_path(tenant_id: int | None, folder: str) -> Path:
    """One teacher's folder."""
    return teachers_root(tenant_id) / folder


def last_run_path(tenant_id: int | None) -> Path:
    """The record of the last run of this tenant."""
    return teachers_root(tenant_id) / "last-run.json"


def progress_path(tenant_id: int | None) -> Path:
    """The live progress of the run of this tenant."""
    return brain_dir() / f"hf-progress-{tenant_key(tenant_id)}.json"


def read_json(path: Path) -> dict[str, Any]:
    """A JSON record, or nothing when it has not been written yet."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a record a reader can always parse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def read_last_run(tenant_id: int | None) -> dict[str, Any]:
    """The numbers the last run measured."""
    return read_json(last_run_path(tenant_id))


def read_progress(tenant_id: int | None) -> dict[str, Any]:
    """The stage and step count of the run that is going, or the last one."""
    return read_json(progress_path(tenant_id))


def size_of(path: Path) -> int:
    """The bytes the teacher's files take on disk."""
    try:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    except OSError:
        return 0


def trained_at(path: Path) -> str | None:
    """When the weights were last written."""
    try:
        stamp = (path / "model.safetensors").stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(stamp, tz=UTC).isoformat()


def list_teachers(tenant_id: int | None) -> list[dict[str, Any]]:
    """Every fine-tuned teacher this tenant has, newest first."""
    root = teachers_root(tenant_id)
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for folder in root.iterdir():
        if not folder.is_dir() or not folder.name.startswith("teacher-"):
            continue
        if not (folder / "model.safetensors").is_file():
            continue
        record = read_json(folder / "training.json")
        rows.append(
            {
                "folder": folder.name,
                "model_id": str(record.get("model_id") or ""),
                "name": str(record.get("name") or folder.name.removeprefix("teacher-")),
                "steps": record.get("steps"),
                "loss": record.get("loss"),
                "trained_at": trained_at(folder),
                "bytes": size_of(folder),
                "ready": True,
            }
        )
    return sorted(rows, key=lambda row: str(row.get("trained_at") or ""), reverse=True)


def remove_teacher(tenant_id: int | None, folder: str) -> bool:
    """Remove one teacher's weights; false when it is not there."""
    if not folder.startswith("teacher-") or "/" in folder or "\\" in folder:
        return False
    path = teacher_path(tenant_id, folder)
    if not path.is_dir():
        return False
    shutil.rmtree(path)
    return True


def now() -> float:
    """The clock the run records use."""
    return time.time()
