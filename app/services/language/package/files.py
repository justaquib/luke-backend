"""Stage, back up and restore the files owned by a package."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from app.config import settings
from app.services import brain_paths
from app.services.foundation import identity
from app.services.language.package.contents import STATE_FILES
from app.services.language.package.manifest import PackageError, root_name, safe_member, sha256_path
from app.services.neural.tokenizer.base import META_FILE, TOKENIZER_FILE

STATE_NAME = re.compile(r"^(?:tenant|model|level|progress|runs|expressions|feedback)-(\d+)\.json$")
CHECKPOINT_NAME = re.compile(r"^(?:words-(\d+)\.pt|brain-(\d+)-[A-Za-z0-9_.-]+\.pt)$")
ARTIFACT_ID = re.compile(r"^[0-9a-f]{64}$")
BASE_FILES = frozenset(
    {
        "LICENSE",
        "config.json",
        "generation_config.json",
        "identity.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "model.safetensors",
    }
)
DATASET_FILES = frozenset(
    f"{name}.jsonl"
    for name in (
        "documents",
        "faqs",
        "pending",
        "feedback",
        "uploads",
        "intents",
        "materials",
        "lessons",
        "passages",
    )
)


def validate(records: list[dict[str, Any]], key: int) -> None:
    """Reject unknown files and symlink destinations before any replacement."""
    for record in records:
        name = record["path"]
        if _destination(name, key) is not None:
            _target(name, key)
    for path in _current(key):
        _no_symlinks(path)


def backup(tenant_id: int | None, destination: Path, records: list[dict[str, Any]]) -> Path:
    """Snapshot replaced files, including global tokenizer and foundation files."""
    key = brain_paths.tenant_key(tenant_id)
    targets = set(_current(key))
    for record in records:
        if _destination(record["path"], key) is None:
            continue
        target = _target(record["path"], key)
        if not _matches(target, record):
            targets.add(target)
    destination.mkdir(parents=True, exist_ok=True)
    inventory = []
    for index, target in enumerate(sorted(targets)):
        existed = target.is_file()
        if existed:
            _copy(target, destination / str(index))
        inventory.append({"target": str(target), "existed": existed, "backup": str(index)})
    (destination / "inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    return destination


def apply(staging: Path, tenant_id: int | None, records: list[dict[str, Any]]) -> dict[str, int]:
    """Replace validated workspace files, retaining identical shared artifacts."""
    key = brain_paths.tenant_key(tenant_id)
    validate(records, key)
    for path in _current(key):
        path.unlink()
    counts = {"state": 0, "checkpoints": 0, "tokenizers": 0, "base": 0, "raw": 0}
    root = staging / root_name(key)
    for record in records:
        name = record["path"]
        kind = _destination(name, key)
        if kind is None:
            continue
        target = _target(name, key)
        if not _matches(target, record):
            _copy(root / name, target)
        counts[kind] += 1
    return counts


def rollback(backup_dir: Path, tenant_id: int | None) -> None:
    """Restore every changed file, or remove it if it did not previously exist."""
    inventory = json.loads((backup_dir / "inventory.json").read_text(encoding="utf-8"))
    for entry in inventory:
        target = Path(entry["target"])
        if entry["existed"]:
            _copy(backup_dir / entry["backup"], target)
        else:
            target.unlink(missing_ok=True)


def cleanup(folder: Path) -> None:
    """Remove an import's temporary staging files."""
    shutil.rmtree(folder)


def _current(key: int) -> list[Path]:
    root = brain_paths.brain_dir()
    paths = [root / f"words-{key}.pt", *root.glob(f"brain-{key}-*.pt")]
    paths.extend(root / name.format(key=key) for name in STATE_FILES)
    uploads = Path(settings.upload_dir) / str(key)
    if uploads.is_dir():
        paths.extend(uploads.iterdir())
    return [path for path in paths if path.is_file() or path.is_symlink()]


def _matches(target: Path, record: dict[str, Any]) -> bool:
    return (
        target.is_file()
        and target.stat().st_size == record["bytes"]
        and sha256_path(target) == record["sha256"]
    )


def _destination(name: str, key: int) -> str | None:
    if not safe_member(name):
        raise PackageError(f"The package names an unsafe file path: {name!r}.")
    parts = name.split("/")
    if parts[0] == "state" and len(parts) == 2:
        _check_state(parts[1], key)
        return "state"
    if parts[:2] == ["weights", "checkpoints"] and len(parts) == 3:
        _check_checkpoint(parts[2], key)
        return "checkpoints"
    if parts[:2] == ["weights", "tokenizers"] and len(parts) == 4:
        _check_tokenizer(parts[2], parts[3])
        return "tokenizers"
    if parts[:2] == ["weights", "base"] and len(parts) == 3 and parts[2] in BASE_FILES:
        return "base"
    if parts[0] == "raw" and len(parts) == 2:
        return "raw"
    if name == "README.md" or (
        len(parts) == 2 and parts[0] == "dataset" and parts[1] in DATASET_FILES
    ):
        return None
    raise PackageError(f"The package carries an unexpected file: {name}.")


def _target(name: str, key: int) -> Path:
    parts = name.split("/")
    root = brain_paths.brain_dir()
    if parts[0] == "state":
        target = root / parts[1]
    elif parts[:2] == ["weights", "checkpoints"]:
        target = root / parts[2]
    elif parts[:2] == ["weights", "tokenizers"]:
        target = root / "tokenizers" / parts[2] / parts[3]
    elif parts[:2] == ["weights", "base"]:
        target = identity.install_dir() / parts[2]
    else:
        target = Path(settings.upload_dir) / str(key) / parts[1]
    _no_symlinks(target)
    return target


def _no_symlinks(target: Path) -> None:
    for path in (target, *target.parents):
        if path.is_symlink():
            raise PackageError(f"The restore destination contains a symlink: {path.name}.")


def _check_state(name: str, key: int) -> None:
    match = STATE_NAME.fullmatch(name)
    if not match or int(match.group(1)) != key:
        raise PackageError(f"The package carries another workspace's state file: {name}.")


def _check_checkpoint(name: str, key: int) -> None:
    match = CHECKPOINT_NAME.fullmatch(name)
    keys = [value for value in match.groups() if value] if match else []
    if not keys or any(int(value) != key for value in keys):
        raise PackageError(f"The package carries another workspace's checkpoint: {name}.")


def _check_tokenizer(artifact_id: str, name: str) -> None:
    if not ARTIFACT_ID.fullmatch(artifact_id) or name not in {TOKENIZER_FILE, META_FILE}:
        raise PackageError(
            f"The package carries an unexpected tokenizer file: {artifact_id}/{name}."
        )


def _copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise PackageError(f"The package is missing {source.name}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
