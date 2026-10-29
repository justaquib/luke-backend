"""Read a package without trusting it: bounded upload, safe names, checked files.

Nothing in the zip is trusted until the manifest is read and every other
member is matched against it. A name that could leave the extraction folder, a
symlink, an encrypted member, an entry count or an expanded size beyond the
limits, an extra file or a missing one is refused before the database or the
learner folder is touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

from app.services.language.package.manifest import (
    MANIFEST_NAME,
    PackageError,
    PackageIntegrityError,
    root_name,
    safe_member,
    validate,
)

BLOCK = 1024 * 1024
MAX_ENTRIES = 100_000
MAX_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024


def receive(source: BinaryIO, limit: int) -> Path:
    """Copy an upload to a temporary file, stopping at the size limit."""
    handle, name = tempfile.mkstemp(prefix="luke-import-", suffix=".zip")
    path = Path(name)
    size = 0
    complete = False
    try:
        with os.fdopen(handle, "wb") as target:
            for block in iter(lambda: source.read(BLOCK), b""):
                size += len(block)
                if size > limit:
                    raise PackageError(f"The package is larger than the {limit} byte import limit.")
                target.write(block)
        complete = True
    finally:
        if not complete:
            path.unlink(missing_ok=True)
    return path


def open_package(path: Path, workspace: int) -> tuple[zipfile.ZipFile, str, dict[str, Any]]:
    """Open the zip, find its one manifest and check the archive guards."""
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as error:
        raise PackageError("The uploaded file is not a readable zip package.") from error
    try:
        root = _inspect(archive)
        if root != root_name(workspace):
            raise PackageError(
                f"The package is rooted at {root}, not this workspace ({root_name(workspace)})."
            )
        payload = _manifest(archive, root)
        validate(payload, workspace)
        return archive, root, payload
    except PackageError:
        archive.close()
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        archive.close()
        raise PackageError(f"The package could not be read: {error}") from error


def _inspect(archive: zipfile.ZipFile) -> str:
    """The one package folder, after checking every member name and size."""
    infos = archive.infolist()
    if len(infos) > MAX_ENTRIES:
        raise PackageError(f"The package holds more than {MAX_ENTRIES} entries.")
    total = 0
    roots: set[str] = set()
    names: set[str] = set()
    for info in infos:
        name = info.filename
        if name in names:
            raise PackageError(f"The package contains a duplicate entry: {name!r}.")
        names.add(name)
        _guard_entry(info)
        if info.is_dir():
            continue
        roots.add(name.split("/", 1)[0])
        total += info.file_size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise PackageError(
                f"The package expands beyond {MAX_UNCOMPRESSED_BYTES} bytes; refusing it."
            )
    if len(roots) != 1:
        raise PackageError("The package must hold exactly one top-level folder.")
    return roots.pop()


def _guard_entry(info: zipfile.ZipInfo) -> None:
    name = info.filename
    if not safe_member(name.removesuffix("/") if info.is_dir() else name):
        raise PackageError(f"The package names an unsafe file path: {name!r}.")
    if name.endswith((".json", ".jsonl")) and info.file_size > MAX_JSON_BYTES:
        raise PackageError(f"The JSON entry {name!r} exceeds the {MAX_JSON_BYTES} byte limit.")
    if info.flag_bits & 0x1:
        raise PackageError("The package is encrypted; plain zip packages only.")
    if (info.external_attr >> 16) & 0o170000 == 0o120000:
        raise PackageError(f"The package contains a symlink: {name!r}.")


def _manifest(archive: zipfile.ZipFile, root: str) -> dict[str, Any]:
    name = f"{root}/{MANIFEST_NAME}"
    if name not in archive.namelist():
        raise PackageError("The package has no manifest.json at its root.")
    try:
        payload = json.loads(archive.read(name))
    except (KeyError, ValueError) as error:
        raise PackageError("The package manifest is not readable JSON.") from error
    return payload


def verify(archive: zipfile.ZipFile, root: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Check every listed file against the zip, and refuse anything unlisted."""
    listed = {f"{root}/{record['path']}": record for record in payload["files"]}
    listed[f"{root}/{MANIFEST_NAME}"] = {}
    actual = {info.filename for info in archive.infolist() if not info.is_dir()}
    missing = sorted(set(listed) - actual)
    extra = sorted(actual - set(listed))
    if missing:
        raise PackageIntegrityError(
            "The package is missing " + ", ".join(record.split("/", 1)[1] for record in missing[:5])
        )
    if extra:
        raise PackageIntegrityError(
            "The package holds files the manifest does not list: " + ", ".join(extra[:5])
        )
    for name, record in listed.items():
        if not record:
            continue
        info = archive.getinfo(name)
        if info.file_size != record["bytes"]:
            raise PackageIntegrityError(f"{record['path']} is not the size the manifest records.")
        if _hash(archive, info) != record["sha256"]:
            raise PackageIntegrityError(f"{record['path']} does not match its SHA-256.")
    return [record for record in payload["files"]]


def _hash(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(info) as handle:
        for block in iter(lambda: handle.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(archive: zipfile.ZipFile, root: str, destination: Path) -> None:
    """Write the verified members under one fresh folder."""
    destination.mkdir(parents=True, exist_ok=True)
    for info in archive.infolist():
        if info.is_dir():
            continue
        target = destination / info.filename
        if destination.resolve() not in target.resolve().parents:
            raise PackageError(f"The package entry {info.filename!r} leaves the staging folder.")
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target.open("wb") as sink:
            for block in iter(lambda: source.read(BLOCK), b""):
                sink.write(block)


def discard(path: Path) -> None:
    """Remove the temporary upload, whatever happened to it."""
    path.unlink(missing_ok=True)
