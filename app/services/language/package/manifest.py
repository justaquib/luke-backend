"""The manifest that says what a package is and proves every file in it.

The manifest is the package's identity: the format and schema it was written
by, the workspace it belongs to, the checkpoint and tokenizer it carries, the
embedding the index was built with, the collection settings, the counts of the
material, and the size and SHA-256 of every other file. An import checks the
manifest before it touches the database, so a damaged or foreign package is
refused instead of half-restored.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.chroma import COLLECTION_CONFIG
from app.services.neural.tokenizer import artifacts
from app.services.neural.tokenizer.base import META_FILE, TOKENIZER_FILE

FORMAT = "luke-package"
FORMAT_VERSION = 2
MANIFEST_NAME = "manifest.json"
README_NAME = "README.md"
CHECKPOINT_SCHEMA = 3
TOKENIZER_SCHEMA = 1
MODE_FULL = "full"
MODE_SLIM = "slim"
MODES = (MODE_FULL, MODE_SLIM)
# The embedding the collections are written with. A restored index is only
# comparable when it was built in the same space, so the package names it.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_PROVIDER = "chroma-onnx"
EMBEDDING_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
# A package keeps at most this many training materials, so one download stays
# reviewable. Every database row is kept; only the derived material list is cut.
MAX_MATERIAL_LINES = 50_000
CHUNK = 1024 * 1024

_REQUIRED = (
    "format",
    "format_version",
    "created",
    "workspace",
    "mode",
    "model",
    "checkpoint",
    "base_model",
    "embedding",
    "collections",
    "tokenizers",
    "revisions",
    "licence",
    "counts",
    "files",
    "excluded",
)


class PackageError(Exception):
    """A package cannot be built, read or restored, with a reason to act on."""


class PackageIntegrityError(PackageError):
    """A file in the package does not match the manifest."""


@dataclass(frozen=True)
class PackageFacts:
    """Everything the manifest records besides its own file list."""

    workspace: int
    mode: str
    model: dict[str, Any]
    checkpoint: dict[str, Any] | None
    base_model: dict[str, Any]
    tokenizers: list[dict[str, Any]]
    counts: dict[str, int]
    files: list[dict[str, Any]]
    excluded: list[str]


def sha256_bytes(data: bytes) -> str:
    """The SHA-256 of a block already in memory."""
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    """The SHA-256 of a file, read in blocks so a weight file fits memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def iso_now() -> str:
    """The moment the package was written, in UTC."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def root_name(workspace: int) -> str:
    """The one folder every package file lives under."""
    return f"luke-{workspace}"


def safe_member(name: str) -> bool:
    """Whether a member name is a plain relative path with no way out."""
    if not isinstance(name, str) or not name or name.startswith("/"):
        return False
    if "\\" in name or "\x00" in name:
        return False
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    return ":" not in parts[0]


def embedding_record() -> dict[str, Any]:
    """The embedding model a restored index must be built with."""
    return {
        "name": EMBEDDING_MODEL,
        "provider": EMBEDDING_PROVIDER,
        "sha256": EMBEDDING_SHA256,
    }


def collections_record() -> dict[str, Any]:
    """The collection names and the index settings the vectors were built with."""
    return {
        "names": {
            "documents": settings.chroma_collection,
            "faqs": settings.chroma_faq_collection,
            "intents": settings.chroma_intents_collection,
        },
        "config": dict(COLLECTION_CONFIG),
    }


def revisions_record(base_revision: str) -> dict[str, Any]:
    """The formats this package was written by, so an older one is refused."""
    return {
        "package_format": FORMAT_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "tokenizer_schema": TOKENIZER_SCHEMA,
        "base_revision": base_revision,
    }


def tokenizer_record(artifact_id: str) -> dict[str, Any]:
    """One tokenizer artifact, read from its folder and its own meta."""
    folder = artifacts.folder(artifact_id)
    try:
        stored = artifacts.meta(artifact_id)
    except artifacts.MissingArtifactError as error:
        raise PackageError(str(error)) from error
    files = []
    for name in (TOKENIZER_FILE, META_FILE):
        path = folder / name
        if path.is_file():
            files.append(
                {
                    "name": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_path(path),
                }
            )
    return {
        "kind": str(stored.get("kind") or ""),
        "id": artifact_id,
        "vocab_size": int(stored.get("vocab_size") or 0),
        "licence": str(stored.get("licence") or ""),
        "implementation": str(stored.get("implementation") or ""),
        "files": files,
    }


def manifest(facts: PackageFacts) -> dict[str, Any]:
    """The whole manifest, assembled from the facts and the file records."""
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "created": iso_now(),
        "workspace": facts.workspace,
        "mode": facts.mode,
        "model": dict(facts.model),
        "checkpoint": facts.checkpoint,
        "base_model": dict(facts.base_model),
        "embedding": embedding_record(),
        "collections": collections_record(),
        "tokenizers": list(facts.tokenizers),
        "revisions": revisions_record(str(facts.base_model.get("revision") or "")),
        "licence": str(facts.base_model.get("license") or "Apache-2.0"),
        "counts": dict(facts.counts),
        "files": list(facts.files),
        "excluded": list(facts.excluded),
    }


def validate(payload: Any, workspace: int) -> None:
    """Refuse a manifest that is not this format, workspace or shape."""
    if not isinstance(payload, dict):
        raise PackageError("The package manifest is not a JSON object.")
    missing = [key for key in _REQUIRED if key not in payload]
    if missing:
        raise PackageError(f"The package manifest is missing {', '.join(missing)}.")
    if payload["format"] != FORMAT:
        raise PackageError(f"The package is not a Luke package ({payload['format']}).")
    if payload["format_version"] != FORMAT_VERSION:
        raise PackageError(
            f"The package format is {payload['format_version']}; this build reads "
            f"format {FORMAT_VERSION}. Export it again from the same version."
        )
    if payload["mode"] not in MODES:
        raise PackageError(f"The package mode '{payload['mode']}' is not known.")
    if type(payload["workspace"]) is not int or payload["workspace"] != workspace:
        raise PackageError(
            f"The package belongs to workspace {payload['workspace']}, "
            f"not this workspace ({workspace}). A package cannot change workspace."
        )
    _validate_sections(payload)
    _validate_files(payload["files"])


def _validate_sections(payload: dict[str, Any]) -> None:
    _validate_objects(payload)
    _validate_revisions(payload["revisions"])
    _validate_counts(payload["counts"])
    if payload["collections"].get("config") != COLLECTION_CONFIG:
        raise PackageError("The package collection configuration is not supported by this build.")
    if payload["embedding"] != embedding_record():
        raise PackageError("The package embedding model does not match this build.")


def _validate_objects(payload: dict[str, Any]) -> None:
    for section in ("model", "base_model", "embedding", "collections", "counts", "revisions"):
        if not isinstance(payload[section], dict):
            raise PackageError(f"The package manifest has no readable {section}.")
    if payload["checkpoint"] is not None and not isinstance(payload["checkpoint"], dict):
        raise PackageError("The package checkpoint record is not an object.")
    if not isinstance(payload["tokenizers"], list):
        raise PackageError("The package manifest has no tokenizer list.")
    for record in payload["tokenizers"]:
        if not isinstance(record, dict):
            raise PackageError("The package has an invalid tokenizer record.")


def _validate_revisions(revisions: dict[str, Any]) -> None:
    for key in ("checkpoint_schema", "tokenizer_schema", "package_format"):
        if type(revisions.get(key)) is not int:
            raise PackageError(f"The package has an invalid {key} revision.")


def _validate_counts(counts: dict[str, Any]) -> None:
    for key in ("documents", "faqs", "pending_total", "feedback", "uploads", "intents"):
        value = counts.get(key)
        if type(value) is not int or value < 0:
            raise PackageError(f"The package has an invalid {key} count.")


def _validate_files(files: Any) -> None:
    """Every file record is a safe relative path with a size and a hash."""
    if not isinstance(files, list) or not files:
        raise PackageError("The package manifest lists no files.")
    seen: set[str] = set()
    for record in files:
        if not isinstance(record, dict):
            raise PackageError("A file record in the manifest is not an object.")
        name = str(record.get("path") or "")
        if not safe_member(name):
            raise PackageError(f"The package names an unsafe file path: {name!r}.")
        if name in seen:
            raise PackageError(f"The package lists {name} twice.")
        seen.add(name)
        if type(record.get("bytes")) is not int or record["bytes"] < 0:
            raise PackageError(f"The package gives {name} no size.")
        digest = record.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise PackageError(f"The package gives {name} no SHA-256.")
