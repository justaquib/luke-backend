"""Luke 0.0.1 and the checkpoint it was built on, measured from the files.

The identity is not a label typed into a config: the parameter count comes from
the safetensors header, the weight hash from the file itself, and the context
limit from the checkpoint's own config. When the files and the record disagree,
the model refuses to load instead of answering with an identity it cannot back.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import settings

MODEL_NAME = "Luke"
MODEL_VERSION = "0.0.1"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_SLUG = "Qwen2.5-0.5B-Instruct"
PINNED_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
LICENSE = "Apache-2.0"
WEIGHT_FILE = "model.safetensors"
IDENTITY_FILE = "identity.json"
CHUNK = 1024 * 1024


class FoundationError(Exception):
    """The foundation cannot be used, with a reason the trainer can act on."""


class FoundationMissingError(FoundationError):
    """The checkpoint was never installed."""


class FoundationCorruptError(FoundationError):
    """The checkpoint on disk does not match its recorded identity."""


@dataclass(frozen=True)
class FoundationIdentity:
    """What the loaded checkpoint really is."""

    name: str
    version: str
    base_model: str
    revision: str
    license: str
    parameters: int
    context_limit: int
    weight_sha256: str
    weight_bytes: int
    dtypes: tuple[str, ...]
    folder: Path

    @property
    def display(self) -> str:
        """The one line the product shows for the loaded model."""
        base = self.base_model.rsplit("/", 1)[-1]
        return f"{self.name} {self.version} · {base} · revision {self.revision[:7]}"

    @property
    def weight_dtype(self) -> str:
        """The dtype the weights were saved in, as a plain word."""
        return self.dtypes[0].casefold() if self.dtypes else "unknown"

    def as_dict(self) -> dict[str, Any]:
        """The identity as the API reports it."""
        return {
            "name": self.name,
            "version": self.version,
            "base_model": self.base_model,
            "revision": self.revision,
            "license": self.license,
            "parameters": self.parameters,
            "context_limit": self.context_limit,
            "weight_sha256": self.weight_sha256,
            "weight_bytes": self.weight_bytes,
            "dtypes": list(self.dtypes),
            "display": self.display,
        }


def install_dir() -> Path:
    """Where the pinned checkpoint is expected to live."""
    return Path(settings.foundation_model_dir) / BASE_SLUG / PINNED_REVISION


def sha256_file(path: Path) -> str:
    """The SHA-256 of a file, read in chunks so a large weight file fits memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def weight_parameters(path: Path) -> int:
    """The parameter count, summed from the safetensors header."""
    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_size))
    total = 0
    for name, row in header.items():
        if name == "__metadata__":
            continue
        count = 1
        for size in row["shape"]:
            count *= int(size)
        total += count
    return total


def load_identity(
    folder: Path | None = None,
    verify_weights: bool | None = None,
) -> FoundationIdentity:
    """The verified identity of the installed checkpoint."""
    target = folder or install_dir()
    verify = settings.foundation_verify_weights if verify_weights is None else verify_weights
    return _verified(str(target), verify)


def inspect_identity(folder: Path) -> FoundationIdentity:
    """Verify a checkpoint folder without caching it, for a staged package.

    A staged folder is temporary, so caching it would evict the installed
    checkpoint's verified identity and make the next answer hash the weights again.
    """
    return _verified.__wrapped__(str(folder), True)


@lru_cache(maxsize=4)
def _verified(folder: str, verify_weights: bool) -> FoundationIdentity:
    """Read and check the identity once per path and verification setting."""
    target = Path(folder)
    record_path = target / IDENTITY_FILE
    weights = target / WEIGHT_FILE
    if not target.is_dir() or not record_path.is_file() or not weights.is_file():
        raise FoundationMissingError(
            "Luke's foundation checkpoint is not installed. Run "
            "`python scripts/install_foundation.py` in assistance_back, then restart the API."
        )
    recorded = _read_json(record_path, "identity record")
    config = _read_json(target / "config.json", "checkpoint config")
    if recorded.get("base_model") != BASE_MODEL or recorded.get("revision") != PINNED_REVISION:
        raise FoundationCorruptError(
            "The installed checkpoint is not the pinned Luke 0.0.1 base model."
        )
    size = weights.stat().st_size
    if int(recorded.get("weight_bytes") or 0) != size:
        raise FoundationCorruptError(
            "The weight file size does not match the recorded identity; reinstall the checkpoint."
        )
    parameters = weight_parameters(weights)
    if int(recorded.get("parameters") or 0) != parameters:
        raise FoundationCorruptError(
            "The parameter count does not match the recorded identity; reinstall the checkpoint."
        )
    context_limit = int(config.get("max_position_embeddings") or 0)
    if context_limit != int(recorded.get("context_limit") or 0):
        raise FoundationCorruptError(
            "The checkpoint context limit does not match the recorded identity."
        )
    digest = str(recorded.get("weight_sha256") or "")
    if verify_weights and sha256_file(weights) != digest:
        raise FoundationCorruptError(
            "The weight file hash does not match the recorded identity; reinstall the checkpoint."
        )
    return FoundationIdentity(
        name=str(recorded.get("name") or MODEL_NAME),
        version=str(recorded.get("version") or MODEL_VERSION),
        base_model=str(recorded["base_model"]),
        revision=str(recorded["revision"]),
        license=str(recorded.get("license") or LICENSE),
        parameters=parameters,
        context_limit=context_limit,
        weight_sha256=digest,
        weight_bytes=size,
        dtypes=tuple(str(item) for item in recorded.get("dtypes") or ()),
        folder=target,
    )


def status() -> dict[str, Any]:
    """Whether Luke's foundation is installed, and the measured identity."""
    try:
        identity = load_identity()
    except FoundationError as error:
        return {"installed": False, "identity": None, "error": str(error)}
    return {"installed": True, "identity": identity.as_dict(), "error": None}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    """One JSON file, with a corrupt-file reason that names the file."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FoundationCorruptError(f"The {label} at {path} cannot be read.") from error
    if not isinstance(value, dict):
        raise FoundationCorruptError(f"The {label} at {path} is not a JSON object.")
    return value
