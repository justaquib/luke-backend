"""Install the pinned foundation checkpoint Luke 0.0.1 is built on.

The checkpoint is downloaded once at an exact revision into a local cache that
is never packaged, merged or fine-tuned in place. The installer measures the
files it wrote: the parameter count and dtypes are read from the safetensors
header, and every file gets a size and a SHA-256 in ``identity.json``.

Run from ``assistance_back/``:

    .venv/bin/python scripts/install_foundation.py
    .venv/bin/python scripts/install_foundation.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SLUG = "Qwen2.5-0.5B-Instruct"
LICENSE = "Apache-2.0"
FILES = (
    "LICENSE",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
WEIGHTS = "model.safetensors"
README = "identity.json"
ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / ".models" / "foundation" / SLUG / REVISION
CHUNK = 1024 * 1024


class InstallError(Exception):
    """The checkpoint is missing or does not match its recorded identity."""


def sha256_file(path: Path) -> str:
    """The SHA-256 of a file, read in chunks so a large weight file fits memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def weight_facts(path: Path) -> tuple[int, list[str]]:
    """The parameter count and dtypes, read from the safetensors header."""
    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_size))
    tensors = {name: row for name, row in header.items() if name != "__metadata__"}
    parameters = 0
    for row in tensors.values():
        count = 1
        for size in row["shape"]:
            count *= int(size)
        parameters += count
    dtypes = sorted({str(row["dtype"]) for row in tensors.values()})
    return parameters, dtypes


def download(target: Path) -> None:
    """Download the allowlisted files at the pinned revision."""
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=target,
        allow_patterns=list(FILES),
        local_dir_use_symlinks=False,
    )


def identity(target: Path) -> dict[str, Any]:
    """The measured identity of the checkpoint on disk."""
    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    weights = target / WEIGHTS
    if not weights.is_file():
        raise InstallError(f"{weights} is missing; run the installer without --check.")
    parameters, dtypes = weight_facts(weights)
    files = []
    for name in sorted(path.name for path in target.glob("*") if path.is_file()):
        if name == README:
            continue
        path = target / name
        files.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "name": "Luke",
        "version": "0.0.1",
        "base_model": REPO_ID,
        "revision": REVISION,
        "license": LICENSE,
        "parameters": parameters,
        "dtypes": dtypes,
        "context_limit": int(config.get("max_position_embeddings") or 0),
        "architectures": list(config.get("architectures") or []),
        "weight_sha256": sha256_file(weights),
        "weight_bytes": weights.stat().st_size,
        "files": files,
    }


def write_identity(target: Path, measured: dict[str, Any]) -> Path:
    """Keep the identity beside the checkpoint, written in one step."""
    path = target / README
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(measured, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def check(target: Path) -> dict[str, Any]:
    """Verify the recorded identity against the files that are there now."""
    path = target / README
    if not path.is_file():
        raise InstallError(f"{path} is missing; install the checkpoint first.")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    measured = identity(target)
    for key in ("base_model", "revision", "parameters", "weight_sha256", "weight_bytes"):
        if recorded.get(key) != measured.get(key):
            raise InstallError(f"{key} does not match the checkpoint on disk.")
    recorded_files = {row["name"]: row for row in recorded.get("files", [])}
    measured_files = {row["name"]: row for row in measured["files"]}
    if recorded_files != measured_files:
        raise InstallError("A checkpoint file was added, removed or changed.")
    return measured


def main(argv: list[str] | None = None) -> int:
    """Install the checkpoint, or check the installed one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without downloading")
    args = parser.parse_args(argv)
    try:
        if args.check:
            measured = check(TARGET)
        else:
            download(TARGET)
            measured = identity(TARGET)
            write_identity(TARGET, measured)
    except InstallError as error:
        print(f"Foundation checkpoint: {error}", file=sys.stderr)
        return 1
    print(f"Foundation checkpoint: {TARGET}")
    print(f"  base model   {measured['base_model']}")
    print(f"  revision     {measured['revision']}")
    print(f"  parameters   {measured['parameters']:,}")
    print(f"  dtypes       {', '.join(measured['dtypes'])}")
    print(f"  context      {measured['context_limit']:,} tokens")
    print(f"  weights      {measured['weight_bytes']:,} bytes")
    print(f"  sha256       {measured['weight_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
