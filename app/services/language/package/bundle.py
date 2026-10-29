"""Build the downloadable package, or count it before it is built.

A package carries the weights the model answers with, every tokenizer artifact
those weights name, the learner state, the database rows and the raw uploads.
A full package also carries the pinned foundation checkpoint; a slim package
records its name, revision and hash so the same checkpoint can be installed
separately. Every entry is listed with its size and SHA-256, and the manifest
is written last so it describes the whole package.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.services import model_settings
from app.services.huggingface import store as teacher_store
from app.services.language.package import contents, dataset
from app.services.language.package import manifest as manifest_module
from app.services.language.package.contents import Entry
from app.services.language.package.manifest import (
    MANIFEST_NAME,
    MODE_FULL,
    MODE_SLIM,
    MODES,
    README_NAME,
    PackageError,
    PackageFacts,
)
from app.services.neural import checkpoint

logger = logging.getLogger(__name__)

# A weight file is already incompressible, so compressing it only costs time.
STORE_THRESHOLD = 16 * 1024 * 1024
BLOCK = 1024 * 1024
EXCLUDED = (
    "Hugging Face teacher weights",
    "the retired writer weights (transformer-<workspace>.pt)",
    "generated answers and chat history",
    "the vector index, which is rebuilt from the database rows on import",
)


@dataclass(frozen=True)
class Plan:
    """Everything one package carries, before it is written."""

    workspace: int
    mode: str
    model: dict[str, Any]
    checkpoints: list[Entry]
    checkpoint_records: list[dict[str, Any]]
    base_model: dict[str, Any]
    base_entries: list[Entry]
    tokenizers: list[Entry]
    tokenizer_records: list[dict[str, Any]]
    state: list[Entry]
    raw: list[Entry]
    dataset: dict[str, bytes]
    counts: dict[str, int]
    material_count: int
    by_kind: dict[str, int]


def preview(tenant_id: int | None, db: Session) -> dict[str, Any]:
    """What the package would hold, in both modes, without writing it."""
    plan = _plan(tenant_id, db, MODE_FULL)
    slim = _total(plan, MODE_SLIM)
    full = _total(plan, MODE_FULL)
    settings = model_settings.read(tenant_id)
    return {
        "model": plan.model,
        "available": plan.material_count,
        "by_kind": plan.by_kind,
        "checkpoints": [
            {"name": record["name"], "bytes": record["bytes"]} for record in plan.checkpoint_records
        ],
        "teachers": _teachers(tenant_id),
        "pending": plan.counts["pending"],
        "bytes": full["bytes"],
        "filename": _filename(tenant_id, settings.name, MODE_FULL),
        "modes": {
            MODE_FULL: {**full, "filename": _filename(tenant_id, settings.name, MODE_FULL)},
            MODE_SLIM: {**slim, "filename": _filename(tenant_id, settings.name, MODE_SLIM)},
        },
        "base": plan.base_model,
        "tokenizers": [
            {
                "id": record["id"],
                "kind": record["kind"],
                "vocab_size": record["vocab_size"],
                "bytes": sum(item["bytes"] for item in record["files"]),
            }
            for record in plan.tokenizer_records
        ],
        "embedding": manifest_module.embedding_record(),
        "collections": manifest_module.collections_record(),
        "counts": plan.counts,
    }


def build(tenant_id: int | None, db: Session, mode: str = MODE_FULL) -> tuple[Path, str]:
    """Write the package to a temporary zip and return it with its file name."""
    if mode not in MODES:
        raise PackageError(f"Unknown package mode '{mode}'; choose full or slim.")
    plan = _plan(tenant_id, db, mode)
    settings = model_settings.read(tenant_id)
    filename = _filename(tenant_id, settings.name, mode)
    root = manifest_module.root_name(plan.workspace)
    files: list[dict[str, Any]] = []
    handle, name = tempfile.mkstemp(prefix="luke-export-", suffix=".zip")
    path = Path(name)
    try:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for entry in _entries(plan):
                _write(archive, files, root, entry)
            readme = _readme(settings.name, plan)
            _write_bytes(archive, files, f"{root}/{README_NAME}", README_NAME, readme)
            payload = manifest_module.manifest(_facts(plan, files))
            _write_bytes(
                archive,
                files,
                f"{root}/{MANIFEST_NAME}",
                MANIFEST_NAME,
                _json(payload),
            )
        return path, filename
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        os.close(handle)


def _plan(tenant_id: int | None, db: Session, mode: str) -> Plan:
    """Collect every entry once, in the mode the package will be written in."""
    key = contents.workspace(tenant_id)
    material = contents.materials(tenant_id, db)
    base_record, base_entries = contents.base(True)
    include_base = mode == MODE_FULL and bool(base_entries)
    if not include_base:
        base_entries = []
    base_record = {**base_record, "included": include_base}
    checkpoints = contents.checkpoints(tenant_id)
    tokenizer_records = contents.tokenizer_records()
    raw = contents.raw(tenant_id)
    dataset_files = contents.dataset_files(tenant_id, db, material)
    counts = {
        **dataset.counts(db, tenant_id),
        "lessons": len(dataset.parse(dataset_files["lessons.jsonl"], "lessons.jsonl")),
        "materials": min(len(material), manifest_module.MAX_MATERIAL_LINES),
        "passages": len(dataset.parse(dataset_files["passages.jsonl"], "passages.jsonl")),
        "checkpoints": len(checkpoints),
        "tokenizers": len(tokenizer_records),
        "raw_files": len(raw),
        "base_files": base_record["files"] if include_base else 0,
    }
    return Plan(
        workspace=key,
        mode=mode,
        model=contents.model_view(tenant_id),
        checkpoints=checkpoints,
        checkpoint_records=_checkpoint_records(checkpoints),
        base_model=base_record,
        base_entries=base_entries,
        tokenizers=contents.tokenizers(),
        tokenizer_records=tokenizer_records,
        state=contents.state(tenant_id),
        raw=raw,
        dataset=dataset_files,
        counts=counts,
        material_count=len(material),
        by_kind=contents.material_kinds(material),
    )


def _checkpoint_records(entries: list[Entry]) -> list[dict[str, Any]]:
    """What each checkpoint is, read from its own header without its weights."""
    records = []
    for entry in entries:
        record: dict[str, Any] = {"name": entry.path.name if entry.path else entry.name}
        record["bytes"] = entry.size
        stored = checkpoint.load(entry.path) if entry.path else None
        if stored is not None:
            record.update(
                {
                    "schema": stored.schema,
                    "task": stored.task,
                    "created": stored.created,
                    "metrics": stored.metrics,
                    "tokenizer": stored.tokenizer,
                }
            )
        records.append(record)
    return records


def _entries(plan: Plan) -> list[Entry]:
    """The files of one package, small ones first and the base weights last."""
    entries = [*plan.state, *plan.checkpoints, *plan.tokenizers]
    entries.extend(Entry(f"dataset/{name}", data=data) for name, data in plan.dataset.items())
    entries.extend(plan.raw)
    entries.extend(plan.base_entries)
    return entries


def _total(plan: Plan, mode: str) -> dict[str, Any]:
    """The measured size and file count of one mode, without writing it."""
    entries = (
        _entries(plan)
        if mode == MODE_FULL
        else [entry for entry in _entries(plan) if not entry.name.startswith("weights/base/")]
    )
    included = any(entry.name.startswith("weights/base/") for entry in entries)
    view = replace(plan, mode=mode, base_model={**plan.base_model, "included": included})
    readme = len(_readme(str(plan.model["name"]), view))
    return {
        "bytes": sum(entry.size for entry in entries) + readme,
        "files": len(entries) + 1,
        "includes_base": included,
    }


def _facts(plan: Plan, files: list[dict[str, Any]]) -> PackageFacts:
    primary = plan.checkpoint_records[0] if plan.checkpoint_records else None
    return PackageFacts(
        workspace=plan.workspace,
        mode=plan.mode,
        model=plan.model,
        checkpoint=primary,
        base_model=plan.base_model,
        tokenizers=plan.tokenizer_records,
        counts={**plan.counts, "files": len(files)},
        files=files,
        excluded=list(EXCLUDED),
    )


def _write(
    archive: zipfile.ZipFile,
    files: list[dict[str, Any]],
    root: str,
    entry: Entry,
) -> None:
    """Write one entry and record what was actually written."""
    if entry.data is not None:
        _write_bytes(archive, files, f"{root}/{entry.name}", entry.name, entry.data)
        return
    if entry.path is None:
        raise PackageError(f"The package entry {entry.name} has no file.")
    compress = zipfile.ZIP_STORED if entry.size > STORE_THRESHOLD else zipfile.ZIP_DEFLATED
    info = zipfile.ZipInfo(f"{root}/{entry.name}")
    info.compress_type = compress
    digest = hashlib.sha256()
    size = 0
    with entry.path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
        for block in iter(lambda: source.read(BLOCK), b""):
            digest.update(block)
            target.write(block)
            size += len(block)
    files.append({"path": entry.name, "bytes": size, "sha256": digest.hexdigest()})


def _write_bytes(
    archive: zipfile.ZipFile,
    files: list[dict[str, Any]],
    archive_name: str,
    path: str,
    data: bytes,
) -> None:
    info = zipfile.ZipInfo(archive_name)
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, data)
    files.append({"path": path, "bytes": len(data), "sha256": manifest_module.sha256_bytes(data)})


def _readme(name: str, plan: Plan) -> bytes:
    """A short note that says what is inside and how to bring it back."""
    base = plan.base_model
    lines = [
        f"# {name} — Luke Lab package",
        "",
        f"Format {manifest_module.FORMAT} v{manifest_module.FORMAT_VERSION}, "
        f"mode **{plan.mode}**, workspace {plan.workspace}.",
        "",
        "## What is inside",
        "",
        "- weights/checkpoints: the word model and its trained heads.",
        "- weights/tokenizers: every tokenizer artifact the checkpoints name.",
        "- state/: the learner lessons, model settings, level, runs and expressions.",
        "- dataset/: documents, answers, pending questions, feedback, uploads,",
        "  taught lessons, training material and the passages cut from the documents.",
        "- raw/: the uploaded files themselves.",
        "- manifest.json: every file with its size and SHA-256, plus the checkpoint",
        "  identity, licence, revisions, counts and embedding model.",
        "",
        "## Base checkpoint",
        "",
    ]
    if base.get("included"):
        lines.extend(
            [
                f"Included: {base.get('base_model')} at revision "
                f"{str(base.get('revision'))[:7]}, {base.get('weight_bytes')} bytes.",
            ]
        )
    else:
        lines.extend(
            [
                f"Not included. Install {base.get('base_model')} at revision "
                f"{str(base.get('revision'))[:7]} and check its weight SHA-256 is",
                f"`{base.get('weight_sha256') or 'not measured'}` before answering with it.",
            ]
        )
    lines.extend(
        [
            "",
            "## Restoring",
            "",
            "Import this zip from the trainer's package card with *Replace* confirmed.",
            "The index is rebuilt from dataset/documents.jsonl and dataset/faqs.jsonl;",
            "no index files are carried.",
            "",
            "Not included: Hugging Face teacher weights, the retired writer weights,",
            "generated answers, chat history and the vector index.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _teachers(tenant_id: int | None) -> list[dict[str, Any]]:
    return [
        {
            "name": str(row.get("name") or row.get("folder") or ""),
            "model_id": str(row.get("model_id") or ""),
            "steps": row.get("steps"),
            "bytes": int(row.get("bytes") or 0),
        }
        for row in teacher_store.list_teachers(tenant_id)
    ]


def _filename(tenant_id: int | None, model_name: str, mode: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model_name.casefold()).strip("-") or "luke"
    key = contents.workspace(tenant_id)
    date = time.strftime("%Y-%m-%d", time.gmtime())
    suffix = "" if mode == MODE_FULL else "-slim"
    return f"{slug}-{key}-{date}{suffix}.zip"


def _json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
