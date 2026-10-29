"""Bring a package back: verify it, stage it, replace this workspace, rebuild.

An import is a dry run unless the caller asks otherwise. The manifest, every
file, the dataset rows, the weights and the tokenizers are checked before
anything is written; the previous rows and files are copied aside before the
replacement; and a failure in the database, the files or the index puts both
back and rebuilds the index from the rows that were there. The package's
workspace must be this workspace: a package never changes which workspace it
belongs to. One import at a time replaces the files this process serves.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path
from typing import Any, BinaryIO

import chromadb
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import settings
from app.services import brain_paths
from app.services.foundation import identity
from app.services.foundation import runtime as foundation_runtime
from app.services.language import runtime
from app.services.language.package import (
    archive,
    compatibility,
    contents,
    dataset,
    files,
    reindex,
)
from app.services.language.package import (
    rows as row_checks,
)
from app.services.language.package.manifest import (
    CHECKPOINT_SCHEMA,
    TOKENIZER_SCHEMA,
    PackageError,
    root_name,
)

logger = logging.getLogger(__name__)

_IMPORT_LOCK = threading.Lock()


class PackageConflictError(PackageError):
    """This workspace already holds a model and the caller did not confirm a replace."""


def import_package(
    tenant_id: int | None,
    db: Session,
    source: BinaryIO,
    dry_run: bool = True,
    replace: bool = False,
) -> dict[str, Any]:
    """Check a package and, when asked, restore it into this workspace."""
    workspace = contents.workspace(tenant_id)
    upload = archive.receive(source, settings.max_package_bytes)
    staging: Path | None = None
    try:
        opened, root, payload = archive.open_package(upload, workspace)
        try:
            records = archive.verify(opened, root, payload)
            _check_payload(payload)
            files.validate(records, workspace)
            report = _report(payload, records)
            report["replace_required"] = _has_state(tenant_id, db)
            staging = Path(tempfile.mkdtemp(prefix="import-", dir=brain_paths.brain_dir()))
            archive.extract(opened, root, staging)
            staged = staging / root
            row_checks.validate(staged, payload["counts"])
            compatibility.validate(staged, payload)
            report["applied"] = False
            if dry_run:
                return report
            with _IMPORT_LOCK:
                report["replace_required"] = _has_state(tenant_id, db)
                _confirm(replace, report["replace_required"])
                report.update(_restore(tenant_id, db, staging, records, payload))
            report["applied"] = True
            logger.info("Restored workspace %s from a package (%s).", workspace, payload["mode"])
            return report
        finally:
            opened.close()
    finally:
        archive.discard(upload)
        if staging is not None:
            files.cleanup(staging)


def _restore(
    tenant_id: int | None,
    db: Session,
    staging: Path,
    records: list[dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Replace the workspace, then prove the rows and the index came back."""
    rows = dataset.load_rows(staging / root_name(contents.workspace(tenant_id)))
    snapshot = dataset.snapshot(db, tenant_id)
    backup = files.backup(tenant_id, staging / "previous", records)
    try:
        restored = files.apply(staging, tenant_id, records)
        dataset.clear(db, tenant_id)
        written = dataset.restore(db, tenant_id, rows)
        db.commit()
        _invalidate()
        indexed = reindex.rebuild(tenant_id, db)
        verified = reindex.verify(tenant_id, db, payload["counts"])
        return {"files": restored, "rows": written, "indexed": indexed, "verified": verified}
    except (
        PackageError,
        OSError,
        ValueError,
        RuntimeError,
        SQLAlchemyError,
        chromadb.errors.ChromaError,
    ) as error:
        db.rollback()
        logger.warning("The import failed and the previous workspace is being restored: %s", error)
        files.rollback(backup, tenant_id)
        dataset.clear(db, tenant_id)
        dataset.restore(db, tenant_id, snapshot, dedupe=False, keep_ids=True)
        db.commit()
        _invalidate()
        reindex.rebuild(tenant_id, db)
        raise


def _invalidate() -> None:
    """Forget cached state and weights after the files they read have changed."""
    runtime._CACHE.clear()
    foundation_runtime.unload()
    identity._verified.cache_clear()


def _confirm(replace: bool, has_state: bool) -> None:
    if has_state and not replace:
        raise PackageConflictError(
            "This workspace already holds a model. Send replace=true to replace it, "
            "or run the import as a dry run first."
        )


def _has_state(tenant_id: int | None, db: Session) -> bool:
    """Whether a replace would overwrite rows or files that are already here."""
    existing = dataset.counts(db, tenant_id)
    tables = ("documents", "faqs", "pending_total", "feedback", "uploads", "intents")
    rows = sum(existing[name] for name in tables)
    kept = bool(contents.state(tenant_id) or contents.checkpoints(tenant_id))
    return bool(rows or kept)


def _check_payload(payload: dict[str, Any]) -> None:
    """Refuse a package written by a newer format or against another foundation."""
    revisions = payload.get("revisions") or {}
    if int(revisions.get("checkpoint_schema") or 0) > CHECKPOINT_SCHEMA:
        raise PackageError("The package holds a newer checkpoint schema than this build reads.")
    if int(revisions.get("tokenizer_schema") or 0) > TOKENIZER_SCHEMA:
        raise PackageError("The package holds a newer tokenizer schema than this build reads.")
    base = payload.get("base_model") or {}
    wrong_model = base.get("base_model") != identity.BASE_MODEL
    wrong_revision = base.get("revision") != identity.PINNED_REVISION
    if base.get("included") and (wrong_model or wrong_revision):
        raise PackageError(
            "The package carries a different base checkpoint than the pinned "
            f"{identity.BASE_MODEL} at revision {identity.PINNED_REVISION[:7]}."
        )


def _report(payload: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """What the package says it holds, before anything is applied."""
    base = payload.get("base_model") or {}
    checkpoint = payload.get("checkpoint") or {}
    return {
        "format": payload["format"],
        "format_version": payload["format_version"],
        "mode": payload["mode"],
        "workspace": payload["workspace"],
        "created": payload["created"],
        "model": payload["model"],
        "licence": payload["licence"],
        "checkpoint": {
            "name": checkpoint.get("name"),
            "schema": checkpoint.get("schema"),
            "tokenizer": checkpoint.get("tokenizer"),
        },
        "base_model": {
            "included": bool(base.get("included")),
            "base_model": base.get("base_model"),
            "revision": base.get("revision"),
            "weight_sha256": base.get("weight_sha256"),
            "installed": bool(base.get("installed")),
        },
        "tokenizers": payload["tokenizers"],
        "counts": payload["counts"],
        "files": {
            "checked": len(records),
            "bytes": sum(int(record["bytes"]) for record in records),
        },
    }
