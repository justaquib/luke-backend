"""Turning Luke into one package a trainer can keep or move to another machine.

The package holds the model Luke answers with, the heads it trained, the
learner state and every piece of material those were trained from. Teacher
weights and the retired writer are left out, and every file is listed with its
size and SHA-256, so the package can be checked without opening it.

The package is written to a temporary file and removed after it is sent, so a
download never leaves a second copy on disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.database import Document, Faq, PendingQuestion
from app.services import brain as brain_service
from app.services import brain_levels, model_settings
from app.services.brain_paths import brain_dir, tenant_key
from app.services.huggingface import store as teacher_store
from app.services.language import training_materials

FORMAT = "luke-package"
FORMAT_VERSION = 1
MAX_MATERIAL_LINES = 50_000
_PENDING_FILE = "data/pending.jsonl"
_STATE_FILES = (
    "tenant-{key}.json",
    "model-{key}.json",
    "level-{key}.json",
    "progress-{key}.json",
    "runs-{key}.json",
)


class ExportError(Exception):
    """Luke cannot be packaged for this caller."""


def preview(tenant_id: int | None, db: Session) -> dict[str, Any]:
    """What the package would hold, without building it."""
    key = _tenant(tenant_id)
    materials = _materials(tenant_id, db)
    checkpoints = _checkpoints(key)
    settings = model_settings.read(tenant_id)
    teachers = teacher_store.list_teachers(tenant_id)
    return {
        "model": {
            "name": settings.name,
            "start_level": settings.start_level,
            "approved_level": brain_levels.approved_level(tenant_id),
        },
        "available": len(materials),
        "by_kind": _count_kinds(materials),
        "checkpoints": [{"name": path.name, "bytes": _size(path)} for path in checkpoints],
        "teachers": [
            {
                "name": str(row.get("name") or row.get("folder") or ""),
                "model_id": str(row.get("model_id") or ""),
                "steps": row.get("steps"),
                "bytes": int(row.get("bytes") or 0),
            }
            for row in teachers
        ],
        "pending": _pending_count(tenant_id, db),
        "bytes": sum(_size(path) for path in checkpoints),
        "filename": _filename(tenant_id, settings.name),
    }


def build(tenant_id: int | None, db: Session) -> tuple[Path, str]:
    """Write the package to a temporary zip and return it with its file name."""
    key = _tenant(tenant_id)
    materials = _materials(tenant_id, db)
    checkpoints = _checkpoints(key)
    settings = model_settings.read(tenant_id)
    filename = _filename(tenant_id, settings.name)
    root = f"luke-{key}"
    files: list[dict[str, Any]] = []
    handle, name = tempfile.mkstemp(prefix="luke-export-", suffix=".zip")
    path = Path(name)
    try:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for checkpoint in checkpoints:
                _add(archive, files, f"{root}/model/{checkpoint.name}", checkpoint.read_bytes())
            for state in _state_paths(key):
                if state.is_file():
                    _add(archive, files, f"{root}/state/{state.name}", state.read_bytes())
            _add(archive, files, f"{root}/data/materials.jsonl", _materials_bytes(materials))
            _add(archive, files, f"{root}/data/pending.jsonl", _pending_bytes(tenant_id, db))
            summary = _summary(tenant_id, db, materials, checkpoints, settings)
            _add(archive, files, f"{root}/data/summary.json", _json(summary))
            _add(archive, files, f"{root}/README.md", _readme(settings.name, key, summary))
            manifest = {
                "format": FORMAT,
                "format_version": FORMAT_VERSION,
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tenant": key,
                "model": summary["model"],
                "material": summary["material"],
                "checkpoints": summary["checkpoints"],
                "files": files,
                "excluded": summary["excluded"],
            }
            _add(archive, files, f"{root}/manifest.json", _json(manifest))
        return path, filename
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        os.close(handle)


def _tenant(tenant_id: int | None) -> int:
    return tenant_key(tenant_id)


def _materials(
    tenant_id: int | None,
    db: Session,
) -> list[training_materials.Material]:
    """The tenant's material, and for the system administrator only its own.

    ``training_materials.gather`` reads every tenant's database rows when it is
    given no tenant; the download narrows that to rows with no tenant, so one
    administrator's package cannot carry another tenant's material.
    """
    if tenant_id is not None:
        return training_materials.gather(tenant_id, db)
    feedback = brain_service.load_feedback(None)
    corrections = [
        (str(entry.get("question") or ""), str(entry.get("correction") or ""))
        for entry in feedback
        if entry.get("correction")
    ]
    approved = db.execute(
        select(Faq.question, Faq.answer).where(Faq.status == "approved", Faq.tenant_id.is_(None))
    ).all()
    documents = db.execute(
        select(Document.source, Document.text).where(Document.tenant_id.is_(None))
    ).all()
    return training_materials.collect(
        None,
        approved=[(question, answer) for question, answer in approved],
        documents=[(source, text) for source, text in documents],
        corrections=corrections,
    )


def _checkpoints(key: int) -> list[Path]:
    """Luke's current checkpoint and trained heads, never another tenant's."""
    root = brain_dir()
    files = [root / f"words-{key}.pt"]
    files.extend(sorted(root.glob(f"brain-{key}-*.pt")))
    return [path for path in files if path.is_file()]


def _state_paths(key: int) -> list[Path]:
    root = brain_dir()
    return [root / name.format(key=key) for name in _STATE_FILES]


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _count_kinds(materials: Sequence[training_materials.Material]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for material in materials:
        counts[material.kind] = counts.get(material.kind, 0) + 1
    return counts


def _pending_count(tenant_id: int | None, db: Session) -> int:
    statement = select(func.count()).select_from(PendingQuestion)
    statement = _scope(statement, PendingQuestion.tenant_id, tenant_id)
    statement = statement.where(PendingQuestion.resolved.is_(False))
    return int(db.execute(statement).scalar_one() or 0)


def _pending_bytes(tenant_id: int | None, db: Session) -> bytes:
    """The unresolved questions Luke could not answer, kept for context."""
    statement = select(PendingQuestion)
    statement = _scope(statement, PendingQuestion.tenant_id, tenant_id)
    statement = (
        statement.where(PendingQuestion.resolved.is_(False))
        .order_by(PendingQuestion.id)
        .limit(MAX_MATERIAL_LINES)
    )
    lines = []
    for row in db.execute(statement).scalars():
        lines.append(
            _json_line(
                {
                    "id": row.id,
                    "question": row.question,
                    "created_at": str(getattr(row, "created_at", "") or ""),
                }
            )
        )
    return "".join(lines).encode("utf-8")


def _scope(statement: Any, column: Any, tenant_id: int | None) -> Any:
    """Keep a query on one tenant's rows, including the administrator's."""
    if tenant_id is None:
        return statement.where(column.is_(None))
    return statement.where(column == tenant_id)


def _materials_bytes(materials: Sequence[training_materials.Material]) -> bytes:
    """Every material line as JSON, capped so one package stays reviewable."""
    lines = [
        _json_line(
            {
                "kind": material.kind,
                "question": material.question,
                "answer": material.answer,
                "subject": material.subject,
            }
        )
        for material in materials[:MAX_MATERIAL_LINES]
    ]
    return "".join(lines).encode("utf-8")


def _summary(
    tenant_id: int | None,
    db: Session,
    materials: Sequence[training_materials.Material],
    checkpoints: Sequence[Path],
    settings: model_settings.ModelSettings,
) -> dict[str, Any]:
    counts = _count_kinds(materials)
    return {
        "model": {
            "name": settings.name,
            "start_level": settings.start_level,
            "approved_level": brain_levels.approved_level(tenant_id),
        },
        "material": {
            "total": len(materials),
            "exported": min(len(materials), MAX_MATERIAL_LINES),
            "by_kind": counts,
            "pending_excluded": _pending_count(tenant_id, db),
        },
        "checkpoints": [
            {
                "name": path.name,
                "bytes": _size(path),
                "sha256": _sha256(path.read_bytes()),
            }
            for path in checkpoints
        ],
        "teachers": teacher_store.list_teachers(tenant_id),
        "excluded": [
            "Hugging Face teacher weights",
            "retired writer weights (transformer-<tenant>.pt)",
            "the index and uploaded document bytes",
        ],
    }


def _readme(name: str, key: int, summary: dict[str, Any]) -> bytes:
    counts = summary["material"]["by_kind"]
    lines = [
        f"# {name} — downloadable package",
        "",
        "This package holds the model Luke answers with, the heads it trained,",
        "the learner state and the material it learned from.",
        "",
        f"- model/: the word model and its trained heads for tenant {key}.",
        "- state/: the learner lessons, the model settings and the level report.",
        "- data/materials.jsonl: one material per line (kind, question, answer, subject).",
        f"- data/pending.jsonl: {summary['material']['pending_excluded']} unresolved",
        "  questions, kept for context only.",
        "- manifest.json: every file with its size and SHA-256.",
        "",
        "Material counts: " + ", ".join(f"{kind} {count}" for kind, count in counts.items()),
        "",
        "Not included: Hugging Face teacher weights, the retired writer weights",
        "and the document files themselves (their indexed chunks are in the material).",
        "",
        "To restore the model on another machine, stop the API, copy `model/` and",
        "`state/` into `.language_model/` for the same tenant number, then restart.",
        "To retrain instead, feed `data/materials.jsonl` to the trainer from a",
        "tenant account with the same material.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _filename(tenant_id: int | None, model_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model_name.casefold()).strip("-") or "luke"
    key = tenant_key(tenant_id)
    date = time.strftime("%Y-%m-%d", time.gmtime())
    return f"{slug}-{key}-{date}.zip"


def _json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _add(
    archive: zipfile.ZipFile,
    files: list[dict[str, Any]],
    name: str,
    data: bytes,
) -> None:
    archive.writestr(name, data)
    files.append({"path": name, "bytes": len(data), "sha256": _sha256(data)})
