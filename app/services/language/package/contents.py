"""What a package is made of: weights, learner state, dataset and raw uploads.

The contents are described as entries, not as a zip, so the preview can count
them without writing a file and the builder can write the same list. Nothing
is read from another workspace: the checkpoints, state, dataset rows and raw
uploads are all selected by the package's own workspace number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.services import brain as brain_service
from app.services import brain_levels, brain_paths, model_settings
from app.services.foundation import identity
from app.services.language import lessons as lesson_service
from app.services.language import training_materials
from app.services.language.package import dataset
from app.services.language.package.manifest import MAX_MATERIAL_LINES, tokenizer_record
from app.services.language.passages import split_passages
from app.services.neural.tokenizer import artifacts
from app.services.neural.tokenizer.base import META_FILE, TOKENIZER_FILE

STATE_FILES = (
    "tenant-{key}.json",
    "model-{key}.json",
    "level-{key}.json",
    "progress-{key}.json",
    "runs-{key}.json",
    "expressions-{key}.json",
    "feedback-{key}.json",
)


@dataclass(frozen=True)
class Entry:
    """One file of the package: bytes in memory or a file to stream."""

    name: str
    data: bytes | None = None
    path: Path | None = None

    @property
    def size(self) -> int:
        if self.data is not None:
            return len(self.data)
        try:
            return self.path.stat().st_size if self.path is not None else 0
        except OSError:
            return 0


def workspace(tenant_id: int | None) -> int:
    """The number a package is written for."""
    return brain_paths.tenant_key(tenant_id)


def checkpoints(tenant_id: int | None) -> list[Entry]:
    """Luke's current checkpoint and trained heads, never another workspace's."""
    key = workspace(tenant_id)
    root = brain_paths.brain_dir()
    paths = [root / f"words-{key}.pt", *sorted(root.glob(f"brain-{key}-*.pt"))]
    return [
        Entry(f"weights/checkpoints/{path.name}", path=path) for path in paths if path.is_file()
    ]


def state(tenant_id: int | None) -> list[Entry]:
    """The learner's own files: lessons, settings, level, runs and feedback."""
    key = workspace(tenant_id)
    root = brain_paths.brain_dir()
    return [
        Entry(f"state/{path.name}", path=path)
        for path in (root / name.format(key=key) for name in STATE_FILES)
        if path.is_file()
    ]


def tokenizers() -> list[Entry]:
    """Every recorded tokenizer artifact, read back by hash."""
    root = artifacts.root()
    if not root.is_dir():
        return []
    entries: list[Entry] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder / TOKENIZER_FILE).is_file():
            continue
        for name in (TOKENIZER_FILE, META_FILE):
            path = folder / name
            if path.is_file():
                entries.append(Entry(f"weights/tokenizers/{folder.name}/{name}", path=path))
    return entries


def tokenizer_records() -> list[dict[str, Any]]:
    """The manifest record of every artifact the entries carry."""
    root = artifacts.root()
    if not root.is_dir():
        return []
    return [
        tokenizer_record(folder.name)
        for folder in sorted(root.iterdir())
        if folder.is_dir() and (folder / TOKENIZER_FILE).is_file()
    ]


def raw(tenant_id: int | None) -> list[Entry]:
    """The uploaded files themselves, under the workspace that owns them."""
    folder = Path(settings.upload_dir) / str(workspace(tenant_id))
    if not folder.is_dir():
        return []
    return [
        Entry(f"raw/{path.name}", path=path) for path in sorted(folder.iterdir()) if path.is_file()
    ]


def base(include: bool) -> tuple[dict[str, Any], list[Entry]]:
    """The pinned foundation checkpoint, as a reference or as carried files."""
    try:
        found = identity.load_identity(verify_weights=False)
    except identity.FoundationError as error:
        return {
            "installed": False,
            "included": False,
            "error": str(error),
            "name": identity.MODEL_NAME,
            "version": identity.MODEL_VERSION,
            "base_model": identity.BASE_MODEL,
            "revision": identity.PINNED_REVISION,
            "license": identity.LICENSE,
            "weight_sha256": "",
            "weight_bytes": 0,
            "files": 0,
            "bytes": 0,
        }, []
    files = sorted(
        path
        for path in found.folder.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(found.folder).parts)
    )
    record = {
        "installed": True,
        "included": include,
        "error": None,
        "name": found.name,
        "version": found.version,
        "base_model": found.base_model,
        "revision": found.revision,
        "license": found.license,
        "parameters": found.parameters,
        "context_limit": found.context_limit,
        "weight_sha256": found.weight_sha256,
        "weight_bytes": found.weight_bytes,
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
    }
    entries = []
    if include:
        entries = [
            Entry(f"weights/base/{path.relative_to(found.folder)}", path=path) for path in files
        ]
    return record, entries


def materials(tenant_id: int | None, db: Session) -> list[training_materials.Material]:
    """The workspace's training material, never another workspace's rows."""
    if tenant_id is not None:
        return training_materials.gather(tenant_id, db)
    feedback = brain_service.load_feedback(None)
    corrections = [
        (str(entry.get("question") or ""), str(entry.get("correction") or ""))
        for entry in feedback
        if entry.get("correction")
    ]
    rows = dataset.dump_rows(db, None)
    return training_materials.collect(
        None,
        approved=[
            (str(row["question"]), str(row["answer"]))
            for row in rows["faqs.jsonl"]
            if row.get("status") == "approved"
        ],
        documents=[
            (str(row["source"]), str(row["text"]), row.get("quality"))
            for row in rows["documents.jsonl"]
        ],
        corrections=corrections,
    )


def dataset_files(
    tenant_id: int | None,
    db: Session,
    material: list[training_materials.Material],
) -> dict[str, bytes]:
    """Every dataset file, keyed by its name inside ``dataset/``."""
    rows = dataset.dump_rows(db, tenant_id)
    files = {name: dataset.jsonl(items) for name, items in rows.items()}
    files["materials.jsonl"] = dataset.jsonl(_material_rows(material))
    files["lessons.jsonl"] = dataset.jsonl(lesson_service.all_lesson_cards(tenant_id))
    files["passages.jsonl"] = dataset.jsonl(_passage_rows(rows["documents.jsonl"]))
    return files


def _material_rows(material: list[training_materials.Material]) -> list[dict[str, Any]]:
    return [
        {
            "kind": item.kind,
            "question": item.question,
            "answer": item.answer,
            "subject": item.subject,
        }
        for item in material[:MAX_MATERIAL_LINES]
    ]


def _passage_rows(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in documents:
        for passage in split_passages(str(document.get("text") or "")):
            rows.append(
                {
                    "source": document.get("source"),
                    "page": document.get("page"),
                    "text": passage,
                }
            )
    return rows


def material_kinds(material: list[training_materials.Material]) -> dict[str, int]:
    """How much material of each kind the package carries."""
    counts: dict[str, int] = {}
    for item in material:
        counts[item.kind] = counts.get(item.kind, 0) + 1
    return counts


def model_view(tenant_id: int | None) -> dict[str, Any]:
    """The name and level the package records for the model."""
    loaded = model_settings.read(tenant_id)
    return {
        "name": loaded.name,
        "start_level": loaded.start_level,
        "approved_level": brain_levels.approved_level(tenant_id),
    }
