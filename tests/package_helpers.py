"""Temporary SQLite, workspace files and archive helpers for package tests."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import package_models
import package_vectors
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db import database
from app.db.database import Document, Faq, Feedback, PendingQuestion, Tenant, UploadedFile
from app.services.language import export, runtime
from app.services.language.package import dataset, restore

WORKSPACE = 7
OTHER = 8
SOURCE = "moon.txt"
QUESTION = "Who wrote the moon?"
ANSWER = "A poet wrote the moon."
TEXT = "The moon reflects sunlight."
MOMENT = datetime(2025, 1, 2, 3, 4, 5)


def write_json(path: Path, value: object) -> None:
    """Write a UTF-8 test state or identity file."""
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def seed_files(brain: Path, uploads: Path) -> None:
    """Two workspaces have disjoint checkpoints, state and raw uploads."""
    package_models.write_scratch(brain, WORKSPACE)
    package_models.write_scratch(brain, OTHER, "word")
    write_json(
        brain / "tenant-7.json",
        {
            "lessons": [{"text": "The moon is bright."}],
            "word_cards": [{"word": "lunar", "meaning": "of the moon"}],
            "lesson_cards": [
                {
                    "id": "drum",
                    "question": "What is a drum?",
                    "answer": "A drum is hit with a stick.",
                    "subject": "Music",
                }
            ],
        },
    )
    write_json(brain / "model-7.json", {"name": "Luna", "start_level": "0.1.0"})
    write_json(brain / "level-7.json", {"approved_level": "0.1.0"})
    write_json(brain / "expressions-7.json", {"items": []})
    write_json(brain / "model-8.json", {"name": "Other", "start_level": "0.0.0"})
    (brain / "transformer-7.pt").write_bytes(b"retired writer: excluded")
    for key, content in ((WORKSPACE, TEXT), (OTHER, "Another workspace's upload")):
        folder = uploads / str(key)
        folder.mkdir(parents=True)
        (folder / SOURCE).write_text(content, encoding="utf-8")


def seed_rows(db: Session, workspace: int | None, start: int) -> None:
    """Real rows include dates, provenance and explicit ids to expose rollback loss."""
    text = TEXT if workspace == WORKSPACE else f"Workspace {workspace} private text."
    digest = hashlib.sha256(text.encode()).hexdigest()
    _seed_corpus(db, workspace, start, text, digest)
    _seed_activity(db, workspace, start)
    db.commit()


def _seed_corpus(db: Session, workspace: int | None, start: int, text: str, digest: str) -> None:
    db.add(
        UploadedFile(
            id=start,
            tenant_id=workspace,
            source=SOURCE,
            content_hash=digest,
            size_bytes=len(text.encode()),
            created_at=MOMENT,
        )
    )
    db.flush()
    db.add(
        Document(
            id=start,
            tenant_id=workspace,
            source=SOURCE,
            chunk_index=0,
            text=text,
            document_id=start,
            char_start=0,
            char_end=len(text),
            page=3,
            content_hash=digest,
            language="en",
            quality=0.875,
            created_at=MOMENT,
        )
    )
    db.add(
        Faq(
            id=start,
            tenant_id=workspace,
            question=QUESTION if workspace == WORKSPACE else f"Private question {workspace}?",
            answer=ANSWER if workspace == WORKSPACE else "Private answer.",
            status="approved",
            confidence=0.91,
            created_at=MOMENT,
            updated_at=MOMENT,
        )
    )


def _seed_activity(db: Session, workspace: int | None, start: int) -> None:
    db.add(
        PendingQuestion(
            id=start,
            tenant_id=workspace,
            question=f"What is the tide {workspace}?",
            occurrences=4,
            resolved=False,
            created_at=MOMENT,
            last_seen_at=MOMENT,
        )
    )
    db.add(
        Feedback(
            id=start,
            tenant_id=workspace,
            message_id=23,
            question=QUESTION,
            answer=ANSWER,
            source="faq",
            rating="up",
            created_at=MOMENT,
        )
    )


@dataclass
class Environment:
    """Everything a package test owns, with no live database or model paths."""

    root: Path
    brain: Path
    uploads: Path
    sessions: sessionmaker[Session]
    collections: dict[str, package_vectors.Collection]

    def package(self, mode: str = "slim", workspace: int | None = WORKSPACE) -> bytes:
        """Build through the real exporter and always discard its temporary zip."""
        with self.sessions() as db:
            path, _ = export.build(workspace, db, mode)
        try:
            return path.read_bytes()
        finally:
            path.unlink()

    def apply(self, data: bytes, *, dry_run: bool = True, replace: bool = False) -> dict:
        """Import with a fresh SQLite session and the public API's default dry run."""
        with self.sessions() as db:
            return restore.import_package(
                WORKSPACE, db, io.BytesIO(data), dry_run=dry_run, replace=replace
            )

    def snapshot(self) -> dict:
        """Capture every row/id, all owned files and all index metadata."""
        with self.sessions() as db:
            rows = {
                name: [
                    {
                        column.name: getattr(row, column.name)
                        for column in table.model.__table__.columns
                    }
                    for row in db.scalars(select(table.model).order_by(table.model.id))
                ]
                for name, table in dataset.TABLES.items()
            }
        return {
            "rows": rows,
            "files": {
                str(path.relative_to(self.root)): path.read_bytes()
                for folder in (self.brain, self.uploads, Path(settings.foundation_model_dir))
                for path in folder.rglob("*")
                if path.is_file()
            },
            "vectors": {
                name: deepcopy(collection.rows) for name, collection in self.collections.items()
            },
        }


@contextmanager
def environment(monkeypatch, root: Path) -> Iterator[Environment]:
    """Create disposable files and real SQLite; collections alone are faked."""
    brain, uploads = root / "brain", root / "raw-uploads"
    brain.mkdir()
    uploads.mkdir()
    monkeypatch.setattr(runtime, "_DATA_DIR", brain)
    monkeypatch.setattr(settings, "upload_dir", str(uploads))
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    runtime._CACHE.clear()
    engine = create_engine(f"sqlite:///{root / 'package.db'}")
    database.Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(database, "SessionLocal", sessions)
    collections = package_vectors.install(monkeypatch)
    try:
        seed_files(brain, uploads)
        with sessions() as db:
            db.add_all([Tenant(id=key, name=str(key), slug=str(key)) for key in (WORKSPACE, OTHER)])
            db.commit()
            for key, start in ((WORKSPACE, 10), (OTHER, 80), (None, 150)):
                seed_rows(db, key, start)
        yield Environment(root, brain, uploads, sessions, collections)
    finally:
        runtime._CACHE.clear()
        engine.dispose()


def unpack(data: bytes, workspace: int = WORKSPACE) -> tuple[dict, dict[str, bytes]]:
    """Read archive-relative members and the manifest without extracting files."""
    prefix = f"luke-{workspace}/"
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = {name.removeprefix(prefix): archive.read(name) for name in archive.namelist()}
    payload = json.loads(entries.pop("manifest.json"))
    return payload, entries


def pack(payload: object, entries: dict[str, bytes], workspace: int = WORKSPACE) -> bytes:
    """Build a mutated package without silently repairing its inventory."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(f"luke-{workspace}/{name}", data)
        archive.writestr(f"luke-{workspace}/manifest.json", json.dumps(payload))
    return stream.getvalue()


def inventory(entries: dict[str, bytes]) -> list[dict]:
    """An independently computed inventory for intentionally edited payloads."""
    return [
        {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in entries.items()
    ]
