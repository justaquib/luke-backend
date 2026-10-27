"""Tests for the downloadable Luke package.

No model is loaded: the checkpoints are small placeholder files and the
material is written into a temporary brain folder, so the package's contents,
its tenant scope and its manifest can be checked without training.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.security import get_bearer_user
from app.db import database
from app.db.database import Document, Faq, PendingQuestion, User, get_session
from app.routers import language_brain
from app.services.language import export, runtime

TENANT = 7


@pytest.fixture()
def brain(monkeypatch, tmp_path: Path) -> Iterator[Path]:
    """A temporary brain folder with Luke's files for this tenant."""
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    runtime._CACHE.clear()
    (tmp_path / "words-7.pt").write_bytes(b"word weights")
    (tmp_path / "brain-7-intent.pt").write_bytes(b"intent head")
    (tmp_path / "words-8.pt").write_bytes(b"another tenant")
    (tmp_path / "tenant-7.json").write_text(
        json.dumps(
            {
                "lessons": [{"text": "The moon is bright."}],
                "word_cards": [{"word": "lunar", "meaning": "of the moon"}],
                "lesson_cards": [
                    {
                        "question": "What is a drum?",
                        "answer": "A drum is hit with a stick.",
                        "subject": "Music",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model-7.json").write_text(
        json.dumps({"name": "Luna", "start_level": "0.1.0"}), encoding="utf-8"
    )
    (tmp_path / "level-7.json").write_text(
        json.dumps({"approved_level": "0.1.0"}), encoding="utf-8"
    )
    yield tmp_path
    runtime._CACHE.clear()


@pytest.fixture()
def sessions(monkeypatch, brain: Path) -> Iterator[sessionmaker[Session]]:
    """A temporary database holding two tenants' content."""
    engine = create_engine(f"sqlite:///{brain / 'luke.db'}")
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as db:
        db.add(
            Faq(
                question="Who wrote the moon?",
                answer="A poet.",
                status="approved",
                tenant_id=TENANT,
            )
        )
        db.add(
            Document(
                source="notes.txt",
                chunk_index=0,
                text="The moon reflects sunlight.",
                tenant_id=TENANT,
            )
        )
        db.add(PendingQuestion(question="What is the tide?", tenant_id=TENANT))
        db.add(
            Faq(
                question="Another tenant's question",
                answer="No.",
                status="approved",
                tenant_id=8,
            )
        )
        db.commit()
    yield factory
    engine.dispose()


def test_preview_counts_the_material_and_luke_files(sessions) -> None:
    with sessions() as db:
        preview = export.preview(TENANT, db)

    counts = preview["by_kind"]
    assert preview["model"]["name"] == "Luna"
    assert preview["model"]["approved_level"] == "0.1.0"
    assert counts["approved_answer"] == 1
    assert counts["document"] == 1
    assert counts["word"] == 1
    assert counts["sentence"] == 1
    assert counts["conversation"] > 0
    assert counts["lesson"] > 0
    assert preview["available"] == sum(counts.values())
    assert preview["pending"] == 1
    assert [item["name"] for item in preview["checkpoints"]] == [
        "words-7.pt",
        "brain-7-intent.pt",
    ]
    assert re.fullmatch(r"luna-7-\d{4}-\d{2}-\d{2}\.zip", preview["filename"])


def test_package_holds_only_this_tenant_and_lists_every_file(sessions) -> None:
    with sessions() as db:
        path, filename = export.build(TENANT, db)

    assert re.fullmatch(r"luna-7-\d{4}-\d{2}-\d{2}\.zip", filename)
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("luke-7/manifest.json"))
        assert {
            "luke-7/model/words-7.pt",
            "luke-7/model/brain-7-intent.pt",
            "luke-7/state/tenant-7.json",
            "luke-7/state/model-7.json",
            "luke-7/state/level-7.json",
            "luke-7/data/materials.jsonl",
            "luke-7/data/pending.jsonl",
            "luke-7/data/summary.json",
            "luke-7/README.md",
        } <= names
        assert not any("words-8" in name for name in names)
        assert not any("teacher" in name.lower() for name in names)
        listed = {item["path"] for item in manifest["files"]}
        assert listed == names - {"luke-7/manifest.json"}
        for item in manifest["files"]:
            assert item["bytes"] == len(archive.read(item["path"]))
        materials = archive.read("luke-7/data/materials.jsonl").decode()
        assert "notes.txt" in materials
        assert "Another tenant's question" not in materials
        pending = archive.read("luke-7/data/pending.jsonl").decode()
        assert "What is the tide?" in pending
    path.unlink()


def test_administrator_package_keeps_only_tenantless_rows(sessions) -> None:
    with sessions() as db:
        preview = export.preview(None, db)
        path, _ = export.build(None, db)

    assert preview["available"] == sum(preview["by_kind"].values())
    assert "approved_answer" not in preview["by_kind"]
    assert "document" not in preview["by_kind"]
    assert preview["pending"] == 0
    with zipfile.ZipFile(path) as archive:
        materials = archive.read("luke-0/data/materials.jsonl").decode()
        assert "Who wrote the moon?" not in materials
        assert "Another tenant's question" not in materials
        pending = archive.read("luke-0/data/pending.jsonl").decode()
        assert "What is the tide?" not in pending
        assert "luke-0/model/words-7.pt" not in archive.namelist()
    path.unlink()


def test_routes_report_the_package_and_send_a_zip(brain: Path) -> None:
    engine = create_engine(f"sqlite:///{brain / 'routes.db'}")
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as db:
        db.add(
            Faq(
                question="What is a drum?",
                answer="A drum is hit.",
                status="approved",
                tenant_id=TENANT,
            )
        )
        db.commit()

    app = FastAPI()
    app.include_router(language_brain.router)

    def sess() -> Iterator[Session]:
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_bearer_user] = lambda: User(
        id=1,
        email="trainer@example.com",
        name="Trainer",
        is_systemadmin=False,
        tenant_id=TENANT,
    )
    app.dependency_overrides[get_session] = sess
    with TestClient(app) as client:
        preview = client.get("/language-model/export/preview")
        assert preview.status_code == 200
        assert preview.json()["by_kind"]["approved_answer"] == 1
        package = client.get("/language-model/export")
        assert package.status_code == 200
        assert package.headers["content-type"] == "application/zip"
        assert "attachment" in package.headers["content-disposition"]
        assert package.content[:2] == b"PK"
    engine.dispose()
