"""Package v2 exports are measured, complete and scoped to one workspace."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections.abc import Iterator

import package_helpers as helpers
import package_models
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import get_bearer_user
from app.core.workspace import CANONICAL_WORKSPACE
from app.db.database import Faq, User, get_session
from app.routers import language_brain
from app.services.foundation import identity
from app.services.language import export
from app.services.language.package import manifest
from app.services.neural import checkpoint
from app.services.neural.tokenizer import artifacts


@pytest.fixture()
def env(monkeypatch, tmp_path) -> Iterator[helpers.Environment]:
    """The package's own files, SQLite rows and offline vector collections."""
    with helpers.environment(monkeypatch, tmp_path) as isolated:
        yield isolated


def test_preview_counts_material_rows_and_actual_checkpoint_bytes(env) -> None:
    with env.sessions() as db:
        preview = export.preview(helpers.WORKSPACE, db)
    counts = preview["by_kind"]
    assert preview["model"] == {"name": "Luna", "start_level": "0.1.0", "approved_level": "0.1.0"}
    for kind in ("approved_answer", "document", "word", "sentence"):
        assert counts[kind] == 1
    assert counts["conversation"] > 0
    assert counts["lesson"] > 0
    assert preview["available"] == sum(counts.values())
    assert preview["pending"] == 1
    assert preview["counts"]["documents"] == 1
    assert preview["counts"]["uploads"] == 1
    assert preview["checkpoints"] == [
        {"name": "words-7.pt", "bytes": (env.brain / "words-7.pt").stat().st_size}
    ]
    assert re.fullmatch(r"luna-7-\d{4}-\d{2}-\d{2}\.zip", preview["filename"])


@pytest.mark.parametrize("mode", ["full", "slim"])
def test_manifest_is_an_exact_relative_sha256_inventory(env, mode) -> None:
    package_models.write_base()
    data = env.package(mode)
    payload, entries = helpers.unpack(data)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert all(name.startswith("luke-7/") for name in archive.namelist())
    assert payload["format"] == "luke-package"
    assert payload["format_version"] == 2
    assert payload["workspace"] == helpers.WORKSPACE
    assert payload["mode"] == mode
    assert {row["path"] for row in payload["files"]} == set(entries)
    assert len(payload["files"]) == len(entries) == payload["counts"]["files"]
    assert sorted(payload["files"], key=lambda row: row["path"]) == sorted(
        helpers.inventory(entries), key=lambda row: row["path"]
    )
    assert {
        "weights/checkpoints/words-7.pt",
        "state/tenant-7.json",
        "state/model-7.json",
        "state/level-7.json",
        "state/expressions-7.json",
        "dataset/documents.jsonl",
        "dataset/faqs.jsonl",
        "dataset/pending.jsonl",
        "dataset/uploads.jsonl",
        "dataset/feedback.jsonl",
        "dataset/lessons.jsonl",
        "dataset/materials.jsonl",
        "dataset/passages.jsonl",
        "raw/moon.txt",
        "README.md",
    } <= set(entries)
    assert "manifest.json" not in {row["path"] for row in payload["files"]}
    assert not any(name.startswith(("model/", "data/")) for name in entries)
    assert payload["embedding"] == manifest.embedding_record()
    assert payload["collections"] == manifest.collections_record()
    assert payload["revisions"]["checkpoint_schema"] == 3
    assert payload["revisions"]["tokenizer_schema"] == 1
    assert payload["licence"] == identity.LICENSE


@pytest.mark.parametrize("mode", ["full", "slim"])
def test_full_and_slim_preview_sizes_equal_the_written_payloads(env, mode) -> None:
    base = package_models.write_base()
    with env.sessions() as db:
        preview = export.preview(helpers.WORKSPACE, db)
        path, filename = export.build(helpers.WORKSPACE, db, mode)
    try:
        payload, entries = helpers.unpack(path.read_bytes())
    finally:
        path.unlink()
    measured = preview["modes"][mode]
    assert measured["bytes"] == sum(len(data) for data in entries.values())
    assert measured["files"] == len(entries)
    assert measured["filename"] == filename
    assert measured["includes_base"] is (mode == "full")
    assert preview["bytes"] == preview["modes"]["full"]["bytes"]
    assert preview["base"]["bytes"] == sum(p.stat().st_size for p in base.iterdir())
    assert payload["base_model"]["included"] is (mode == "full")
    assert payload["counts"]["base_files"] == (len(list(base.iterdir())) if mode == "full" else 0)
    assert filename.endswith("-slim.zip") is (mode == "slim")


def test_full_carries_base_and_slim_records_the_same_pinned_identity(env) -> None:
    folder = package_models.write_base()
    full, full_entries = helpers.unpack(env.package("full"))
    slim, slim_entries = helpers.unpack(env.package("slim"))
    for key in ("base_model", "revision", "weight_sha256", "weight_bytes", "license"):
        assert full["base_model"][key] == slim["base_model"][key]
    weights = (folder / "model.safetensors").read_bytes()
    assert full["base_model"]["weight_sha256"] == hashlib.sha256(weights).hexdigest()
    assert full["base_model"]["weight_bytes"] == len(weights)
    assert full["base_model"]["parameters"] == 6
    assert full["base_model"]["revision"] == identity.PINNED_REVISION
    assert not any(name.startswith("weights/base/") for name in slim_entries)
    for path in folder.iterdir():
        assert full_entries[f"weights/base/{path.name}"] == path.read_bytes()


def test_missing_base_is_reported_not_filled_with_invented_bytes(env) -> None:
    with env.sessions() as db:
        preview = export.preview(helpers.WORKSPACE, db)
    assert preview["base"]["installed"] is False
    assert preview["base"]["weight_bytes"] == preview["base"]["bytes"] == 0
    assert preview["base"]["weight_sha256"] == ""
    assert preview["modes"]["full"]["includes_base"] is False
    assert preview["modes"]["slim"]["includes_base"] is False


def test_bpe_artifact_matches_the_checkpoint_and_keeps_base_tokenizer_separate(env) -> None:
    package_models.write_base()
    stored = checkpoint.load(env.brain / "words-7.pt")
    assert stored is not None and stored.schema == 3
    payload, entries = helpers.unpack(env.package("full"))
    assert payload["checkpoint"]["tokenizer"] == stored.tokenizer
    (record,) = payload["tokenizers"]
    assert record["id"] == stored.tokenizer["id"]
    assert record["vocab_size"] == len(stored.vocabulary)
    assert record["kind"] == "bpe"
    assert record["licence"] == "Apache-2.0"
    prefix = f"weights/tokenizers/{record['id']}"
    for item in record["files"]:
        content = entries[f"{prefix}/{item['name']}"]
        assert item["bytes"] == len(content)
        assert item["sha256"] == hashlib.sha256(content).hexdigest()
    assert hashlib.sha256(entries[f"{prefix}/tokenizer.json"]).hexdigest() == record["id"]
    assert entries[f"{prefix}/tokenizer.json"] != entries["weights/base/tokenizer.json"]
    assert (
        artifacts.load(stored.tokenizer).decode(
            artifacts.load(stored.tokenizer).encode(package_models.TEXT)
        )
        == package_models.TEXT
    )


@pytest.mark.parametrize("workspace", [helpers.WORKSPACE, helpers.OTHER, None])
def test_package_keeps_only_the_selected_workspaces_rows_and_files(env, workspace) -> None:
    payload, entries = helpers.unpack(env.package(workspace=workspace), workspace or 0)
    assert payload["workspace"] == (workspace or 0)
    rows = [json.loads(line) for line in entries["dataset/documents.jsonl"].splitlines()]
    expected = (
        helpers.TEXT if workspace == helpers.WORKSPACE else f"Workspace {workspace} private text."
    )
    assert [row["text"] for row in rows] == [expected]
    assert payload["counts"]["documents"] == payload["counts"]["faqs"] == 1
    for name in entries:
        assert "teacher" not in name.casefold()
        assert "transformer" not in name
        if name.startswith(("state/", "weights/checkpoints/")):
            assert f"-{workspace or 0}." in name
    material = entries["dataset/materials.jsonl"].decode()
    if workspace != helpers.WORKSPACE:
        assert helpers.QUESTION not in material
        assert helpers.TEXT not in material


def test_unknown_export_mode_is_rejected(env) -> None:
    with env.sessions() as db, pytest.raises(manifest.PackageError, match="mode"):
        export.build(helpers.WORKSPACE, db, "unknown")


def test_routes_preview_v2_and_send_both_modes_in_canonical_workspace(env) -> None:
    with env.sessions() as db:
        db.add(
            Faq(
                question="What is a drum?",
                answer="A drum is hit.",
                status="approved",
                tenant_id=CANONICAL_WORKSPACE,
            )
        )
        db.commit()
    app = FastAPI()
    app.include_router(language_brain.router)

    def session() -> Iterator[Session]:
        with env.sessions() as db:
            yield db

    app.dependency_overrides[get_bearer_user] = lambda: User(
        id=1,
        email="trainer@example.com",
        name="Trainer",
        is_systemadmin=False,
        tenant_id=helpers.OTHER,
    )
    app.dependency_overrides[get_session] = session
    with TestClient(app) as client:
        preview = client.get("/language-model/export/preview")
        assert preview.status_code == 200
        assert preview.json()["by_kind"]["approved_answer"] == 1
        for mode in ("full", "slim"):
            response = client.get("/language-model/export", params={"mode": mode})
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/zip"
            assert "attachment" in response.headers["content-disposition"]
            payload, _ = helpers.unpack(response.content, CANONICAL_WORKSPACE)
            assert payload["format_version"] == 2
            assert payload["workspace"] == CANONICAL_WORKSPACE
            assert payload["mode"] == mode
