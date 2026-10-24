"""Who owns material uploaded from the training room."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile

from app.db.database import Document, User
from app.services import documents


class _Session:
    def __init__(self) -> None:
        self.added: list[Document] = []

    def add(self, row: Document) -> None:
        self.added.append(row)

    def commit(self) -> None:
        pass


class _VectorStore:
    def __init__(self) -> None:
        self.tenant_ids: list[int | None] = []

    def collection_size(self) -> int:
        return 0

    def add_chunks(
        self,
        source: str,
        chunks: list[str],
        ids: list[str],
        tenant_id: int | None,
    ) -> None:
        self.tenant_ids.append(tenant_id)


def _prepare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, vector: _VectorStore) -> None:
    monkeypatch.setattr(documents, "settings", SimpleNamespace(upload_dir=str(tmp_path)))
    monkeypatch.setattr(
        documents, "extract_text_from_file", lambda path, content_type: "alpha beta"
    )
    monkeypatch.setattr(documents, "chunk_text", lambda text: ["alpha beta"])
    monkeypatch.setattr(documents, "vector_store", vector)


def _upload() -> UploadFile:
    return UploadFile(filename="notes.txt", file=io.BytesIO(b"alpha beta"))


def _user(*, is_systemadmin: bool, tenant_id: int | None) -> User:
    return User(
        id=1,
        email="person@example.com",
        name="Person",
        is_systemadmin=is_systemadmin,
        tenant_id=tenant_id,
    )


def test_system_admin_uploads_land_in_the_platform_bucket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vector = _VectorStore()
    _prepare(monkeypatch, tmp_path, vector)
    db = _Session()

    result = documents.upload_document(_user(is_systemadmin=True, tenant_id=None), _upload(), db)

    assert result == {"source": "notes.txt", "chunks": 1, "status": "ingested"}
    assert vector.tenant_ids == [None]
    assert [row.tenant_id for row in db.added] == [None]
    assert (tmp_path / "notes.txt").read_bytes() == b"alpha beta"


def test_tenant_uploads_keep_their_tenant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vector = _VectorStore()
    _prepare(monkeypatch, tmp_path, vector)

    documents.upload_document(_user(is_systemadmin=False, tenant_id=1), _upload(), _Session())

    assert vector.tenant_ids == [1]


def test_a_user_without_a_tenant_cannot_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _prepare(monkeypatch, tmp_path, _VectorStore())

    with pytest.raises(HTTPException) as error:
        documents.upload_document(
            _user(is_systemadmin=False, tenant_id=None), _upload(), _Session()
        )

    assert error.value.status_code == 403
    assert not (tmp_path / "notes.txt").exists()
