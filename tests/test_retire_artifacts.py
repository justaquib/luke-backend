"""Retiring the lessons collection is backed up, removable and reversible."""

from __future__ import annotations

from pathlib import Path

from scripts import retire_artifacts
from scripts.single_user import vectors

ROWS = {
    "lesson-1": ("The moon is bright.", {"tenant_id": "0"}),
    "lesson-2": ("The sky is clear.", {}),
}


class _Collection:
    name = retire_artifacts.COLLECTION

    def __init__(self) -> None:
        self.rows = dict(ROWS)

    def count(self) -> int:
        return len(self.rows)

    def get(self, include: list[str] | None = None) -> dict:
        return {
            "ids": list(self.rows),
            "documents": [text for text, _ in self.rows.values()],
            "metadatas": [dict(metadata) for _, metadata in self.rows.values()],
        }

    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        for identifier, document, metadata in zip(ids, documents, metadatas, strict=True):
            self.rows[identifier] = (document, dict(metadata))


class _Client:
    def __init__(self) -> None:
        self.collections: dict[str, _Collection] = {}

    def list_collections(self) -> list[_Collection]:
        return list(self.collections.values())

    def create_collection(self, name: str) -> _Collection:
        return self.collections.setdefault(name, _Collection())

    def delete_collection(self, name: str) -> None:
        del self.collections[name]


def _store(tmp_path: Path) -> tuple[_Client, Path]:
    client = _Client()
    client.collections[retire_artifacts.COLLECTION] = _Collection()
    checkpoint = tmp_path / retire_artifacts.CHECKPOINT
    checkpoint.write_bytes(b"retired writer")
    return client, checkpoint


def test_the_plan_counts_what_would_go(tmp_path: Path) -> None:
    client, checkpoint = _store(tmp_path)

    found = retire_artifacts.plan(client, checkpoint)

    assert found["collection"] == {"name": "lessons", "present": True, "records": 2}
    assert found["checkpoint"]["present"] is True
    assert found["checkpoint"]["bytes"] == len(b"retired writer")


def test_the_backup_keeps_the_text_and_moves_the_checkpoint(tmp_path: Path) -> None:
    client, checkpoint = _store(tmp_path)
    folder = tmp_path / "backup"

    kept = retire_artifacts.backup(client, folder, checkpoint)

    assert kept["collection"] == {"file": "lessons.json", "records": 2}
    assert kept["checkpoint"] == {"file": "transformer-0.pt", "bytes": 14}
    assert (folder / "lessons.json").is_file()
    assert not checkpoint.exists()


def test_removing_takes_both_away(tmp_path: Path) -> None:
    client, checkpoint = _store(tmp_path)
    retire_artifacts.backup(client, tmp_path / "backup", checkpoint)

    removed = retire_artifacts.remove(client, checkpoint)

    assert removed == {"collection": True, "checkpoint": False}
    assert client.list_collections() == []
    assert retire_artifacts.plan(client, checkpoint)["collection"]["present"] is False


def test_a_rollback_puts_the_lessons_and_the_writer_back(tmp_path: Path) -> None:
    client, checkpoint = _store(tmp_path)
    folder = tmp_path / "backup"
    retire_artifacts.backup(client, folder, checkpoint)
    retire_artifacts.remove(client, checkpoint)

    restored = retire_artifacts.restore(client, folder, checkpoint)

    assert restored == {"collection": True, "checkpoint": True}
    assert client.list_collections()[0].rows == ROWS
    assert checkpoint.read_bytes() == b"retired writer"


def test_a_rollback_of_an_empty_backup_changes_nothing(tmp_path: Path) -> None:
    client, checkpoint = _store(tmp_path)

    restored = retire_artifacts.restore(client, tmp_path / "missing", checkpoint)

    assert restored == {"collection": False, "checkpoint": False}
    assert checkpoint.is_file()


def test_the_single_user_migration_no_longer_plans_the_lessons_collection() -> None:
    assert "lessons" not in vectors.COLLECTIONS
