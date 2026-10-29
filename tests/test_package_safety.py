"""Damaged, foreign or hostile packages are refused before any write."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Iterator

import package_helpers as helpers
import pytest

from app.config import settings
from app.services.language.package.manifest import PackageError


@pytest.fixture()
def env(monkeypatch, tmp_path) -> Iterator[helpers.Environment]:
    """A real temporary database with offline index collections."""
    with helpers.environment(monkeypatch, tmp_path) as isolated:
        yield isolated


def _repack(env: helpers.Environment, path: str, data: bytes) -> bytes:
    """Replace one verified member and keep the manifest's record of it honest."""
    payload, entries = helpers.unpack(env.package())
    entries[path] = data
    record = next(row for row in payload["files"] if row["path"] == path)
    record["bytes"] = len(data)
    record["sha256"] = hashlib.sha256(data).hexdigest()
    return helpers.pack(payload, entries)


def _assert_refused(env: helpers.Environment, data: bytes, match: str) -> None:
    before = env.snapshot()
    with pytest.raises(PackageError, match=match):
        env.apply(data)
    assert env.snapshot() == before
    assert not list(env.root.glob("luke-import-*"))
    assert not list(env.brain.glob("import-*"))


def _archive(entries: list[tuple[str, bytes]]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return stream.getvalue()


def test_a_package_for_another_workspace_is_refused(env) -> None:
    _assert_refused(env, env.package(workspace=helpers.OTHER), "not this workspace")


def test_a_file_the_manifest_does_not_list_is_refused(env) -> None:
    payload, entries = helpers.unpack(env.package())
    entries["dataset/extra.jsonl"] = b"{}\n"
    _assert_refused(env, helpers.pack(payload, entries), "does not list")


def test_a_member_that_climbs_out_of_the_archive_is_refused(env) -> None:
    data = _archive(
        [
            ("luke-7/manifest.json", b"{}"),
            ("luke-7/../../escape.txt", b"escape"),
        ]
    )
    _assert_refused(env, data, "unsafe file path")


def test_a_duplicate_archive_entry_is_refused(env) -> None:
    with pytest.warns(UserWarning, match="Duplicate name"):
        data = _archive(
            [
                ("luke-7/manifest.json", b"{}"),
                ("luke-7/manifest.json", b"{}"),
            ]
        )
    _assert_refused(env, data, "duplicate entry")


def test_a_tampered_file_fails_its_sha256(env) -> None:
    payload, entries = helpers.unpack(env.package())
    entries["state/model-7.json"] = b'{"name": "Evil", "start_level": "0.0.0"}'
    _assert_refused(env, helpers.pack(payload, entries), "SHA-256")


def test_a_dataset_count_that_disagrees_with_the_manifest_is_refused(env) -> None:
    data = _repack(env, "dataset/faqs.jsonl", b"")
    _assert_refused(env, data, "dataset count")


def test_a_document_referring_to_a_missing_upload_is_refused(env) -> None:
    _, entries = helpers.unpack(env.package())
    rows = [
        {**json.loads(line), "document_id": 999}
        for line in entries["dataset/documents.jsonl"].splitlines()
    ]
    data = "".join(json.dumps(row) + "\n" for row in rows).encode()
    _assert_refused(env, _repack(env, "dataset/documents.jsonl", data), "missing document_id")


def test_a_raw_upload_that_does_not_match_its_hash_is_refused(env) -> None:
    _assert_refused(env, _repack(env, "raw/moon.txt", b"x" * len(helpers.TEXT)), "content hash")


def test_a_corrupt_checkpoint_is_refused(env) -> None:
    _assert_refused(env, _repack(env, "weights/checkpoints/words-7.pt", b"junk"), "checkpoint")


def test_a_tokenizer_artifact_that_does_not_match_its_id_is_refused(env) -> None:
    _, entries = helpers.unpack(env.package())
    name = next(
        name
        for name in entries
        if name.endswith("/tokenizer.json") and name.startswith("weights/tokenizers/")
    )
    _assert_refused(env, _repack(env, name, b"{}"), "Tokenizer artifact hash")


def test_a_checkpoint_for_another_workspace_is_refused(env) -> None:
    payload, entries = helpers.unpack(env.package())
    entries["weights/checkpoints/words-8.pt"] = entries["weights/checkpoints/words-7.pt"]
    payload["files"].append(
        helpers.inventory(
            {"weights/checkpoints/words-8.pt": entries["weights/checkpoints/words-8.pt"]}
        )[0]
    )
    _assert_refused(env, helpers.pack(payload, entries), "another workspace's checkpoint")


def test_an_upload_beyond_the_size_limit_is_refused(env, monkeypatch) -> None:
    data = env.package()
    monkeypatch.setattr(settings, "max_package_bytes", 128)
    _assert_refused(env, data, "import limit")
