"""Package v2 restores preserve content, provenance, ids and transactional safety."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator

import package_helpers as helpers
import package_models
import pytest
from sqlalchemy import select

from app.db.database import Document, Faq, Feedback, Intent, UploadedFile
from app.services import model_settings, vector_store
from app.services.foundation import identity
from app.services.language import runtime
from app.services.language.package import dataset, reindex, restore
from app.services.language.package.manifest import PackageError
from app.services.neural import checkpoint
from app.services.neural.tokenizer import artifacts
from app.services.neural.words import WordTask


@pytest.fixture()
def env(monkeypatch, tmp_path) -> Iterator[helpers.Environment]:
    """A real temporary database with offline index collections."""
    with helpers.environment(monkeypatch, tmp_path) as isolated:
        yield isolated


def _index_workspaces(env: helpers.Environment) -> None:
    with env.sessions() as db:
        for workspace in (helpers.WORKSPACE, helpers.OTHER):
            reindex.rebuild(workspace, db)


def _erase_target(env: helpers.Environment) -> None:
    """Make the destination empty without calling production restore helpers."""
    with env.sessions() as db:
        dataset.clear(db, helpers.WORKSPACE)
        db.commit()
    for path in env.brain.iterdir():
        if path.is_file() and (path.name.endswith("-7.json") or path.name == "words-7.pt"):
            path.unlink()
    shutil.rmtree(env.uploads / "7")
    shutil.rmtree(artifacts.root())
    runtime._CACHE.clear()


def _queries() -> dict:
    """Keep answer content and provenance; uploaded-file ids may be remapped."""
    documents = vector_store.query(helpers.TEXT, 1, helpers.WORKSPACE)
    faqs = vector_store.search_faq(helpers.QUESTION, 1, helpers.WORKSPACE)
    assert documents and faqs
    for document in documents:
        document["metadata"].pop("document_id", None)
    for faq in faqs:
        faq.pop("faq_id", None)
    return {"documents": documents, "faqs": faqs}


def test_import_defaults_to_dry_run_and_does_not_touch_rows_files_or_vectors(env) -> None:
    data = env.package()
    payload, _ = helpers.unpack(data)
    _index_workspaces(env)
    before = env.snapshot()
    report = env.apply(data)
    assert report["applied"] is False
    assert report["replace_required"] is True
    assert report["format_version"] == 2
    assert report["workspace"] == helpers.WORKSPACE
    assert report["mode"] == "slim"
    assert report["counts"] == payload["counts"]
    assert report["files"] == {
        "checked": len(payload["files"]),
        "bytes": sum(row["bytes"] for row in payload["files"]),
    }
    assert env.snapshot() == before
    assert not list(env.root.glob("luke-import-*"))
    assert not list(env.brain.glob("import-*"))


def test_nonempty_workspace_requires_explicit_replace(env) -> None:
    data = env.package()
    _index_workspaces(env)
    before = env.snapshot()
    with pytest.raises(restore.PackageConflictError, match="replace=true"):
        env.apply(data, dry_run=False)
    assert env.snapshot() == before
    assert not list(env.root.glob("luke-import-*"))


@pytest.mark.parametrize("kind", ["word", "bpe"])
def test_successful_restore_into_empty_workspace_keeps_provenance_and_queries(env, kind) -> None:
    package_models.write_scratch(env.brain, helpers.WORKSPACE, kind)
    _index_workspaces(env)
    before_queries = _queries()
    with env.sessions() as db:
        original_rows = dataset.dump_rows(db, helpers.WORKSPACE)
    data = env.package()
    _, entries = helpers.unpack(data)
    _erase_target(env)
    report = env.apply(data, dry_run=False)
    assert report["applied"] is True
    assert report["replace_required"] is False
    assert report["rows"] == {
        "documents": 1,
        "faqs": 1,
        "pending": 1,
        "feedback": 1,
        "uploads": 1,
        "intents": 0,
    }
    assert report["verified"]["samples"] == {"documents": 1, "faqs": 1}
    assert report["indexed"]["documents"] == 1
    _assert_rows_restored(env, original_rows)
    assert _queries() == before_queries
    assert model_settings.read(helpers.WORKSPACE).name == "Luna"
    assert (env.uploads / "7" / helpers.SOURCE).read_bytes() == entries["raw/moon.txt"]
    assert (env.brain / "words-7.pt").read_bytes() == entries["weights/checkpoints/words-7.pt"]
    loaded = WordTask.load(env.brain / "words-7.pt")
    assert loaded is not None and loaded.tokenizer.kind == kind
    assert not list(env.brain.glob("import-*"))


def test_intent_keeps_its_answer_link_and_is_indexed_again(env) -> None:
    with env.sessions() as db:
        db.add(
            Intent(
                id=12,
                tenant_id=helpers.WORKSPACE,
                name="moon_fact",
                response_template="The moon reflects sunlight.",
                faq_id=10,
                created_at=helpers.MOMENT,
                updated_at=helpers.MOMENT,
            )
        )
        db.commit()
    _index_workspaces(env)
    data = env.package()
    _, entries = helpers.unpack(data)
    exported = [json.loads(line) for line in entries["dataset/intents.jsonl"].splitlines()]
    assert exported == [
        {
            "name": "moon_fact",
            "response_template": "The moon reflects sunlight.",
            "faq_id": 10,
            "created_at": helpers.MOMENT.isoformat(),
            "updated_at": helpers.MOMENT.isoformat(),
        }
    ]
    _erase_target(env)
    report = env.apply(data, dry_run=False)
    assert report["rows"]["intents"] == 1
    assert report["verified"]["counts"]["intents"] == 1
    with env.sessions() as db:
        intent = db.scalars(select(Intent).where(Intent.tenant_id == helpers.WORKSPACE)).one()
        faq = db.scalars(select(Faq).where(Faq.tenant_id == helpers.WORKSPACE)).one()
        assert intent.name == "moon_fact"
        assert intent.response_template == "The moon reflects sunlight."
        assert intent.faq_id == faq.id
        assert intent.faq_id != 10
        assert (intent.created_at, intent.updated_at) == (helpers.MOMENT, helpers.MOMENT)
    hits = vector_store.search_intents("moon_fact", 1, helpers.WORKSPACE)
    assert hits == [
        {
            "name": "moon_fact",
            "response_template": "The moon reflects sunlight.",
            "faq_id": faq.id,
            "distance": 0.0,
        }
    ]


def test_feedback_without_a_question_is_exported_and_restored(env) -> None:
    """A rating is real data even when the question behind it was not kept."""
    with env.sessions() as db:
        db.add(
            Feedback(
                id=11,
                tenant_id=helpers.WORKSPACE,
                question="",
                answer=helpers.ANSWER,
                source="faq",
                rating="like",
                created_at=helpers.MOMENT,
            )
        )
        db.commit()
    _index_workspaces(env)
    data = env.package()
    _, entries = helpers.unpack(data)
    exported = [json.loads(line) for line in entries["dataset/feedback.jsonl"].splitlines()]
    assert exported[-1]["question"] == ""
    _erase_target(env)
    assert env.apply(data, dry_run=False)["rows"]["feedback"] == 2
    _assert_blank_feedback(env)
    assert env.apply(_without_questions(data), dry_run=False, replace=True)["rows"]["feedback"] == 2
    _assert_blank_feedback(env)


def _without_questions(data: bytes) -> bytes:
    """The same package with the empty question column left out of the rating."""
    payload, entries = helpers.unpack(data)
    rows = [json.loads(line) for line in entries["dataset/feedback.jsonl"].splitlines()]
    entries["dataset/feedback.jsonl"] = dataset.jsonl(
        [{key: value for key, value in row.items() if key != "question" or value} for row in rows]
    )
    payload["files"] = helpers.inventory(entries)
    return helpers.pack(payload, entries)


def _assert_blank_feedback(env: helpers.Environment) -> None:
    with env.sessions() as db:
        rows = db.scalars(select(Feedback).where(Feedback.tenant_id == helpers.WORKSPACE)).all()
    assert len(rows) == 2
    blank = [row for row in rows if row.question == ""]
    assert len(blank) == 1
    assert (blank[0].answer, blank[0].source, blank[0].rating) == (helpers.ANSWER, "faq", "like")


def _assert_rows_restored(env: helpers.Environment, original: dict) -> None:
    with env.sessions() as db:
        restored = dataset.dump_rows(db, helpers.WORKSPACE)
        upload = db.scalars(
            select(UploadedFile).where(UploadedFile.tenant_id == helpers.WORKSPACE)
        ).one()
        document = db.scalars(select(Document).where(Document.tenant_id == helpers.WORKSPACE)).one()
        assert document.document_id == upload.id
        assert document.content_hash == hashlib.sha256(helpers.TEXT.encode()).hexdigest()
        assert (document.page, document.char_start, document.char_end) == (3, 0, len(helpers.TEXT))
        assert (document.language, document.quality, document.created_at) == (
            "en",
            0.875,
            helpers.MOMENT,
        )
    original["uploads.jsonl"][0]["upload_id"] = restored["uploads.jsonl"][0]["upload_id"]
    original["documents.jsonl"][0]["document_id"] = restored["documents.jsonl"][0]["document_id"]
    original["faqs.jsonl"][0]["faq_id"] = restored["faqs.jsonl"][0]["faq_id"]
    assert restored == original


def test_explicit_replace_restores_files_and_leaves_other_workspace_untouched(env) -> None:
    data = env.package()
    _index_workspaces(env)
    before = env.snapshot()
    helpers.write_json(env.brain / "model-7.json", {"name": "Previous", "start_level": "0.0.0"})
    (env.uploads / "7" / "stale.txt").write_bytes(b"not in the imported package")
    report = env.apply(data, dry_run=False, replace=True)
    after = env.snapshot()
    assert report["applied"] is True and report["replace_required"] is True
    assert model_settings.read(helpers.WORKSPACE).name == "Luna"
    assert not (env.uploads / "7" / "stale.txt").exists()
    for name, rows in before["rows"].items():
        assert [row for row in after["rows"][name] if row["tenant_id"] != helpers.WORKSPACE] == [
            row for row in rows if row["tenant_id"] != helpers.WORKSPACE
        ]
    for name, data in before["files"].items():
        if "-8." in name or "/8/" in name:
            assert after["files"][name] == data
    for name, rows in before["vectors"].items():
        for identifier, row in rows.items():
            if row[1].get("tenant_id") == str(helpers.OTHER):
                assert after["vectors"][name][identifier] == row


def test_importing_same_package_twice_keeps_all_row_and_vector_ids_stable(env) -> None:
    data = env.package()
    _index_workspaces(env)
    first = env.apply(data, dry_run=False, replace=True)
    before = env.snapshot()
    second = env.apply(data, dry_run=False, replace=True)
    assert first["rows"] == second["rows"]
    assert first["indexed"] == second["indexed"]
    assert first["verified"] == second["verified"]
    assert env.snapshot() == before


def test_full_import_restores_the_actual_base_and_tokenizer_artifacts(env) -> None:
    folder = package_models.write_base()
    data = env.package("full")
    payload, entries = helpers.unpack(data)
    shutil.rmtree(folder)
    shutil.rmtree(artifacts.root())
    report = env.apply(data, dry_run=False, replace=True)
    assert report["applied"] is True
    assert report["files"]["base"] == payload["counts"]["base_files"]
    assert report["files"]["tokenizers"] == 2
    for name, content in entries.items():
        if name.startswith("weights/base/"):
            assert (folder / name.removeprefix("weights/base/")).read_bytes() == content
    identity._verified.cache_clear()
    assert identity.load_identity(verify_weights=True).parameters == 6
    stored = checkpoint.load(env.brain / "words-7.pt")
    assert stored is not None
    tokenizer = artifacts.load(stored.tokenizer)
    assert tokenizer.decode(tokenizer.encode(package_models.TEXT)) == package_models.TEXT


def test_full_import_replaces_same_size_base_with_different_hash(env) -> None:
    folder = package_models.write_base()
    data = env.package("full")
    expected = (folder / "model.safetensors").read_bytes()
    package_models.write_base(offset=1)
    assert (folder / "model.safetensors").stat().st_size == len(expected)
    assert (folder / "model.safetensors").read_bytes() != expected
    env.apply(data, dry_run=False, replace=True)
    assert (folder / "model.safetensors").read_bytes() == expected
    identity._verified.cache_clear()
    identity.load_identity(verify_weights=True)


def test_slim_import_never_overwrites_an_installed_base(env) -> None:
    package_models.write_base()
    data = env.package("slim")
    folder = package_models.write_base(offset=1)
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    env.apply(data, dry_run=False, replace=True)
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before


@pytest.mark.parametrize("section", ["rows", "files", "vectors"])
@pytest.mark.parametrize("existing_shared", [True, False])
def test_verify_failure_rolls_back_original_rows_ids_files_and_index(
    env, monkeypatch, section, existing_shared
) -> None:
    folder = package_models.write_base()
    data = env.package("full")
    if existing_shared:
        package_models.write_base(offset=1)
    else:
        shutil.rmtree(folder)
        shutil.rmtree(artifacts.root())
    helpers.write_json(env.brain / "model-7.json", {"name": "Previous", "start_level": "0.0.0"})
    (env.uploads / "7" / "previous.txt").write_bytes(b"previous upload")
    with env.sessions() as db:
        faq = db.get(Faq, 10)
        faq.answer = "The previous answer."
        db.commit()
    _index_workspaces(env)
    before = env.snapshot()
    real_verify = reindex.verify
    reached = []

    def fail_after_verify(workspace, db, counts) -> dict:
        real_verify(workspace, db, counts)
        reached.append(True)
        assert model_settings.read(workspace).name == "Luna"
        raise PackageError("injected post-reindex verification failure")

    monkeypatch.setattr(reindex, "verify", fail_after_verify)
    with pytest.raises(PackageError, match="injected post-reindex"):
        env.apply(data, dry_run=False, replace=True)
    assert reached == [True]
    assert env.snapshot()[section] == before[section]
    assert not list(env.root.glob("luke-import-*"))
    assert not list(env.brain.glob("import-*"))
