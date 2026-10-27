"""Tests for the knowledge transfer from a fine-tuned teacher into Luke.

No test loads a real model: the teacher is a small stand-in that writes from
its reference (or leaves it), so the seeds, the grounding rule, the stored FAQs
and the undo are tested without waiting on a generation.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.security import get_bearer_user
from app.db import database
from app.db.database import Faq, User, get_session
from app.routers import huggingface as huggingface_router
from app.services.huggingface import seeds as seed_service
from app.services.huggingface import service, serving, store, transfer
from app.services.huggingface.service import BaseModelBusyError
from app.services.language import runtime, training_materials

MODEL = "HuggingFaceTB/SmolLM2-135M"
TENANT = 7


class _Teacher:
    """A teacher that writes what it was given, or the text it was told."""

    def __init__(self, written: str | None = None) -> None:
        self.written = written

    def generate(self, question: str, reference: str = "") -> str | None:
        return self.written if self.written is not None else reference


@pytest.fixture()
def brain(monkeypatch, tmp_path: Path) -> Iterator[Path]:
    """A temporary brain folder and no run left behind."""
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    runtime._CACHE.clear()
    transfer._RUNNING.clear()
    service._RUNNING.clear()
    yield tmp_path
    transfer._RUNNING.clear()
    service._RUNNING.clear()
    runtime._CACHE.clear()


@pytest.fixture()
def sessions(monkeypatch, brain: Path) -> Iterator[sessionmaker[Session]]:
    """A temporary database, shared with the transfer's own sessions.

    It lives in a file, not in memory, so each session uses its own connection:
    the transfer writes on its thread while a request reads on another.
    """
    engine = create_engine(
        f"sqlite:///{brain / 'luke.db'}",
        connect_args={"check_same_thread": False},
    )
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(database, "SessionLocal", factory)
    monkeypatch.setattr(transfer, "SessionLocal", factory)
    yield factory
    engine.dispose()


@pytest.fixture()
def vectors(monkeypatch) -> tuple[list[int], list[int]]:
    """The FAQ ids the transfer indexed and removed."""
    upserted: list[int] = []
    deleted: list[int] = []
    monkeypatch.setattr(
        transfer.vector_store, "upsert_faq", lambda faq_id, *_args: upserted.append(faq_id)
    )
    monkeypatch.setattr(transfer.vector_store, "delete_faq", deleted.append)
    return upserted, deleted


def _teacher(tenant_id: int = TENANT) -> str:
    """Put one installed teacher's weights on the temporary machine."""
    folder = store.teacher_folder(MODEL)
    path = store.teacher_path(tenant_id, folder)
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.safetensors").write_bytes(b"weights")
    store.write_json(path / "training.json", {"model_id": MODEL, "name": "SmolLM2 135M"})
    return folder


def _materials() -> list[training_materials.Material]:
    """One lesson, one taught word and one uploaded document."""
    return [
        training_materials.Material(
            training_materials.LESSON, "What is a drum?", "A drum is hit with a stick."
        ),
        training_materials.Material(training_materials.WORD, "lunar", "of the moon"),
        training_materials.Material(
            training_materials.DOCUMENT, "notes.txt", "The moon reflects sunlight."
        ),
    ]


def _prepare(monkeypatch, written: str | None = None) -> None:
    monkeypatch.setattr(
        seed_service.training_materials, "gather", lambda tenant_id, db: _materials()
    )
    monkeypatch.setattr(serving, "load", lambda tenant_id, folder: _Teacher(written))


def _wait(tenant_id: int = TENANT) -> None:
    deadline = time.time() + 5
    while transfer.running(tenant_id) and time.time() < deadline:
        time.sleep(0.01)
    assert not transfer.running(tenant_id)


def _faqs(factory: sessionmaker[Session]) -> list[Faq]:
    with factory() as session:
        return list(session.execute(select(Faq)).scalars().all())


def test_plan_counts_each_source_and_skips_a_question_luke_has(
    monkeypatch, sessions, vectors, brain: Path
) -> None:
    _prepare(monkeypatch)
    with sessions() as db:
        db.add(
            Faq(
                question="what is a drum?",
                answer="A drum is hit.",
                status="pending",
                tenant_id=TENANT,
            )
        )
        db.commit()
        state = transfer.plan(TENANT, db)

    assert state["available"] == 2
    assert state["by_kind"] == {
        training_materials.WORD: 1,
        training_materials.DOCUMENT: 1,
    }
    questions = [sample["question"] for sample in state["samples"]]
    assert "What does the word lunar mean?" in questions
    assert "Tell me about notes." in questions
    assert state["running"] is False
    assert state["last"] is None


def test_start_keeps_grounded_answers_as_approved_faqs(
    monkeypatch, sessions, vectors, brain: Path
) -> None:
    _prepare(monkeypatch)
    folder = _teacher()
    with sessions() as db:
        started = transfer.start(TENANT, db, folder, 10)
    assert started["running"] is True
    _wait()

    record = store.read_json(store.transfer_path(TENANT))
    assert record["stage"] == "done"
    assert (record["done"], record["kept"], record["skipped"]) == (3, 3, 0)
    assert record["by_kind"] == {
        training_materials.LESSON: 1,
        training_materials.WORD: 1,
        training_materials.DOCUMENT: 1,
    }

    rows = _faqs(sessions)
    assert {row.status for row in rows} == {"approved"}
    assert {row.question for row in rows} == {
        "What is a drum?",
        "What does the word lunar mean?",
        "Tell me about notes.",
    }
    upserted, _deleted = vectors
    assert sorted(upserted) == sorted(row.id for row in rows)
    assert record["faq_ids"] == sorted(row.id for row in rows)


def test_start_skips_an_answer_that_leaves_the_material(
    monkeypatch, sessions, vectors, brain: Path
) -> None:
    _prepare(monkeypatch, written="Zebras gallop across the plain.")
    folder = _teacher()
    with sessions() as db:
        transfer.start(TENANT, db, folder, 10)
    _wait()

    record = store.read_json(store.transfer_path(TENANT))
    assert record["stage"] == "done"
    assert (record["done"], record["kept"], record["skipped"]) == (3, 0, 3)
    assert record["faq_ids"] == []
    assert _faqs(sessions) == []
    assert vectors[0] == []


def test_undo_removes_only_the_answers_the_last_transfer_made(
    monkeypatch, sessions, vectors, brain: Path
) -> None:
    _prepare(monkeypatch)
    folder = _teacher()
    with sessions() as db:
        db.add(
            Faq(
                question="Where is the office?",
                answer="On the first floor.",
                status="approved",
            )
        )
        db.commit()
        kept = db.execute(select(Faq.id)).scalars().all()
        transfer.start(TENANT, db, folder, 10)
    _wait()
    created = store.read_json(store.transfer_path(TENANT))["faq_ids"]

    with sessions() as db:
        undone = transfer.undo(TENANT, db)
    assert undone["last"]["stage"] == "undone"
    assert undone["last"]["undone"] == 3
    assert sorted(row.id for row in _faqs(sessions)) == list(kept)
    assert sorted(vectors[1]) == sorted(created)

    with sessions() as db, pytest.raises(transfer.TransferError):
        transfer.undo(TENANT, db)


def test_each_run_waits_for_the_other(monkeypatch, sessions, vectors, brain: Path) -> None:
    _prepare(monkeypatch)
    transfer._RUNNING.add(TENANT)
    with sessions() as db, pytest.raises(BaseModelBusyError):
        service.start(TENANT, MODEL, 1, [("What is a drum?", "A drum is hit.")])
    transfer._RUNNING.clear()

    monkeypatch.setattr(service, "running", lambda tenant_id: True)
    folder = _teacher()
    with sessions() as db, pytest.raises(transfer.TransferBusyError):
        transfer.start(TENANT, db, folder, 5)


def test_routes_plan_start_and_undo_a_transfer(monkeypatch, sessions, vectors, brain: Path) -> None:
    _prepare(monkeypatch)
    folder = _teacher()
    app = FastAPI()
    app.include_router(huggingface_router.router)
    app.dependency_overrides[get_bearer_user] = lambda: User(
        id=1, email="trainer@example.com", name="Trainer", is_systemadmin=False, tenant_id=TENANT
    )

    def _session() -> Iterator[Session]:
        with sessions() as db:
            yield db

    app.dependency_overrides[get_session] = _session
    with TestClient(app) as client:
        planned = client.get("/language-model/huggingface/transfer")
        assert planned.status_code == 200
        assert planned.json()["available"] == 3
        assert planned.json()["teachers"][0]["folder"] == folder

        started = client.post(
            "/language-model/huggingface/transfer", json={"folder": folder, "limit": 5}
        )
        assert started.status_code == 200
        assert started.json()["running"] is True

        _wait()
        report = client.get("/language-model/huggingface/transfer").json()
        assert report["running"] is False
        assert report["last"]["kept"] == 3

        undone = client.post("/language-model/huggingface/transfer/undo")
        assert undone.status_code == 200
        assert undone.json()["last"]["stage"] == "undone"

        missing = client.post(
            "/language-model/huggingface/transfer", json={"folder": "teacher-gone", "limit": 5}
        )
        assert missing.status_code == 404

        again = client.post("/language-model/huggingface/transfer/undo")
        assert again.status_code == 400
