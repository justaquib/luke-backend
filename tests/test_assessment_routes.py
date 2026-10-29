"""The assessment routes: start, follow, read the cards and review a rubric."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import get_bearer_user
from app.db import database
from app.db.database import AssessmentResult, AssessmentRun, User, get_session
from app.routers import assessment as assessment_router
from app.services.assessment import bank, runner

PROVENANCE = {
    "model_id": "Luke 0.0.1",
    "model_artifact": "Qwen/Qwen2.5-0.5B-Instruct@revision#0123456789abcdef",
    "tokenizer_id": "Qwen2.5-0.5B-Instruct@7ae5576:tokenizer.json:0123456789abcdef",
    "generation": '{"max_new_tokens": 220, "temperature": 0.0}',
    "bank_version": 1,
    "bank_salt": "luke-assessment-v1",
}


def _user() -> User:
    return User(id=1, email="trainer@example.com", name="Trainer", tenant_id=0)


@pytest.fixture()
def engine() -> Iterator[Engine]:
    bound = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    database.Base.metadata.create_all(bound)
    yield bound
    bound.dispose()


@pytest.fixture()
def factory(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> sessionmaker[Session]:
    made = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(database, "SessionLocal", made)
    monkeypatch.setattr(runner, "_provenance", lambda: dict(PROVENANCE))
    monkeypatch.setattr(runner, "submit", lambda run_id: None)
    return made


@pytest.fixture()
def client(factory: sessionmaker[Session]) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(assessment_router.router)
    app.dependency_overrides[get_bearer_user] = _user
    app.dependency_overrides[get_session] = lambda: None
    with TestClient(app) as test_client:
        yield test_client


def _done_run(factory: sessionmaker[Session]) -> int:
    with factory() as db:
        run = AssessmentRun(mode="unaided", status="done", total=2, tenant_id=0)
        db.add(run)
        db.commit()
        return int(run.id)


def test_no_run_still_shows_eight_unassessed_cards(client: TestClient) -> None:
    response = client.get("/assessment/profiles")

    assert response.status_code == 200
    body = response.json()
    assert body["run"] is None
    assert len(body["profiles"]) == 8
    assert {profile["status"] for profile in body["profiles"]} == {"unassessed"}
    assert all(profile["score"] is None for profile in body["profiles"])


def test_starting_a_run_returns_202_with_its_provenance(client: TestClient) -> None:
    response = client.post("/assessment/runs", json={"mode": "unaided"})

    assert response.status_code == 202
    body = response.json()
    assert body["run"]["status"] == "queued"
    assert body["run"]["mode"] == "unaided"
    assert body["run"]["bank"]["eval_items"] == 240
    assert body["run"]["model_artifact"] == PROVENANCE["model_artifact"]
    assert client.get("/assessment/runs").json()[0]["id"] == body["run_id"]


def test_an_unknown_mode_is_refused(client: TestClient) -> None:
    assert client.post("/assessment/runs", json={"mode": "cheating"}).status_code == 422
    assert client.get("/assessment/profiles?mode=cheating").status_code == 400


def test_a_completed_run_fills_the_skill_cards(
    client: TestClient, factory: sessionmaker[Session]
) -> None:
    with factory() as db:
        run = AssessmentRun(mode="unaided", status="done", total=2, tenant_id=0)
        db.add(run)
        db.flush()
        for index, score in enumerate((1.0, 0.0)):
            db.add(
                AssessmentResult(
                    run_id=run.id,
                    item_id=f"grammar:blimp:{index}",
                    skill="grammar",
                    split="eval",
                    kind="option",
                    stage=1,
                    topic="agreement",
                    method="option",
                    score=score,
                )
            )
        db.commit()
        run_id = run.id

    body = client.get(f"/assessment/profiles?run_id={run_id}").json()
    grammar = next(card for card in body["profiles"] if card["skill"] == "grammar")

    assert body["run"]["id"] == run_id
    assert grammar["status"] == "assessed"
    assert grammar["score"] == 0.5
    assert grammar["verdict"] == "benchmark not yet passed"


def test_only_a_waiting_rubric_answer_can_be_reviewed(
    client: TestClient, factory: sessionmaker[Session]
) -> None:
    with factory() as db:
        run = AssessmentRun(mode="unaided", status="done", total=2, tenant_id=0)
        db.add(run)
        db.flush()
        rubric = AssessmentResult(
            run_id=run.id,
            item_id="creative_writing:prompt:1",
            skill="creative_writing",
            split="eval",
            kind="rubric",
            method="rubric",
            stage=1,
            topic="story",
        )
        option = AssessmentResult(
            run_id=run.id,
            item_id="grammar:blimp:0",
            skill="grammar",
            split="eval",
            kind="option",
            method="option",
            stage=1,
            topic="agreement",
            score=1.0,
            correct=True,
        )
        db.add_all([rubric, option])
        db.commit()
        run_id, rubric_id, option_id = run.id, rubric.id, option.id

    reviewed = client.post(
        f"/assessment/results/{rubric_id}/review", json={"score": 0.9, "note": "clear story"}
    )

    assert reviewed.status_code == 200
    assert reviewed.json()["score"] == 0.9
    assert reviewed.json()["correct"] is True
    assert reviewed.json()["reviewed_by"] == "trainer@example.com"
    assert reviewed.json()["review_note"] == "clear story"
    refused = client.post(f"/assessment/results/{option_id}/review", json={"score": 0.9})
    assert refused.status_code == 400
    creative = next(
        card
        for card in client.get(f"/assessment/profiles?run_id={run_id}").json()["profiles"]
        if card["skill"] == "creative_writing"
    )
    assert creative["score"] == 0.9


def test_a_rubric_result_carries_what_the_trainer_judges(
    client: TestClient, factory: sessionmaker[Session]
) -> None:
    item = bank.items("creative_writing", "eval")[0]
    with factory() as db:
        run = AssessmentRun(mode="unaided", status="done", total=1, tenant_id=0)
        db.add(run)
        db.flush()
        db.add(
            AssessmentResult(
                run_id=run.id,
                item_id=str(item["id"]),
                skill="creative_writing",
                split="eval",
                kind="rubric",
                method="rubric",
                stage=1,
                topic="story",
            )
        )
        db.commit()
        run_id = run.id

    rows = client.get(f"/assessment/runs/{run_id}/results").json()

    assert rows[0]["prompt"] == item["prompt"]
    assert rows[0]["criteria"] == item["expected"]["criteria"]
    assert rows[0]["score"] is None


def test_a_done_run_streams_its_terminal_state(
    client: TestClient, factory: sessionmaker[Session]
) -> None:
    run_id = _done_run(factory)

    response = client.get(f"/assessment/runs/{run_id}/events")

    assert response.status_code == 200
    assert "event: run" in response.text
    assert '"status": "done"' in response.text


def test_cancel_marks_a_queued_run_stopped(
    client: TestClient, factory: sessionmaker[Session]
) -> None:
    with factory() as db:
        run = AssessmentRun(mode="assisted", status="queued", total=2, tenant_id=0)
        db.add(run)
        db.commit()
        run_id = run.id

    response = client.post(f"/assessment/runs/{run_id}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert response.json()["step"] == "Stopped by the trainer."
