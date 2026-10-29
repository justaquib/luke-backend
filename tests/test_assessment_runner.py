"""A run measures every held-out item and keeps what produced the number."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import database
from app.db.database import AssessmentResult, AssessmentRun
from app.services.assessment import bank, profiles, runner, skills
from app.services.foundation.identity import FoundationMissingError

PROVENANCE = {
    "model_id": "Luke 0.0.1",
    "model_artifact": "Qwen/Qwen2.5-0.5B-Instruct@revision#0123456789abcdef",
    "tokenizer_id": "Qwen2.5-0.5B-Instruct@7ae5576:tokenizer.json:0123456789abcdef",
    "generation": '{"max_new_tokens": 220, "temperature": 0.0}',
    "bank_version": 1,
    "bank_salt": "luke-assessment-v1",
}


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
    return made


@pytest.fixture()
def answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "_provenance", lambda: dict(PROVENANCE))
    monkeypatch.setattr(runner, "_load_model", lambda: "model")
    monkeypatch.setattr(runner, "_generate", lambda model, messages, cancel: ("B", 0.01))


def _run(factory: sessionmaker[Session], run_id: int) -> AssessmentRun:
    with factory() as db:
        stored = db.get(AssessmentRun, run_id)
        assert stored is not None
        return stored


def test_a_run_answers_every_held_out_item_and_records_its_provenance(
    factory: sessionmaker[Session], answer: None
) -> None:
    run_id = runner.create(None, runner.UNAIDED)

    runner.execute(run_id)

    snapshot = runner.snapshot(run_id, None)
    assert snapshot["status"] == "done"
    assert snapshot["done"] == 240
    assert snapshot["total"] == 240
    assert snapshot["progress"] == 1.0
    assert snapshot["model_id"] == PROVENANCE["model_id"]
    assert snapshot["model_artifact"] == PROVENANCE["model_artifact"]
    assert snapshot["tokenizer_id"] == PROVENANCE["tokenizer_id"]
    assert snapshot["generation"] == {"max_new_tokens": 220, "temperature": 0.0}
    assert snapshot["bank"] == {"version": 1, "salt": "luke-assessment-v1", "eval_items": 240}
    with factory() as db:
        assert db.query(AssessmentResult).count() == 240


def test_each_kind_is_marked_the_way_it_says(factory: sessionmaker[Session], answer: None) -> None:
    run_id = runner.create(None, runner.UNAIDED)

    runner.execute(run_id)

    rows = runner.results(run_id, None)
    methods = {row.kind: row.method for row in rows}
    assert methods[skills.OPTION] == "option"
    assert methods[skills.SPAN] == "span"
    assert methods[skills.NUMBER] == "number"
    assert methods[skills.LITERAL] == "literal"
    assert methods[skills.RULES] in {"rules", "unassessed"}
    rubric = next(row for row in rows if row.kind == skills.RUBRIC)
    assert rubric.method == "rubric"
    assert rubric.score is None
    assert rubric.correct is None


def test_an_unaided_run_never_reads_the_corpus(
    factory: sessionmaker[Session], answer: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(prompt: str, tenant_id: int | None) -> tuple[dict[str, str], ...]:
        raise AssertionError("an unaided run must not ask the corpus")

    monkeypatch.setattr(runner, "_evidence", refuse)
    run_id = runner.create(None, runner.UNAIDED)

    runner.execute(run_id)

    assert runner.snapshot(run_id, None)["status"] == "done"


def test_an_assisted_run_reads_the_corpus_for_every_item(
    factory: sessionmaker[Session], answer: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []
    monkeypatch.setattr(
        runner,
        "_evidence",
        lambda prompt, tenant_id: asked.append(prompt) or (),
    )
    run_id = runner.create(None, runner.ASSISTED)

    runner.execute(run_id)

    assert len(asked) == 240


def test_cancel_stops_after_the_item_in_flight(
    factory: sessionmaker[Session], answer: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = runner.create(None, runner.UNAIDED)

    def stop(model: object, messages: list[dict[str, str]], cancel: object) -> tuple[str, float]:
        runner.cancel(run_id, None)
        return "B", 0.01

    monkeypatch.setattr(runner, "_generate", stop)

    runner.execute(run_id)

    snapshot = runner.snapshot(run_id, None)
    assert snapshot["status"] == "cancelled"
    assert snapshot["done"] == 1
    assert len(runner.results(run_id, None)) == 1


def test_a_run_cancelled_before_it_started_never_answers(
    factory: sessionmaker[Session], answer: None
) -> None:
    run_id = runner.create(None, runner.UNAIDED)
    runner.cancel(run_id, None)

    runner.execute(run_id)

    assert runner.snapshot(run_id, None)["status"] == "cancelled"
    assert runner.results(run_id, None) == []


def test_a_crash_marks_the_run_failed_without_blaming_the_model(
    factory: sessionmaker[Session], answer: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(model: object, messages: list[dict[str, str]], cancel: object) -> tuple[str, float]:
        raise RuntimeError("no memory")

    monkeypatch.setattr(runner, "_generate", crash)
    run_id = runner.create(None, runner.UNAIDED)

    runner.execute(run_id)

    snapshot = runner.snapshot(run_id, None)
    assert snapshot["status"] == "failed"
    assert snapshot["error"] == runner.SAFE_FAILURE


def test_a_missing_foundation_keeps_its_actionable_reason(
    factory: sessionmaker[Session], answer: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(model: object, messages: list[dict[str, str]], cancel: object) -> tuple[str, float]:
        raise FoundationMissingError("the checkpoint is not installed")

    monkeypatch.setattr(runner, "_generate", missing)
    run_id = runner.create(None, runner.UNAIDED)

    runner.execute(run_id)

    assert runner.snapshot(run_id, None)["error"] == (
        "Luke's foundation could not answer: the checkpoint is not installed"
    )


def test_recover_marks_what_a_restart_interrupted(
    factory: sessionmaker[Session], answer: None
) -> None:
    runner.create(None, runner.UNAIDED)
    runner.create(None, runner.ASSISTED)
    with factory() as db:
        db.get(AssessmentRun, 2).status = "running"
        db.commit()

    recovered = runner.recover()

    assert recovered == 2
    for run_id in (1, 2):
        snapshot = runner.snapshot(run_id, None)
        assert snapshot["status"] == "failed"
        assert snapshot["error"] == runner.INTERRUPTED


def test_start_queues_without_running(
    factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    queued: list[int] = []
    monkeypatch.setattr(runner, "_provenance", lambda: dict(PROVENANCE))
    monkeypatch.setattr(runner, "submit", queued.append)

    run_id = runner.start(None, runner.UNAIDED)

    assert queued == [run_id]
    assert runner.snapshot(run_id, None)["status"] == "queued"


def test_an_unknown_mode_is_refused(factory: sessionmaker[Session]) -> None:
    with pytest.raises(ValueError, match="unknown assessment mode"):
        runner.create(None, "cheating")


def test_the_previous_completed_score_is_per_skill_and_per_mode(
    factory: sessionmaker[Session],
) -> None:
    with factory() as db:
        older = AssessmentRun(mode="unaided", status="done", total=2)
        db.add(older)
        db.flush()
        for score in (1.0, 0.0):
            db.add(
                AssessmentResult(
                    run_id=older.id,
                    item_id=f"grammar:blimp:{score}",
                    skill="grammar",
                    split="eval",
                    kind="option",
                    score=score,
                )
            )
        newer = AssessmentRun(mode="unaided", status="queued", total=2)
        db.add(newer)
        db.commit()
        older_id, newer_id = older.id, newer.id

    assert profiles.previous_score("grammar", "unaided", newer_id) == 0.5
    assert profiles.previous_score("grammar", "assisted", newer_id) is None
    assert profiles.previous_score("reading", "unaided", newer_id) is None
    assert profiles.previous_score("grammar", "unaided", older_id) is None


def test_a_finished_run_builds_profiles_and_rubrics_wait(
    factory: sessionmaker[Session], answer: None
) -> None:
    run_id = runner.create(None, runner.UNAIDED)
    runner.execute(run_id)

    rows = runner.results(run_id, None)
    creative = skills.build_profile(
        "creative_writing",
        [profiles.outcome(row) for row in rows if row.skill == "creative_writing"],
        bank_items=bank.items("creative_writing", "eval"),
    )
    grammar = skills.build_profile(
        "grammar",
        [profiles.outcome(row) for row in rows if row.skill == "grammar"],
        bank_items=bank.items("grammar", "eval"),
    )

    assert creative.status == "unassessed"
    assert creative.score is None
    assert creative.waiting == 30
    assert grammar.status == "assessed"
    assert grammar.score is not None
