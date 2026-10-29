"""The assessment tables can be added to an existing workspace, and undone."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import database
from app.db.database import AssessmentResult, AssessmentRun
from scripts import migrate_assessment


@pytest.fixture()
def engine() -> Iterator[Engine]:
    """An empty database, as a workspace from before Phase 6 would look."""
    bound = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    yield bound
    bound.dispose()


def _present(bound: Engine) -> set[str]:
    return set(inspect(bound).get_table_names())


def _run(**values: object) -> AssessmentRun:
    run = AssessmentRun(mode="unaided", model_id="luke-0.0.1", **values)
    return run


def test_the_plan_names_the_missing_tables(engine: Engine) -> None:
    plan = migrate_assessment.plan(engine)

    assert plan["tables"] == []
    assert plan["missing"] == ["assessment_runs", "assessment_results"]
    assert plan["rows"] == {"assessment_runs": 0, "assessment_results": 0}


def test_creating_adds_both_tables_and_the_second_run_is_a_no_op(engine: Engine) -> None:
    created = migrate_assessment.create(engine)

    assert created == ["assessment_runs", "assessment_results"]
    assert _present(engine) >= {"assessment_runs", "assessment_results"}
    assert migrate_assessment.create(engine) == []
    assert migrate_assessment.plan(engine)["missing"] == []


def test_the_plan_counts_the_rows_that_are_there(engine: Engine) -> None:
    migrate_assessment.create(engine)
    factory = sessionmaker(bind=engine, autoflush=False)
    with factory() as db:
        run = _run()
        run.results.append(
            AssessmentResult(
                item_id="grammar:blimp:0",
                skill="grammar",
                split="eval",
                kind="option",
                method="exact",
                score=1.0,
                correct=True,
            )
        )
        db.add(run)
        db.commit()

    assert migrate_assessment.plan(engine)["rows"] == {
        "assessment_runs": 1,
        "assessment_results": 1,
    }


def test_a_new_run_is_unaided_and_a_rubric_result_starts_unassessed(engine: Engine) -> None:
    migrate_assessment.create(engine)
    factory = sessionmaker(bind=engine, autoflush=False)
    with factory() as db:
        run = _run()
        run.results.append(
            AssessmentResult(
                item_id="creative_writing:dolly_creative:0",
                skill="creative_writing",
                split="eval",
                kind="rubric",
                method="rubric",
            )
        )
        db.add(run)
        db.commit()
        stored = db.query(AssessmentResult).one()

        assert run.mode == "unaided"
        assert run.status == "queued"
        assert stored.score is None
        assert stored.correct is None
        assert stored.reviewed_at is None


def test_dropping_the_run_removes_its_results(engine: Engine) -> None:
    migrate_assessment.create(engine)
    factory = sessionmaker(bind=engine, autoflush=False)
    with factory() as db:
        run = _run()
        run.results.append(
            AssessmentResult(
                item_id="mathematics:gsm8k:1",
                skill="mathematics",
                split="eval",
                kind="number",
                method="exact",
                score=0.0,
                correct=False,
            )
        )
        db.add(run)
        db.commit()
        db.delete(run)
        db.commit()

        assert db.query(AssessmentResult).count() == 0


def test_rollback_drops_the_tables_again(engine: Engine) -> None:
    migrate_assessment.create(engine)

    dropped = migrate_assessment.drop(engine)

    assert dropped == ["assessment_runs", "assessment_results"]
    assert not _present(engine) & set(migrate_assessment.TABLES)
    assert migrate_assessment.plan(engine)["missing"] == list(migrate_assessment.TABLES)


def test_a_fresh_install_creates_the_tables_too(engine: Engine) -> None:
    database.Base.metadata.create_all(engine)

    assert {"assessment_runs", "assessment_results"} <= _present(engine)
