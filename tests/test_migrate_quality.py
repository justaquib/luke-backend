"""The stored passages get the score the ingest path would have given them."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import database
from app.db.database import Document
from app.services.language.identify import guess
from app.services.quality.score import passage_quality
from scripts import migrate_quality

CLEAN = "The little dog ran across the wet grass and into the tall grass again."
SHORT = "wet grass"


@pytest.fixture()
def engine() -> Iterator[Engine]:
    """A documents table with two unscored rows and one row already measured."""
    bound = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    database.Base.metadata.create_all(bound)
    factory = sessionmaker(bind=bound, autoflush=False)
    with factory() as db:
        db.add_all(
            [
                Document(source="a.txt", chunk_index=0, text=CLEAN, tenant_id=0),
                Document(source="a.txt", chunk_index=1, text=SHORT, tenant_id=0),
                Document(
                    source="b.txt",
                    chunk_index=0,
                    text="already measured",
                    tenant_id=0,
                    language="de",
                    quality=0.25,
                ),
            ]
        )
        db.commit()
    yield bound
    bound.dispose()


def _values(bound: Engine) -> list[tuple[str, str, float]]:
    with bound.connect() as connection:
        rows = connection.execute(
            text("SELECT text, language, quality FROM documents ORDER BY id")
        ).all()
    return [(row.text, row.language, row.quality) for row in rows]


def test_the_plan_counts_the_rows_without_a_score(engine: Engine) -> None:
    plan = migrate_quality.plan(engine)

    assert plan["rows"] == 3
    assert plan["language"] == 2
    assert plan["quality"] == 2
    assert plan["missing"] == []


def test_filling_measures_the_rows_it_has_never_seen(engine: Engine) -> None:
    filled = migrate_quality.fill(engine)

    assert filled == 2
    assert _values(engine) == [
        (CLEAN, guess(CLEAN), passage_quality(CLEAN)),
        (SHORT, guess(SHORT), passage_quality(SHORT)),
        ("already measured", "de", 0.25),
    ]


def test_a_second_run_fills_nothing(engine: Engine) -> None:
    migrate_quality.fill(engine)

    assert migrate_quality.fill(engine) == 0
    assert migrate_quality.plan(engine)["quality"] == 0
