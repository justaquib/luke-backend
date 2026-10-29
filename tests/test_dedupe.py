"""A passage the workspace already holds is not stored a second time."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base, Document
from app.services import dedupe
from app.services.nlp import ChunkSpan
from app.services.provenance import content_hash
from app.services.quality.simhash import hamming, simhash, to_signed
from scripts import migrate_simhash

ORIGINAL = (
    "The rain falls on the green hills every morning, and the river runs to the sea "
    "through the valley where the old mill stands beside the road. Farmers bring their "
    "sheep down from the high fields before the winter, and the village prepares for "
    "the long cold months ahead with wood and grain."
)
NEAR = ORIGINAL.replace("every morning", "every evening")
OTHER = (
    "A spreadsheet lists invoices, totals and the date each payment is due. The finance "
    "team checks every row against the bank statement before the quarter closes, and "
    "marks the accounts that are still open so the next review starts from the right place."
)


def _span(text: str, start: int = 0) -> ChunkSpan:
    return ChunkSpan(text, start, start + len(text))


def _session() -> Session:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False)()


def test_the_same_passage_is_stored_once_even_inside_one_run():
    result = dedupe.collapse([_span(ORIGINAL), _span(ORIGINAL, 10)])
    assert result.stored == 1
    assert result.exact == 1
    second = dedupe.collapse([_span(ORIGINAL)], known_hashes={content_hash(ORIGINAL)})
    assert second.exact == 1
    assert second.stored == 0


def test_a_near_repeat_is_recognised_by_its_fingerprint():
    assert hamming(simhash(ORIGINAL), simhash(NEAR)) <= dedupe.NEAR_DUPLICATE_DISTANCE

    result = dedupe.collapse([_span(ORIGINAL), _span(NEAR)])

    assert result.stored == 1
    assert result.near == 1


def test_a_passage_about_something_else_is_kept():
    result = dedupe.collapse([_span(ORIGINAL), _span(OTHER)])

    assert result.stored == 2
    assert (result.exact, result.near) == (0, 0)


def test_the_hashes_and_fingerprints_of_a_workspace_are_read_from_the_rows():
    db = _session()
    db.add_all(
        [
            Document(
                source="notes.txt",
                chunk_index=0,
                text=ORIGINAL,
                tenant_id=0,
                content_hash=content_hash(ORIGINAL),
                simhash=to_signed(simhash(ORIGINAL)),
            ),
            Document(source="other.txt", chunk_index=0, text=OTHER, tenant_id=1),
        ]
    )
    db.commit()

    hashes, fingerprints = dedupe.known(db, 0)

    assert hashes == {content_hash(ORIGINAL)}
    assert fingerprints == [simhash(ORIGINAL)]
    db.close()


def _documents_table() -> Any:
    engine = create_engine("sqlite://")
    metadata = MetaData()
    Table(
        "documents",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("source", String, nullable=False),
        Column("text", String, nullable=True),
    )
    metadata.create_all(engine)
    return engine


def test_the_migration_adds_backfills_and_keeps_the_text():
    engine = _documents_table()
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO documents (id, source, text) VALUES (1, 'a.txt', :text)"),
            {"text": ORIGINAL},
        )
        connection.execute(
            text("INSERT INTO documents (id, source, text) VALUES (2, 'b.txt', :text)"),
            {"text": OTHER},
        )

    assert migrate_simhash.plan(engine)["missing"] == ["simhash"]
    assert migrate_simhash.add(engine) == ["simhash"]
    assert migrate_simhash.backfill(engine) == 2

    after = migrate_simhash.plan(engine)
    assert after["missing"] == []
    assert after["unbackfilled"] == 0
    with engine.connect() as connection:
        rows = connection.execute(text("SELECT id, text, simhash FROM documents")).all()
    assert rows[0].text == ORIGINAL
    assert rows[0].simhash is not None
    assert rows[1].simhash is not None
    assert migrate_simhash.add(engine) == []


def test_the_migration_rolls_back_the_fingerprint_only():
    engine = _documents_table()
    migrate_simhash.add(engine)

    assert migrate_simhash.drop(engine) == ["simhash"]
    assert migrate_simhash.drop(engine) == []
    assert migrate_simhash.plan(engine)["missing"] == ["simhash"]
