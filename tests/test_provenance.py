"""A chunk remembers its place in the source and its own quality."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, inspect, text

from app.core.extraction import ExtractedText, PageText, extract_document
from app.services import chunk_vectors
from app.services.nlp import ChunkSpan, chunk_spans
from app.services.provenance import content_hash, facts_for
from scripts import migrate_provenance


def test_facts_name_the_page_and_exact_characters_of_each_chunk():
    text = "First page words.\nSecond page words."
    extracted = ExtractedText(
        text=text,
        pages=(
            PageText(1, "First page words.", 0, 17),
            PageText(2, "Second page words.", 18, 36),
        ),
    )

    spans = chunk_spans(text, max_chars=18, overlap=0)
    facts = facts_for(spans, extracted)

    assert [(fact.char_start, fact.char_end, fact.page) for fact in facts] == [
        (0, 17, 1),
        (18, 36, 2),
    ]
    assert [fact.content_hash for fact in facts] == [
        content_hash("First page words."),
        content_hash("Second page words."),
    ]
    assert all(0.0 <= fact.quality <= 1.0 for fact in facts)


def test_a_pdf_page_span_holds_the_text_of_that_page(tmp_path):
    import pymupdf

    document = pymupdf.open()
    for words in ("First page words.", "Second page words."):
        page = document.new_page()
        page.insert_text((72, 72), words)
    path = tmp_path / "two-pages.pdf"
    document.save(path)
    document.close()

    extracted = extract_document(str(path))

    assert [page.page for page in extracted.pages] == [1, 2]
    for page in extracted.pages:
        assert extracted.text[page.char_start : page.char_end] == page.text
        assert page.text.strip()


def test_a_format_without_pages_gets_no_page_number():
    spans = [ChunkSpan("alpha beta gamma", 0, 16)]

    facts = facts_for(spans)

    assert facts[0].page is None
    assert facts[0].char_start == 0
    assert facts[0].char_end == 16


def test_chunk_notes_carry_provenance_beside_the_vector(monkeypatch):
    collection = _collection(monkeypatch)
    facts = facts_for([ChunkSpan("Avery wrote this.", 3, 20)])

    chunk_vectors.add_chunks(
        "voucher.pdf",
        ["Avery wrote this."],
        ["voucher-1"],
        tenant_id=7,
        notes=chunk_vectors.ChunkNotes.from_facts(facts, document_id=12),
    )

    metadata = collection.upsert.call_args.kwargs["metadatas"][0]
    assert metadata["document_id"] == 12
    assert metadata["chunk_index"] == 0
    assert metadata["char_start"] == 3
    assert metadata["char_end"] == 20
    assert metadata["content_hash"] == content_hash("Avery wrote this.")
    assert metadata["language"] in {"en", "unknown"}
    assert 0.0 <= metadata["quality"] <= 1.0


def _collection(monkeypatch) -> Any:
    from unittest.mock import Mock

    collection = Mock()
    monkeypatch.setattr(chunk_vectors, "get_collection", lambda: collection)
    return collection


def _documents_database() -> Any:
    engine = create_engine("sqlite://")
    metadata = MetaData()
    Table(
        "documents",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("source", String, nullable=False),
    )
    metadata.create_all(engine)
    return engine


def _names() -> list[str]:
    return [name for name, _ in migrate_provenance.COLUMNS]


def test_the_migration_adds_each_missing_column_once():
    engine = _documents_database()

    assert migrate_provenance.plan(engine)["missing"] == _names()
    assert migrate_provenance.add(engine) == _names()
    assert migrate_provenance.plan(engine)["missing"] == []
    assert migrate_provenance.add(engine) == []


def test_the_migration_leaves_existing_rows_untouched():
    engine = _documents_database()
    with engine.begin() as connection:
        connection.execute(text("INSERT INTO documents (id, source) VALUES (1, 'notes.txt')"))

    migrate_provenance.add(engine)

    with engine.connect() as connection:
        row = connection.execute(text("SELECT source, page FROM documents")).one()
    assert row.source == "notes.txt"
    assert row.page is None


def test_rollback_removes_only_the_provenance_columns():
    engine = _documents_database()
    migrate_provenance.add(engine)

    dropped = migrate_provenance.drop(engine)

    assert dropped == list(reversed(_names()))
    assert {column["name"] for column in inspect(engine).get_columns("documents")} == {
        "id",
        "source",
    }
