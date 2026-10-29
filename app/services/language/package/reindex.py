"""Rebuild a workspace's vectors from the database after a restore.

The index is not carried in the package: every chunk id is deterministic from
the workspace, the source and the chunk's own text, so the same rows produce
the same ids. The rebuild clears only this workspace, writes the document
chunks with their provenance, and indexes the approved and pending questions.
A sample query then proves the index answers from the restored rows.
"""

from __future__ import annotations

from itertools import groupby
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import Document, Faq, Intent, PendingQuestion
from app.services import chunk_vectors, vector_store
from app.services.chroma import get_collection, get_faq_collection, get_intents_collection, where
from app.services.chunk_vectors import ChunkNotes
from app.services.language.package.dataset import counts
from app.services.language.package.manifest import PackageError


def clear(tenant_id: int | None) -> None:
    """Remove this workspace's document, answer, pending and intent vectors, and no other's."""
    get_collection().delete(where=where({}, tenant_id))
    get_faq_collection().delete(where=where({}, tenant_id))
    get_intents_collection().delete(where=where({}, tenant_id))


def rebuild(tenant_id: int | None, db: Session) -> dict[str, int]:
    """Write every workspace row back into the index."""
    clear(tenant_id)
    documents = _documents(tenant_id, db)
    _faqs(tenant_id, db)
    _pending(tenant_id, db)
    _intents(tenant_id, db)
    return {"documents": documents, **_counts(tenant_id, db)}


def _documents(tenant_id: int | None, db: Session) -> int:
    statement = select(Document).where(_scope(Document.tenant_id, tenant_id))
    rows = db.execute(statement.order_by(Document.source, Document.chunk_index)).scalars()
    written = 0
    for source, group in groupby(rows, key=lambda row: row.source):
        batch = list(group)
        notes = ChunkNotes(
            page_numbers=[row.page for row in batch],
            document_ids=[row.document_id for row in batch],
            chunk_indexes=[row.chunk_index for row in batch],
            char_starts=[row.char_start for row in batch],
            char_ends=[row.char_end for row in batch],
            content_hashes=[row.content_hash for row in batch],
            languages=[row.language for row in batch],
            qualities=[row.quality for row in batch],
        )
        texts = [row.text for row in batch]
        ids = [chunk_vectors.chunk_id(tenant_id, source, text) for text in texts]
        written += chunk_vectors.add_chunks(source, texts, ids, tenant_id, notes)
    return written


def _faqs(tenant_id: int | None, db: Session) -> None:
    statement = select(Faq).where(_scope(Faq.tenant_id, tenant_id)).order_by(Faq.id)
    for row in db.execute(statement).scalars():
        vector_store.upsert_faq(row.id, row.question, row.answer, row.status, tenant_id)


def _pending(tenant_id: int | None, db: Session) -> None:
    statement = (
        select(PendingQuestion)
        .where(_scope(PendingQuestion.tenant_id, tenant_id))
        .order_by(PendingQuestion.id)
    )
    for row in db.execute(statement).scalars():
        vector_store.index_pending(row.id, row.question, tenant_id)


def _intents(tenant_id: int | None, db: Session) -> None:
    statement = select(Intent).where(_scope(Intent.tenant_id, tenant_id)).order_by(Intent.id)
    for row in db.execute(statement).scalars():
        vector_store.upsert_intent(row.id, row.name, row.response_template, row.faq_id, tenant_id)


def verify(tenant_id: int | None, db: Session, expected: dict[str, Any]) -> dict[str, Any]:
    """Compare the restored rows and prove the index answers a sample query."""
    actual = _counts(tenant_id, db)
    _compare(expected, actual)
    samples = _samples(tenant_id, db)
    return {"counts": actual, "samples": samples}


def _compare(expected: dict[str, Any], actual: dict[str, int]) -> None:
    for name in ("documents", "faqs", "pending_total", "feedback", "uploads", "intents"):
        if name not in expected:
            continue
        if int(expected[name]) != actual[name]:
            raise PackageError(
                f"The restore wrote {actual[name]} {name} but the package records {expected[name]}."
            )


def _samples(tenant_id: int | None, db: Session) -> dict[str, Any]:
    """One document and one answer, searched back through the rebuilt index."""
    checked: dict[str, Any] = {"documents": 0, "faqs": 0}
    document = (
        db.execute(
            select(Document).where(_scope(Document.tenant_id, tenant_id)).order_by(Document.id)
        )
        .scalars()
        .first()
    )
    if document is not None:
        hits = vector_store.query(document.text, 1, tenant_id)
        if not hits:
            raise PackageError("The rebuilt index answered no document query.")
        if hits[0]["metadata"].get("source") != document.source:
            raise PackageError("The rebuilt index answered with the wrong source.")
        checked["documents"] = 1
    faq = (
        db.execute(select(Faq).where(_scope(Faq.tenant_id, tenant_id)).order_by(Faq.id))
        .scalars()
        .first()
    )
    if faq is not None:
        hits = vector_store.search_faq_in(faq.question, [faq.id], 1, tenant_id)
        if not hits or int(hits[0].get("faq_id") or -1) != int(faq.id):
            raise PackageError("The rebuilt index did not find the restored answer.")
        checked["faqs"] = 1
    return checked


def _counts(tenant_id: int | None, db: Session) -> dict[str, int]:
    return counts(db, tenant_id)


def _scope(column: Any, tenant_id: int | None) -> Any:
    return column.is_(None) if tenant_id is None else column == tenant_id
