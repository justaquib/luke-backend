"""Store the passages of one document, in batches that can report progress.

Both the synchronous ingest and the background job use this one path, so the
rows, the vectors and the duplicate counts cannot drift apart. A batch is
embedded at a time, and the trainer's cancel is checked before each batch, so
a stopped job never keeps the vectors it had already written.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.extraction import ExtractedText
from app.db.database import UploadedFile
from app.services import chunk_vectors, vector_store
from app.services.chunk_vectors import ChunkNotes
from app.services.dedupe import Collapse, collapse, known
from app.services.nlp import chunk_spans
from app.services.provenance import rows_for

# How many passages are embedded and written in one call. It is small enough
# that a cancel is noticed within a moment on a large file.
VECTOR_BATCH = 32

Progress = Callable[[int, int], None]
Cancelled = Callable[[], bool]


class IngestCancelledError(Exception):
    """The trainer stopped the job before it finished."""


@dataclass(frozen=True)
class Sink:
    """Where one upload writes its passages, and who watches the work."""

    db: Session
    workspace: int | None
    source: str
    record: UploadedFile
    progress: Progress | None = None
    cancelled: Cancelled | None = None


def store_chunks(sink: Sink, extracted: ExtractedText) -> Collapse:
    """Write the passages the workspace does not already hold, and their vectors."""
    spans = chunk_spans(extracted.text)
    if not spans:
        return Collapse()
    hashes, fingerprints = known(sink.db, sink.workspace)
    stored = collapse(spans, extracted, hashes, fingerprints)
    if not stored.spans:
        return stored
    sink.db.add_all(
        rows_for(sink.workspace, sink.source, stored.spans, stored.facts, sink.record.id)
    )
    sink.db.flush()
    _embed(sink, stored)
    return stored


def _embed(sink: Sink, stored: Collapse) -> None:
    """Embed the stored passages a batch at a time, stopping when asked."""
    total = stored.stored
    for start in range(0, total, VECTOR_BATCH):
        if sink.cancelled is not None and sink.cancelled():
            _abandon(sink)
        _embed_batch(sink, stored, start)
        if sink.progress is not None:
            sink.progress(min(start + VECTOR_BATCH, total), total)
    if sink.cancelled is not None and sink.cancelled():
        _abandon(sink)


def _embed_batch(sink: Sink, stored: Collapse, start: int) -> None:
    spans = stored.spans[start : start + VECTOR_BATCH]
    facts = stored.facts[start : start + VECTOR_BATCH]
    texts = [span.text for span in spans]
    ids = [chunk_vectors.chunk_id(sink.workspace, sink.source, text) for text in texts]
    chunk_vectors.add_chunks(
        sink.source,
        texts,
        ids,
        sink.workspace,
        ChunkNotes.from_facts(facts, sink.record.id, start),
    )


def _abandon(sink: Sink) -> None:
    """Remove the vectors a stopped job had already written, then stop."""
    vector_store.delete_chunks_by_source(sink.source, sink.workspace)
    raise IngestCancelledError
