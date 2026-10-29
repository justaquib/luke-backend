"""Turn pasted text into stored chunks and indexed vectors."""

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.db.database import Document
from app.services import chunk_vectors, vector_store
from app.services.chunk_vectors import ChunkNotes
from app.services.dedupe import Collapse, collapse, known
from app.services.nlp import chunk_spans
from app.services.provenance import rows_for
from app.services.quality.pipeline import prepare_text


def ingest_text(db: Session, tenant_id: int | None, source: str, text: str) -> dict:
    """Index one source, replacing any earlier text stored under its name."""
    spans = chunk_spans(prepare_text(text))
    if not spans:
        return {"source": source, "chunks": 0, "start_id": 0, "duplicates": Collapse().as_dict()}

    condition = Document.source == source
    condition = condition if tenant_id is None else condition & (Document.tenant_id == tenant_id)
    db.execute(delete(Document).where(condition))
    vector_store.delete_chunks_by_source(source, tenant_id)

    hashes, fingerprints = known(db, tenant_id)
    stored = collapse(spans, None, hashes, fingerprints)
    if stored.spans:
        db.add_all(rows_for(tenant_id, source, stored.spans, stored.facts))
        db.flush()

        texts = [span.text for span in stored.spans]
        ids = [chunk_vectors.chunk_id(tenant_id, source, chunk) for chunk in texts]
        chunk_vectors.add_chunks(
            source, texts, ids, tenant_id, ChunkNotes.from_facts(list(stored.facts))
        )
    db.commit()

    return {
        "source": source,
        "chunks": stored.stored,
        "start_id": 0,
        "duplicates": stored.as_dict(),
    }
