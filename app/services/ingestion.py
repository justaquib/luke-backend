from sqlalchemy.orm import Session

from app.db.database import Document
from app.services import vector_store
from app.services.nlp import chunk_text


def ingest_text(db: Session, source: str, text: str) -> dict:
    chunks = chunk_text(text)
    if not chunks:
        return {"source": source, "chunks": 0}

    start_id = vector_store.collection_size()
    ids = [f"{source}-{start_id + i}" for i in range(len(chunks))]
    vector_store.add_chunks(source, chunks, ids)

    for i, chunk in enumerate(chunks):
        db.add(Document(source=source, chunk_index=start_id + i, text=chunk))
    db.commit()

    return {"source": source, "chunks": len(chunks), "start_id": start_id}
