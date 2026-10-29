"""Write chunk vectors with the facts that say where each chunk came from."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.services.chroma import get_collection, tenant_metadata
from app.services.provenance import ChunkFacts, content_hash

_CHUNK_NAMESPACE = uuid.UUID("2f8f5d4e-5b7a-4f3c-9a6e-0d1c2b3a4f5e")

# The note field on the left is stored under the metadata key on the right.
_NOTES = (
    ("page_number", "page_numbers"),
    ("document_id", "document_ids"),
    ("chunk_index", "chunk_indexes"),
    ("char_start", "char_starts"),
    ("char_end", "char_ends"),
    ("content_hash", "content_hashes"),
    ("language", "languages"),
    ("quality", "qualities"),
)


def chunk_id(tenant_id: int | None, source: str, text: str) -> str:
    """The stable id of one chunk, so re-indexing overwrites instead of duplicating."""
    workspace = 0 if tenant_id is None else tenant_id
    return str(uuid.uuid5(_CHUNK_NAMESPACE, f"{workspace}:{source}:{content_hash(text)}"))


@dataclass(frozen=True)
class ChunkNotes:
    """The per-chunk facts stored beside the text: its place and its fields."""

    page_numbers: list[int | None] | None = None
    pdf_fields: list[dict[str, str]] | None = None
    document_ids: list[int | None] | None = None
    chunk_indexes: list[int | None] | None = None
    char_starts: list[int | None] | None = None
    char_ends: list[int | None] | None = None
    content_hashes: list[str | None] | None = None
    languages: list[str | None] | None = None
    qualities: list[float | None] | None = None

    @classmethod
    def from_facts(
        cls,
        facts: Sequence[ChunkFacts],
        document_id: int | None = None,
        start_index: int = 0,
    ) -> ChunkNotes:
        """The notes of a run of chunks, in the order they were stored."""
        return cls(
            page_numbers=[fact.page for fact in facts],
            document_ids=[document_id for _ in facts],
            chunk_indexes=list(range(start_index, start_index + len(facts))),
            char_starts=[fact.char_start for fact in facts],
            char_ends=[fact.char_end for fact in facts],
            content_hashes=[fact.content_hash for fact in facts],
            languages=[fact.language for fact in facts],
            qualities=[fact.quality for fact in facts],
        )


def _notes_metadata(notes: ChunkNotes, index: int) -> dict[str, Any]:
    """The notes of one chunk, leaving out what the file did not provide."""
    stored: dict[str, Any] = {}
    for key, field in _NOTES:
        values = getattr(notes, field)
        if values is not None and index < len(values) and values[index] is not None:
            stored[key] = values[index]
    if notes.pdf_fields is not None and index < len(notes.pdf_fields):
        stored["pdf_fields"] = json.dumps(notes.pdf_fields[index])
    return stored


def add_chunks(
    source: str,
    chunks: list[str],
    ids: list[str],
    tenant_id: int | None = None,
    notes: ChunkNotes | None = None,
) -> int:
    """Store document chunks with source, tenant, place, and field metadata."""
    notes = notes or ChunkNotes()
    metadatas = [
        {"source": source, **tenant_metadata(tenant_id), **_notes_metadata(notes, index)}
        for index in range(len(chunks))
    ]
    unique: dict[str, tuple[str, dict[str, Any]]] = {}
    for identifier, document, metadata in zip(ids, chunks, metadatas, strict=True):
        unique.setdefault(identifier, (document, metadata))
    get_collection().upsert(
        ids=list(unique),
        documents=[document for document, _ in unique.values()],
        metadatas=[metadata for _, metadata in unique.values()],
    )
    return len(unique)
