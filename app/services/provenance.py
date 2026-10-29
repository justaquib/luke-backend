"""Where a chunk sits in its source, and how much prose it reads as.

A chunk is more than its words: the index remembers the upload it came from,
the characters it covers in the extracted text, the page it was found on, the
hash of its own text, the fingerprint of its wording, the language it is
written in and the share of it that reads as prose. Those facts are stored on
the document row and beside the vector, so an answer can point at the passage
it used and an ingest can recognise the passage it already has.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.extraction import ExtractedText
from app.db.database import Document
from app.services.language.identify import guess
from app.services.nlp import ChunkSpan
from app.services.quality.score import passage_quality
from app.services.quality.simhash import simhash, to_signed


@dataclass(frozen=True)
class ChunkFacts:
    """What the index remembers about one chunk beyond its text."""

    char_start: int
    char_end: int
    page: int | None
    content_hash: str
    simhash: int
    language: str
    quality: float


def content_hash(text: str) -> str:
    """The SHA-256 of a chunk's own text, so equal text has one hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def facts_for(
    spans: Sequence[ChunkSpan], extracted: ExtractedText | None = None
) -> list[ChunkFacts]:
    """The facts of every span, with its page when the format has pages."""
    return [
        ChunkFacts(
            char_start=span.char_start,
            char_end=span.char_end,
            page=extracted.page_at(span.char_start) if extracted is not None else None,
            content_hash=content_hash(span.text),
            simhash=simhash(span.text),
            language=guess(span.text),
            quality=passage_quality(span.text),
        )
        for span in spans
    ]


def rows_for(
    tenant_id: int | None,
    source: str,
    spans: Sequence[ChunkSpan],
    facts: Sequence[ChunkFacts],
    document_id: int | None = None,
) -> list[Document]:
    """The document rows for a run of chunks, in their stored order."""
    return [
        Document(
            source=source,
            chunk_index=index,
            text=span.text,
            tenant_id=tenant_id,
            document_id=document_id,
            char_start=fact.char_start,
            char_end=fact.char_end,
            page=fact.page,
            content_hash=fact.content_hash,
            simhash=to_signed(fact.simhash),
            language=fact.language,
            quality=fact.quality,
        )
        for index, (span, fact) in enumerate(zip(spans, facts, strict=True))
    ]
