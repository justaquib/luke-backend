"""Drop the passages the workspace already holds, exactly or nearly.

The index stores one copy of a passage. A second copy costs a row, a vector
and training steps without adding a word the model did not already have. The
SHA-256 of the text catches the exact repeat; the 64-bit SimHash catches the
same passage with a few words changed. A near repeat is counted, not stored,
so the trainer can see what the upload skipped.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.extraction import ExtractedText
from app.db.database import Document
from app.services.nlp import ChunkSpan
from app.services.provenance import ChunkFacts, facts_for
from app.services.quality.simhash import hamming, to_unsigned

# How many of the 64 fingerprint bits may differ and still be the same passage.
# A changed word moves a few bits in a chunk-sized passage; an unrelated passage
# of the same language stays tens of bits away.
NEAR_DUPLICATE_DISTANCE = 5


@dataclass(frozen=True)
class Collapse:
    """The passages worth storing, and how many repeats were left out."""

    spans: tuple[ChunkSpan, ...] = ()
    facts: tuple[ChunkFacts, ...] = ()
    exact: int = 0
    near: int = 0

    @property
    def stored(self) -> int:
        return len(self.spans)

    def as_dict(self) -> dict[str, int]:
        return {"stored": self.stored, "exact": self.exact, "near": self.near}


def known(db: Session, tenant_id: int | None) -> tuple[set[str], list[int]]:
    """The hashes and fingerprints of the passages the workspace already holds."""
    statement = select(Document.content_hash, Document.simhash)
    if tenant_id is not None:
        statement = statement.where(Document.tenant_id == tenant_id)
    hashes: set[str] = set()
    fingerprints: list[int] = []
    for content_hash, fingerprint in db.execute(statement).all():
        if content_hash:
            hashes.add(content_hash)
        if fingerprint is not None:
            fingerprints.append(to_unsigned(int(fingerprint)))
    return hashes, fingerprints


def collapse(
    spans: Sequence[ChunkSpan],
    extracted: ExtractedText | None = None,
    known_hashes: Iterable[str] = (),
    known_simhashes: Iterable[int] = (),
) -> Collapse:
    """The passages whose text and fingerprint the workspace does not hold yet.

    A passage is also compared with the passages kept earlier in the same run,
    so a repeated page inside one upload is stored once.
    """
    facts = facts_for(spans, extracted)
    hashes = set(known_hashes)
    fingerprints = list(known_simhashes)
    kept_spans: list[ChunkSpan] = []
    kept_facts: list[ChunkFacts] = []
    exact = 0
    near = 0
    for span, fact in zip(spans, facts, strict=True):
        if fact.content_hash in hashes:
            exact += 1
            continue
        if any(hamming(fact.simhash, other) <= NEAR_DUPLICATE_DISTANCE for other in fingerprints):
            near += 1
            continue
        hashes.add(fact.content_hash)
        fingerprints.append(fact.simhash)
        kept_spans.append(span)
        kept_facts.append(fact)
    return Collapse(tuple(kept_spans), tuple(kept_facts), exact, near)
