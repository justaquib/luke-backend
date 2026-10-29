"""The passages the model learns from the uploaded documents.

A chunk is the unit the index stores and the passage is the unit the model
learns from, so an 800-character chunk is cut into 3-50-word passages here.
The passages are chosen by the quality measured when the chunk was extracted,
best first, and the union is cut at the token budget, so a poor scan is
dropped before a clean page and two runs over the same material agree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.services.language.passages import split_passages

# How many words of uploaded documents the training keeps. The default word
# tokenizer counts one word as one token, so this is also the token budget.
MAX_DOCUMENT_TOKENS = 100_000


@dataclass(frozen=True)
class DocumentPassages:
    """The passages kept from the uploaded documents, and what was dropped."""

    texts: tuple[str, ...] = ()
    chunks: int = 0
    kept_chunks: int = 0
    dropped_chunks: int = 0
    dropped_passages: int = 0
    tokens: int = 0
    budget: int = MAX_DOCUMENT_TOKENS

    def as_dict(self) -> dict[str, int]:
        return {
            "chunks": self.chunks,
            "kept_chunks": self.kept_chunks,
            "dropped_chunks": self.dropped_chunks,
            "passages": len(self.texts),
            "dropped_passages": self.dropped_passages,
            "tokens": self.tokens,
            "budget": self.budget,
        }


def document_passages(
    materials: Sequence,
    budget: int = MAX_DOCUMENT_TOKENS,
) -> DocumentPassages:
    """The passages of the documents, best quality first, cut at the budget."""
    ordered = sorted(materials, key=lambda material: -material.quality)
    texts: list[str] = []
    tokens = 0
    kept = 0
    for index, material in enumerate(ordered):
        pieces = split_passages(material.answer)
        cost = sum(len(piece.split()) for piece in pieces)
        if tokens + cost > budget:
            remaining = ordered[index:]
            return DocumentPassages(
                texts=tuple(texts),
                chunks=len(ordered),
                kept_chunks=kept,
                dropped_chunks=len(remaining),
                dropped_passages=sum(len(split_passages(item.answer)) for item in remaining),
                tokens=tokens,
                budget=budget,
            )
        texts.extend(pieces)
        tokens += cost
        kept += 1
    return DocumentPassages(
        texts=tuple(texts),
        chunks=len(ordered),
        kept_chunks=kept,
        tokens=tokens,
        budget=budget,
    )
