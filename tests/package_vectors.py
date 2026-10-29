"""Offline Chroma-shaped collections; production reindex and query stay real."""

from __future__ import annotations

from copy import deepcopy

from app.services import chroma


def matches(metadata: dict, condition: dict | None) -> bool:
    """Evaluate the equality, conjunction and membership filters used by Chroma."""
    if not condition:
        return True
    if "$and" in condition:
        return all(matches(metadata, item) for item in condition["$and"])
    if "$or" in condition:
        return any(matches(metadata, item) for item in condition["$or"])
    return all(_match(metadata.get(key), wanted) for key, wanted in condition.items())


def _match(actual: object, wanted: object) -> bool:
    if not isinstance(wanted, dict):
        return actual == wanted
    if "$in" in wanted:
        return actual in wanted["$in"]
    if "$eq" in wanted:
        return actual == wanted["$eq"]
    if "$ne" in wanted:
        return actual != wanted["$ne"]
    for operator, check in (
        ("$gte", lambda value, bound: value >= bound),
        ("$gt", lambda value, bound: value > bound),
        ("$lte", lambda value, bound: value <= bound),
        ("$lt", lambda value, bound: value < bound),
    ):
        if operator in wanted and isinstance(actual, (int, float)):
            return check(actual, wanted[operator])
    return True


class Collection:
    """Keep vectors' text and metadata, with deterministic lexical query order."""

    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, dict]] = {}
        self.metadata = dict(chroma.COLLECTION_CONFIG)

    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        """Mirror Chroma's stable-id replacement, not append-only insertion."""
        for identifier, document, metadata in zip(ids, documents, metadatas, strict=True):
            self.rows[identifier] = (document, deepcopy(metadata))

    def get(
        self, ids: list[str] | None = None, where: dict | None = None, **kwargs: object
    ) -> dict:
        """Read matching ids without embedding any text."""
        kept = [
            identifier
            for identifier, (_, metadata) in self.rows.items()
            if (ids is None or identifier in ids) and matches(metadata, where)
        ]
        return {
            "ids": kept,
            "documents": [self.rows[key][0] for key in kept],
            "metadatas": [deepcopy(self.rows[key][1]) for key in kept],
        }

    def delete(self, ids: list[str] | None = None, where: dict | None = None) -> None:
        """Delete only rows selected by the real service's filter."""
        for key in self.get(ids=ids, where=where)["ids"]:
            del self.rows[key]

    def count(self) -> int:
        """The actual number of stored ids."""
        return len(self.rows)

    def query(self, query_texts: list[str], n_results: int, where: dict | None = None) -> dict:
        """Supply Chroma's nested result shape without an external embedding model."""
        assert len(query_texts) == 1
        wanted = set(query_texts[0].casefold().split())
        found = self.get(where=where)
        ranked = sorted(
            found["ids"],
            key=lambda key: (-len(wanted & set(self.rows[key][0].casefold().split())), key),
        )[:n_results]
        return {
            "ids": [ranked],
            "documents": [[self.rows[key][0] for key in ranked]],
            "metadatas": [[deepcopy(self.rows[key][1]) for key in ranked]],
            "distances": [[0.0 if self.rows[key][0] == query_texts[0] else 0.5 for key in ranked]],
        }


def install(monkeypatch) -> dict[str, Collection]:
    """Replace collection opening, leaving the package's actual indexing intact."""
    collections: dict[str, Collection] = {}

    def open_collection(name: str) -> Collection:
        return collections.setdefault(name, Collection())

    monkeypatch.setattr(chroma, "open_collection", open_collection)
    return collections
