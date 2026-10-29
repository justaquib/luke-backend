"""The stored passages of one workspace, kept for the word ranking.

The word ranking reads every passage of the collection, so reading them on
every question would repeat the same work. A corpus is kept until a write
changes the store: the Chroma file's change time is the signal, and the
counters of the words are ready for the next question.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.config import settings
from app.services.chroma import where
from app.services.retrieval import LexicalIndex

# The cache holds a few workspaces, and stops keeping a corpus larger than
# the character limit, so a big collection is read afresh instead of filling
# the memory.
CACHE_ENTRIES = 4
CACHE_CHARS = 2_000_000


@dataclass(frozen=True)
class Corpus:
    """The stored passages of one workspace, ready to be ranked by words."""

    ids: list[str]
    documents: list[str]
    metadatas: list[dict]
    index: LexicalIndex


_CACHE: OrderedDict[tuple[str, int | None], tuple[int, Corpus]] = OrderedDict()
_LOCK = threading.Lock()


def _store_stamp() -> int:
    """When the Chroma store last changed; any write makes a corpus stale."""
    try:
        return (Path(settings.chroma_persist_dir) / "chroma.sqlite3").stat().st_mtime_ns
    except OSError:
        return 0


def read(collection: chromadb.Collection, tenant_id: int | None) -> Corpus:
    """Every workspace passage, read once from its collection."""
    stored = collection.get(where=where({}, tenant_id), include=["documents", "metadatas"])
    ids = [str(identifier) for identifier in stored.get("ids") or []]
    documents = [str(document) for document in stored.get("documents") or []]
    metadatas = list(stored.get("metadatas") or [])
    return Corpus(ids=ids, documents=documents, metadatas=metadatas, index=LexicalIndex(documents))


def corpus(collection: chromadb.Collection, tenant_id: int | None) -> Corpus:
    """The workspace's passages, read once and kept until the store changes.

    A collection without a stable id is a test double, and is read fresh so
    one double's passages can never be served for another.
    """
    identifier = str(getattr(collection, "id", ""))
    if not identifier:
        return read(collection, tenant_id)
    key = (identifier, tenant_id)
    stamp = _store_stamp()
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None and cached[0] == stamp:
            _CACHE.move_to_end(key)
            return cached[1]
    stored = read(collection, tenant_id)
    if sum(len(document) for document in stored.documents) <= CACHE_CHARS:
        with _LOCK:
            _CACHE[key] = (stamp, stored)
            _CACHE.move_to_end(key)
            while len(_CACHE) > CACHE_ENTRIES:
                _CACHE.popitem(last=False)
    return stored


def forget() -> None:
    """Drop every cached corpus, so the next read goes to the store."""
    with _LOCK:
        _CACHE.clear()
