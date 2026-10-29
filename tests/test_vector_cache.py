"""The word ranking reads a workspace once and reuses it until a write."""

from __future__ import annotations

import pytest

from app.services import corpus_cache, vector_store

TEXTS = {
    "a": ("The ferry leaves at noon.", {"source": "ferry.txt"}),
    "b": ("The voucher holds the number.", {"source": "voucher.txt"}),
}


class _Collection:
    """A Chroma-shaped collection that counts how often it is read."""

    id = "docs-in-a-test"

    def __init__(self) -> None:
        self.reads = 0
        self.stamp = {"value": 1}

    def count(self) -> int:
        return len(TEXTS)

    def get(self, where: dict | None = None, include: list[str] | None = None) -> dict:
        self.reads += 1
        return {
            "ids": list(TEXTS),
            "documents": [text for text, _ in TEXTS.values()],
            "metadatas": [metadata for _, metadata in TEXTS.values()],
        }

    def query(self, query_texts: list[str], n_results: int, where: dict | None = None) -> dict:
        identifiers = list(TEXTS)
        return {
            "ids": [identifiers],
            "documents": [[TEXTS[key][0] for key in identifiers]],
            "metadatas": [[TEXTS[key][1] for key in identifiers]],
            "distances": [[0.4 for _ in identifiers]],
        }


@pytest.fixture()
def collection(monkeypatch) -> _Collection:
    stored = _Collection()
    stamp = {"value": 1}
    monkeypatch.setattr(vector_store, "get_collection", lambda: stored)
    monkeypatch.setattr(corpus_cache, "_store_stamp", lambda: stamp["value"])
    corpus_cache.forget()
    stored.stamp = stamp
    return stored


def test_a_second_question_reads_no_workspace_again(collection) -> None:
    vector_store.query("where is the voucher", n_results=1, tenant_id=7)
    vector_store.query("when does the ferry leave", n_results=1, tenant_id=7)

    assert collection.reads == 1


def test_a_write_makes_the_workspace_be_read_again(collection) -> None:
    vector_store.query("where is the voucher", n_results=1, tenant_id=7)
    collection.stamp["value"] = 2

    vector_store.query("where is the voucher", n_results=1, tenant_id=7)

    assert collection.reads == 2


def test_a_large_workspace_is_not_kept(collection, monkeypatch) -> None:
    monkeypatch.setattr(corpus_cache, "CACHE_CHARS", 5)

    vector_store.query("where is the voucher", n_results=1, tenant_id=7)
    vector_store.query("where is the voucher", n_results=1, tenant_id=7)

    assert collection.reads == 2


def test_the_word_ranking_still_finds_the_named_passage(collection) -> None:
    hits = vector_store.query("voucher", n_results=1, tenant_id=7)

    assert hits[0]["text"] == TEXTS["b"][0]
