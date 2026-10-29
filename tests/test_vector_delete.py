"""Tests for removing a vector that may already be gone.

Deleting an id that was never stored is not an error: the vector store
answers quietly instead of printing a complaint from chroma.
"""

import logging

import pytest

from app.services import chroma, vector_store

# The chroma logger that complains when a asked-for id was never stored.
CHROMA_DELETE_LOGGER = "chromadb.segment.impl.vector.local_persistent_hnsw"


@pytest.fixture()
def fresh_store(monkeypatch, tmp_path):
    monkeypatch.setattr(vector_store.settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    chroma._CLIENTS.clear()
    yield
    chroma._CLIENTS.clear()


def test_a_repeated_delete_stays_quiet(fresh_store, caplog):
    vector_store.upsert_faq(27, "What is a kite?", "A kite flies on the wind.", "approved", None)

    vector_store.delete_faq(27)
    vector_store.delete_faq(27)

    stored = vector_store.get_faq_collection()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=CHROMA_DELETE_LOGGER):
        vector_store.delete_faq(27)
    assert caplog.records == []
    assert stored.get(ids=["27"])["ids"] == []


def test_a_delete_of_an_id_that_was_never_stored_stays_quiet(fresh_store, caplog):
    with caplog.at_level(logging.ERROR, logger=CHROMA_DELETE_LOGGER):
        vector_store.delete_faq(27)

    assert caplog.records == []
    assert vector_store.get_faq_collection().count() == 0
