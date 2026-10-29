"""The one persistent Chroma client, its collections, and the workspace filter."""

from __future__ import annotations

import os
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.services.embeddings import cached_embedding_function

_CLIENTS: list[chromadb.PersistentClient] = []

# The index every new collection is built with. Cosine distance is the one the
# embedding's similarity is defined in; the HNSW sizes below are Chroma's
# defaults made explicit, so a rebuild and a fresh install agree.
COLLECTION_CONFIG = {
    "hnsw:space": "cosine",
    "hnsw:M": 16,
    "hnsw:construction_ef": 200,
    "hnsw:search_ef": 100,
}


def get_client() -> chromadb.PersistentClient:
    """Return the process-wide persistent Chroma client."""
    if not _CLIENTS:
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        _CLIENTS.append(
            chromadb.PersistentClient(
                path=settings.chroma_persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        )
    return _CLIENTS[0]


def reset_clients() -> None:
    """Forget the open client so the store can be copied or vacuumed."""
    _CLIENTS.clear()


def open_collection(name: str) -> chromadb.Collection:
    """Open a collection as it was built, creating it with the index config.

    ``get_or_create_collection`` rewrites the metadata of an existing
    collection without rebuilding its index, so a collection that already
    exists must be opened without metadata. Otherwise it would report cosine
    while its vectors are still L2.
    """
    client = get_client()
    embedding = cached_embedding_function()
    try:
        return client.get_collection(name=name, embedding_function=embedding)
    except ValueError:
        return client.create_collection(
            name=name, metadata=COLLECTION_CONFIG, embedding_function=embedding
        )


def get_collection() -> chromadb.Collection:
    return open_collection(settings.chroma_collection)


def collection_settings(name: str) -> dict[str, Any]:
    """The index a collection was built with, read from its own metadata.

    A collection keeps the settings it was created with. Reading them back is
    how a measurement can say what the search really used instead of what the
    current code would create.
    """
    metadata = open_collection(name).metadata or {}
    return {key: metadata.get(key, COLLECTION_CONFIG[key]) for key in COLLECTION_CONFIG}


def get_faq_collection() -> chromadb.Collection:
    return open_collection(settings.chroma_faq_collection)


def get_intents_collection() -> chromadb.Collection:
    return open_collection(settings.chroma_intents_collection)


def tenant_metadata(tenant_id: int | None) -> dict[str, Any]:
    """The workspace stored beside a vector; absent for the shared workspace."""
    return {} if tenant_id is None else {"tenant_id": str(tenant_id)}


def where(conditions: dict[str, Any], tenant_id: int | None) -> dict[str, Any] | None:
    """The Chroma filter that holds some conditions inside one workspace."""
    combined = {**conditions, **tenant_metadata(tenant_id)}
    if not combined:
        return None
    if len(combined) == 1:
        return combined
    return {"$and": [{key: value} for key, value in combined.items()]}
