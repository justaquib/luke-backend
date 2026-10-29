"""Give every vector the workspace it belongs to.

The rows were written before the workspace id was stored, so their metadata
has no ``tenant_id`` and every workspace-scoped search skips them. The
migration writes ``tenant_id = "0"`` in place; nothing is re-embedded.
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.services import vector_store

WORKSPACE = "0"
COLLECTIONS = (
    settings.chroma_collection,
    settings.chroma_faq_collection,
    settings.chroma_intents_collection,
)


def _collection(name: str) -> Any:
    return vector_store.open_collection(name)


def plan(collection: Any) -> dict[str, Any]:
    """The ids in one collection whose metadata lacks the workspace."""
    stored = collection.get(include=["metadatas"])
    ids = stored.get("ids") or []
    metadatas = stored.get("metadatas") or []
    stale = [
        identifier
        for identifier, metadata in zip(ids, metadatas, strict=False)
        if str((metadata or {}).get("tenant_id", "")) != WORKSPACE
    ]
    return {
        "collection": collection.name,
        "total": collection.count(),
        "stale": len(stale),
        "ids": stale,
    }


def apply(collection: Any, ids: list[str]) -> int:
    """Add the workspace to the given ids, keeping their other metadata."""
    stored = collection.get(ids=ids, include=["metadatas"])
    found = stored.get("ids") or []
    metadatas = stored.get("metadatas") or []
    updated = [{**(metadata or {}), "tenant_id": WORKSPACE} for metadata in metadatas]
    if found:
        collection.update(ids=found, metadatas=updated)
    return len(found)


def drop(collection: Any, ids: list[str]) -> int:
    """Remove vectors whose row was merged into another."""
    if not ids:
        return 0
    collection.delete(ids=ids)
    return len(ids)


def inspect() -> dict[str, Any]:
    """The workspace metadata state of every collection."""
    return {"collections": [plan(_collection(name)) for name in COLLECTIONS]}


def apply_all(plan_payload: dict[str, Any], dropped_faq_ids: list[int]) -> dict[str, Any]:
    """Write the workspace into every stale vector and drop merged FAQs."""
    results = []
    for entry in plan_payload["collections"]:
        collection = _collection(entry["collection"])
        results.append({"collection": collection.name, "updated": apply(collection, entry["ids"])})
    if dropped_faq_ids:
        faq = _collection(settings.chroma_faq_collection)
        removed = drop(faq, [str(identifier) for identifier in dropped_faq_ids])
        results.append({"collection": faq.name, "removed": removed})
    return {"collections": results}


def sample(question: str) -> dict[str, Any]:
    """One fixed search, run the same way the app runs it."""
    hits = vector_store.query(question, n_results=3, tenant_id=0)
    faqs = vector_store.search_faq(question, n_results=1, tenant_id=0)
    return {
        "question": question,
        "documents": [
            {"source": (hit.get("metadata") or {}).get("source"), "distance": hit.get("distance")}
            for hit in hits
        ],
        "faq": (faqs[0] if faqs else None),
    }
