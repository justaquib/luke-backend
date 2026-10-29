"""ChromaDB access for documents, FAQs, intents, and pending questions."""

from __future__ import annotations

import logging
from typing import Any

import chromadb

from app.config import settings
from app.services import corpus_cache
from app.services.chroma import (
    get_collection,
    get_faq_collection,
    get_intents_collection,
    open_collection,
    tenant_metadata,
    where,
)
from app.services.rerank import rerank
from app.services.retrieval import fuse_rankings, terms

logger = logging.getLogger(__name__)


def collection_space(collection_name: str | None = None) -> str:
    """The distance space a collection was built with; Chroma defaults to l2."""
    name = collection_name or settings.chroma_collection
    collection = open_collection(name)
    return str((collection.metadata or {}).get("hnsw:space", "l2"))


def _semantic_hits(result: dict) -> dict[str, dict]:
    """The vector search hits of one query, keyed by their stable id."""
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    return {
        identifier: {
            "text": documents[index],
            "metadata": metadatas[index] if index < len(metadatas) else {},
            "distance": distances[index] if index < len(distances) else None,
        }
        for index, identifier in enumerate(ids)
    }


def query(question: str, n_results: int = 4, tenant_id: int | None = None) -> list[dict]:
    """Search chunks by meaning and by words, then fuse both rankings."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    fetch = max(n_results * 4, 8)
    result = collection.query(query_texts=[question], n_results=fetch, where=where({}, tenant_id))
    semantic = _semantic_hits(result)
    corpus = corpus_cache.corpus(collection, tenant_id)
    by_id = {
        identifier: {
            "text": document,
            "metadata": corpus.metadatas[index] if index < len(corpus.metadatas) else {},
            "distance": None,
        }
        for index, (identifier, document) in enumerate(
            zip(corpus.ids, corpus.documents, strict=False)
        )
    }
    by_id.update(semantic)
    lexical = [corpus.ids[index] for index in corpus.index.ranking(question)]
    fused = fuse_rankings([list(semantic), lexical])
    candidates = [by_id[identifier] for identifier in fused[: max(n_results * 5, 20)]]
    return rerank(question, candidates, n_results)


def collection_size() -> int:
    return get_collection().count()


def upsert_faq(
    faq_id: int,
    question: str,
    answer: str,
    status: str = "pending",
    tenant_id: int | None = None,
) -> None:
    """Insert or update one FAQ in its vector collection."""
    get_faq_collection().upsert(
        ids=[str(faq_id)],
        documents=[question],
        metadatas=[
            {
                "answer": answer,
                "faq_id": faq_id,
                "status": status,
                **tenant_metadata(tenant_id),
            }
        ],
    )


def _delete(collection: chromadb.Collection, ids: list[str]) -> None:
    try:
        stored = set(collection.get(ids=ids).get("ids") or [])
        missing = [id_ for id_ in ids if id_ not in stored]
        if missing:
            logger.debug("Vector ids %s were already gone; nothing to delete.", missing)
        if stored:
            collection.delete(ids=sorted(stored))
    except chromadb.errors.ChromaError as error:
        logger.warning("Could not delete vector ids %s: %s", ids, error)


def delete_faq(faq_id: int) -> None:
    _delete(get_faq_collection(), [str(faq_id)])


def delete_chunks_by_source(source: str, tenant_id: int | None = None) -> None:
    """Remove every chunk that came from one document in one workspace."""
    collection = get_collection()
    if collection.count() == 0:
        return
    try:
        collection.delete(where=where({"source": source}, tenant_id))
    except chromadb.errors.ChromaError as error:
        logger.warning("Could not delete the chunks of %s: %s", source, error)


def search_lessons(question: str, n_results: int = 3, tenant_id: int | None = None) -> list[dict]:
    """The learned lessons that look like the question that was asked.

    The lessons live in the tenant's own learner file, so they are matched on
    the words they share with the question.
    """
    from app.services import language_logic

    try:
        cards = language_logic.all_lesson_cards(tenant_id)
    except (OSError, ValueError) as error:
        logger.warning("Could not read the lessons of tenant %s: %s", tenant_id, error)
        return []
    wanted = terms(question)
    hits: list[dict] = []
    for card in cards:
        card_terms = terms(str(card.get("question", "")))
        if not card_terms or not wanted:
            continue
        overlap = len(wanted & card_terms) / len(wanted | card_terms)
        if overlap <= 0:
            continue
        hits.append(
            {
                "lesson_id": card.get("id"),
                "question": card.get("question", ""),
                "answer": card.get("answer", ""),
                "distance": round(1 - overlap, 4),
            }
        )
    hits.sort(key=lambda hit: hit["distance"])
    return hits[:n_results]


def _faq_hits(result: dict, include_status: bool) -> list[dict]:
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    hits: list[dict] = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        hit = {
            "question": document,
            "answer": metadata.get("answer", ""),
            "distance": distances[index] if index < len(distances) else None,
            "faq_id": metadata.get("faq_id"),
        }
        if include_status:
            hit["status"] = metadata.get("status")
        hits.append(hit)
    return hits


def search_faq(question: str, n_results: int = 1, tenant_id: int | None = None) -> list[dict]:
    """Search approved FAQs for the closest answer."""
    collection = get_faq_collection()
    if collection.count() == 0:
        return []
    result = collection.query(
        query_texts=[question],
        n_results=n_results,
        where=where({"status": "approved"}, tenant_id),
    )
    return _faq_hits(result, include_status=False)


def search_faq_loose(question: str, n_results: int = 5, tenant_id: int | None = None) -> list[dict]:
    """Search every FAQ regardless of status."""
    collection = get_faq_collection()
    if collection.count() == 0:
        return []
    result = collection.query(
        query_texts=[question], n_results=n_results, where=where({}, tenant_id)
    )
    return _faq_hits(result, include_status=True)


def search_faq_in(
    question: str, faq_ids: list[int], n_results: int = 50, tenant_id: int | None = None
) -> list[dict]:
    """Search only the given FAQ ids."""
    collection = get_faq_collection()
    if collection.count() == 0 or not faq_ids:
        return []
    condition = where({"faq_id": {"$in": [int(faq_id) for faq_id in faq_ids]}}, tenant_id)
    result = collection.query(query_texts=[question], n_results=n_results, where=condition)
    return _faq_hits(result, include_status=True)


def search_pending(
    question: str,
    candidates: list[str],
    n_results: int = 50,
    tenant_id: int | None = None,
) -> list[dict]:
    """Search pending questions and keep only the supplied candidates."""
    if not candidates or get_intents_collection().count() == 0:
        return []
    matches = search_pending_all(question, n_results, tenant_id)
    return [hit for hit in matches if hit["question"] in candidates]


def _pending_hits(result: dict) -> list[dict]:
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    return [
        {
            "pending_id": metadatas[index].get("pending_id") if index < len(metadatas) else None,
            "question": document,
            "distance": distances[index] if index < len(distances) else None,
        }
        for index, document in enumerate(documents)
    ]


def search_pending_all(
    question: str, n_results: int = 25, tenant_id: int | None = None
) -> list[dict]:
    """Search every pending question."""
    collection = get_intents_collection()
    if collection.count() == 0:
        return []
    result = collection.query(
        query_texts=[question],
        n_results=n_results,
        where=where({"kind": "pending"}, tenant_id),
    )
    return _pending_hits(result)


def index_pending(pending_id: int, question: str, tenant_id: int | None = None) -> None:
    """Insert or update a pending question in the intent index."""
    get_intents_collection().upsert(
        ids=[f"pending-{pending_id}"],
        documents=[question],
        metadatas=[{"kind": "pending", "pending_id": pending_id, **tenant_metadata(tenant_id)}],
    )


def delete_pending_index(pending_id: int) -> None:
    _delete(get_intents_collection(), [f"pending-{pending_id}"])


def upsert_intent(
    intent_id: int,
    name: str,
    response_template: str,
    faq_id: int | None,
    tenant_id: int | None = None,
) -> None:
    """Insert or update one intent in its vector collection."""
    metadata: dict[str, Any] = {
        "kind": "intent",
        "name": name,
        "response_template": response_template,
        **tenant_metadata(tenant_id),
    }
    if faq_id is not None:
        metadata["faq_id"] = faq_id
    get_intents_collection().upsert(
        ids=[f"intent-{intent_id}"], documents=[name], metadatas=[metadata]
    )


def delete_intent(intent_id: int) -> None:
    _delete(get_intents_collection(), [f"intent-{intent_id}"])


def search_intents(question: str, n_results: int = 1, tenant_id: int | None = None) -> list[dict]:
    """Search named intents."""
    collection = get_intents_collection()
    if collection.count() == 0:
        return []
    result = collection.query(
        query_texts=[question],
        n_results=n_results,
        where=where({"kind": "intent"}, tenant_id),
    )
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    return [
        {
            "name": document,
            "response_template": metadatas[index].get("response_template", "")
            if index < len(metadatas)
            else "",
            "faq_id": metadatas[index].get("faq_id") if index < len(metadatas) else None,
            "distance": distances[index] if index < len(distances) else None,
        }
        for index, document in enumerate(documents)
    ]


def faq_size() -> int:
    return get_faq_collection().count()


def intents_size() -> int:
    return get_intents_collection().count()
