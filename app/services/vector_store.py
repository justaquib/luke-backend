"""ChromaDB access for documents, FAQs, intents, and pending questions."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings

logger = logging.getLogger(__name__)
_CLIENTS: list[chromadb.PersistentClient] = []


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


def get_collection():
    return get_client().get_or_create_collection(name=settings.chroma_collection)


def get_faq_collection():
    return get_client().get_or_create_collection(name=settings.chroma_faq_collection)


def get_intents_collection():
    return get_client().get_or_create_collection(name=settings.chroma_intents_collection)


def _tenant_metadata(tenant_id: int | None) -> dict[str, Any]:
    return {} if tenant_id is None else {"tenant_id": str(tenant_id)}


def add_chunks(
    source: str,
    chunks: list[str],
    ids: list[str],
    tenant_id: int | None = None,
    page_numbers: list[int] | None = None,
    pdf_fields: list[dict[str, str]] | None = None,
) -> int:
    """Store document chunks with source, tenant, page, and field metadata."""
    metadatas = [
        {
            "source": source,
            **_tenant_metadata(tenant_id),
            **(
                {"page_number": page_numbers[index]}
                if page_numbers is not None and index < len(page_numbers)
                else {}
            ),
            **(
                {"pdf_fields": json.dumps(pdf_fields[index])}
                if pdf_fields is not None and index < len(pdf_fields)
                else {}
            ),
        }
        for index in range(len(chunks))
    ]
    get_collection().add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def _where(conditions: dict[str, Any], tenant_id: int | None) -> dict[str, Any] | None:
    combined = {**conditions, **_tenant_metadata(tenant_id)}
    if not combined:
        return None
    if len(combined) == 1:
        return combined
    return {"$and": [{key: value} for key, value in combined.items()]}


def _terms(text: str) -> set[str]:
    return {
        token
        for token in "".join(character if character.isalnum() else " " for character in text.lower()).split()
        if len(token) > 2
    }


def _rerank(question: str, hits: list[dict]) -> list[dict]:
    wanted = _terms(question)
    return sorted(
        hits,
        key=lambda hit: (
            -len(wanted & _terms(str(hit.get("text", "")))),
            float(hit.get("distance") or 1.0),
        ),
    )


def query(question: str, n_results: int = 4, tenant_id: int | None = None) -> list[dict]:
    """Search document chunks and rerank semantic candidates by question terms."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    fetch = max(n_results * 2, 2)
    result = collection.query(
        query_texts=[question], n_results=fetch, where=_where({}, tenant_id)
    )
    documents = result.get("documents") or [[]]
    metadatas = result.get("metadatas") or [[]]
    distances = result.get("distances") or [[]]
    hits = [
        {
            "text": documents[0][index],
            "metadata": metadatas[0][index] if metadatas and index < len(metadatas[0]) else {},
            "distance": distances[0][index] if distances and index < len(distances[0]) else None,
        }
        for index in range(len(documents[0]))
    ]
    return _rerank(question, hits)[:n_results]


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
                **_tenant_metadata(tenant_id),
            }
        ],
    )


def _delete(collection, ids: list[str]) -> None:
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


def delete_chunks_by_source(source: str) -> None:
    """Remove every chunk that came from one document."""
    collection = get_collection()
    if collection.count() == 0:
        return
    try:
        collection.delete(where={"source": source})
    except chromadb.errors.ChromaError as error:
        logger.warning("Could not delete the chunks of %s: %s", source, error)


def search_lessons(
    question: str, n_results: int = 3, tenant_id: int | None = None
) -> list[dict]:
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
    wanted = _terms(question)
    hits: list[dict] = []
    for card in cards:
        terms = _terms(str(card.get("question", "")))
        if not terms or not wanted:
            continue
        overlap = len(wanted & terms) / len(wanted | terms)
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


def search_faq(
    question: str, n_results: int = 1, tenant_id: int | None = None
) -> list[dict]:
    """Search approved FAQs for the closest answer."""
    collection = get_faq_collection()
    if collection.count() == 0:
        return []
    result = collection.query(
        query_texts=[question],
        n_results=n_results,
        where=_where({"status": "approved"}, tenant_id),
    )
    return _faq_hits(result, include_status=False)


def search_faq_loose(
    question: str, n_results: int = 5, tenant_id: int | None = None
) -> list[dict]:
    """Search every FAQ regardless of status."""
    collection = get_faq_collection()
    if collection.count() == 0:
        return []
    result = collection.query(
        query_texts=[question], n_results=n_results, where=_where({}, tenant_id)
    )
    return _faq_hits(result, include_status=True)


def search_faq_in(
    question: str, faq_ids: list[int], n_results: int = 50, tenant_id: int | None = None
) -> list[dict]:
    """Search only the given FAQ ids."""
    collection = get_faq_collection()
    if collection.count() == 0 or not faq_ids:
        return []
    where = _where({"faq_id": {"$in": [int(faq_id) for faq_id in faq_ids]}}, tenant_id)
    result = collection.query(
        query_texts=[question], n_results=n_results, where=where
    )
    return _faq_hits(result, include_status=True)


def search_pending(
    question: str,
    candidates: list[str],
    n_results: int = 50,
    tenant_id: int | None = None,
) -> list[dict]:
    """Search pending questions and keep only the supplied candidates."""
    collection = get_intents_collection()
    if collection.count() == 0 or not candidates:
        return []
    result = collection.query(
        query_texts=[question],
        n_results=n_results,
        where=_where({"kind": "pending"}, tenant_id),
    )
    return [
        hit
        for hit in _pending_hits(result)
        if hit["question"] in candidates
    ]


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
        where=_where({"kind": "pending"}, tenant_id),
    )
    return _pending_hits(result)


def index_pending(pending_id: int, question: str, tenant_id: int | None = None) -> None:
    """Insert or update a pending question in the intent index."""
    get_intents_collection().upsert(
        ids=[f"pending-{pending_id}"],
        documents=[question],
        metadatas=[
            {"kind": "pending", "pending_id": pending_id, **_tenant_metadata(tenant_id)}
        ],
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
        **_tenant_metadata(tenant_id),
    }
    if faq_id is not None:
        metadata["faq_id"] = faq_id
    get_intents_collection().upsert(ids=[f"intent-{intent_id}"], documents=[name], metadatas=[metadata])


def delete_intent(intent_id: int) -> None:
    _delete(get_intents_collection(), [f"intent-{intent_id}"])


def search_intents(
    question: str, n_results: int = 1, tenant_id: int | None = None
) -> list[dict]:
    """Search named intents."""
    collection = get_intents_collection()
    if collection.count() == 0:
        return []
    result = collection.query(
        query_texts=[question],
        n_results=n_results,
        where=_where({"kind": "intent"}, tenant_id),
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
