import os
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings


_client = None


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_collection():
    return get_client().get_or_create_collection(name=settings.chroma_collection)


def get_faq_collection():
    return get_client().get_or_create_collection(name=settings.chroma_faq_collection)


def get_intents_collection():
    return get_client().get_or_create_collection(name=settings.chroma_intents_collection)


def add_chunks(source: str, chunks: list[str], ids: list[str]) -> int:
    coll = get_collection()
    coll.add(documents=chunks, ids=ids, metadatas=[{"source": source} for _ in chunks])
    return len(chunks)


def query(question: str, n_results: int = 4) -> list[dict]:
    coll = get_collection()
    if coll.count() == 0:
        return []
    res = coll.query(query_texts=[question], n_results=n_results)
    out = []
    for i, doc in enumerate(res["documents"][0]):
        meta = res["metadatas"][0][i] if res.get("metadatas") else {}
        dist = res["distances"][0][i] if res.get("distances") else None
        out.append({"text": doc, "metadata": meta, "distance": dist})
    return out


def collection_size() -> int:
    return get_collection().count()


def upsert_faq(faq_id: int, question: str, answer: str, status: str = "pending") -> None:
    coll = get_faq_collection()
    coll.upsert(
        ids=[str(faq_id)],
        documents=[question],
        metadatas=[{"answer": answer, "faq_id": faq_id, "status": status}],
    )


def delete_faq(faq_id: int) -> None:
    coll = get_faq_collection()
    try:
        coll.delete(ids=[str(faq_id)])
    except Exception:
        pass


def search_faq(question: str, n_results: int = 1) -> list[dict]:
    coll = get_faq_collection()
    if coll.count() == 0:
        return []
    res = coll.query(
        query_texts=[question],
        n_results=n_results,
        where={"status": "approved"},
    )
    out = []
    for i, doc in enumerate(res["documents"][0]):
        meta = res["metadatas"][0][i] if res.get("metadatas") else {}
        dist = res["distances"][0][i] if res.get("distances") else None
        out.append({"question": doc, "answer": meta.get("answer", ""), "distance": dist, "faq_id": meta.get("faq_id")})
    return out


def search_faq_loose(question: str, n_results: int = 5) -> list[dict]:
    coll = get_faq_collection()
    if coll.count() == 0:
        return []
    res = coll.query(
        query_texts=[question],
        n_results=n_results,
    )
    out = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) else {}
        dist = dists[i] if i < len(dists) else None
        out.append({
            "question": docs[i],
            "answer": meta.get("answer", ""),
            "distance": dist,
            "faq_id": meta.get("faq_id"),
            "status": meta.get("status"),
        })
    return out


def search_faq_in(question: str, faq_ids: list[int], n_results: int = 50) -> list[dict]:
    coll = get_faq_collection()
    if coll.count() == 0 or not faq_ids:
        return []
    res = coll.query(
        query_texts=[question],
        n_results=n_results,
        where={"faq_id": {"$in": [str(i) for i in faq_ids]}},
    )
    out = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) else {}
        out.append({
            "question": docs[i],
            "answer": meta.get("answer", ""),
            "distance": dists[i] if i < len(dists) else None,
            "faq_id": meta.get("faq_id"),
            "status": meta.get("status"),
        })
    return out


def search_pending(question: str, candidates: list[str], n_results: int = 50) -> list[dict]:
    coll = get_intents_collection()
    if coll.count() == 0 or not candidates:
        return []
    res = coll.query(
        query_texts=[question],
        n_results=n_results,
        where={"kind": "pending"},
    )
    out = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) else {}
        text = docs[i]
        if text not in candidates:
            continue
        out.append({
            "pending_id": meta.get("pending_id"),
            "question": text,
            "distance": dists[i] if i < len(dists) else None,
        })
    return out


def search_pending_all(question: str, n_results: int = 25) -> list[dict]:
    coll = get_intents_collection()
    if coll.count() == 0:
        return []
    res = coll.query(
        query_texts=[question],
        n_results=n_results,
        where={"kind": "pending"},
    )
    out = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) else {}
        out.append({
            "pending_id": meta.get("pending_id"),
            "question": docs[i],
            "distance": dists[i] if i < len(dists) else None,
        })
    return out


def index_pending(pending_id: int, question: str) -> None:
    coll = get_intents_collection()
    coll.upsert(
        ids=[f"pending-{pending_id}"],
        documents=[question],
        metadatas=[{"kind": "pending", "pending_id": pending_id}],
    )


def delete_pending_index(pending_id: int) -> None:
    coll = get_intents_collection()
    try:
        coll.delete(ids=[f"pending-{pending_id}"])
    except Exception:
        pass


def upsert_intent(intent_id: int, name: str, response_template: str, faq_id: int | None) -> None:
    coll = get_intents_collection()
    meta = {"kind": "intent", "name": name, "response_template": response_template}
    if faq_id is not None:
        meta["faq_id"] = faq_id
    coll.upsert(
        ids=[f"intent-{intent_id}"],
        documents=[name],
        metadatas=[meta],
    )


def delete_intent(intent_id: int) -> None:
    coll = get_intents_collection()
    try:
        coll.delete(ids=[f"intent-{intent_id}"])
    except Exception:
        pass


def search_intents(question: str, n_results: int = 1) -> list[dict]:
    coll = get_intents_collection()
    if coll.count() == 0:
        return []
    res = coll.query(
        query_texts=[question],
        n_results=n_results,
        where={"kind": "intent"},
    )
    out = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) else {}
        out.append({
            "name": docs[i],
            "response_template": meta.get("response_template", ""),
            "faq_id": meta.get("faq_id"),
            "distance": dists[i] if i < len(dists) else None,
        })
    return out


def faq_size() -> int:
    return get_faq_collection().count()


def intents_size() -> int:
    return get_intents_collection().count()