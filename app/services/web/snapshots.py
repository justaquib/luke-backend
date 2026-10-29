"""Keeping each read page as a snapshot, and searching the trusted ones.

A snapshot records the address, the title, the readable text, the hash of that
text, the trust tier, when it was fetched and when it was last checked. Official
pages are chunked into the unverified ``web`` collection so an answer can quote
them later; a general page is kept as a candidate for the trainer and never
enters the collection. An unchanged re-read keeps the text and moves the date
forward, so the citation stays reproducible without storing the page twice.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import UTC, datetime, timedelta

import chromadb
from sqlalchemy import func, select
from sqlalchemy.orm import InstrumentedAttribute, Session
from sqlalchemy.sql import Select

from app.config import settings
from app.db.database import Faq, WebSnapshot, WebSource
from app.services.chroma import open_collection, tenant_metadata, where
from app.services.nlp import chunk_spans
from app.services.retrieval import similarity_from_distance
from app.services.web import trust
from app.services.web.fetch import FetchedPage

logger = logging.getLogger(__name__)

STATUS_INDEXED = "indexed"
STATUS_CANDIDATE = "candidate"
STATUS_SUPERSEDED = "superseded"
MAX_CHUNKS = 24
CANDIDATE_CHARS = 900
_WEB_NAMESPACE = uuid.UUID("6c1d0f7e-3a4b-5c8d-9e2f-1a3b5c7d9e0f")


def now_utc() -> datetime:
    """The current moment, stored without a zone like every other row."""
    return datetime.now(UTC).replace(tzinfo=None)


def content_hash(text: str) -> str:
    """The hash that recognises the same page text on a later read."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fresh_after(now: datetime, ttl_hours: int) -> int:
    """The oldest fetch time a citation may still use, as epoch seconds."""
    moment = now.replace(tzinfo=UTC) if now.tzinfo is None else now
    return int((moment - timedelta(hours=ttl_hours)).timestamp())


def web_chunk_id(tenant_id: int | None, url: str, text: str) -> str:
    """The stable id of one web passage, so a re-index overwrites it."""
    workspace = 0 if tenant_id is None else tenant_id
    return str(uuid.uuid5(_WEB_NAMESPACE, f"{workspace}:{url}:{content_hash(text)}"))


def _collection() -> chromadb.Collection:
    return open_collection(settings.chroma_web_collection)


def remove_vectors(url: str, tenant_id: int | None) -> None:
    """Remove every passage indexed from one address in one workspace."""
    collection = _collection()
    if collection.count() == 0:
        return
    try:
        collection.delete(where=where({"source": url}, tenant_id))
    except chromadb.errors.ChromaError as error:
        logger.warning("Could not delete the web vectors of %s: %s", url, error)


def index_snapshot(snapshot: WebSnapshot) -> int:
    """Chunk an official page into the unverified web collection."""
    spans = chunk_spans(snapshot.text)[:MAX_CHUNKS]
    if not spans:
        return 0
    texts = [span.text for span in spans]
    fetched = int(snapshot.fetched_at.replace(tzinfo=UTC).timestamp())
    ids = [web_chunk_id(snapshot.tenant_id, snapshot.url, text) for text in texts]
    metadatas = [
        {
            "source": snapshot.url,
            "url": snapshot.url,
            "title": snapshot.title,
            "trust": snapshot.trust,
            "verified": "false",
            "fetched_at": fetched,
            "snapshot_id": snapshot.id,
            "content_hash": snapshot.content_hash,
            **tenant_metadata(snapshot.tenant_id),
        }
        for _ in texts
    ]
    remove_vectors(snapshot.url, snapshot.tenant_id)
    _collection().upsert(ids=ids, documents=texts, metadatas=metadatas)
    return len(texts)


def search_web(
    question: str,
    tenant_id: int | None,
    min_score: float,
    fresh_from: int,
    n_results: int = 3,
) -> list[dict]:
    """The freshest trusted passages that are close enough to quote."""
    collection = _collection()
    if collection.count() == 0:
        return []
    condition = where({"fetched_at": {"$gte": fresh_from}}, tenant_id)
    result = collection.query(query_texts=[question], n_results=n_results, where=condition)
    space = str((collection.metadata or {}).get("hnsw:space", "l2"))
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    hits: list[dict] = []
    for index, text in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        score = similarity_from_distance(distance, space)
        if score is None or score < min_score:
            continue
        hits.append({"text": text, "metadata": metadata, "distance": distance, "score": score})
    return hits


def _workspace(stmt: Select, column: InstrumentedAttribute, tenant_id: int | None) -> Select:
    return stmt.where(column.is_(None)) if tenant_id is None else stmt.where(column == tenant_id)


def latest_snapshot(db: Session, url: str, tenant_id: int | None) -> WebSnapshot | None:
    """The newest snapshot of one address that has not been superseded."""
    stmt = select(WebSnapshot).where(
        WebSnapshot.url == url, WebSnapshot.status != STATUS_SUPERSEDED
    )
    stmt = _workspace(stmt, WebSnapshot.tenant_id, tenant_id)
    return db.execute(stmt.order_by(WebSnapshot.id.desc())).scalars().first()


def _supersede(db: Session, url: str, tenant_id: int | None) -> None:
    stmt = select(WebSnapshot).where(
        WebSnapshot.url == url, WebSnapshot.status != STATUS_SUPERSEDED
    )
    stmt = _workspace(stmt, WebSnapshot.tenant_id, tenant_id)
    for row in db.execute(stmt).scalars().all():
        row.status = STATUS_SUPERSEDED


def save_snapshot(
    db: Session,
    page: FetchedPage,
    tier: str,
    tenant_id: int | None,
    source_id: int | None = None,
) -> tuple[WebSnapshot, bool]:
    """Store one read page; returns the snapshot and whether its text changed."""
    digest = content_hash(page.text)
    latest = latest_snapshot(db, page.url, tenant_id)
    if latest is not None and latest.content_hash == digest:
        latest.checked_at = page.fetched_at
        latest.fetched_at = page.fetched_at
        db.commit()
        db.refresh(latest)
        return latest, False
    _supersede(db, page.url, tenant_id)
    snapshot = WebSnapshot(
        source_id=source_id,
        url=page.url,
        title=page.title or page.url,
        text=page.text,
        content_hash=digest,
        trust=tier,
        status=STATUS_INDEXED if tier == trust.OFFICIAL else STATUS_CANDIDATE,
        fetched_at=page.fetched_at,
        checked_at=page.fetched_at,
        tenant_id=tenant_id,
    )
    db.add(snapshot)
    db.flush()
    if tier == trust.OFFICIAL:
        index_snapshot(snapshot)
    else:
        remove_vectors(page.url, tenant_id)
        add_candidate(db, snapshot, page.url)
    db.commit()
    db.refresh(snapshot)
    return snapshot, True


def add_candidate(db: Session, snapshot: WebSnapshot, origin: str) -> Faq | None:
    """Offer a general page to the trainer, without training on it."""
    question = f"{snapshot.title} — {origin}"[:500]
    stmt = select(Faq).where(Faq.question == question)
    stmt = _workspace(stmt, Faq.tenant_id, snapshot.tenant_id)
    if db.execute(stmt).scalars().first() is not None:
        return None
    faq = Faq(
        question=question,
        answer=snapshot.text[:CANDIDATE_CHARS].strip(),
        status="pending",
        tenant_id=snapshot.tenant_id,
        confidence=settings.base_answer_confidence,
    )
    db.add(faq)
    db.flush()
    return faq


def add_source(db: Session, url: str, tenant_id: int | None) -> WebSource:
    """Allowlist one page and its site for reading and refreshing."""
    domain = trust.domain_of(url)
    if not domain or not trust.clean_domain(domain):
        raise ValueError("The address must be a public web page.")
    stmt = select(WebSource).where(WebSource.domain == domain)
    stmt = _workspace(stmt, WebSource.tenant_id, tenant_id)
    existing = db.execute(stmt).scalars().first()
    if existing is not None:
        return existing
    source = WebSource(url=url.strip(), domain=domain, tenant_id=tenant_id)
    db.add(source)
    db.commit()
    db.refresh(source)
    return source


def list_sources(db: Session, tenant_id: int | None) -> list[dict]:
    """The allowlisted sources with what was last read from each."""
    stmt = _workspace(select(WebSource), WebSource.tenant_id, tenant_id)
    rows = db.execute(stmt.order_by(WebSource.id)).scalars().all()
    return [_source_view(db, row) for row in rows]


def _source_view(db: Session, source: WebSource) -> dict:
    latest = latest_snapshot(db, source.url, source.tenant_id)
    checked = latest.checked_at if latest is not None else None
    return {
        "id": source.id,
        "url": source.url,
        "domain": source.domain,
        "enabled": bool(source.enabled),
        "title": latest.title if latest is not None else "",
        "status": latest.status if latest is not None else "new",
        "checked_at": checked.isoformat() if checked is not None else None,
    }


def remove_source(db: Session, source_id: int, tenant_id: int | None) -> str | None:
    """Forget an allowlisted source and its vectors; returns its domain."""
    stmt = select(WebSource).where(WebSource.id == source_id)
    stmt = _workspace(stmt, WebSource.tenant_id, tenant_id)
    source = db.execute(stmt).scalars().first()
    if source is None:
        return None
    remove_vectors(source.url, tenant_id)
    domain = source.domain
    db.delete(source)
    db.commit()
    return domain


def remove_expired(db: Session, tenant_id: int | None, ttl_hours: int) -> int:
    """Delete snapshots that have not been checked inside the time to live."""
    cutoff = now_utc() - timedelta(hours=ttl_hours)
    stmt = select(WebSnapshot).where(WebSnapshot.checked_at < cutoff)
    stmt = _workspace(stmt, WebSnapshot.tenant_id, tenant_id)
    rows = db.execute(stmt).scalars().all()
    for row in rows:
        if row.status == STATUS_INDEXED:
            remove_vectors(row.url, tenant_id)
        db.delete(row)
    if rows:
        db.commit()
    return len(rows)


def counts(db: Session, tenant_id: int | None) -> dict[str, int]:
    """How many snapshots are indexed, waiting for a trainer, or superseded."""
    stmt = select(WebSnapshot.status, func.count(WebSnapshot.id)).group_by(WebSnapshot.status)
    stmt = _workspace(stmt, WebSnapshot.tenant_id, tenant_id)
    found = {str(status): int(total) for status, total in db.execute(stmt).all()}
    return {
        STATUS_INDEXED: found.get(STATUS_INDEXED, 0),
        STATUS_CANDIDATE: found.get(STATUS_CANDIDATE, 0),
        STATUS_SUPERSEDED: found.get(STATUS_SUPERSEDED, 0),
    }
