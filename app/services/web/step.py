"""The web evidence step of the answer flow.

The step runs only when the reader turned web evidence on for the question and
the workspace policy allows it. It first looks in the snapshots already stored,
so a repeated question is answered from the same reproducible page; only when
nothing is close enough does it search, read and store a page. A page outside
the allowlist is offered to the trainer and never answers.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.db.database import WebSnapshot
from app.schemas import CitationOut, QueryHit, QueryRequest, QueryResponse
from app.services.connections import Connection
from app.services.web import ingest, snapshots, trust


def _as_of(metadata: dict) -> str:
    fetched = int(metadata.get("fetched_at") or 0)
    return datetime.fromtimestamp(fetched, UTC).date().isoformat()


def _response(req: QueryRequest, entities: list, best: dict) -> QueryResponse:
    metadata = best.get("metadata") or {}
    text = " ".join(str(best.get("text") or "").split())
    title = str(metadata.get("title") or metadata.get("url") or "web page")
    url = str(metadata.get("url") or "")
    as_of = _as_of(metadata)
    return QueryResponse(
        question=req.question,
        source="web",
        answer=text,
        hits=[
            QueryHit(
                text=text,
                metadata={
                    "source": "web",
                    "url": url,
                    "title": title,
                    "as_of": as_of,
                    "verified": "false",
                },
                distance=best.get("distance"),
            )
        ],
        entities=entities,
        distance=best.get("distance"),
        confidence=best.get("score"),
        citations=[CitationOut(index=1, source="web", label=title, excerpt=text[:240])],
        app={
            "title": title,
            "url": url,
            "as_of": as_of,
            "trust": str(metadata.get("trust") or trust.OFFICIAL),
            "verified": "false",
        },
    )


def _official(snapshot: WebSnapshot) -> bool:
    return snapshot.trust == trust.OFFICIAL


def _connection(best: dict) -> Connection:
    metadata = best.get("metadata") or {}
    title = str(metadata.get("title") or metadata.get("url") or "web page")
    as_of = _as_of(metadata)
    return Connection(
        "web",
        "match",
        f"Quoted “{title}” fetched on {as_of}; it is unverified and not taught material.",
        title,
    )


def _live(db: Session, policy: trust.WebPolicy, tenant_id: int | None, query: str) -> list[dict]:
    """Search and read, then look again at what the allowlisted pages added."""
    stored = ingest.search_and_store(db, policy, tenant_id, query)
    if not any(_official(snapshot) for snapshot in stored):
        return []
    cutoff = snapshots.fresh_after(snapshots.now_utc(), policy.ttl_hours)
    return snapshots.search_web(query, tenant_id, policy.min_score, cutoff, policy.results)


def web_answer(
    req: QueryRequest,
    entities: list,
    tenant_id: int | None,
    db: Session | None,
) -> tuple[QueryResponse | None, tuple[Connection, ...]]:
    """Quote a stored trusted page, or report why the web had nothing to add."""
    if not req.web:
        return None, ()
    policy = trust.read(tenant_id)
    if not policy.enabled:
        return None, (Connection("web", "skipped", "Web evidence is turned off in Settings."),)
    if db is None:
        return None, (Connection("web", "fail", "The web search could not be saved."),)
    cutoff = snapshots.fresh_after(snapshots.now_utc(), policy.ttl_hours)
    hits = snapshots.search_web(req.question, tenant_id, policy.min_score, cutoff, policy.results)
    if hits:
        return _response(req, entities, hits[0]), (_connection(hits[0]),)
    hits = _live(db, policy, tenant_id, req.question)
    if hits:
        return _response(req, entities, hits[0]), (_connection(hits[0]),)
    return None, (
        Connection(
            "web",
            "miss",
            "No stored page was close enough and no allowlisted page answered; "
            "general pages are offered to the trainer instead.",
        ),
    )
