"""Searching the web, storing what was read, and keeping sources fresh.

The search result says which address to read; the page itself is always
fetched through the guarded reader, so a search provider can never make the
engine read a private address. An allowlisted site is indexed as unverified
evidence; any other page is offered to the trainer as a candidate. Refreshing
re-reads only the allowlisted sources whose snapshot has passed its time to
live, and records whether the text actually changed.
"""

from __future__ import annotations

import logging

import httpx
from sqlalchemy import select
from sqlalchemy.orm import InstrumentedAttribute, Session
from sqlalchemy.sql import Select

from app.db.database import WebSnapshot, WebSource
from app.services.web import fetch, search, snapshots, trust

logger = logging.getLogger(__name__)


def _workspace(stmt: Select, column: InstrumentedAttribute, tenant_id: int | None) -> Select:
    return stmt.where(column.is_(None)) if tenant_id is None else stmt.where(column == tenant_id)


def ingest_page(
    db: Session,
    url: str,
    policy: trust.WebPolicy,
    tenant_id: int | None,
    client: httpx.Client | None = None,
) -> WebSnapshot:
    """Read one address through the guard and store it under its trust tier."""
    page = fetch.fetch_page(url, client=client)
    tier = policy.tier_for(page.final_url)
    snapshot, _ = snapshots.save_snapshot(db, page, tier, tenant_id)
    return snapshot


def search_and_store(
    db: Session,
    policy: trust.WebPolicy,
    tenant_id: int | None,
    query: str,
    client: httpx.Client | None = None,
) -> list[WebSnapshot]:
    """Search, read the first usable results, and return what was stored."""
    provider = search.provider_for(policy, client)
    if provider is None:
        return []
    try:
        results = provider.search(query, policy.results)
    except search.SearchError as error:
        logger.warning("The web search failed: %s", error)
        return []
    stored: list[WebSnapshot] = []
    for result in results[: policy.results]:
        try:
            stored.append(ingest_page(db, result.url, policy, tenant_id, client))
        except fetch.FetchError as error:
            logger.info("The web result %s was not read: %s", result.url, error)
    return stored


def _source_due(db: Session, source: WebSource, policy: trust.WebPolicy) -> bool:
    latest = snapshots.latest_snapshot(db, source.url, source.tenant_id)
    if latest is None:
        return True
    moment = snapshots.now_utc()
    age = moment - latest.checked_at
    return age.total_seconds() >= policy.ttl_hours * 3600


def _refresh_one(
    db: Session,
    source: WebSource,
    policy: trust.WebPolicy,
    client: httpx.Client | None,
) -> str:
    """Re-read one source; returns changed, unchanged or failed."""
    if not policy.allows(source.url):
        return "failed"
    try:
        page = fetch.fetch_page(source.url, client=client)
    except fetch.FetchError as error:
        logger.info("The source %s could not be refreshed: %s", source.url, error)
        return "failed"
    tier = policy.tier_for(page.final_url)
    _, changed = snapshots.save_snapshot(db, page, tier, source.tenant_id, source.id)
    return "changed" if changed else "unchanged"


def run_refresh(
    db: Session,
    policy: trust.WebPolicy,
    tenant_id: int | None,
    client: httpx.Client | None = None,
) -> dict:
    """Re-read every allowlisted source that is due, then drop expired pages."""
    stmt = select(WebSource).where(WebSource.enabled.is_(True))
    stmt = _workspace(stmt, WebSource.tenant_id, tenant_id)
    sources = db.execute(stmt.order_by(WebSource.id)).scalars().all()
    report = {"checked": 0, "changed": 0, "unchanged": 0, "failed": 0, "expired": 0}
    for source in sources:
        if not _source_due(db, source, policy):
            continue
        report["checked"] += 1
        outcome = _refresh_one(db, source, policy, client)
        report[outcome] += 1
    report["expired"] = snapshots.remove_expired(db, tenant_id, policy.ttl_hours)
    return report
