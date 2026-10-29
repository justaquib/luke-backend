"""Routes for reading the web as evidence: policy, sources and refresh."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import get_bearer_user, tenant_id_of
from app.db.database import User, get_session
from app.schemas import WebSettingsBody, WebSourceBody
from app.services.web import ingest, snapshots, trust

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/web", tags=["web"])


@router.get("/status")
def web_status(
    user: User = Depends(get_bearer_user),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """The policy, the allowlisted sources and how many pages each state holds."""
    tenant_id = tenant_id_of(user)
    sources = snapshots.list_sources(db, tenant_id)
    return {
        "policy": trust.view(tenant_id),
        "sources": sources,
        "counts": snapshots.counts(db, tenant_id),
    }


@router.put("/settings")
def save_web_settings(
    body: WebSettingsBody,
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """Keep the web evidence policy of this workspace."""
    tenant_id = tenant_id_of(user)
    policy = trust.WebPolicy(
        enabled=body.enabled,
        provider=body.provider,
        endpoint=body.endpoint,
        allowlist=tuple(body.allowlist),
        ttl_hours=body.ttl_hours,
        min_score=body.min_score,
        results=body.results,
    )
    try:
        saved = trust.save(tenant_id, policy)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return trust.view(tenant_id) | {"enabled": saved.enabled}


@router.post("/sources")
def add_web_source(
    body: WebSourceBody,
    user: User = Depends(get_bearer_user),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Allowlist one page and remember it for refresh."""
    tenant_id = tenant_id_of(user)
    domain = trust.clean_domain(body.url)
    if not domain:
        raise HTTPException(400, "The address must be a public web page.")
    source = snapshots.add_source(db, body.url, tenant_id)
    policy = trust.read(tenant_id)
    if domain not in policy.allowlist:
        updated = trust.WebPolicy(
            enabled=policy.enabled,
            provider=policy.provider,
            endpoint=policy.endpoint,
            allowlist=(*policy.allowlist, domain),
            ttl_hours=policy.ttl_hours,
            min_score=policy.min_score,
            results=policy.results,
        )
        try:
            trust.save(tenant_id, updated)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
    return {"source": {"id": source.id, "url": source.url, "domain": source.domain}}


@router.delete("/sources/{source_id}")
def delete_web_source(
    source_id: int,
    user: User = Depends(get_bearer_user),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Forget a source, its vectors, and a domain no other source still uses."""
    tenant_id = tenant_id_of(user)
    domain = snapshots.remove_source(db, source_id, tenant_id)
    if domain is None:
        raise HTTPException(404, "That source is not in this workspace.")
    _forget_domain(db, tenant_id, domain)
    return {"deleted": source_id}


def _forget_domain(db: Session, tenant_id: int | None, domain: str) -> None:
    """Drop an allowlist entry when the last source of that domain is gone."""
    if any(item["domain"] == domain for item in snapshots.list_sources(db, tenant_id)):
        return
    policy = trust.read(tenant_id)
    if domain not in policy.allowlist:
        return
    kept = tuple(entry for entry in policy.allowlist if entry != domain)
    try:
        trust.save(tenant_id, replace(policy, allowlist=kept))
    except ValueError as error:
        logger.warning("Could not drop the allowlist entry %s: %s", domain, error)


@router.post("/refresh")
def refresh_web_sources(
    user: User = Depends(get_bearer_user),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Re-read the allowlisted sources that are due, and drop expired pages."""
    tenant_id = tenant_id_of(user)
    policy = trust.read(tenant_id)
    if not policy.enabled:
        raise HTTPException(400, "Turn web evidence on before refreshing sources.")
    report = ingest.run_refresh(db, policy, tenant_id)
    return {"report": report, "sources": snapshots.list_sources(db, tenant_id)}
