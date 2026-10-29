"""The web routes keep one workspace's policy and its sources."""

from __future__ import annotations

from collections.abc import Iterator

import package_vectors
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import get_bearer_user
from app.db import database
from app.db.database import User, get_session
from app.routers import web
from app.services import brain_paths
from app.services.web import snapshots


@pytest.fixture()
def client(monkeypatch, tmp_path) -> Iterator[TestClient]:
    monkeypatch.setattr(brain_paths, "brain_dir", lambda: tmp_path)
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    database.Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False)()
    collections: dict[str, package_vectors.Collection] = {}
    monkeypatch.setattr(
        snapshots,
        "open_collection",
        lambda name: collections.setdefault(name, package_vectors.Collection()),
    )

    app = FastAPI()
    app.include_router(web.router)
    app.dependency_overrides[get_bearer_user] = lambda: User(
        id=1, email="trainer@example.com", name="Trainer", tenant_id=0
    )
    app.dependency_overrides[get_session] = lambda: session
    with TestClient(app) as test_client:
        yield test_client
    session.close()


def test_the_status_starts_off(client: TestClient) -> None:
    body = client.get("/web/status").json()
    assert body["policy"]["enabled"] is False
    assert body["sources"] == []
    assert body["counts"]["indexed"] == 0


def test_settings_are_saved_and_read_back(client: TestClient) -> None:
    response = client.put(
        "/web/settings",
        json={
            "enabled": True,
            "provider": "searxng",
            "endpoint": "http://search.local:8888",
            "allowlist": ["https://docs.example.com/guide"],
            "ttl_hours": 48,
            "min_score": 0.5,
            "results": 4,
        },
    )
    assert response.status_code == 200
    body = client.get("/web/status").json()
    assert body["policy"]["enabled"] is True
    assert body["policy"]["allowlist"] == ["docs.example.com"]
    assert body["policy"]["ttl_hours"] == 48


def test_a_searxng_setting_without_an_address_is_refused(client: TestClient) -> None:
    response = client.put("/web/settings", json={"enabled": True, "provider": "searxng"})
    assert response.status_code == 400
    assert "SearXNG" in response.json()["detail"]


def test_adding_a_source_allowlists_its_domain(client: TestClient) -> None:
    response = client.post("/web/sources", json={"url": "https://docs.example.com/guide/page"})
    assert response.status_code == 200
    assert response.json()["source"]["domain"] == "docs.example.com"
    status = client.get("/web/status").json()
    assert status["policy"]["allowlist"] == ["docs.example.com"]
    assert [item["domain"] for item in status["sources"]] == ["docs.example.com"]


def test_a_source_without_a_public_name_is_refused(client: TestClient) -> None:
    response = client.post("/web/sources", json={"url": "http://127.0.0.1/page"})
    assert response.status_code == 400


def test_an_unknown_source_cannot_be_deleted(client: TestClient) -> None:
    assert client.delete("/web/sources/99").status_code == 404


def test_a_known_source_can_be_deleted(client: TestClient) -> None:
    source = client.post("/web/sources", json={"url": "https://docs.example.com/guide"}).json()
    assert client.delete(f"/web/sources/{source['source']['id']}").status_code == 200
    status = client.get("/web/status").json()
    assert status["sources"] == []
    assert status["policy"]["allowlist"] == []


def test_refresh_requires_web_evidence_to_be_on(client: TestClient) -> None:
    response = client.post("/web/refresh")
    assert response.status_code == 400
    assert "Turn web evidence on" in response.json()["detail"]
