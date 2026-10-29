"""Shared fixtures for the language route tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.security import get_bearer_user
from app.db.database import User, get_session
from app.routers import language_brain, language_model
from app.services.language import runtime


@pytest.fixture(autouse=True)
def no_installed_foundation(monkeypatch, tmp_path) -> Iterator[None]:
    """No test loads the real checkpoint; a test that needs one builds a fake."""
    from app.config import settings
    from app.services.foundation import identity
    from app.services.foundation import runtime as foundation_runtime

    monkeypatch.setattr(settings, "foundation_model_dir", str(tmp_path / "foundation"))
    identity._verified.cache_clear()
    foundation_runtime.unload()
    yield
    identity._verified.cache_clear()
    foundation_runtime.unload()


class _EmptyResult:
    """A query result with no rows, so the test needs no database."""

    def scalars(self) -> _EmptyResult:
        return self

    def all(self) -> list:
        return []

    def scalar_one(self) -> int:
        return 0


class _EmptySession:
    """A session that answers every query with nothing."""

    def execute(self, statement) -> _EmptyResult:
        return _EmptyResult()


@pytest.fixture()
def language_client(monkeypatch, tmp_path) -> Iterator[TestClient]:
    """A client for the language model routes with an empty tenant.

    The services that open their own session read the module attribute, so a
    temporary SQLite database replaces Postgres here: the routes are tested
    without needing a database server to be running.
    """
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    runtime._CACHE.clear()

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.db import database

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    database.Base.metadata.create_all(engine)
    monkeypatch.setattr(
        database,
        "SessionLocal",
        sessionmaker(bind=engine, autoflush=False, autocommit=False),
    )

    app = FastAPI()
    app.include_router(language_model.router)
    app.include_router(language_brain.router)
    app.dependency_overrides[get_bearer_user] = lambda: User(
        id=1,
        email="trainer@example.com",
        name="Trainer",
        is_systemadmin=False,
        tenant_id=None,
    )
    app.dependency_overrides[get_session] = _EmptySession
    with TestClient(app) as test_client:
        yield test_client
    runtime._CACHE.clear()
