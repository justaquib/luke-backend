"""Allowlisted pages are indexed unverified; other pages wait for a trainer."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from types import SimpleNamespace

import package_vectors
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import database
from app.db.database import Faq, WebSnapshot, WebSource
from app.schemas import QueryRequest
from app.services import brain_paths
from app.services.web import ingest, snapshots, step, trust
from app.services.web.fetch import FetchedPage

URL = "https://docs.example.com/tokenizer/bpe"
TEXT = "Byte pair encoding replaces the most frequent pair with a new symbol."
OTHER = "A byte-level model keeps every character addressable."


@pytest.fixture()
def db() -> Iterator[Session]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    database.Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False)()
    yield session
    session.close()


@pytest.fixture()
def vectors(monkeypatch, tmp_path) -> dict[str, package_vectors.Collection]:
    """A Chroma-shaped collection, and a temporary folder for the policy."""
    monkeypatch.setattr(brain_paths, "brain_dir", lambda: tmp_path)
    collections: dict[str, package_vectors.Collection] = {}

    def open_collection(name: str) -> package_vectors.Collection:
        return collections.setdefault(name, package_vectors.Collection())

    monkeypatch.setattr(snapshots, "open_collection", open_collection)
    return collections


def page(text: str = TEXT, when: datetime | None = None, url: str = URL) -> FetchedPage:
    return FetchedPage(
        url=url,
        final_url=url,
        title="Byte pair encoding",
        text=text,
        size=len(text),
        fetched_at=when or snapshots.now_utc(),
    )


def test_an_official_page_is_indexed_unverified(db, vectors) -> None:
    snapshot, changed = snapshots.save_snapshot(db, page(), trust.OFFICIAL, 0)
    assert changed
    assert snapshot.status == snapshots.STATUS_INDEXED
    stored = vectors["web"].rows
    assert len(stored) == 1
    metadata = next(iter(stored.values()))[1]
    assert metadata["verified"] == "false"
    assert metadata["trust"] == trust.OFFICIAL
    assert metadata["url"] == URL
    assert isinstance(metadata["fetched_at"], int)
    hits = snapshots.search_web(TEXT, 0, 0.45, 0)
    assert hits and hits[0]["metadata"]["snapshot_id"] == snapshot.id


def test_a_general_page_waits_for_a_trainer_and_is_not_indexed(db, vectors) -> None:
    snapshot, _ = snapshots.save_snapshot(db, page(), trust.GENERAL, 0)
    assert snapshot.status == snapshots.STATUS_CANDIDATE
    assert not vectors.get("web") or vectors["web"].count() == 0
    candidate = db.execute(select(Faq)).scalars().one()
    assert candidate.status == "pending"
    assert URL in candidate.question
    assert candidate.answer.startswith("Byte pair encoding")


def test_the_same_text_is_not_stored_twice(db, vectors) -> None:
    first, changed = snapshots.save_snapshot(db, page(), trust.OFFICIAL, 0)
    later = snapshots.now_utc() + timedelta(hours=2)
    again, changed_again = snapshots.save_snapshot(db, page(when=later), trust.OFFICIAL, 0)
    assert changed and not changed_again
    assert again.id == first.id
    assert again.checked_at == later
    assert snapshots.counts(db, 0)[snapshots.STATUS_INDEXED] == 1
    assert vectors["web"].count() == 1


def test_changed_text_supersedes_the_old_page(db, vectors) -> None:
    snapshots.save_snapshot(db, page(), trust.OFFICIAL, 0)
    fresh, changed = snapshots.save_snapshot(db, page(OTHER), trust.OFFICIAL, 0)
    assert changed
    counts = snapshots.counts(db, 0)
    assert counts[snapshots.STATUS_INDEXED] == 1
    assert counts[snapshots.STATUS_SUPERSEDED] == 1
    texts = [text for text, _ in vectors["web"].rows.values()]
    assert texts == [OTHER]
    assert snapshots.latest_snapshot(db, URL, 0).id == fresh.id


def test_an_expired_page_is_removed(db, vectors) -> None:
    snapshot, _ = snapshots.save_snapshot(db, page(), trust.OFFICIAL, 0)
    snapshot.checked_at = snapshots.now_utc() - timedelta(days=3)
    db.commit()
    assert snapshots.remove_expired(db, 0, 24) == 1
    assert db.execute(select(WebSnapshot)).scalars().all() == []
    assert vectors["web"].count() == 0


def test_a_disallowed_source_is_not_refreshed(db, vectors, monkeypatch) -> None:
    db.add(WebSource(url=URL, domain="docs.example.com", tenant_id=0))
    db.commit()

    def refuse(*args: object, **kwargs: object) -> FetchedPage:
        raise AssertionError("a disallowed source must not be read")

    monkeypatch.setattr(ingest.fetch, "fetch_page", refuse)
    report = ingest.run_refresh(db, trust.WebPolicy(enabled=True, allowlist=()), 0)
    assert report["failed"] == 1


def test_refresh_reports_a_changed_and_an_unchanged_page(db, vectors, monkeypatch) -> None:
    source = WebSource(url=URL, domain="docs.example.com", tenant_id=0)
    db.add(source)
    db.commit()
    old, _ = snapshots.save_snapshot(db, page(), trust.OFFICIAL, 0)
    old.checked_at = snapshots.now_utc() - timedelta(days=8)
    db.commit()
    policy = trust.WebPolicy(
        enabled=True, endpoint="http://search.local:8888", allowlist=("docs.example.com",)
    )

    monkeypatch.setattr(ingest.fetch, "fetch_page", lambda url, client=None: page())
    assert ingest.run_refresh(db, policy, 0)["unchanged"] == 1

    old = db.execute(select(WebSnapshot)).scalars().first()
    old.checked_at = snapshots.now_utc() - timedelta(days=8)
    db.commit()
    monkeypatch.setattr(ingest.fetch, "fetch_page", lambda url, client=None: page(OTHER))
    report = ingest.run_refresh(db, policy, 0)
    assert report["changed"] == 1
    assert report["expired"] == 1
    latest = snapshots.latest_snapshot(db, URL, 0)
    assert latest is not None and latest.text == OTHER


def test_a_fresh_source_is_not_checked_again(db, vectors, monkeypatch) -> None:
    db.add(WebSource(url=URL, domain="docs.example.com", tenant_id=0))
    db.commit()
    snapshots.save_snapshot(db, page(), trust.OFFICIAL, 0)
    monkeypatch.setattr(ingest.fetch, "fetch_page", lambda url, client=None: page(OTHER))
    report = ingest.run_refresh(
        db,
        trust.WebPolicy(
            enabled=True, endpoint="http://search.local:8888", allowlist=("docs.example.com",)
        ),
        0,
    )
    assert report["checked"] == 0


def test_a_stored_page_answers_with_its_address_and_date(db, vectors) -> None:
    trust.save(
        0,
        trust.WebPolicy(
            enabled=True, endpoint="http://search.local:8888", allowlist=("docs.example.com",)
        ),
    )
    snapshots.save_snapshot(db, page(), trust.OFFICIAL, 0)
    response, connections = step.web_answer(QueryRequest(question=TEXT, web=True), [], 0, db)
    assert response is not None
    assert response.source == "web"
    assert response.app["url"] == URL
    assert response.app["as_of"] == snapshots.now_utc().date().isoformat()
    assert response.app["verified"] == "false"
    assert response.citations[0].source == "web"
    assert connections[0].outcome == "match"
    assert "unverified" in connections[0].detail


def test_web_evidence_is_skipped_when_it_is_off(db, vectors) -> None:
    response, connections = step.web_answer(QueryRequest(question=TEXT, web=True), [], 0, db)
    assert response is None
    assert connections[0].outcome == "skipped"


def test_web_evidence_is_not_asked_for_unless_requested(db, vectors) -> None:
    trust.save(
        0,
        trust.WebPolicy(
            enabled=True, endpoint="http://search.local:8888", allowlist=("docs.example.com",)
        ),
    )
    snapshots.save_snapshot(db, page(), trust.OFFICIAL, 0)
    response, connections = step.web_answer(QueryRequest(question=TEXT), [], 0, db)
    assert response is None
    assert connections == ()


def test_only_general_results_leave_the_question_unanswered(db, vectors, monkeypatch) -> None:
    trust.save(
        0,
        trust.WebPolicy(
            enabled=True, endpoint="http://search.local:8888", allowlist=("docs.example.com",)
        ),
    )
    found = [SimpleNamespace(trust=trust.GENERAL)]
    monkeypatch.setattr(step.ingest, "search_and_store", lambda *args, **kwargs: found)
    response, connections = step.web_answer(QueryRequest(question=TEXT, web=True), [], 0, db)
    assert response is None
    assert connections[0].outcome == "miss"
    assert "trainer" in connections[0].detail
