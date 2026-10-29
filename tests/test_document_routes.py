"""Uploaded files are only served to a signed-in request."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_viewing_a_file_requires_a_login() -> None:
    response = TestClient(app).get("/documents/notes.txt/view")

    assert response.status_code == 401


def test_downloading_a_file_requires_a_login() -> None:
    response = TestClient(app).get("/documents/notes.txt/download")

    assert response.status_code == 401
