"""The dev server may pick any free loopback port, so CORS must allow it."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:4200",
        "http://localhost:5199",
        "http://127.0.0.1:4304",
    ],
)
def test_a_loopback_dev_origin_is_allowed(origin: str) -> None:
    client = TestClient(create_app())
    response = client.options(
        "/query/stream",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_a_foreign_origin_is_not_allowed() -> None:
    client = TestClient(create_app())
    response = client.options(
        "/query/stream",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") is None
