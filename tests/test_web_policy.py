"""Who may be read, and how much each page is trusted."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services import brain_paths
from app.services.web import trust


@pytest.fixture(autouse=True)
def policy_folder(monkeypatch, tmp_path) -> Path:
    """Keep workspace policies in a temporary folder."""
    monkeypatch.setattr(brain_paths, "brain_dir", lambda: tmp_path)
    return tmp_path


def test_a_domain_is_read_from_a_pasted_address() -> None:
    assert trust.clean_domain("https://Docs.Example.com/guide/page?x=1") == "docs.example.com"
    assert trust.clean_domain("  example.com.  ") == "example.com"


def test_an_address_is_not_an_allowlist_entry() -> None:
    assert trust.clean_domain("127.0.0.1") == ""
    assert trust.clean_domain("http://[::1]/") == ""
    assert trust.clean_domain("not a domain") == ""
    assert trust.clean_domain("") == ""


def test_an_allowlisted_site_covers_its_subdomains() -> None:
    policy = trust.WebPolicy(enabled=True, allowlist=("example.com",))
    assert policy.allows("https://example.com/a")
    assert policy.allows("https://docs.example.com/b")
    assert not policy.allows("https://notexample.com/c")
    assert not policy.allows("https://example.com.evil.net/d")


def test_trust_follows_the_allowlist() -> None:
    policy = trust.WebPolicy(allowlist=("example.com",))
    assert policy.tier_for("https://docs.example.com/a") == trust.OFFICIAL
    assert policy.tier_for("https://other.net/a") == trust.GENERAL


def test_an_allowlist_is_cleaned_and_deduplicated() -> None:
    cleaned = trust.clean_allowlist(["https://Example.com/x", "example.com", "  ", "a.org"])
    assert cleaned == ("example.com", "a.org")


def test_a_saved_policy_is_read_back(policy_folder: Path) -> None:
    saved = trust.save(
        7,
        trust.WebPolicy(
            enabled=True,
            provider="searxng",
            endpoint="http://search.local:8888/",
            allowlist=("example.com",),
            ttl_hours=48,
            min_score=0.6,
            results=5,
        ),
    )
    assert saved.endpoint == "http://search.local:8888"
    assert trust.read(7) == saved
    assert (policy_folder / "web-7.json").exists()


def test_a_policy_without_a_file_is_off() -> None:
    policy = trust.read(3)
    assert not policy.enabled
    assert policy.provider == "searxng"
    assert policy.allowlist == ()


def test_a_searxng_policy_needs_its_address() -> None:
    with pytest.raises(ValueError, match="SearXNG"):
        trust.save(0, trust.WebPolicy(enabled=True, provider="searxng"))


def test_an_unknown_provider_is_refused() -> None:
    with pytest.raises(ValueError, match="SearXNG"):
        trust.save(0, trust.WebPolicy(enabled=True, provider="browser"))


def test_a_hand_edited_file_is_bounded() -> None:
    path = trust.policy_path(0)
    path.write_text(
        '{"provider": "brave", "ttl_hours": 99999, "min_score": 5, "results": 99,'
        ' "endpoint": "ftp://bad", "allowlist": ["example.com"]}',
        encoding="utf-8",
    )
    policy = trust.read(0)
    assert policy.provider == "brave"
    assert policy.endpoint == ""
    assert policy.ttl_hours == trust.MAX_TTL_HOURS
    assert policy.min_score == 1.0
    assert policy.results == trust.MAX_RESULTS


def test_the_view_never_carries_a_search_key() -> None:
    view = trust.view(0)
    assert "api_key" not in view
    assert {item["id"] for item in view["providers"]} == set(trust.PROVIDERS)
