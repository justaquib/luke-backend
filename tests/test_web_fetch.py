"""The guarded reader refuses every address that is not a public page."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

from app.config import settings
from app.services.web import fetch, search, trust

FIXTURES = Path(__file__).parent / "fixtures" / "web"
REAL_ADDRESSES = fetch._addresses


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def fresh_reader(monkeypatch) -> Iterator[list[float]]:
    """A public test site, a still clock, and no real sleeping."""
    slept: list[float] = []
    fetch.reset_caches()
    monkeypatch.setattr(fetch, "_addresses", lambda host, port: ["93.184.216.34"])
    monkeypatch.setattr(fetch, "_CLOCK", lambda: 1000.0)
    monkeypatch.setattr(fetch, "_SLEEP", slept.append)
    yield slept
    fetch.reset_caches()


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def site_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/robots.txt":
        return httpx.Response(200, text=fixture_text("robots.txt"), headers=typed("text/plain"))
    return httpx.Response(200, text=fixture_text("page.html"), headers=typed("text/html"))


def typed(kind: str) -> dict[str, str]:
    return {"content-type": kind}


def test_a_page_is_read_as_text_with_its_title() -> None:
    page = fetch.fetch_page(
        "https://docs.example.com/tokenizer/bpe", client=client_for(site_handler)
    )
    assert page.title == "Training a small word model"
    assert "byte-level model learns" in page.text
    assert "<p>" not in page.text
    assert page.final_url == "https://docs.example.com/tokenizer/bpe"
    assert page.size > 0


def test_robots_disallow_stops_the_read() -> None:
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        return site_handler(request)

    with pytest.raises(fetch.FetchError, match=r"robots\.txt"):
        fetch.fetch_page("https://docs.example.com/private/secret", client=client_for(handler))
    assert asked == ["/robots.txt"]


def test_a_missing_robots_file_allows_the_read() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return site_handler(request)

    page = fetch.fetch_page("https://docs.example.com/guide", client=client_for(handler))
    assert "Training a small word model" in page.title


def test_robots_parser_uses_the_longest_rule() -> None:
    rules = fetch.parse_robots(fixture_text("robots.txt"))
    assert rules.allows("/guide")
    assert not rules.allows("/private/secret")
    assert rules.allows("/private/public/one")
    assert not rules.allows("/search?q=1")


def test_a_named_agent_group_wins_over_the_wildcard() -> None:
    rules = fetch.parse_robots(
        "User-agent: *\nDisallow: /\n\nUser-agent: lukelab\nDisallow: /draft"
    )
    assert rules.allows("/guide")
    assert not rules.allows("/draft/page")


def test_a_redirect_chain_is_capped() -> None:
    seen = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        seen += 1
        return httpx.Response(302, headers={"location": f"/step-{seen}"})

    with pytest.raises(fetch.FetchError, match="too many times"):
        fetch.fetch_page("https://docs.example.com/start", client=client_for(handler))
    assert seen == settings.web_max_redirects + 1


def test_a_file_that_is_not_a_page_is_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, content=b"%PDF-1.7", headers=typed("application/pdf"))

    with pytest.raises(fetch.FetchError, match="pdf"):
        fetch.fetch_page("https://docs.example.com/file.pdf", client=client_for(handler))


def test_a_page_larger_than_the_cap_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(settings, "web_max_bytes", 64)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, text="x" * 500, headers=typed("text/plain"))

    with pytest.raises(fetch.FetchError, match="larger than"):
        fetch.fetch_page("https://docs.example.com/big.txt", client=client_for(handler))


def test_two_reads_of_one_site_are_kept_apart(fresh_reader: list[float]) -> None:
    fetch.fetch_page("https://docs.example.com/a", client=client_for(site_handler))
    fetch.fetch_page("https://docs.example.com/b", client=client_for(site_handler))
    assert fresh_reader == [fetch.MIN_INTERVAL_SECONDS]


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://10.0.0.5/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://0.0.0.0/",
    ],
)
def test_an_address_on_a_private_network_is_refused(monkeypatch, url: str) -> None:
    monkeypatch.setattr(fetch, "_addresses", REAL_ADDRESSES)
    with pytest.raises(fetch.FetchError, match="private network"):
        fetch.ensure_public(url)


def test_a_public_address_is_accepted() -> None:
    assert fetch._public_ip("93.184.216.34")
    assert not fetch._public_ip("127.0.0.1")
    assert not fetch._public_ip("192.168.1.2")
    assert not fetch._public_ip("not-an-address")


def test_a_name_that_resolves_inward_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(fetch, "_addresses", lambda host, port: ["10.0.0.5"])
    with pytest.raises(fetch.FetchError, match="private network"):
        fetch.ensure_public("https://docs.example.com/guide")


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/",
        "http://service.local/",
        "https://user:secret@example.com/",
        "http://example.com:8080/",
        "file:///etc/passwd",
    ],
)
def test_an_address_that_is_not_a_plain_web_page_is_refused(url: str) -> None:
    with pytest.raises(fetch.FetchError):
        fetch.validate_url(url)


def searxng_client() -> httpx.Client:
    payload = fixture_text("searxng.json")
    return client_for(lambda request: httpx.Response(200, text=payload))


def test_searxng_results_are_read_best_first() -> None:
    found = search.SearxngProvider("http://search.local:8080/", searxng_client()).search("bpe", 5)
    assert [item.url for item in found] == [
        "https://docs.example.com/tokenizer/bpe",
        "https://docs.example.com/tokenizer/specials",
    ]
    assert found[0].title == "Byte-level BPE - Example Docs"
    assert "256 byte symbols" in found[0].snippet


def test_brave_sends_its_key_and_drops_unreadable_urls() -> None:
    payload = fixture_text("brave.json")
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("X-Subscription-Token", "")
        seen["path"] = request.url.path
        return httpx.Response(200, text=payload)

    found = search.BraveProvider("secret-key", client_for(handler)).search("bpe", 5)
    assert [item.url for item in found] == ["https://example.org/byte-tokenizer"]
    assert seen["key"] == "secret-key"
    assert seen["path"] == "/res/v1/web/search"


def test_tavily_posts_the_query_and_drops_hostless_urls() -> None:
    payload = fixture_text("tavily.json")
    body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body.update(json.loads(request.content))
        return httpx.Response(200, text=payload)

    found = search.TavilyProvider("tvly-test", client_for(handler)).search("bpe", 2)
    assert [item.url for item in found] == ["https://example.net/bpe"]
    assert body["query"] == "bpe"
    assert body["max_results"] == 2
    assert body["api_key"] == "tvly-test"


def test_a_failing_provider_is_a_search_error() -> None:
    client = client_for(lambda request: httpx.Response(500))
    with pytest.raises(search.SearchError):
        search.SearxngProvider("http://search.local", client).search("bpe", 3)


def test_the_provider_follows_the_policy(monkeypatch) -> None:
    monkeypatch.setattr(settings, "web_search_api_key", "")
    assert search.provider_for(trust.WebPolicy(enabled=False)) is None
    assert search.provider_for(trust.WebPolicy(enabled=True, endpoint="")) is None
    assert search.provider_for(trust.WebPolicy(enabled=True, provider="brave")) is None
    named = trust.WebPolicy(enabled=True, provider="brave")
    assert isinstance(search.provider_for(named, api_key="key"), search.BraveProvider)
    endpoint = trust.WebPolicy(enabled=True, endpoint="http://search.local")
    assert isinstance(search.provider_for(endpoint), search.SearxngProvider)
