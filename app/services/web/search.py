"""Finding pages about a question, through one configured search provider.

SearXNG is the default because an instance can be self-hosted, so the query
never leaves the operator's own network. Brave Search and Tavily are offered
for an operator who already has a key; a key is read from the environment and
is never stored in the workspace settings or sent to the browser.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from app.config import settings
from app.services.web import trust

logger = logging.getLogger(__name__)

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
TAVILY_ENDPOINT = "https://api.tavily.com/search"


class SearchError(RuntimeError):
    """A search that could not be made, or whose answer could not be read."""


@dataclass(frozen=True)
class SearchResult:
    """One page a search found, and the words it was found with."""

    title: str
    url: str
    snippet: str = ""


class SearchProvider(Protocol):
    """One way of asking the web a question."""

    def search(self, query: str, limit: int = 3) -> list[SearchResult]:
        """The pages that answer the query, best first."""


def _usable(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.hostname)


def _clean(results: Any, fields: tuple[str, str, str], limit: int) -> list[SearchResult]:
    """Keep the results that have a public address, in the provider's order."""
    if not isinstance(results, list):
        return []
    found: list[SearchResult] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        url = str(item.get(fields[0]) or "")
        if not _usable(url):
            continue
        found.append(
            SearchResult(
                title=" ".join(str(item.get(fields[1]) or "").split())[:200],
                url=url,
                snippet=" ".join(str(item.get(fields[2]) or "").split())[:400],
            )
        )
        if len(found) == limit:
            break
    return found


def _client(client: httpx.Client | None) -> httpx.Client:
    return client if client is not None else httpx.Client(timeout=settings.web_timeout_seconds)


def _get_json(
    url: str,
    params: dict[str, Any],
    client: httpx.Client | None,
    headers: dict[str, str] | None = None,
) -> Any:
    owned = client is None
    chosen = _client(client)
    try:
        response = chosen.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise SearchError("The search provider could not be reached.") from error
    finally:
        if owned:
            chosen.close()


def _post_json(url: str, body: dict[str, Any], client: httpx.Client | None) -> Any:
    owned = client is None
    chosen = _client(client)
    try:
        response = chosen.post(url, json=body)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise SearchError("The search provider could not be reached.") from error
    finally:
        if owned:
            chosen.close()


class SearxngProvider:
    """A self-hosted SearXNG instance, asked for its JSON answer."""

    def __init__(self, endpoint: str, client: httpx.Client | None = None) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.client = client

    def search(self, query: str, limit: int = 3) -> list[SearchResult]:
        payload = _get_json(
            f"{self.endpoint}/search",
            {"q": query, "format": "json", "language": "en", "safesearch": "1"},
            self.client,
        )
        results = payload.get("results") if isinstance(payload, dict) else None
        return _clean(results, ("url", "title", "content"), limit)


class BraveProvider:
    """Brave's web search API, with the key kept in the header."""

    def __init__(self, api_key: str, client: httpx.Client | None = None) -> None:
        self.api_key = api_key
        self.client = client

    def search(self, query: str, limit: int = 3) -> list[SearchResult]:
        payload = _get_json(
            BRAVE_ENDPOINT,
            {"q": query, "count": limit, "safesearch": "moderate"},
            self.client,
            {"X-Subscription-Token": self.api_key, "Accept": "application/json"},
        )
        found = payload.get("web") if isinstance(payload, dict) else None
        results = found.get("results") if isinstance(found, dict) else None
        return _clean(results, ("url", "title", "description"), limit)


class TavilyProvider:
    """Tavily's search API, which also returns a readable snippet per page."""

    def __init__(self, api_key: str, client: httpx.Client | None = None) -> None:
        self.api_key = api_key
        self.client = client

    def search(self, query: str, limit: int = 3) -> list[SearchResult]:
        payload = _post_json(
            TAVILY_ENDPOINT,
            {
                "api_key": self.api_key,
                "query": query,
                "max_results": limit,
                "search_depth": "basic",
            },
            self.client,
        )
        results = payload.get("results") if isinstance(payload, dict) else None
        return _clean(results, ("url", "title", "content"), limit)


def provider_for(
    policy: trust.WebPolicy, client: httpx.Client | None = None, api_key: str | None = None
) -> SearchProvider | None:
    """The configured provider, or none when web evidence is not set up."""
    if not policy.enabled:
        return None
    key = settings.web_search_api_key if api_key is None else api_key
    if policy.provider == "searxng":
        return SearxngProvider(policy.endpoint, client) if policy.endpoint else None
    if policy.provider == "brave" and key:
        return BraveProvider(key, client)
    if policy.provider == "tavily" and key:
        return TavilyProvider(key, client)
    return None
