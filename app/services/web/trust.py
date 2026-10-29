"""Which pages the engine may read, and how much each one is trusted.

An allowlisted site is an official source: its pages may be indexed into the
unverified web collection and quoted with their address. A page found by a
search outside the allowlist is a general source: it is kept as a candidate
for the trainer and never answers a question on its own. The policy is stored
per workspace as JSON, the same way the model name is.
"""

from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlsplit

from app.services import brain_paths

OFFICIAL = "official"
GENERAL = "general"
TIERS = (OFFICIAL, GENERAL)

PROVIDERS = ("searxng", "brave", "tavily")
PROVIDER_NAMES = {
    "searxng": "SearXNG",
    "brave": "Brave Search",
    "tavily": "Tavily",
}
DEFAULT_PROVIDER = "searxng"
DEFAULT_TTL_HOURS = 24 * 7
DEFAULT_MIN_SCORE = 0.45
DEFAULT_RESULTS = 3
MIN_TTL_HOURS = 1
MAX_TTL_HOURS = 24 * 30
MAX_SOURCES = 50
MAX_RESULTS = 10

_DOMAIN = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+$")


@dataclass(frozen=True)
class WebPolicy:
    """The web settings of one workspace: on or off, and what it may read."""

    enabled: bool = False
    provider: str = DEFAULT_PROVIDER
    endpoint: str = ""
    allowlist: tuple[str, ...] = ()
    ttl_hours: int = DEFAULT_TTL_HOURS
    min_score: float = DEFAULT_MIN_SCORE
    results: int = DEFAULT_RESULTS

    def allows(self, url: str) -> bool:
        """Whether a page belongs to a site the workspace has allowlisted."""
        domain = domain_of(url)
        return bool(domain) and any(_same_site(domain, entry) for entry in self.allowlist)

    def tier_for(self, url: str) -> str:
        """The trust a page gets: official when allowlisted, general otherwise."""
        return OFFICIAL if self.allows(url) else GENERAL


def domain_of(url: str) -> str:
    """The lower-case host of a URL, or an empty string when it has none."""
    try:
        host = urlsplit(url.strip()).hostname
    except ValueError:
        return ""
    return (host or "").rstrip(".").lower()


def _same_site(host: str, entry: str) -> bool:
    """Whether a host is the allowlisted site or one of its subdomains."""
    return host == entry or host.endswith(f".{entry}")


def clean_domain(value: str) -> str:
    """An allowlist entry as a plain lower-case domain, or an empty string.

    A full URL is accepted and reduced to its host, so a trainer can paste the
    address of a documentation page. Internationalised names are refused
    rather than silently compared in two different forms.
    """
    text = value.strip()
    if not text:
        return ""
    candidate = (urlsplit(text).hostname or "") if "://" in text else text
    candidate = candidate.strip().strip(".").lower()
    if _is_address(candidate):
        return ""
    return candidate if _DOMAIN.match(candidate) else ""


def _is_address(value: str) -> bool:
    """A bare address is not an allowlist entry; sites are named by domain."""
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def clean_allowlist(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """The usable domains of a list, without repeats, in the order given."""
    cleaned: list[str] = []
    for value in values:
        domain = clean_domain(str(value))
        if domain and domain not in cleaned:
            cleaned.append(domain)
        if len(cleaned) == MAX_SOURCES:
            break
    return tuple(cleaned)


def policy_path(tenant_id: int | None) -> Path:
    """Where one workspace's web policy is kept."""
    return brain_paths.brain_dir() / f"web-{brain_paths.tenant_key(tenant_id)}.json"


def read(tenant_id: int | None) -> WebPolicy:
    """The workspace policy, or the defaults when none was saved."""
    try:
        stored = json.loads(policy_path(tenant_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return WebPolicy()
    if not isinstance(stored, dict):
        return WebPolicy()
    return _policy_from(stored)


def _policy_from(stored: dict) -> WebPolicy:
    provider = str(stored.get("provider") or DEFAULT_PROVIDER)
    allowlist = stored.get("allowlist")
    try:
        endpoint = _clean_endpoint(str(stored.get("endpoint") or ""))
    except ValueError:
        endpoint = ""
    return WebPolicy(
        enabled=bool(stored.get("enabled")),
        provider=provider if provider in PROVIDERS else DEFAULT_PROVIDER,
        endpoint=endpoint,
        allowlist=clean_allowlist(allowlist if isinstance(allowlist, list) else ()),
        ttl_hours=_bounded_int(stored.get("ttl_hours"), DEFAULT_TTL_HOURS, 1, MAX_TTL_HOURS),
        min_score=_bounded_score(stored.get("min_score"), DEFAULT_MIN_SCORE),
        results=_bounded_int(stored.get("results"), DEFAULT_RESULTS, 1, MAX_RESULTS),
    )


def _clean_endpoint(value: str) -> str:
    text = value.strip().rstrip("/")
    if not text:
        return ""
    parts = urlsplit(text)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("The search address must start with http:// or https://.")
    return text


def _bounded_int(value: object, default: int, low: int, high: int) -> int:
    if not isinstance(value, (int, float, str)):
        return default
    try:
        number = int(value)
    except ValueError:
        return default
    return min(high, max(low, number))


def _bounded_score(value: object, default: float) -> float:
    if not isinstance(value, (int, float, str)):
        return default
    try:
        score = float(value)
    except ValueError:
        return default
    return min(1.0, max(0.0, score))


def save(tenant_id: int | None, policy: WebPolicy) -> WebPolicy:
    """Keep a workspace policy, after checking what can be checked on it."""
    if policy.provider not in PROVIDERS:
        raise ValueError("Choose SearXNG, Brave Search or Tavily as the search source.")
    if policy.provider == "searxng" and policy.enabled and not _clean_endpoint(policy.endpoint):
        raise ValueError("SearXNG needs the address of the instance to search.")
    checked = replace(
        policy,
        endpoint=_clean_endpoint(policy.endpoint),
        allowlist=clean_allowlist(policy.allowlist),
        ttl_hours=min(MAX_TTL_HOURS, max(MIN_TTL_HOURS, int(policy.ttl_hours))),
        min_score=min(1.0, max(0.0, float(policy.min_score))),
        results=min(MAX_RESULTS, max(1, int(policy.results))),
    )
    path = policy_path(tenant_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "enabled": checked.enabled,
                "provider": checked.provider,
                "endpoint": checked.endpoint,
                "allowlist": list(checked.allowlist),
                "ttl_hours": checked.ttl_hours,
                "min_score": checked.min_score,
                "results": checked.results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
    return checked


def view(tenant_id: int | None) -> dict:
    """The policy as the settings page reads it. The search key is never sent."""
    policy = read(tenant_id)
    return {
        "enabled": policy.enabled,
        "provider": policy.provider,
        "provider_name": PROVIDER_NAMES[policy.provider],
        "endpoint": policy.endpoint,
        "allowlist": list(policy.allowlist),
        "ttl_hours": policy.ttl_hours,
        "min_score": policy.min_score,
        "results": policy.results,
        "providers": [{"id": key, "name": PROVIDER_NAMES[key]} for key in PROVIDERS],
        "max_sources": MAX_SOURCES,
    }
