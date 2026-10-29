"""Read one public page, and refuse anything that is not one.

A page is read only when its address is a plain http(s) URL to a public
server, every redirect on the way stays public, the site's robots.txt allows
it, the answer is a readable text type, and the body stays inside the size
limit. The readable text is extracted with trafilatura; raw markup is never
kept.
"""

from __future__ import annotations

import html as html_text
import ipaddress
import logging
import re
import socket
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urljoin, urlsplit

import httpx
import trafilatura

from app.config import settings

logger = logging.getLogger(__name__)

USER_AGENT = "LukeLab/0.1 (local knowledge assistant)"
ALLOWED_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
REDIRECT_CODES = (301, 302, 303, 307, 308)
ROBOTS_MAX_BYTES = 64 * 1024
MIN_INTERVAL_SECONDS = 0.5
BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_CHARSET = re.compile(r"charset\s*=\s*[\"']?([\w-]+)", re.IGNORECASE)

_LAST_FETCH: dict[str, float] = {}
_ROBOTS: dict[str, RobotsRules] = {}
_CLOCK = time.time
_SLEEP = time.sleep


class FetchError(ValueError):
    """A page that may not be read, or could not be read safely."""


@dataclass(frozen=True)
class FetchedPage:
    """One page as it was read: where it came from and what it said."""

    url: str
    final_url: str
    title: str
    text: str
    size: int
    fetched_at: datetime


def validate_url(url: str) -> str:
    """Refuse an address that is not a plain public web address."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError as error:
        raise FetchError("The address could not be read.") from error
    if parts.scheme not in ("http", "https"):
        raise FetchError("Only http and https pages can be read.")
    host = (parts.hostname or "").lower()
    if not host:
        raise FetchError("The address has no site name.")
    if parts.username or parts.password:
        raise FetchError("An address that carries a login is not read.")
    if port not in (None, 80, 443):
        raise FetchError("Only the standard web ports are read.")
    if host == "localhost" or host.endswith(BLOCKED_SUFFIXES):
        raise FetchError("Local addresses are not read.")
    return url.strip()


def _public_ip(address: str) -> bool:
    try:
        parsed = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped is not None:
        parsed = parsed.ipv4_mapped
    return parsed.is_global


def _addresses(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as error:
        raise FetchError(f"The site {host} could not be looked up.") from error
    addresses = sorted({str(info[4][0]) for info in infos})
    if not addresses:
        raise FetchError(f"The site {host} could not be looked up.")
    return addresses


def ensure_public(url: str) -> str:
    """The address, once it is known to point at a public server right now."""
    checked = validate_url(url)
    parts = urlsplit(checked)
    host = (parts.hostname or "").lower()
    port = parts.port or (443 if parts.scheme == "https" else 80)
    refused = [address for address in _addresses(host, port) if not _public_ip(address)]
    if refused:
        raise FetchError("The address points at a private network and is not read.")
    return checked


@dataclass(frozen=True)
class RobotsRules:
    """The allow and disallow rules that apply to this reader."""

    rules: tuple[tuple[str, str], ...] = ()

    def allows(self, path: str) -> bool:
        """Whether the longest matching rule allows the path; allow wins a tie."""
        target = path or "/"
        best = -1
        allowed = True
        for kind, rule in self.rules:
            if not rule or not target.startswith(rule):
                continue
            if len(rule) > best or (len(rule) == best and kind == "allow"):
                best = len(rule)
                allowed = kind == "allow"
        return allowed


def parse_robots(text: str, agent: str = "lukelab") -> RobotsRules:
    """Read robots.txt into the rules of this reader and the wildcard group."""
    groups: list[tuple[list[str], list[tuple[str, str]]]] = []
    agents: list[str] = []
    rules: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "user-agent":
            if agents and rules:
                groups.append((agents, rules))
                agents, rules = [], []
            agents.append(value.lower())
        elif key in ("allow", "disallow") and agents:
            rules.append((key, value))
    if agents and rules:
        groups.append((agents, rules))
    chosen = [rule for named, rows in groups if agent in named for rule in rows]
    if not chosen:
        chosen = [rule for named, rows in groups if "*" in named for rule in rows]
    return RobotsRules(tuple(chosen))


def _robots(client: httpx.Client, scheme: str, netloc: str) -> RobotsRules:
    """The site's robots rules, fetched once per process and kept."""
    origin = f"{scheme}://{netloc}"
    cached = _ROBOTS.get(origin)
    if cached is not None:
        return cached
    rules = RobotsRules((("disallow", "/"),))
    try:
        response = client.get(
            f"{origin}/robots.txt",
            headers={"User-Agent": USER_AGENT},
            follow_redirects=False,
            timeout=settings.web_timeout_seconds,
        )
    except httpx.HTTPError as error:
        logger.warning("robots.txt of %s could not be read: %s", origin, error)
    else:
        if response.status_code < 400:
            rules = parse_robots(response.text[:ROBOTS_MAX_BYTES])
        elif response.status_code < 500:
            rules = RobotsRules()
    _ROBOTS[origin] = rules
    return rules


def _wait(host: str, now: float) -> None:
    """Keep two reads of one site apart, so no site is hammered."""
    last = _LAST_FETCH.get(host)
    if last is not None:
        gap = MIN_INTERVAL_SECONDS - (now - last)
        if gap > 0:
            _SLEEP(gap)
            now += gap
    _LAST_FETCH[host] = now


def _content_type(headers: httpx.Headers) -> str:
    return headers.get("content-type", "").split(";", 1)[0].strip().lower()


def _check_type(kind: str) -> None:
    if kind not in ALLOWED_TYPES:
        raise FetchError(f"A {kind or 'unknown'} file is not read as a page.")


def _read(client: httpx.Client, url: str) -> tuple[int, httpx.Headers, bytes]:
    """One request, reading no more than the size limit allows."""
    headers = {"User-Agent": USER_AGENT, "Accept": ", ".join(ALLOWED_TYPES)}
    with client.stream(
        "GET",
        url,
        headers=headers,
        follow_redirects=False,
        timeout=settings.web_timeout_seconds,
    ) as response:
        status = response.status_code
        if status >= 300:
            return status, response.headers, b""
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) > settings.web_max_bytes:
                raise FetchError(
                    f"The page is larger than {settings.web_max_bytes} bytes and was not read."
                )
        return status, response.headers, bytes(body)


def _follow(client: httpx.Client, url: str, now: float) -> tuple[str, int, httpx.Headers, bytes]:
    """Read the address, walking a capped number of public redirects."""
    current = ensure_public(url)
    for _ in range(settings.web_max_redirects + 1):
        _wait((urlsplit(current).hostname or "").lower(), now)
        status, headers, body = _read(client, current)
        if status not in REDIRECT_CODES:
            return current, status, headers, body
        target = headers.get("location")
        if not target:
            raise FetchError("The page redirected without an address.")
        current = ensure_public(urljoin(current, target))
    raise FetchError("The page redirected too many times.")


def _charset(headers: httpx.Headers) -> str:
    found = _CHARSET.search(headers.get("content-type", ""))
    return found.group(1) if found else "utf-8"


def _tidy(text: str) -> str:
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _page_title(markup: str) -> str:
    found = _TITLE.search(markup)
    if not found:
        return ""
    return " ".join(html_text.unescape(found.group(1)).split())[:200]


def extract(markup: str, url: str, content_type: str) -> tuple[str, str]:
    """The readable text and title of a page, or a refusal when it has none."""
    if content_type == "text/plain":
        text = _tidy(markup)
        return text, url
    text = trafilatura.extract(
        markup,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )
    if not text:
        text = trafilatura.baseline(markup)
    cleaned = _tidy(text or "")
    if not cleaned:
        raise FetchError("The page had no readable text.")
    return cleaned, _page_title(markup)


def fetch_page(
    url: str, client: httpx.Client | None = None, now: float | None = None
) -> FetchedPage:
    """Read one page, or raise ``FetchError`` with the reason it was refused."""
    moment = _CLOCK() if now is None else now
    owned = client is None
    chosen = client if client is not None else httpx.Client()
    try:
        parts = urlsplit(ensure_public(url))
        if not _robots(chosen, parts.scheme, parts.netloc).allows(parts.path):
            raise FetchError("The site's robots.txt does not allow this page to be read.")
        final_url, status, headers, body = _follow(chosen, url, moment)
    except httpx.HTTPError as error:
        logger.warning("The page %s could not be read: %s", url, error)
        raise FetchError("The page could not be reached.") from error
    finally:
        if owned:
            chosen.close()
    if status >= 400:
        raise FetchError(f"The page answered with HTTP {status}.")
    kind = _content_type(headers)
    _check_type(kind)
    text, title = extract(body.decode(_charset(headers), errors="replace"), final_url, kind)
    fetched_at = datetime.fromtimestamp(moment, UTC).replace(tzinfo=None)
    return FetchedPage(
        url=url,
        final_url=final_url,
        title=title,
        text=text,
        size=len(body),
        fetched_at=fetched_at,
    )


def reset_caches() -> None:
    """Forget the robots and rate-limit state, so a test starts clean."""
    _ROBOTS.clear()
    _LAST_FETCH.clear()
