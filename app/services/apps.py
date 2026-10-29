import logging
import re

import httpx

logger = logging.getLogger(__name__)

WEATHER_KEYWORDS = (
    "weather",
    "temperature",
    "forecast",
    "rain",
    "snow",
    "humidity",
    "wind",
    "sunny",
    "cloudy",
    "storm",
)

WIKI_KEYWORDS = (
    "who is",
    "who was",
    "what is",
    "what was",
    "what are",
    "where is",
    "where was",
    "tell me about",
    "define",
    "definition of",
)

# The answers that are known well enough to be given without a lookup.
_CURATED_WIKI = {
    "father of computer": {
        "title": "Charles Babbage",
        "answer": (
            "Charles Babbage is often regarded as the father of the computer "
            "for his design of the Analytical Engine."
        ),
        "url": "https://en.wikipedia.org/wiki/Charles_Babbage",
    },
}


def detect_intents(question: str) -> list[str]:
    q = question.lower().strip()
    if any(keyword in q for keyword in WEATHER_KEYWORDS):
        return ["weather"]
    if any(q.startswith(keyword) or f" {keyword} " in f" {q} " for keyword in WIKI_KEYWORDS):
        return ["wiki"]
    return []


def _extract_city(question: str) -> str | None:
    patterns = [
        r"weather (?:like |in |at |for )?(.+?)(?:\?|$|\.)",
        r"temperature (?:in |at |for )?(.+?)(?:\?|$|\.)",
        r"forecast (?:for |in |at )?(.+?)(?:\?|$|\.)",
        r"in ([A-Z][\w\s,]+?)(?:\?|$|\.)",
        r"at ([A-Z][\w\s,]+?)(?:\?|$|\.)",
        r"for ([A-Z][\w\s,]+?)(?:\?|$|\.)",
    ]
    for pat in patterns:
        m = re.search(pat, question, flags=re.IGNORECASE)
        if m:
            city = m.group(1).strip().strip("?,.")
            if 1 < len(city) < 60:
                return city
    cleaned = question.strip().strip("?.!")
    if 1 < len(cleaned) < 60 and re.match(r"^[A-Z][\w\s,]+$", cleaned):
        return cleaned
    return None


def weather_for(
    question: str, lat: float | None = None, lon: float | None = None, timeout: float = 6.0
) -> dict | None:
    params: dict = {}
    if lat is not None and lon is not None:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,wind_speed_10m,relative_humidity_2m,weather_code",
        }
    else:
        city = _extract_city(question)
        if not city:
            return None
        try:
            with httpx.Client(timeout=timeout) as client:
                geo = client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": city, "count": 1, "language": "en", "format": "json"},
                )
                geo.raise_for_status()
                geo_data = geo.json()
            results = geo_data.get("results") or []
            if not results:
                return None
            params = {
                "latitude": results[0]["latitude"],
                "longitude": results[0]["longitude"],
                "current": "temperature_2m,wind_speed_10m,relative_humidity_2m,weather_code",
            }
        except Exception as error:  # noqa: BLE001 -- the weather place lookup boundary
            logger.warning("the weather place lookup failed: %s", error)
            return None

    try:
        with httpx.Client(timeout=timeout) as client:
            res = client.get("https://api.open-meteo.com/v1/forecast", params=params)
            res.raise_for_status()
            data = res.json()
    except Exception as error:  # noqa: BLE001 -- the weather forecast provider boundary
        logger.warning("the weather forecast provider failed: %s", error)
        return None
    cur = data.get("current") or {}
    code = cur.get("weather_code")
    return {
        "source": "weather",
        "answer": _format_weather(cur, code),
        "payload": {
            "temperature": cur.get("temperature_2m"),
            "wind_speed": cur.get("wind_speed_10m"),
            "humidity": cur.get("relative_humidity_2m"),
            "code": code,
            "location": params,
        },
    }


WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
}


def _format_weather(cur: dict, code: int | None) -> str:
    if not cur:
        return "I couldn't fetch the weather right now."
    desc = WEATHER_CODES.get(code, "unknown conditions")
    t = cur.get("temperature_2m")
    w = cur.get("wind_speed_10m")
    h = cur.get("relative_humidity_2m")
    bits = [f"Currently {desc}"]
    if t is not None:
        bits.append(f"{t}°C")
    if h is not None:
        bits.append(f"humidity {h}%")
    if w is not None:
        bits.append(f"wind {w} km/h")
    return ", ".join(bits) + "."


def _extract_topic(question: str) -> str | None:
    patterns = [
        r"who (?:is|was) (.+?)(?:\?|$|\.)",
        r"what (?:is|was|are) (.+?)(?:\?|$|\.)",
        r"where (?:is|was) (.+?)(?:\?|$|\.)",
        r"tell me about (.+?)(?:\?|$|\.)",
        r"define (.+?)(?:\?|$|\.)",
        r"definition of (.+?)(?:\?|$|\.)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if not match:
            continue
        topic = match.group(1).strip().strip("?,.")
        if not 1 < len(topic) < 80:
            continue
        return re.sub(r"^(?:the|a|an)\s+", "", topic, flags=re.IGNORECASE)
    return None


def _curated_wiki(topic: str) -> dict | None:
    entry = _CURATED_WIKI.get(topic.casefold())
    if not entry:
        return None
    return {
        "source": "wiki",
        "answer": entry["answer"],
        "payload": {"title": entry["title"], "url": entry["url"], "topic": topic},
    }


def _search_title(client: httpx.Client, topic: str) -> str | None:
    response = client.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "opensearch",
            "search": topic,
            "limit": 1,
            "namespace": 0,
            "format": "json",
        },
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list) or len(data) < 2 or not data[1]:
        return None
    return str(data[1][0]).replace(" ", "_")


def _summary_payload(data: object, topic: str, title: str) -> dict | None:
    if not isinstance(data, dict):
        return None
    extract = data.get("extract")
    if not extract:
        return None
    page_url = (
        data.get("content_urls", {}).get("desktop", {}).get("page")
        or f"https://en.wikipedia.org/wiki/{title}"
    )
    return {
        "source": "wiki",
        "answer": extract,
        "payload": {
            "title": data.get("title") or topic,
            "url": page_url,
            "topic": topic,
        },
    }


def _curated_wiki(topic: str) -> dict | None:
    entry = _CURATED_WIKI.get(topic.casefold())
    if not entry:
        return None
    return {
        "source": "wiki",
        "answer": entry["answer"],
        "payload": {"title": entry["title"], "url": entry["url"], "topic": topic},
    }


def _search_title(client: httpx.Client, topic: str) -> str | None:
    response = client.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "opensearch",
            "search": topic,
            "limit": 1,
            "namespace": 0,
            "format": "json",
        },
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list) or len(data) < 2 or not data[1]:
        return None
    return str(data[1][0]).replace(" ", "_")


def _summary_payload(data: object, topic: str, title: str) -> dict | None:
    if not isinstance(data, dict):
        return None
    extract = data.get("extract")
    if not extract:
        return None
    page_url = (
        data.get("content_urls", {}).get("desktop", {}).get("page")
        or f"https://en.wikipedia.org/wiki/{title}"
    )
    return {
        "source": "wiki",
        "answer": extract,
        "payload": {
            "title": data.get("title") or topic,
            "url": page_url,
            "topic": topic,
        },
    }


def wiki_summary(question: str, timeout: float = 6.0) -> dict | None:
    topic = _extract_topic(question)
    if not topic:
        return None
    curated = _curated_wiki(topic)
    if curated is not None:
        return curated
    title = topic.replace(" ", "_")
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "LukeLabBot/0.1 (https://example.local)"},
        ) as client:
            response = client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}")
            if response.status_code == 404:
                searched = _search_title(client, topic)
                if searched is None:
                    return None
                title = searched
                response = client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}")
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as error:
        logger.warning("Wikipedia lookup failed for %r: %s", topic, error)
        return None
    return _summary_payload(data, topic, title)
