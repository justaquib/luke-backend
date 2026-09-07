import re
from typing import Optional
import httpx


WEATHER_KEYWORDS = (
    "weather", "temperature", "forecast", "rain", "snow",
    "humidity", "wind", "sunny", "cloudy", "storm",
)

WIKI_KEYWORDS = (
    "who is", "who was", "what is", "what was", "what are",
    "tell me about", "define", "definition of",
)


def detect_intents(question: str) -> list[str]:
    q = question.lower().strip()
    intents: list[str] = []
    if any(kw in q for kw in WEATHER_KEYWORDS):
        intents.append("weather")
    if any(q.startswith(kw) or f" {kw} " in f" {q} " for kw in WIKI_KEYWORDS):
        if "weather" not in intents or "temperature" not in q:
            intents.append("wiki")
    return intents


def _extract_city(question: str) -> Optional[str]:
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


def weather_for(question: str, lat: Optional[float] = None, lon: Optional[float] = None, timeout: float = 6.0) -> Optional[dict]:
    params: dict = {}
    if lat is not None and lon is not None:
        params = {"latitude": lat, "longitude": lon, "current": "temperature_2m,wind_speed_10m,relative_humidity_2m,weather_code"}
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
        except Exception:
            return None

    try:
        with httpx.Client(timeout=timeout) as client:
            res = client.get("https://api.open-meteo.com/v1/forecast", params=params)
            res.raise_for_status()
            data = res.json()
    except Exception:
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


def _format_weather(cur: dict, code) -> str:
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


def _extract_topic(question: str) -> Optional[str]:
    patterns = [
        r"who (?:is|was) (.+?)(?:\?|$|\.)",
        r"what (?:is|was|are) (.+?)(?:\?|$|\.)",
        r"tell me about (.+?)(?:\?|$|\.)",
        r"define (.+?)(?:\?|$|\.)",
        r"definition of (.+?)(?:\?|$|\.)",
    ]
    for pat in patterns:
        m = re.search(pat, question, flags=re.IGNORECASE)
        if m:
            t = m.group(1).strip().strip("?,.")
            if 1 < len(t) < 80:
                return t
    return None


def wiki_summary(question: str, timeout: float = 6.0) -> Optional[dict]:
    topic = _extract_topic(question)
    if not topic:
        return None
    title = topic.replace(" ", "_")
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "AssistanceBot/0.1 (https://example.local)"},
        ) as client:
            res = client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}")
            if res.status_code == 404:
                res = client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "opensearch",
                        "search": topic,
                        "limit": 1,
                        "namespace": 0,
                        "format": "json",
                    },
                )
                res.raise_for_status()
                data = res.json()
                if len(data) >= 2 and data[1]:
                    new_title = data[1][0].replace(" ", "_")
                    res = client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{new_title}")
            res.raise_for_status()
            data = res.json()
    except Exception:
        return None
    extract = data.get("extract")
    if not extract:
        return None
    page_url = (
        data.get("content_urls", {}).get("desktop", {}).get("page")
        or f"https://en.wikipedia.org/wiki/{title}"
    )
    title_text = data.get("title") or topic
    return {
        "source": "wiki",
        "answer": extract,
        "payload": {
            "title": title_text,
            "url": page_url,
            "topic": topic,
        },
    }