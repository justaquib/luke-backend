"""Weather tool: fetch current weather for a city or coordinates.

Wraps the existing weather_for function from apps.py and registers it as a
tool. No confirmation needed since it's a read-only API call.
"""

from __future__ import annotations

from typing import Any

from app.services import apps
from app.services.tools.registry import Tool, register


def get_weather(
    question: str | None = None, lat: float | None = None, lon: float | None = None
) -> dict[str, Any]:
    """Fetch the current weather for the given location."""
    if question is None:
        question = "weather"
    result = apps.weather_for(question, lat=lat, lon=lon)
    if not result:
        raise ValueError("No weather data found for the given location.")
    return result


register(
    Tool(
        name="weather",
        description="Fetch current weather for a city or coordinates.",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The weather question (e.g., 'weather in Paris'). "
                        "Defaults to 'weather' if not provided."
                    ),
                },
                "lat": {
                    "type": "number",
                    "description": "Latitude (optional; extracted from question if not provided).",
                },
                "lon": {
                    "type": "number",
                    "description": "Longitude (optional; extracted from question if not provided).",
                },
            },
            "required": [],
        },
        requires_confirmation=False,
        handler=get_weather,
    )
)
