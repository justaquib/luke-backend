"""Wikipedia tool: fetch a summary for a topic.

Wraps the existing wiki_summary function from apps.py and registers it as a
tool. No confirmation needed since it's a read-only API call.
"""

from __future__ import annotations

from typing import Any

from app.services import apps
from app.services.tools.registry import Tool, register


def get_wiki_summary(question: str) -> dict[str, Any]:
    """Fetch a Wikipedia summary for the topic in the question."""
    result = apps.wiki_summary(question)
    if not result:
        raise ValueError("No Wikipedia summary found for the given topic.")
    return result


register(
    Tool(
        name="wiki",
        description="Fetch a Wikipedia summary for a topic.",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question (e.g., 'who is Albert Einstein').",
                }
            },
            "required": ["question"],
        },
        requires_confirmation=False,
        handler=get_wiki_summary,
    )
)
