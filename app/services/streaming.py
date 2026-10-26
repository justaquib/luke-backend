"""Server-sent events for the answer streams.

Every stage the page draws is one frame the backend produced. A failure
becomes a single error frame, and the stream never continues after it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator

logger = logging.getLogger(__name__)

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}
SAFE_FAILURE = "The answer stopped before it was finished. Please ask again."


def frame(event: str, data: dict) -> str:
    """One server-sent event frame."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def frames(events: Iterable[tuple[str, dict]]) -> Iterator[str]:
    """The frames of an answer stream, with one safe error on failure."""
    try:
        for event, data in events:
            yield frame(event, data)
    except Exception:
        logger.exception("An answer stream failed")
        yield frame("error", {"message": SAFE_FAILURE})
