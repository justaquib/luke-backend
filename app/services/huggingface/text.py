"""The material a fine-tuned teacher is given, and the limits it is read within.

The shape a teacher is trained and asked in lives in
``app.services.training.formats``; what is left here is the reference line the
retriever hands a teacher, kept to the size the model can read.
"""

from __future__ import annotations

from collections.abc import Sequence

MAX_REFERENCE = 1200


def _clean(text: str, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


def grounding(answer: str, passages: Sequence[str] = ()) -> str:
    """The material an answer is grounded on, as one reference line."""
    parts = [_clean(answer, MAX_REFERENCE)]
    for passage in passages:
        clean = _clean(passage, 400)
        if clean and clean.casefold() not in {part.casefold() for part in parts}:
            parts.append(clean)
    return " ".join(parts)[:MAX_REFERENCE]
