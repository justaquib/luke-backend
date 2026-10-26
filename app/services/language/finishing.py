"""The last touch on an answer: a capital letter and a full stop.

The words are the trainer's. Finishing changes nothing about them beyond the
two marks that make a text read as a sentence, so a taught answer and a
dictionary entry come out the same way on every page.
"""

from __future__ import annotations

import re

_FIRST_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def finish(text: str) -> str:
    """Trim, capitalize the first letter, and close a sentence that is open."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    first = _FIRST_LETTER.search(cleaned)
    if first is not None:
        index = first.start()
        cleaned = f"{cleaned[:index]}{cleaned[index].upper()}{cleaned[index + 1 :]}"
    if cleaned[-1].isalnum():
        cleaned = f"{cleaned}."
    return cleaned
