"""Mask the personal details that should not enter the index.

The scrub is optional and off by default. When the trainer turns it on, an
email, a phone number, a social-security number and a long card-like number
are replaced with a fixed marker, so the passage still reads and the index
does not keep the detail.
"""

from __future__ import annotations

import re

EMAIL = "[email]"
PHONE = "[phone]"
IDENTIFIER = "[id]"

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_SOCIAL_SECURITY = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PHONE = re.compile(r"(?<![\w.])(?:\+?\d[\d ()-]{7,}\d)(?![\w.])")


def scrub(text: str) -> str:
    """The text with emails, phone numbers and long identifiers masked."""
    text = _EMAIL.sub(EMAIL, text)
    text = _SOCIAL_SECURITY.sub(IDENTIFIER, text)
    text = _CARD.sub(IDENTIFIER, text)
    return _PHONE.sub(PHONE, text)
