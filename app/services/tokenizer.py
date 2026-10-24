"""The words of a text, as the language services compare and count them.

The neural models tokenize with ``neural.tokens`` as well: their vocabulary
and their loss are defined over numbers, commas and line breaks. The language
services only need the words themselves, so they share this smaller reader.
"""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?|[.!?]")


def tokenize(text: str) -> list[str]:
    """The words of a text, in lower case, with the marks that end a sentence."""
    return [match.group(0).lower().replace("’", "'") for match in TOKEN_RE.finditer(text or "")]
