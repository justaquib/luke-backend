"""The lines a page carries on every page, and the page numbers.

A header, a footer, a copyright line and a "Page 3 of 10" are on the page but
are not what the page says. A short line that repeats on at least two pages,
with its digits ignored, is furniture; so is a page number in a document that
has numbered pages. A text without pages keeps every line, because the same
line twice in one run is not evidence of furniture.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from app.core.extraction import ExtractedText, PageText

# A footer is short. A sentence that repeats on two pages is the document
# saying the same thing, not furniture.
MAX_FURNITURE_WORDS = 12

_DIGITS = re.compile(r"\d+")
_SPACE = re.compile(r"\s+")
_PAGE_NUMBER = re.compile(
    r"^(?:page\s+\d+(?:\s*(?:of|/)\s*\d+)?"
    r"|\d+\s*(?:of|/)\s*\d+"
    r"|[-–—]\s*\d+\s*[-–—])$",
    re.IGNORECASE,
)
_COPYRIGHT = re.compile(
    r"^(?:©|\(c\)\s|copyright\b|all rights reserved\b|confidential\b|internal use only\b)",
    re.IGNORECASE,
)


def _key(line: str) -> str:
    """A line's identity with its digits ignored, so a page number still matches."""
    return _DIGITS.sub("#", _SPACE.sub(" ", line).strip().casefold())


def _repeated(pages: Sequence[PageText]) -> set[str]:
    """The short lines that appear on more than one page."""
    seen: Counter[str] = Counter()
    for page in pages:
        seen.update({_key(line) for line in page.text.splitlines() if line.strip()})
    return {
        key for key, count in seen.items() if count >= 2 and len(key.split()) <= MAX_FURNITURE_WORDS
    }


def _is_furniture(line: str, repeated: set[str], page: PageText) -> bool:
    """Whether a line is furniture, given the lines that repeat across pages."""
    stripped = line.strip()
    if _COPYRIGHT.match(stripped):
        return True
    if page.page is not None and (_PAGE_NUMBER.match(stripped) or stripped == str(page.page)):
        return True
    return _key(line) in repeated


def without_boilerplate(extracted: ExtractedText) -> ExtractedText:
    """The document with the lines that are furniture removed."""
    pages = list(extracted.pages)
    if not pages:
        return extracted
    repeated = _repeated(pages) if len(pages) > 1 else set()
    texts = [
        "\n".join(
            line for line in page.text.splitlines() if not _is_furniture(line, repeated, page)
        )
        for page in pages
    ]
    if texts == [page.text for page in pages]:
        return extracted
    return join_pages(pages, texts)


def join_pages(pages: Sequence[PageText], texts: Sequence[str]) -> ExtractedText:
    """Put page texts back together, with the character spans they now cover."""
    rebuilt: list[PageText] = []
    offset = 0
    for page, text in zip(pages, texts, strict=True):
        if rebuilt:
            offset += 1
        rebuilt.append(PageText(page.page, text, offset, offset + len(text)))
        offset += len(text)
    return ExtractedText(text="\n".join(texts), pages=tuple(rebuilt))
