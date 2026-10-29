"""The text of an upload after the furniture and the personal details are out.

Every upload goes through here before it is chunked, so the chunker, the hash
and the vector all see the same text. The page furniture is always removed;
the personal details are masked only when the trainer turns that on.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.config import settings
from app.core.extraction import ExtractedText, PageText
from app.services.quality.boilerplate import join_pages, without_boilerplate
from app.services.quality.scrub import scrub


def prepare_document(extracted: ExtractedText) -> ExtractedText:
    """Remove the page furniture, and mask personal details when enabled."""
    clean = without_boilerplate(extracted)
    if not settings.pii_scrub_enabled:
        return clean
    return _masked(clean)


def prepare_text(text: str) -> str:
    """The text of a pasted source, with personal details masked when enabled."""
    return scrub(text) if settings.pii_scrub_enabled else text


def _masked(extracted: ExtractedText) -> ExtractedText:
    """The document with every page's personal details masked."""
    pages: Sequence[PageText] = extracted.pages
    texts = [scrub(page.text) for page in pages]
    if texts == [page.text for page in pages]:
        return extracted
    return join_pages(pages, texts)
