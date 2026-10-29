"""How much a passage is worth reading, and which passage it repeats.

The checks live beside each other: the shape of a passage, its language and
its repetition make one score; the furniture a page repeats is removed before
chunking; the fingerprints tell an exact repeat from a near one.
"""

from app.services.quality.boilerplate import join_pages, without_boilerplate
from app.services.quality.density import text_density
from app.services.quality.pipeline import prepare_document, prepare_text
from app.services.quality.score import passage_quality
from app.services.quality.scrub import scrub
from app.services.quality.simhash import hamming, simhash, to_signed, to_unsigned

__all__ = [
    "hamming",
    "join_pages",
    "passage_quality",
    "prepare_document",
    "prepare_text",
    "scrub",
    "simhash",
    "text_density",
    "to_signed",
    "to_unsigned",
    "without_boilerplate",
]
