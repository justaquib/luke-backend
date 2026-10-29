"""A 64-bit fingerprint of a passage, so a near repeat can be recognised.

Every word is hashed, and a bit is set when more than half of the words agree
it is set. Two passages with a few words changed share most of their bits, so
the Hamming distance between their fingerprints is small. Words are used one
by one, not in shingles: an added word must move one feature, not every
window after it.
"""

from __future__ import annotations

import hashlib
import re

SIMHASH_BITS = 64
_WORD = re.compile(r"[a-z0-9']+")


def _features(text: str) -> list[str]:
    """The words of a text, in lower case."""
    return _WORD.findall(text.casefold())


def simhash(text: str) -> int:
    """The 64-bit SimHash of a text, as an unsigned integer."""
    features = _features(text)
    if not features:
        return 0
    weights = [0] * SIMHASH_BITS
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(SIMHASH_BITS):
            weights[bit] += 1 if value >> bit & 1 else -1
    return sum(1 << bit for bit, weight in enumerate(weights) if weight > 0)


def to_signed(value: int) -> int:
    """The same 64-bit value as a signed integer, the range a BIGINT holds."""
    unsigned = to_unsigned(value)
    if unsigned >= 1 << (SIMHASH_BITS - 1):
        return unsigned - (1 << SIMHASH_BITS)
    return unsigned


def to_unsigned(value: int) -> int:
    """The unsigned 64-bit value of a fingerprint read from the database."""
    return value + (1 << SIMHASH_BITS) if value < 0 else value


def hamming(first: int, second: int) -> int:
    """How many of the 64 bits two fingerprints differ in."""
    return (to_unsigned(first) ^ to_unsigned(second)).bit_count()
