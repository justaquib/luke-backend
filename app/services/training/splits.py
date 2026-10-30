"""Which sources the model learns from and which it is measured on.

The split is taken by source, never by sentence: every passage cut out of one
document lands on the same side, so a held-out document is one the model has
never read a word of. The same salt always picks the same sources, so a source
cannot drift into the training side when the corpus grows or a file is
re-uploaded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

TRAIN = "train"
HELD_OUT = "held_out"
SPLITS = (TRAIN, HELD_OUT)

# The share of sources held out. A fifth tells a remembered answer from a real
# one on a corpus of a few dozen documents and still leaves the rest to learn
# from; it is a parameter, not a promise about the final corpus.
DEFAULT_SHARE = 0.2

# A split needs two sides. One source held out leaves nothing to train on, so
# below this the split is simply not taken and the dataset says so.
MIN_SOURCES_TO_SPLIT = 2

# Changing the salt re-draws the split, so every measured run records it.
DEFAULT_SALT = "luke-training-v1"


def rank(salt: str, source: str) -> str:
    """The stable digest a source is placed by."""
    return hashlib.sha256(f"{salt}:{source}".encode()).hexdigest()


def _bucket(digest: str) -> float:
    """Where a source sits between 0 and 1, read from the front of its digest."""
    return int(digest[:8], 16) / 0x1_0000_0000


def assign(
    sources: Iterable[str],
    salt: str = DEFAULT_SALT,
    share: float = DEFAULT_SHARE,
) -> dict[str, str]:
    """The split tag of each source.

    A source is placed by its own digest, so the tags of the sources already in
    the corpus do not move when a document is added or re-uploaded: a held-out
    source cannot drift into the training side as the corpus grows. Two sides
    are always kept once there are two sources to split, and with fewer the
    split is simply not taken. A share outside ``(0, 1)`` asks for a side that
    cannot exist, so it is refused rather than quietly clamped.
    """
    if not 0.0 < share < 1.0:
        raise ValueError(f"share must be between 0 and 1, got {share}")
    ordered = sorted(set(sources), key=lambda source: rank(salt, source))
    if len(ordered) < MIN_SOURCES_TO_SPLIT:
        return dict.fromkeys(ordered, TRAIN)
    held = {source for source in ordered if _bucket(rank(salt, source)) < share}
    if not held:
        held = {ordered[0]}
    if len(held) == len(ordered):
        held = set(ordered[:-1])
    return {source: (HELD_OUT if source in held else TRAIN) for source in ordered}


def is_split_taken(assignment: dict[str, str]) -> bool:
    """Whether the split gave the model one side and the measurement the other."""
    return any(tag == HELD_OUT for tag in assignment.values())


def summary(assignment: dict[str, str]) -> dict[str, int]:
    """How many sources sit on each side."""
    return {
        "sources": len(assignment),
        "train_sources": sum(1 for tag in assignment.values() if tag == TRAIN),
        "held_out_sources": sum(1 for tag in assignment.values() if tag == HELD_OUT),
    }
