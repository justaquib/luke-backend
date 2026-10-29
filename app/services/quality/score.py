"""How much a passage is worth learning from, from 0 to 1.

A passage scores high when it is written in the language Luke serves, most of
its tokens are words, and it does not say the same phrase again. The three
checks multiply, so a passage has to pass all of them to score high; the
language check is the only one that keeps a share instead of zeroing, because
a short line may simply lack the common words the check looks for.
"""

from __future__ import annotations

from app.services.language.identify import guess
from app.services.neural.quality import MAX_REPETITION, MIN_WORDS, repetition
from app.services.neural.tokens import tokenize
from app.services.quality.density import text_density

# A passage whose language is confirmed keeps its whole score.
ENGLISH_FACTOR = 1.0

# A passage whose language cannot be confirmed keeps half of its shape score.
UNCONFIRMED_FACTOR = 0.5


def passage_quality(text: str) -> float:
    """The quality of a passage, from 0 to 1.

    A passage shorter than the shortest sentence worth learning scores 0, so
    a fragment of a sentence is never the best material in the corpus.
    """
    words = [token for token in tokenize(text) if token[:1].isalpha()]
    if len(words) < MIN_WORDS:
        return 0.0
    repeated = repetition(words)
    if repeated >= MAX_REPETITION:
        return 0.0
    factor = ENGLISH_FACTOR if guess(text) == "en" else UNCONFIRMED_FACTOR
    return round(factor * text_density(text) * (1.0 - repeated), 4)
