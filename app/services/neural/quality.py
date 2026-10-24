"""The training data that is worth the training steps.

A sentence that repeats itself, a sentence that is too short or too long, and
a sentence made of words the model will never see again are not worth the
steps they take, so they are left out of the training.

The rule about the words the model has never seen needs a corpus to mean
anything, so it is only used when there are enough sentences to count.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from itertools import pairwise

from app.services.neural.tokens import tokenize

# The shortest sentence that is worth learning from.
MIN_WORDS = 3

# The longest sentence that is worth learning from.
MAX_WORDS = 50

# The share of the words that may be words the model has never seen.
MAX_UNKNOWN_SHARE = 0.2

# The share of the word pairs that may be the same pair again and again.
MAX_REPETITION = 0.6

# How many sentences there must be before the share of unknown words counts.
MIN_CORPUS = 20

# How many times a pair of words may be said in one written answer.
# A poem says the same pair again on purpose, but never three times: a pair
# said more than twice is the model stuck in a loop, not a poem.
MAX_PAIR_SAYS = 2


def _rows(sentences: Sequence[str]) -> list[list[str]]:
    """The words of every sentence, without the punctuation."""
    return [
        [token for token in tokenize(sentence) if token[:1].isalpha()] for sentence in sentences
    ]


def _hapax(words: list[list[str]]) -> set[str]:
    """The words that are only in one sentence of the whole corpus."""
    counts: Counter[str] = Counter()
    for row in words:
        counts.update(set(row))
    return {word for word, count in counts.items() if count <= 1}


def repetition(words: list[str]) -> float:
    """The share of the word pairs that are said again in the same sentence.

    A poem says the same pair again on purpose, so the rule only catches the
    text that says nothing but one phrase.
    """
    if len(words) < 2:
        return 0.0
    pairs = list(pairwise(words))
    seen: Counter[tuple[str, str]] = Counter()
    for pair in pairs:
        seen[pair] += 1
    repeated = sum(count - 1 for count in seen.values() if count > 1)
    return repeated / len(pairs)


def repeats_itself(text: str) -> bool:
    """Whether the answer says the same short pair again and again.

    "Rain, rain, go away" says a pair twice on purpose, and a poem may end
    with its line again, so two are allowed. "go away, go away, go away,
    go away" is the model stuck in its track, so an answer that says a pair
    more times than that is not worth giving away.
    """
    words = _rows([text])[0]
    counts: Counter[tuple[str, str]] = Counter(pairwise(words))
    return any(count > MAX_PAIR_SAYS for count in counts.values())


def _is_worth(words: list[str], unknown: set[str], use_unknown_rule: bool) -> bool:
    """Whether a sentence is worth learning from."""
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return False
    if repetition(words) >= MAX_REPETITION:
        return False
    if not use_unknown_rule:
        return True
    unknown_words = sum(1 for word in words if word in unknown)
    return unknown_words / len(words) <= MAX_UNKNOWN_SHARE


def keep(sentences: Sequence[str]) -> list[str]:
    """The sentences that are worth learning from, in the order they came."""
    rows = _rows(sentences)
    unknown = _hapax(rows)
    use_unknown_rule = len(rows) >= MIN_CORPUS
    return [
        sentence
        for sentence, words in zip(sentences, rows, strict=True)
        if _is_worth(words, unknown, use_unknown_rule)
    ]


def dropped(sentences: Sequence[str]) -> int:
    """How many sentences the filter leaves out."""
    return len(sentences) - len(keep(sentences))
