"""The answers written again in the model's own words.

The model is taught the answer in more than one wording, so it can answer a
question that is worded differently. A rewrite is only used when it says the
same thing and reads like a sentence: the model must not learn a different
answer, and it must not learn bad grammar.
"""

from __future__ import annotations

import re

from app.services.neural import grammar

# The words that never take an article.
NO_ARTICLE = frozenset(
    {
        "advice",
        "equipment",
        "furniture",
        "homework",
        "information",
        "knowledge",
        "news",
        "progress",
        "research",
        "water",
        "weather",
        "work",
    }
)

_ARTICLE = re.compile(r"\b(a|an)\s+([A-Za-z0-9][A-Za-z'’-]*)", re.IGNORECASE)


def _same_words(left: str, right: str) -> bool:
    """Whether two answers say the same thing, word by word."""
    from app.services.language.helpers import _content_terms
    from app.services.language.patterns import _QUESTION_STOP_WORDS

    return _content_terms(left, _QUESTION_STOP_WORDS) == _content_terms(right, _QUESTION_STOP_WORDS)


def _articles_are_right(text: str) -> bool:
    """Whether every article in the text is the right one."""
    for match in _ARTICLE.finditer(text):
        article = match.group(1).lower()
        word = match.group(2)
        if word[0].isupper() or word[0].isdigit():
            return False
        if word.lower() in NO_ARTICLE:
            return False
        wants_an = word[0].lower() in "aeiou"
        if wants_an != (article == "an"):
            return False
    return True


def is_a_good_rewrite(original: str, variant: str) -> bool:
    """Whether the rewrite says the same thing and reads like a sentence."""
    original = (original or "").strip()
    variant = (variant or "").strip()
    if not original or not variant:
        return False
    # The subject was written twice when the rewrite keeps the same first word.
    same_subject = variant.split()[0] == original.split()[0]
    checks = (
        variant.casefold() != original.casefold(),
        _same_words(original, variant),
        variant.endswith("."),
        variant[:1].isupper(),
        not same_subject,
        _articles_are_right(variant),
    )
    return all(checks) and not grammar.problems(variant)
