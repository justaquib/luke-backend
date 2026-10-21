"""Small text helpers: normalising, comparing, and stemming words."""

from __future__ import annotations

import hashlib
import re

_STEM_EXCEPTIONS = {"does": "do", "goes": "go", "was": "be", "has": "have", "is": "be"}

# The words and the numbers of a text, without the punctuation.
_TERM = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")


def _normalize_phrase(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _phrase_similarity(left: str, right: str) -> float:
    left_tokens = set(_normalize_phrase(left).split())
    right_tokens = set(_normalize_phrase(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _words_of(text: str) -> set[str]:
    """The words of a text, one by one.

    Only the punctuation and the case are dropped. Every word is kept, because
    one word is enough to make a different question: the opposite of "in" is
    not the opposite of "loud".
    """
    return set(_normalize_phrase(text).split())


def _same_words(left: str, right: str) -> bool:
    """Two texts that say the same thing, word by word."""
    left_words = _words_of(left)
    right_words = _words_of(right)
    return bool(left_words) and left_words == right_words


def _word_by_word_similarity(left: str, right: str) -> float:
    """How close two questions are, compared word by word.

    The same words mean the same question, whatever order they are in. One
    extra word is still the same question. A word that is different makes it a
    different question, so it is not reported as something already known.
    """
    left_words = _words_of(left)
    right_words = _words_of(right)
    if not left_words or not right_words:
        return 0.0
    smaller, larger = (
        (left_words, right_words)
        if len(left_words) <= len(right_words)
        else (right_words, left_words)
    )
    if not smaller <= larger:
        return 0.0
    if len(larger) - len(smaller) > 1:
        return 0.0
    return 1.0 if len(larger) == len(smaller) else 0.9


def _parse_lesson_pairs(raw: str) -> list[tuple[str, str]]:
    pairs = re.findall(r"(?ims)^\s*q\s*:\s*(.+?)\s*\n\s*a\s*:\s*(.+?)(?=\n\s*q\s*:|\Z)", raw)
    return [
        (question.strip(), answer.strip())
        for question, answer in pairs
        if question.strip() and answer.strip()
    ]


def _parse_word_pairs(raw: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in raw.splitlines():
        if ":" not in line:
            continue
        word, meaning = line.split(":", 1)
        if word.strip() and meaning.strip():
            pairs.append((word.strip().lower(), meaning.strip()))
    return pairs


def _stem(token: str) -> str:
    """The word without its ending, so "gives" and "give" are the same word."""
    if token in _STEM_EXCEPTIONS:
        return _STEM_EXCEPTIONS[token]
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    if token.endswith("ss"):
        return token
    if token.endswith("es") and len(token) > 4 and token[-3] in "sxz":
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _content_terms(text: str, stop_words: set[str]) -> set[str]:
    """Meaningful tokens of a text.

    Punctuation is dropped on purpose. A question like "who is daraksha?"
    otherwise shares the "?" token with every lesson that ends in a question
    mark, which made unrelated curriculum answers match.

    The numbers are kept: "the square root of 16" and "the square root of 25"
    are two different lessons, so a question that names a number is answered
    by the lesson that names the same one.
    """
    return {
        token
        for token in (match.group(0).lower().replace("’", "'") for match in _TERM.finditer(text))
        if token not in stop_words
    }


def _foundation_lesson_id(question: str) -> str:
    """A stable id for a starter lesson so it is only embedded once."""
    digest = hashlib.md5(question.strip().lower().encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return f"foundation-{digest}"


def _token_matches(token: str, word: str) -> bool:
    """Whether a token in a sentence is the word that was asked for.

    "bee" is found in "bee's", because that is still the word bee.
    """
    if token == word:
        return True
    return token.startswith(word + "'")
