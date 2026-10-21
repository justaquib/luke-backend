"""Answers to questions about letters, their forms, and their words."""

from __future__ import annotations

import re

from app.services.language.curriculum import ALPHABET_WORDS


def _alphabet_forms(normalized: str) -> str | None:
    letter_forms = re.search(
        r"\b(?:capital|uppercase)\s+(?:and\s+small\s+)?(?:forms?\s+of\s+)?([a-z])\b", normalized
    )
    explicit_forms = re.search(
        r"\b(?:capital|uppercase)(?:\s+letter)?\s+([a-z])\b.*"
        r"\b(?:small|lowercase)(?:\s+letter)?\s+([a-z])\b",
        normalized,
    )
    if explicit_forms:
        capital, small = explicit_forms.group(1).upper(), explicit_forms.group(2).lower()
        return f"Capital {capital} is {capital}; small {small} is {small}."
    if re.search(
        r"\b(?:capital|uppercase)\s+and\s+(?:small|lowercase)\s+forms?\s+of\s+[a-z]\b", normalized
    ):
        match = re.search(r"\bof\s+([a-z])\b", normalized)
        if match:
            capital = match.group(1).upper()
            return f"Capital {capital} is {capital}; small {capital.lower()} is {capital.lower()}."
    if letter_forms:
        capital = letter_forms.group(1).upper()
        return f"Capital {capital} is the uppercase form of {capital}."
    small_form = re.search(r"\b(?:small|lowercase)\s+(?:letter\s+)?([a-z])\b", normalized)
    if small_form:
        small = small_form.group(1).lower()
        return f"Small {small} is the lowercase form of {small.upper()}."
    return None


def _alphabet_starts_word(normalized: str, word_to_letter: dict[str, str]) -> str | None:
    match = re.search(
        r"\b(?:what|which)\s+letter\s+(?:does\s+)?([a-z]+(?:-[a-z]+)?)\s+"
        r"(?:start|begin)s?\s+(?:with\s*)?\b",
        normalized,
    )
    if not match:
        return None
    word = match.group(1)
    letter = word_to_letter.get(word)
    if letter:
        return f"{word.capitalize()} starts with {letter}."
    return None


def _alphabet_spelling(normalized: str, word_to_letter: dict[str, str]) -> str | None:
    match = re.search(
        r"\b(?:what\s+is\s+)?(?:the\s+)?spelling\s+of\s+([a-z-]+)\b|\bspell\s+([a-z-]+)\b",
        normalized,
    )
    if not match:
        return None
    word = match.group(1) or match.group(2)
    plain_word = word.replace("-", "")
    if plain_word in word_to_letter:
        return f"{word.capitalize()} is spelled {'-'.join(plain_word.upper())}."
    return None


def _alphabet_letter_for(normalized: str, letter_to_word: dict[str, str]) -> str | None:
    match = re.search(r"\b(?:what\s+is\s+)?([a-z])\s+for\b|\b([a-z])\s+for\s+what\b", normalized)
    if not match:
        return None
    letter = (match.group(1) or match.group(2)).upper()
    word = letter_to_word.get(letter)
    if word:
        return f"{letter} is for {word}."
    return None


def _alphabet_word_starts(normalized: str, letter_to_word: dict[str, str]) -> str | None:
    match = re.search(r"\b(?:what|which)\s+word\s+(?:starts|begins)\s+with\s+([a-z])\b", normalized)
    if not match:
        return None
    letter = match.group(1).upper()
    word = letter_to_word.get(letter)
    if word:
        return f"{letter} is for {word}."
    return None


def _answer_alphabet_question(question: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9 -]", " ", question.casefold())
    normalized = " ".join(normalized.split())
    letter_to_word = ALPHABET_WORDS
    word_to_letter = {
        variant: letter
        for letter, word in letter_to_word.items()
        for variant in (word, word.replace("-", ""))
    }
    checks = (
        _alphabet_forms(normalized),
        _alphabet_starts_word(normalized, word_to_letter),
        _alphabet_spelling(normalized, word_to_letter),
        _alphabet_letter_for(normalized, letter_to_word),
        _alphabet_word_starts(normalized, letter_to_word),
    )
    return next((answer for answer in checks if answer), None)
