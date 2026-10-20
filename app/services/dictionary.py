"""Small offline English dictionary answers backed by the NLTK WordNet corpus."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any


_WORD = r"[A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*){0,3}"
_SEARCH_STOP_WORDS = {
    "what", "who", "where", "when", "which", "is", "are", "was", "were",
    "do", "does", "did", "the", "a", "an", "of", "for", "please", "tell",
    "me", "give", "show", "find", "in", "on", "from", "about", "to", "with",
    "name", "number", "no", "date", "email", "phone", "mobile", "contact", "address", "code", "id", "s",
}


@lru_cache(maxsize=1)
def _wordnet():
    # Import lazily so the rest of the API can start even before the corpus is
    # installed. Setup instructions download the data for offline use.
    from nltk.corpus import wordnet

    return wordnet


def _pos_label(pos: str) -> str:
    return {
        "n": "noun",
        "v": "verb",
        "a": "adjective",
        "s": "adjective",
        "r": "adverb",
    }.get(pos, "word")


def _unique_synsets(word: str) -> list[Any]:
    wn = _wordnet()
    normalized = word.strip().replace("’", "'").replace(" ", "_")
    synsets = wn.synsets(normalized)
    if synsets:
        return synsets
    # Handle ordinary inflections such as "apples" or "running".
    lemma = wn.morphy(normalized)
    return wn.synsets(lemma) if lemma else []


def first_part_of_speech(word: str) -> str:
    """The kind of word it is, by its first sense: "noun", "adjective" and so on."""
    normalized = word.strip().replace("\u2019", "'")
    if not re.fullmatch(r"[A-Za-z][A-Za-z'\u2019-]{0,79}", normalized):
        return ""
    try:
        synsets = _unique_synsets(normalized)
    except (ImportError, LookupError):
        return ""
    if not synsets:
        return ""
    return _pos_label(synsets[0].pos())


def lookup_word(word: str) -> dict[str, Any] | None:
    """Return a few local WordNet senses for a word-learning lesson."""
    normalized = word.strip().replace("’", "'")
    if not re.fullmatch(r"[A-Za-z][A-Za-z'’-]{0,79}", normalized):
        return None
    try:
        synsets = _unique_synsets(normalized)
    except (ImportError, LookupError):
        return None
    if not synsets:
        return None
    definitions: list[str] = []
    for synset in synsets:
        definition = synset.definition().strip()
        if definition and definition.casefold() not in {item.casefold() for item in definitions}:
            definitions.append(definition)
        if len(definitions) == 3:
            break
    if not definitions:
        return None
    labels = list(dict.fromkeys(_pos_label(synset.pos()) for synset in synsets[:3]))
    return {
        "word": normalized,
        "part_of_speech": ", ".join(labels),
        "definitions": definitions,
    }


def search_expansions(text: str, per_word: int = 2, max_terms: int = 8) -> list[str]:
    """Return a few common-sense synonyms to help local retrieval match wording."""
    terms = re.findall(r"[a-z]+", text.lower())
    expansions: list[str] = []
    try:
        for term in terms:
            if term in _SEARCH_STOP_WORDS:
                continue
            synsets = _unique_synsets(term)
            if not synsets:
                continue
            added_for_word = 0
            for synset in synsets[:3]:
                for lemma in synset.lemmas():
                    candidate = lemma.name().replace("_", " ").lower()
                    if candidate != term and candidate not in expansions:
                        expansions.append(candidate)
                        added_for_word += 1
                        if len(expansions) >= max_terms or added_for_word >= per_word:
                            break
                if len(expansions) >= max_terms or added_for_word >= per_word:
                    break
            if len(expansions) >= max_terms:
                break
    except (ImportError, LookupError):
        return []
    return expansions


_TRAILING_HINTS = (
    r"\s+(?:here|there|in\s+this\s+sentence|in\s+that\s+sentence|in\s+the\s+sentence|"
    r"in\s+this\s+example|in\s+the\s+text)"
)
_LEADING_HINTS = (
    r"^(?:in\s+this\s+sentence|in\s+that\s+sentence|in\s+the\s+sentence|here|there)\s*[,:]?\s*"
)


def _question_part(question: str) -> str:
    """The question part of an input that may start with a sentence.

    "I am a boy. What is boy here?" asks about the word in the first
    sentence, so only the question part is parsed.
    """
    text = question.strip()
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not parts:
        return text
    for part in reversed(parts):
        if part.endswith("?"):
            return part
    return parts[-1]


def _parse_request(question: str) -> tuple[str, str] | None:
    q = _question_part(question).strip().rstrip("?.! ")
    q = re.sub(_LEADING_HINTS, "", q, flags=re.IGNORECASE)
    q = re.sub(_TRAILING_HINTS, "", q, flags=re.IGNORECASE)
    patterns = [
        ("letter", rf"what\s+(?:letter|alphabet)\s+(?:does\s+)?({_WORD})\s+(?:start|begin)\s+with"),
        ("letter", rf"({_WORD})\s+(?:starts|begins)\s+with\s+which\s+(?:letter|alphabet)"),
        ("letter", rf"what\s+is\s+the\s+first\s+letter\s+of\s+({_WORD})"),
        ("definition", rf"what\s+does\s+({_WORD})\s+mean"),
        ("definition", rf"(?:define|meaning\s+of)\s+({_WORD})"),
        ("definition", rf"what\s+is\s+(?:the\s+)?(?:definition|meaning)\s+of\s+({_WORD})"),
        ("definition", rf"what\s+is\s+(?:(?:a|an|the)\s+)?([A-Za-z][A-Za-z'’-]*)"),
        ("synonym", rf"(?:what\s+are\s+)?(?:the\s+)?synonyms?\s+(?:of|for)\s+({_WORD})"),
        ("synonym", rf"what\s+is\s+(?:a\s+|the\s+)?synonym\s+(?:of|for)\s+({_WORD})"),
        ("synonym", rf"(?:give|show|list|tell)\s+me\s+(?:some\s+)?synonyms?\s+(?:of|for)\s+({_WORD})"),
        ("synonym", rf"what\s+are\s+some\s+synonyms?\s+(?:of|for)\s+({_WORD})"),
        ("antonym", rf"(?:what\s+are\s+)?(?:the\s+)?antonyms?\s+(?:of|for)\s+({_WORD})"),
        ("antonym", rf"what\s+is\s+(?:an?\s+|the\s+)?antonym\s+(?:of|for)\s+({_WORD})"),
        ("antonym", rf"(?:give|show|list|tell)\s+me\s+(?:some\s+)?antonyms?\s+(?:of|for)\s+({_WORD})"),
        ("antonym", rf"what\s+is\s+the\s+opposite\s+of\s+({_WORD})"),
        ("part_of_speech", rf"what\s+part\s+of\s+speech\s+is\s+({_WORD})"),
    ]
    for kind, pattern in patterns:
        match = re.fullmatch(pattern, q, flags=re.IGNORECASE)
        if match:
            return kind, match.group(1).strip()
    return None


def _answer_alphabet_sequence(question: str) -> str | None:
    q = question.strip().rstrip("?.! ")
    patterns = (
        r"what\s+(?:letter\s+)?comes\s+after\s+(?:the\s+)?(?:letter\s+)?([a-z])",
        r"which\s+letter\s+comes\s+after\s+(?:the\s+)?(?:letter\s+)?([a-z])",
        r"what\s+letter\s+follows\s+(?:the\s+)?(?:letter\s+)?([a-z])",
        r"what\s+comes\s+after\s+(?:the\s+)?(?:letter\s+)?([a-z])",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, q, flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter == "Z":
                return "There is no letter after Z in the English alphabet."
            return f"The letter after {letter} is {chr(ord(letter) + 1)}."
    return None


def answer_dictionary_question(question: str) -> str | None:
    """Return a concise offline lexical answer, or None if unsupported/unavailable."""
    alphabet_answer = _answer_alphabet_sequence(question)
    if alphabet_answer is not None:
        return alphabet_answer

    parsed = _parse_request(question)
    if not parsed:
        return None
    kind, word = parsed

    if kind == "letter":
        first = next((char for char in word if char.isalpha()), None)
        return f"{word.capitalize()} starts with the letter {first.upper()}." if first else None

    try:
        synsets = _unique_synsets(word)
    except (ImportError, LookupError):
        return None
    if not synsets:
        return f"I couldn't find \"{word}\" in the offline English dictionary."

    if kind == "definition":
        # WordNet orders senses by frequency; show up to two distinct concise
        # definitions to avoid silently choosing among common meanings.
        definitions: list[str] = []
        for synset in synsets:
            definition = synset.definition().strip()
            if definition and definition.casefold() not in {d.casefold() for d in definitions}:
                definitions.append(definition)
            if len(definitions) == 2:
                break
        if not definitions:
            return f"I couldn't find a definition for \"{word}\" in the offline English dictionary."
        if len(definitions) == 1:
            return f"{word.capitalize()} ({_pos_label(synsets[0].pos())}): {definitions[0]}"
        return f"{word.capitalize()} has multiple meanings. For example: {definitions[0]}; it can also mean {definitions[1]}"

    if kind == "part_of_speech":
        labels = list(dict.fromkeys(_pos_label(synset.pos()) for synset in synsets))
        return f"\"{word}\" can be used as a {', '.join(labels[:-1])} or {labels[-1]}." if len(labels) > 1 else f"\"{word}\" is used as a {labels[0]}."

    related: list[str] = []
    for synset in synsets[:3]:
        lemmas = synset.lemmas()
        for lemma in lemmas:
            candidates = lemma.antonyms() if kind == "antonym" else lemmas
            for candidate in candidates:
                value = candidate.name().replace("_", " ")
                if value.casefold() != word.casefold() and value.casefold() not in {item.casefold() for item in related}:
                    related.append(value)
                    if len(related) >= 5:
                        break
            if len(related) >= 5:
                break
        if len(related) >= 5:
            break
    if not related:
        label = "antonyms" if kind == "antonym" else "synonyms"
        return f"I couldn't find {label} for \"{word}\" in the offline English dictionary."
    label = "Antonyms" if kind == "antonym" else "Synonyms"
    return f"{label} of {word}: {', '.join(related)}."


def is_dictionary_question(question: str) -> bool:
    """Return whether the question has a supported dictionary-style shape."""
    return _answer_alphabet_sequence(question) is not None or _parse_request(question) is not None
