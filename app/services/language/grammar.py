"""Grammar repair, article choice, and rewriting answers as sentences."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.services.language.patterns import _COUNT_REQUEST, _COUNT_WORDS
from app.services.tokenizer import tokenize

logger = logging.getLogger(__name__)

_FUNCTION_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "because",
    "so",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "from",
    "by",
    "is",
    "are",
    "was",
    "were",
    "am",
    "be",
    "been",
    "it",
    "this",
    "that",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "what",
    "which",
    "who",
    "how",
    "can",
    "do",
    "does",
    "did",
    "make",
    "makes",
    "made",
    "sentence",
}

_VERBS_AFTER_POSSESSIVE = {
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "make",
    "makes",
    "made",
    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "say",
    "says",
    "said",
    "eat",
    "eats",
    "live",
    "lives",
    "run",
    "runs",
    "fly",
    "flies",
    "build",
    "builds",
    "give",
    "gives",
    "come",
    "comes",
    "go",
    "goes",
    "like",
    "likes",
    "love",
    "loves",
    "help",
    "helps",
    "need",
    "needs",
    "want",
    "wants",
    "work",
    "works",
    "play",
    "plays",
    "sleep",
    "sleeps",
    "swim",
    "swims",
    "sing",
    "sings",
    "write",
    "writes",
    "read",
    "reads",
    "grow",
    "grows",
    "walk",
    "walks",
    "jump",
    "jumps",
    "drink",
    "drinks",
    "dance",
    "dances",
    "cook",
    "cooks",
    "clean",
    "cleans",
    "laugh",
    "laughs",
    "cry",
    "cries",
    "smile",
    "smiles",
    "bring",
    "brings",
    "keep",
    "keeps",
    "sell",
    "sells",
    "buy",
    "buys",
    "find",
    "finds",
    "know",
    "knows",
    "think",
    "thinks",
    "see",
    "sees",
    "hear",
    "hears",
}
_POSSESSIVE_TYPED = re.compile(
    r"\b([A-Za-z]+)'s\s+(" + "|".join(sorted(_VERBS_AFTER_POSSESSIVE)) + r")\b",
    re.IGNORECASE,
)


def apply_requested_count(question: str, answer: str) -> str:
    """When the question asks for a number of things, give that many.

    "Name 5 animals" with a lesson that lists every animal gives five of them.
    """
    match = _COUNT_REQUEST.search(question or "")
    if not match:
        return answer
    token = match.group(1) or match.group(2) or ""
    wanted = int(token) if token.isdecimal() else _COUNT_WORDS.get(token.casefold(), 0)
    if wanted < 1 or wanted > 50:
        return answer
    head, separator, tail = (answer or "").partition(" are ")
    if not separator:
        return answer
    items = [item.strip() for item in tail.rstrip(". ").split(",") if item.strip()]
    if len(items) <= wanted:
        return answer
    return f"{head} are {', '.join(items[:wanted])}."


def _is_naming_word(word: str) -> bool:
    """Whether a word names a thing, so an article belongs before it."""
    try:
        from app.services.nlp import get_nlp

        doc = get_nlp()(word)
    except Exception as error:  # noqa: BLE001 -- the optional grammar-model boundary
        logger.debug("the grammar model could not read %s: %s", word, error)
        return False
    if not doc or not doc[0].pos_:
        return False
    return doc[0].pos_ in {"NOUN", "PROPN"}


def _insert_articles(sentence: str, document: Any) -> str:
    insertions: list[tuple[int, str]] = []
    for index, token in enumerate(document):
        if token.lower_ not in {"is", "are", "was", "were"}:
            continue
        if index + 1 >= len(document):
            continue
        following = document[index + 1]
        if following.pos_ not in {"NOUN", "PROPN"} or following.tag_ in {"NNS", "NNPS"}:
            continue
        article = "an" if following.text[:1].lower() in "aeiou" else "a"
        insertions.append((following.idx, article))
    text = sentence
    for position, article in reversed(insertions):
        text = text[:position] + f"{article} " + text[position:]
    return text


def _repair_sentence(sentence: str) -> str:
    """Fix the small grammar slips a sentence can carry."""
    if not sentence:
        return sentence
    repaired = _POSSESSIVE_TYPED.sub(lambda match: f"{match.group(1)}s {match.group(2)}", sentence)
    return _add_missing_articles(repaired)


def _add_missing_articles(sentence: str) -> str:
    """Put the article back when a naming word stands on its own."""
    try:
        from app.services.nlp import get_nlp

        doc = get_nlp()(sentence)
    except Exception as error:  # noqa: BLE001 -- the optional grammar-model boundary
        logger.debug("the grammar model could not read a sentence: %s", error)
        return sentence
    return _insert_articles(sentence, doc)


def _repair_with_document(sentence: str, document: Any) -> str:
    """Put back the article that a naming word is missing."""
    return _insert_articles(sentence, document)


def _repair_sentences(texts: list[str]) -> list[str]:
    """Clean a whole list of sentences, with the tagger looking at them all
    at once so it does not have to be loaded for every sentence."""
    if not texts:
        return []
    try:
        from app.services.nlp import get_nlp

        nlp = get_nlp()
        documents = list(nlp.pipe(texts, batch_size=64))
    except Exception as error:  # noqa: BLE001 -- the optional grammar-model boundary
        logger.debug("the grammar model could not read %s sentences: %s", len(texts), error)
        documents = [None] * len(texts)
    repaired: list[str] = []
    for text, document in zip(texts, documents, strict=True):
        if document is None:
            repaired.append(
                _POSSESSIVE_TYPED.sub(
                    lambda match: f"{match.group(1)}s {match.group(2)}",
                    text or "",
                )
            )
            continue
        repaired.append(_repair_with_document(text, document))
    return repaired


def _sentence_starts_with(sentence: str, word: str) -> bool:
    """Whether a sentence is about the word, because it begins with it."""
    tokens = tokenize(sentence)
    if not tokens:
        return False
    first = tokens[1] if tokens[0] in {"a", "an", "the"} and len(tokens) > 1 else tokens[0]
    return first == word or first.startswith(word + "'") or first.startswith(word + "s")
