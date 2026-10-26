"""Reading a question before it is answered.

The reading is one parse of the question: its words, what each word is, and
what it means in the offline dictionary. The answer flow shows the reading
before it looks anywhere for an answer, so a question that was not understood
looks different from a question that was understood and not taught.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services import dictionary, nlp
from app.services.reasoning import entities, intent

# How many synonyms a read key word carries into the page.
SYNONYMS_PER_WORD = 2


@dataclass(frozen=True)
class WordReading:
    """One word of the question, and what the offline dictionary says of it."""

    word: str
    lemma: str
    part_of_speech: str
    meaning: str = ""
    synonyms: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "word": self.word,
            "lemma": self.lemma,
            "part_of_speech": self.part_of_speech,
            "meaning": self.meaning,
            "synonyms": list(self.synonyms),
        }


@dataclass(frozen=True)
class QuestionReading:
    """What the reader understood of the question."""

    question: str
    words: tuple[WordReading, ...]
    question_type: str
    key_words: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "words": [word.as_dict() for word in self.words],
            "question_type": self.question_type,
            "key_words": list(self.key_words),
        }


def _parse(text: str) -> tuple[list, dict[str, tuple[str, str]]]:
    """One parse: its words, and the lemma and part of speech of each."""
    try:
        document = nlp.get_nlp()(text)
    except (ImportError, LookupError):
        return [], {}
    tokens = [token for token in document if token.is_alpha]
    found: dict[str, tuple[str, str]] = {}
    for token in tokens:
        lemma = (token.lemma_ or token.text).casefold()
        found.setdefault(token.text.casefold(), (lemma, token.pos_ or ""))
        found.setdefault(lemma, (lemma, token.pos_ or ""))
    return tokens, found


def _read_word(
    word: str, parsed: dict[str, tuple[str, str]], key_words: frozenset[str]
) -> WordReading:
    """One word with its parse, and a key word with its meaning and synonyms."""
    lemma, part_of_speech = parsed.get(word.casefold(), (word.casefold(), ""))
    lookup = None
    synonyms: tuple[str, ...] = ()
    if word.casefold() in key_words or lemma in key_words:
        try:
            lookup = dictionary.lookup_word(word)
            synonyms = tuple(
                dictionary.search_expansions(
                    word, per_word=SYNONYMS_PER_WORD, max_terms=SYNONYMS_PER_WORD
                )
            )
        except (ImportError, LookupError):
            lookup, synonyms = None, ()
    meaning = ""
    if lookup:
        meaning = lookup["definitions"][0]
        part_of_speech = lookup["part_of_speech"] or part_of_speech
    return WordReading(
        word=word,
        lemma=lemma,
        part_of_speech=part_of_speech,
        meaning=meaning,
        synonyms=synonyms,
    )


def read_question(question: str) -> QuestionReading:
    """Read a question: its words, their meanings, and the kind of question."""
    text = (question or "").strip()
    key_words = tuple(entities.entities_of(text))
    keys = frozenset(key_words)
    tokens, parsed = _parse(text)
    words = (
        tuple(_read_word(token.text, parsed, keys) for token in tokens)
        if tokens
        else tuple(_read_word(word, parsed, keys) for word in key_words)
    )
    return QuestionReading(
        question=text,
        words=words,
        question_type=intent.shape_kind(text),
        key_words=key_words,
    )
