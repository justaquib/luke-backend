import re
from functools import lru_cache
from typing import Any

import spacy

from app.config import settings


@lru_cache(maxsize=1)
def get_nlp() -> Any:
    try:
        return spacy.load(settings.spacy_model)
    except OSError:
        return spacy.blank("en")


def preprocess(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """Split a text into sentence-aligned chunks, keeping questions whole."""
    sentences = _sentences(text)
    if not sentences:
        return [text]
    chunks: list[str] = []
    current = ""
    kept: list[str] = []
    for sentence in sentences:
        for unit in _split_long(sentence, max_chars):
            if current and _QUESTION_START.match(unit):
                chunks.append(current)
                current, kept = unit, [unit]
            elif current and len(current) + len(unit) + 1 > max_chars:
                chunks.append(current)
                tail = _overlap(kept, overlap)
                current = f"{tail} {unit}".strip()
                kept = [*([tail] if tail else []), unit]
            else:
                current = f"{current} {unit}".strip()
                kept.append(unit)
    if current:
        chunks.append(current)
    return chunks


_QUESTION_START = re.compile(r"^(?:Q[.:]|Question\b)", re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    """The sentences of a text, with its line breaks kept for the sentencizer."""
    cleaned = text.replace("\u00a0", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    nlp = get_nlp()
    if "sentencizer" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return [sentence.text.strip() for sentence in nlp(cleaned).sents if sentence.text.strip()]


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """A sentence that fits, or its words packed without cutting a word."""
    if len(sentence) <= max_chars:
        return [sentence]
    parts: list[str] = []
    current = ""
    for word in sentence.split():
        if current and len(current) + len(word) + 1 > max_chars:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts


def _overlap(sentences: list[str], overlap: int) -> str:
    """The tail kept between chunks, at a sentence or word boundary."""
    if overlap <= 0 or not sentences:
        return ""
    last = sentences[-1]
    if len(last) <= overlap:
        kept = [last]
        for sentence in reversed(sentences[:-1]):
            if sum(len(item) + 1 for item in kept) + len(sentence) > overlap:
                break
            kept.insert(0, sentence)
        return " ".join(kept)
    tail = last[-overlap:]
    if " " in tail:
        tail = tail.split(" ", 1)[1]
    return tail.strip(" ,.;:")


def extract_entities(text: str) -> list[dict]:
    nlp = get_nlp()
    if "ner" not in nlp.pipe_names:
        return []
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
