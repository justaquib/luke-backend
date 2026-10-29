"""Reading a text: its words, its sentences, its passages and its names.

A passage that is indexed or learned from keeps the characters of the source
it covers, so the passage can name its page and its offsets. The splitter is
sentence-aligned and keeps a question whole.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import spacy

from app.config import settings


@dataclass(frozen=True)
class ChunkSpan:
    """One passage, and the characters of the source it covers.

    ``text`` is the passage with its spacing normalised; ``char_start`` and
    ``char_end`` are the offsets of the region in the cleaned source the
    passage was cut from, so a reader can point at where it was found.
    """

    text: str
    char_start: int
    char_end: int


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


def chunk_spans(text: str, max_chars: int = 800, overlap: int = 100) -> list[ChunkSpan]:
    """Split a text into sentence-aligned passages, with their source spans."""
    cleaned = _clean(text)
    units = _unit_spans(cleaned, max_chars)
    if not units:
        return [ChunkSpan(text, 0, len(text))]
    groups = _assemble(units, cleaned, max_chars, overlap)
    return [
        ChunkSpan(" ".join(unit.text for unit in group), group[0].char_start, group[-1].char_end)
        for group in groups
    ]


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """Split a text into sentence-aligned chunks, keeping questions whole."""
    return [span.text for span in chunk_spans(text, max_chars, overlap)]


_QUESTION_START = re.compile(r"^(?:Q[.:]|Question\b)", re.IGNORECASE)
_WORD = re.compile(r"\S+")


def _clean(text: str) -> str:
    cleaned = text.replace("\u00a0", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _unit_spans(cleaned: str, max_chars: int) -> list[ChunkSpan]:
    """The sentences of the cleaned text, long ones split between words."""
    nlp = get_nlp()
    if "sentencizer" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    units: list[ChunkSpan] = []
    for sentence in nlp(cleaned).sents:
        text = sentence.text.strip()
        if not text:
            continue
        start = sentence.start_char + sentence.text.find(text)
        units.extend(_split_long(ChunkSpan(text, start, start + len(text)), max_chars))
    return units


def _split_long(unit: ChunkSpan, max_chars: int) -> list[ChunkSpan]:
    """A sentence that fits, or its words packed without cutting a word."""
    if len(unit.text) <= max_chars:
        return [unit]
    words = [
        (match.group(), unit.char_start + match.start()) for match in _WORD.finditer(unit.text)
    ]
    parts: list[ChunkSpan] = []
    current: list[tuple[str, int]] = []
    for word, offset in words:
        width = sum(len(item[0]) + 1 for item in current) + len(word)
        if current and width > max_chars:
            parts.append(_packed(current))
            current = []
        current.append((word, offset))
    if current:
        parts.append(_packed(current))
    return parts


def _packed(words: list[tuple[str, int]]) -> ChunkSpan:
    text = " ".join(word for word, _ in words)
    return ChunkSpan(text, words[0][1], words[-1][1] + len(words[-1][0]))


def _assemble(
    units: list[ChunkSpan], cleaned: str, max_chars: int, overlap: int
) -> list[list[ChunkSpan]]:
    """Pack units into passages, starting a new one at a question or the budget."""
    groups: list[list[ChunkSpan]] = []
    current: list[ChunkSpan] = []
    for unit in units:
        if current and _QUESTION_START.match(unit.text):
            groups.append(current)
            current = [unit]
        elif current and _width(current) + len(unit.text) + 1 > max_chars:
            groups.append(current)
            tail = _overlap_spans(current, cleaned, overlap)
            current = [*tail, unit]
        else:
            current.append(unit)
    if current:
        groups.append(current)
    return groups


def _width(units: list[ChunkSpan]) -> int:
    return sum(len(unit.text) for unit in units) + max(len(units) - 1, 0)


def _overlap_spans(units: list[ChunkSpan], cleaned: str, overlap: int) -> list[ChunkSpan]:
    """The tail kept between chunks, at a sentence or word boundary."""
    if overlap <= 0 or not units:
        return []
    last = units[-1]
    if len(last.text) <= overlap:
        kept = [last]
        for unit in reversed(units[:-1]):
            if _width(kept) + len(unit.text) + 1 > overlap:
                break
            kept.insert(0, unit)
        return kept
    start = max(last.char_start, last.char_end - overlap)
    space = cleaned.find(" ", start, last.char_end)
    if space != -1:
        start = space + 1
    text = cleaned[start : last.char_end]
    stripped = text.strip(" ,.;:")
    if not stripped:
        return []
    start += len(text) - len(text.lstrip(" ,.;:"))
    return [ChunkSpan(stripped, start, start + len(stripped))]


def extract_entities(text: str) -> list[dict]:
    nlp = get_nlp()
    if "ner" not in nlp.pipe_names:
        return []
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
