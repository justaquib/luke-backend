"""The wording a fine-tuned teacher is trained on, and asked with.

Training and serving must use the same shape, or the model reads a prompt at
answer time that it never saw during training. Every pair is written as a
question, the taught answer as the reference, and the answer to write.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

MAX_QUESTION = 500
MAX_REFERENCE = 1200
MAX_TEXTS = 4000


def _clean(text: str, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


def training_text(question: str, answer: str) -> str:
    """One example: the question, the taught answer and the answer to write."""
    asked = _clean(question, MAX_QUESTION)
    taught = _clean(answer, MAX_REFERENCE)
    return f"Question: {asked}\nReference: {taught}\nAnswer: {taught}"


def prompt(question: str, reference: str = "") -> str:
    """The prompt the teacher answers, with the material it may ground on."""
    asked = _clean(question, MAX_QUESTION)
    supplied = _clean(reference, MAX_REFERENCE)
    if supplied:
        return f"Question: {asked}\nReference: {supplied}\nAnswer:"
    return f"Question: {asked}\nAnswer:"


def training_texts(pairs: Iterable[tuple[str, str]]) -> list[str]:
    """The examples of one run, without the pairs that say nothing."""
    texts: list[str] = []
    seen: set[str] = set()
    for question, answer in pairs:
        if not _clean(question, MAX_QUESTION) or not _clean(answer, MAX_REFERENCE):
            continue
        text = training_text(question, answer)
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        texts.append(text)
        if len(texts) == MAX_TEXTS:
            break
    return texts


def grounding(answer: str, passages: Sequence[str] = ()) -> str:
    """The material an answer is grounded on, as one reference line."""
    parts = [_clean(answer, MAX_REFERENCE)]
    for passage in passages:
        clean = _clean(passage, 400)
        if clean and clean.casefold() not in {part.casefold() for part in parts}:
            parts.append(clean)
    return " ".join(parts)[:MAX_REFERENCE]
