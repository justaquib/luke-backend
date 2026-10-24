"""Writing an answer in the model's own words, from what it has learned."""

from __future__ import annotations

import re
from typing import Any

from app.services.language import runtime
from app.services.language.grammar import _add_missing_articles
from app.services.language.helpers import _content_terms, _stem
from app.services.language.patterns import _QUESTION_STOP_WORDS


def _is_a_list(answer: str) -> bool:
    """Whether an answer is a list of things, like "Some animals are ant, bear, cat."

    A list is the answer itself, so it is not put into other words.
    """
    _, separator, tail = (answer or "").partition(" are ")
    if not separator:
        return False
    items = [item.strip() for item in tail.rstrip(". ").split(",") if item.strip()]
    return len(items) >= 3


def _subject_of(answer: str) -> str:
    """The subject of an answer, such as "The cow" in "The cow is an animal."

    It is the first few words, up to the first word that tells what is said
    about it.
    """
    words = (answer or "").strip().split()
    if not words:
        return ""
    stop = {
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "gives",
        "give",
        "makes",
        "make",
        "can",
        "will",
        "does",
        "do",
        "said",
        "says",
        "means",
        "mean",
    }
    for index, word in enumerate(words[:4]):
        if word.strip(".,!?").lower() in stop:
            return " ".join(words[:index]).strip(".,!?")
    return " ".join(words[:2]).strip(".,!?")


def _predicates_of(answer: str, subject: str) -> list[str]:
    """The facts the answer tells about its subject.

    "The cow is animal. It is a farm animal that gives milk." holds the facts
    "is animal", "is a farm animal" and "gives milk".
    """
    facts: list[str] = []
    head_words = re.sub(
        r"^(?:the|a|an)\s+", "", " ".join(subject.split()), flags=re.IGNORECASE
    ).strip(".,!?")
    if not head_words:
        return facts
    subject_pattern = re.compile(
        r"\b(?:(?:the|a|an)\s+)?" + re.escape(head_words) + r"\b",
        re.IGNORECASE,
    )
    for sentence in re.split(r"(?<=[.!?])\s+", (answer or "").strip()):
        text = sentence.strip()
        if not text:
            continue
        if text.endswith("?"):
            # A question is not something that can be said about a subject.
            continue
        text = subject_pattern.sub("they", text)
        if re.match(r"^(?:it|they|he|she|this|that)\b", text, re.IGNORECASE):
            text = re.sub(
                r"^(?:it|they|he|she|this|that)\b",
                "",
                text,
                count=1,
                flags=re.IGNORECASE,
            ).strip()
        for raw in re.split(r"\s+(?:that|which|and)\s+|,\s+", text):
            part = raw.strip(" .!?")
            if part:
                facts.append(part)
    return facts


def _fix_article(sentence: str) -> str:
    """Put the article back when the taught answer left it out."""
    return _add_missing_articles(sentence)


def _wording_score(
    tenant_id: int | None,
    model: Any,
    sentence: str,
    allow_writer: bool = True,
) -> float:
    """How well the model knows a wording.

    The writer is left out while the training corpus is built: a corpus that
    depends on the model being trained cannot be compared across runs. With
    neither a writer nor a word model, every wording scores the same and the
    shortest one is chosen.
    """
    writer = None
    if allow_writer:
        try:
            from app.services import brain

            writer = brain.load_writer(tenant_id)
        except (OSError, RuntimeError, ValueError):
            writer = None
    if writer is not None:
        return writer.score(sentence)
    if model is not None:
        return model.score_words(sentence.rstrip(".").split())
    return 0.0


def paraphrase_answer(
    tenant_id: int | None,
    answer: str,
    subject: str | None = None,
    allow_writer: bool = True,
) -> str | None:
    """Write the answer in the model's own words, from what it has learned."""
    answer = (answer or "").strip()
    if not answer or _is_a_list(answer):
        return None
    subject = subject or _subject_of(answer)
    if not subject:
        return None
    with runtime._LOCK:
        model = runtime._model(runtime._read(tenant_id))
    facts = _predicates_of(answer, subject)
    if not facts:
        return None
    scored: list[tuple[float, str]] = []
    for fact in facts:
        sentence = _fix_article(f"{subject} {fact}.")
        scored.append((_wording_score(tenant_id, model, sentence, allow_writer), sentence))
    scored.sort(key=lambda item: -item[0])
    original_words = {_stem(word) for word in _content_terms(answer, _QUESTION_STOP_WORDS)}
    for _, sentence in scored:
        if sentence.strip().lower() == answer.strip().lower():
            continue
        sentence_words = {_stem(word) for word in _content_terms(sentence, _QUESTION_STOP_WORDS)}
        if not sentence_words:
            continue
        kept = len(sentence_words & original_words) / max(len(original_words), 1)
        if kept < 0.5:
            continue
        return sentence
    return None
