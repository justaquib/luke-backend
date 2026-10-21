"""Grammar repair, article choice, and rewriting answers as sentences."""

from __future__ import annotations

import re

from app.services.language import runtime
from app.services.language.helpers import _content_terms, _stem
from app.services.language.patterns import _COUNT_REQUEST, _COUNT_WORDS, _QUESTION_STOP_WORDS
from app.services.tiny_language_model import tokenize

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


def _is_a_list(answer: str) -> bool:
    """Whether an answer is a list of things, like "Some animals are ant, bear, cat."

    A list is the answer itself, so it is not put into other words.
    """
    head, separator, tail = (answer or "").partition(" are ")
    if not separator:
        return False
    items = [item.strip() for item in tail.rstrip(". ").split(",") if item.strip()]
    return len(items) >= 3


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
        for part in re.split(r"\s+(?:that|which|and)\s+|,\s+", text):
            part = part.strip(" .!?")
            if part:
                facts.append(part)
    return facts


def _article_for(model, word: str) -> str:
    """The article that belongs before a word, such as "an" before "animal".

    The article the model has seen is used, and when it has not seen one the
    usual rule of the language is used.
    """
    lowered = word.lower()
    for article in ("an", "a"):
        counts = model.backoff.get((article, lowered)) or model.transitions.get((article, lowered))
        if counts:
            return article
    if lowered[:1] in "aeiou":
        return "an"
    return "a"


def _fix_article(model, sentence: str) -> str:
    """Put the article back when the taught answer left it out."""
    return _add_missing_articles(sentence)


def _facts_known_about(model, subject: str) -> list[str]:
    """What the model knows about a subject, as facts."""
    head = subject.rsplit(maxsplit=1)[-1].strip(".,!?").lower()
    if not head:
        return []
    facts: list[str] = []
    for sentence in model.sentences:
        if sentence.strip().endswith("?"):
            continue
        tokens = tokenize(sentence)
        if not any(
            token == head or token.startswith(head + "'") or token.startswith(head + "s")
            for token in tokens
        ):
            continue
        facts.extend(_predicates_of(sentence, subject))
    return facts


def _wording_score(tenant_id: int | None, model, sentence: str) -> float:
    """How well the model knows a wording."""
    writer = None
    try:
        from app.services import brain

        writer = brain.load_writer(tenant_id)
    except Exception:
        writer = None
    if writer is not None:
        return writer.score(sentence)
    return model.score_words(sentence.rstrip(".").split())


def paraphrase_answer(tenant_id: int | None, answer: str, subject: str | None = None) -> str | None:
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
        sentence = _fix_article(model, f"{subject} {fact}.")
        scored.append((_wording_score(tenant_id, model, sentence), sentence))
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


def _is_naming_word(word: str) -> bool:
    """Whether a word names a thing, so an article belongs before it."""
    try:
        from app.services.nlp import get_nlp

        doc = get_nlp()(word)
    except Exception:
        return False
    if not doc or not doc[0].pos_:
        return False
    return doc[0].pos_ in {"NOUN", "PROPN"}


def _insert_articles(sentence: str, document) -> str:
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
    except Exception:
        return sentence
    return _insert_articles(sentence, doc)


def _repair_with_document(sentence: str, document) -> str:
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
    except Exception:
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
    if tokens[0] in {"a", "an", "the"} and len(tokens) > 1:
        first = tokens[1]
    else:
        first = tokens[0]
    return first == word or first.startswith(word + "'") or first.startswith(word + "s")

