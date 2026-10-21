"""Turning words and lessons into sentences, and matching taught examples."""

from __future__ import annotations

import re

from app.services.language import runtime
from app.services.language.curriculum import (
    FOUNDATION_LESSON_CARDS,
    FOUNDATION_SENTENCES_BY_GRADE,
    _foundation_words_for_stage,
    _tenant_words_for_stage,
)
from app.services.language.grammar import (
    _FUNCTION_WORDS,
    _repair_sentence,
    _sentence_starts_with,
    _wording_score,
)
from app.services.language.helpers import _content_terms, _normalize_phrase, _token_matches
from app.services.language.lessons import all_lesson_cards
from app.services.language.patterns import _NEXT_WORD_REQUEST, _SENTENCE_REQUEST
from app.services.tiny_language_model import tokenize

_LESSON_QUESTION_STATEMENT = re.compile(
    r"^(?:what|which|who)\s+is\s+(.+?)\s*\??$",
    re.IGNORECASE,
)
_LESSON_WHICH_SUBJECT = re.compile(
    r"^which\s+(.+?)\s+(has|have|is|are|was|were|does|do|did|can|could|will|would|should)\s+"
    r"(.+?)\s*\??$",
    re.IGNORECASE,
)
_LESSON_NAME_REQUEST = re.compile(
    r"^name\s+(the|a|an)\s+(.+?)\s*\??$",
    re.IGNORECASE,
)
_LESSON_NAME_LIST = re.compile(
    r"^name\s+(?:some|all|the\s+list\s+of)\s+(.+?)\s*\??$",
    re.IGNORECASE,
)


def _taught_word_meanings(tenant_id: int | None) -> dict[str, str]:
    """The words this tenant has learned and the meaning it was given."""
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        current_stage = next(
            (item["id"] for item in runtime.STAGES if item["id"] not in state["passed_stages"]),
            "complete",
        )
        meanings: dict[str, str] = {}
        for item in [
            *_tenant_words_for_stage(state, current_stage),
            *_foundation_words_for_stage(current_stage),
        ]:
            word = (item.get("word") or "").strip().lower()
            meaning = (item.get("meaning") or "").strip()
            if word and meaning:
                meanings[word] = meaning
        for item in state.get("word_cards", []):
            word = (item.get("word") or "").strip().lower()
            meaning = (item.get("meaning") or "").strip()
            if word and meaning:
                meanings[word] = meaning
        return meanings


def _is_describing_word(meaning: str) -> bool:
    """Words like red, blue, or happy describe a noun instead of naming one."""
    text = (meaning or "").strip().lower()
    return (
        text.startswith("the color") or text.startswith("feeling") or text.startswith("a feeling")
    )


def _taught_sentences(tenant_id: int | None) -> list[str]:
    """The sentence examples the learner has been taught."""
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        sentences = [item.get("text", "") for item in state.get("lessons", [])]
    for grade in FOUNDATION_SENTENCES_BY_GRADE.values():
        sentences.extend(grade)
    return [sentence for sentence in sentences if sentence]


def _taught_answers(tenant_id: int | None) -> list[str]:
    """The answers of the starter lessons, which read like sentences."""
    return [lesson["answer"] for lesson in FOUNDATION_LESSON_CARDS if lesson.get("answer")]


def _tenant_lesson_answers(tenant_id: int | None) -> list[str]:
    """The answers the trainer taught, which read like sentences as well."""
    try:
        cards = all_lesson_cards(tenant_id)
    except Exception:
        return []
    return [
        card["answer"] for card in cards if card.get("kind") != "starter" and card.get("answer")
    ]


def _clean_phrase(text: str) -> str:
    """A phrase without its end mark."""
    return (text or "").strip().rstrip(".!?").strip()


def _statement_from_lesson(question: str, answer: str) -> str | None:
    """Turn a lesson question into a statement that holds the answer."""
    answer = re.sub(r"\s*\([^)]*\)", "", answer or "").strip().rstrip(".!?").strip()
    if not answer or len(answer.split()) > 5:
        return None
    text = (question or "").strip()
    match = _LESSON_QUESTION_STATEMENT.match(text)
    if match:
        rest = _clean_phrase(match.group(1))
        return f"{answer} is {rest}." if rest else None
    match = _LESSON_WHICH_SUBJECT.match(text)
    if match:
        verb = match.group(2).strip()
        rest = _clean_phrase(match.group(3))
        return f"{answer} {verb} {rest}." if rest else None
    match = _LESSON_NAME_REQUEST.match(text)
    if match:
        article = match.group(1).strip()
        rest = _clean_phrase(match.group(2))
        return f"{answer} is {article} {rest}." if rest else None
    return None


def _list_statement_for_word(tenant_id: int | None, word: str) -> str | None:
    """A word taken from a list lesson becomes a sentence of its own."""
    target = _normalize_phrase(word)
    if not target:
        return None
    try:
        cards = all_lesson_cards(tenant_id)
    except Exception:
        return None
    for card in cards:
        match = _LESSON_NAME_LIST.match((card.get("question") or "").strip())
        if not match:
            continue
        category = _clean_phrase(match.group(1))
        answer = card.get("answer") or ""
        parts = re.split(r"\bare\b", answer, maxsplit=1)
        items_text = parts[1] if len(parts) > 1 else answer
        items = [item.strip(" .!") for item in items_text.split(",")]
        if not any(_normalize_phrase(item) == target for item in items):
            continue
        singular = (
            category[:-1] if category.endswith("s") and not category.endswith("ss") else category
        )
        return f"{word.strip().capitalize()} is a {singular}."
    return None


def _lesson_statement_for_word(tenant_id: int | None, word: str) -> str | None:
    """The sentence the learner's lesson makes for this word."""
    target = _normalize_phrase(word)
    if not target:
        return None
    try:
        cards = all_lesson_cards(tenant_id)
    except Exception:
        return None
    for card in cards:
        card_answer = card.get("answer", "")
        if _normalize_phrase(card_answer) != target:
            continue
        statement = _statement_from_lesson(card.get("question", ""), card_answer)
        if statement:
            return statement
    return None


def _meaning_sentence(
    cleaned: list[str], meanings: dict[str, str], describing: list[str], nouns: list[str], others: list[str]
) -> str | None:
    if not nouns:
        return None
    described = [f"the {noun} is {meanings[noun]}" for noun in nouns]
    modifiers = describing + others
    if modifiers:
        return _repair_sentence(f"The {' '.join(modifiers)} {nouns[0]} is {meanings[nouns[0]]}.")
    if len(described) == 1:
        return _repair_sentence(f"{described[0][0].upper()}{described[0][1:]}.")
    return _repair_sentence(f"{', and '.join(described).capitalize()}.")


def _sentence_corpus(tenant_id: int | None) -> list[str]:
    return [
        sentence
        for group in (
            _taught_sentences(tenant_id),
            _taught_answers(tenant_id),
            _tenant_lesson_answers(tenant_id),
        )
        for sentence in group
    ]


def _about_sentence(content: list[str], corpus: list[str]) -> str | None:
    if len(content) != 1:
        return None
    about = [
        sentence
        for sentence in corpus
        if _sentence_starts_with(sentence, content[0])
        and len(sentence.split()) >= 5
        and "/" not in sentence
    ]
    if not about:
        return None
    return _repair_sentence(min(about, key=lambda sentence: len(sentence.split())))


def _best_matching_sentence(
    tenant_id: int | None, content: list[str], corpus: list[str]
) -> str | None:
    matches = [
        sentence
        for sentence in corpus
        if all(any(_token_matches(token, word) for token in tokenize(sentence)) for word in content)
        and sentence.count(",") < 2
    ]
    if not matches:
        return None
    with runtime._LOCK:
        model = runtime._model(runtime._read(tenant_id))
    best = max(
        matches,
        key=lambda sentence: (
            _wording_score(tenant_id, model, sentence),
            -len(sentence.split()),
        ),
    )
    return _repair_sentence(best)


def make_sentence_from_words(
    tenant_id: int | None, words: list[str], strict: bool = False
) -> str | None:
    """Build a simple grammatical sentence from the given words."""
    cleaned = []
    for word in words:
        for part in str(word).replace(",", " ").split():
            cleaned.append(part.strip().lower().strip("?.!"))
    cleaned = [word for word in cleaned if word and word not in {".", "!", "?"}]
    if not cleaned:
        return None
    content = [word for word in cleaned if word not in _FUNCTION_WORDS]
    if not content:
        content = cleaned
    meanings = _taught_word_meanings(tenant_id)
    describing = [
        word for word in cleaned if word in meanings and _is_describing_word(meanings[word])
    ]
    nouns = [word for word in cleaned if word in meanings and word not in describing]
    others = [word for word in cleaned if word not in meanings and word not in _FUNCTION_WORDS]
    sentence = _meaning_sentence(cleaned, meanings, describing, nouns, others)
    if sentence:
        return sentence
    corpus = _sentence_corpus(tenant_id)
    sentence = _about_sentence(content, corpus)
    if sentence:
        return sentence
    if len(content) == 1:
        statement = _lesson_statement_for_word(tenant_id, content[0])
        if statement:
            return statement
    sentence = _best_matching_sentence(tenant_id, content, corpus)
    if sentence:
        return sentence
    if len(content) == 1:
        statement = _list_statement_for_word(tenant_id, content[0])
        if statement:
            return statement
    if others:
        if strict:
            return None
        return f"The {' '.join(others)} is a word I am still learning."
    return None


def make_sentence(tenant_id: int | None, seed: str | None = None) -> str | None:
    if seed:
        statement = _lesson_statement_for_word(tenant_id, seed)
        if statement:
            return statement
    with runtime._LOCK:
        sentence = runtime._model(runtime._read(tenant_id)).generate_sentence(seed=seed)
    return _repair_sentence(sentence) if sentence else sentence


def answer_sentence_request(question: str, tenant_id: int | None) -> str | None:
    match = _SENTENCE_REQUEST.search(question.strip())
    if not match:
        return None
    requested = re.sub(r"[^A-Za-z'’-].*$", "", match.group(1)).strip(" '’-")
    if not requested:
        return "Tell me which word you want me to use in the sentence."
    sentence = make_sentence(tenant_id, requested)
    if sentence:
        return sentence
    return (
        f"I haven't learned a sentence pattern for “{requested}” yet. Add a few example "
        "sentences with that word in Trainer → Language learner, then ask me again."
    )


def answer_next_word_request(question: str, tenant_id: int | None) -> str | None:
    match = _NEXT_WORD_REQUEST.search(question.strip())
    if not match:
        return None
    prefix = (match.group(1) or match.group(2) or "").strip(" \t\r\n'\".,?!")
    words = predict_next(tenant_id, prefix)
    if not words:
        return (
            f"I haven't learned what usually follows “{prefix}” yet. Add a sentence with "
            "those words in Trainer → Language learner."
        )
    options = ", ".join(
        f"{item['word']} ({item['count']} example{'s' if item['count'] != 1 else ''})"
        for item in words[:3]
    )
    return (
        f"After “{prefix}”, my learned examples suggest: {options}. This is a word-pattern "
        "prediction, not a guaranteed continuation."
    )


def predict_next(tenant_id: int | None, prefix: str) -> list[dict[str, int | str]]:
    with runtime._LOCK:
        return runtime._model(runtime._read(tenant_id)).predict_next(prefix)


def _answer_from_sentence_examples(question: str, examples: list[str]) -> str | None:
    """A taught sentence that covers the question."""
    stop_words = {
        "what",
        "who",
        "where",
        "when",
        "which",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "the",
        "a",
        "an",
        "of",
        "for",
        "please",
        "tell",
        "me",
        "in",
        "on",
        "from",
        "about",
        "to",
        "with",
    }
    terms = _content_terms(question, stop_words)
    best: tuple[float, str] | None = None
    for sentence in examples:
        overlap = terms & set(tokenize(sentence))
        if overlap:
            score = len(overlap) / max(len(terms), 1)
            if best is None or score > best[0]:
                best = (score, sentence)
    return best[1] if best and best[0] >= 0.6 else None
