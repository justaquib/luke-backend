"""Finding, checking, and changing the tenant's lessons.

The teaching itself lives in `teaching.py`. This module holds what the
learner already has: the lessons it can answer with, the ones that repeat
each other, and the ways a lesson is edited or removed.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.config import settings
from app.services.language import runtime
from app.services.language.curriculum import (
    _DUPLICATE_QUESTION_THRESHOLD,
    _GRADE_IDS,
    FOUNDATION_LESSON_CARDS,
    STAGES,
    _allowed_grades,
)
from app.services.language.helpers import (
    _content_terms,
    _foundation_lesson_id,
    _normalize_phrase,
    _parse_lesson_pairs,
    _parse_word_pairs,
    _same_words,
    _stem,
    _word_by_word_similarity,
)
from app.services.language.patterns import _QUESTION_STOP_WORDS
from app.services.language.runtime import parse_subjects, subject_of


def _existing_lesson_cards(state: dict, stage: str) -> list[dict]:
    cards = [card for card in state.get("lesson_cards", []) if card.get("question")]
    if stage in _GRADE_IDS:
        for lesson in FOUNDATION_LESSON_CARDS:
            if lesson["grade_id"] == stage:
                cards.append(
                    {
                        "question": lesson["question"],
                        "answer": lesson["answer"],
                        "subject": lesson.get("subject", ""),
                        "grade_id": stage,
                        "kind": "starter",
                    }
                )
    return cards


def _lesson_duplicates(
    pairs: list[tuple[str, str]], cards: list[dict], stage: str
) -> tuple[list[dict], list[dict], list[dict]]:
    exact: list[dict] = []
    same_question: list[dict] = []
    similar: list[dict] = []
    for question, answer in pairs:
        question_key = _normalize_phrase(question)
        answer_key = _normalize_phrase(answer)
        best: tuple[float, dict] | None = None
        for card in cards:
            card_question = card.get("question", "")
            # The question is the same when the words are the same, word by word.
            if question_key == _normalize_phrase(card_question) or _same_words(
                question, card_question
            ):
                entry = {
                    "question": card_question,
                    "answer": card.get("answer", ""),
                    "subject": card.get("subject", ""),
                    "grade_id": card.get("grade_id", stage),
                    "kind": card.get("kind", "trainer"),
                }
                # The answer is the same when its words are the same, word by word.
                if answer_key == _normalize_phrase(entry["answer"]) or _same_words(
                    answer, entry["answer"]
                ):
                    exact.append(entry)
                else:
                    same_question.append(entry)
                continue
            score = _word_by_word_similarity(question, card_question)
            if score >= _DUPLICATE_QUESTION_THRESHOLD and (best is None or score > best[0]):
                best = (score, card)
        if best is not None:
            score, card = best
            similar.append(
                {
                    "question": card.get("question", ""),
                    "answer": card.get("answer", ""),
                    "subject": card.get("subject", ""),
                    "grade_id": card.get("grade_id", stage),
                    "kind": card.get("kind", "trainer"),
                    "similarity": round(score, 2),
                }
            )
    return exact, same_question, similar


def _word_duplicates(word_pairs: list[tuple[str, str]], state: dict) -> list[dict]:
    duplicate_words: list[dict] = []
    for word, meaning in word_pairs:
        for card in state.get("word_cards", []):
            if card.get("word", "").lower() == word and _normalize_phrase(
                card.get("meaning", "")
            ) == _normalize_phrase(meaning):
                duplicate_words.append({"word": word, "meaning": meaning})
                break
    return duplicate_words


def check_duplicate_lessons(
    tenant_id: int | None, text: str, stage: str = "nursery", subject: str = "General"
) -> dict:
    """Report lessons that already cover the question and answer being taught.

    The report separates an exact duplicate (same question and same answer),
    a repeated question that already has a different answer, and close
    paraphrases that are worth a "did you mean this?" hint.
    """
    raw = (text or "").strip()
    with runtime._LOCK:
        state = runtime._read(tenant_id)
    pairs = _parse_lesson_pairs(raw)
    word_pairs: list[tuple[str, str]] = []
    if not pairs and stage == "words":
        word_pairs = _parse_word_pairs(raw)
    cards = _existing_lesson_cards(state, stage)
    exact, same_question, similar = _lesson_duplicates(pairs, cards, stage)
    duplicate_words = _word_duplicates(word_pairs, state)
    return {
        "pairs": [{"question": question, "answer": answer} for question, answer in pairs],
        "exact": exact,
        "same_question": same_question,
        "similar": similar,
        "duplicate_words": duplicate_words,
        "duplicate": bool(exact or duplicate_words),
    }


def get_lesson_card(tenant_id: int | None, lesson_id: str) -> dict | None:
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        return next((item for item in state["lesson_cards"] if item.get("id") == lesson_id), None)


def all_lesson_cards(tenant_id: int | None) -> list[dict]:
    """Every question and answer this tenant can answer with.

    That is the lessons taught in the language learner plus the starter
    lessons of the stages the tenant has reached.
    """
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        current_stage = next(
            (item["id"] for item in STAGES if item["id"] not in state["passed_stages"]),
            "complete",
        )
        cards = [
            {**item, "subject": subject_of(item), "subjects": parse_subjects(subject_of(item))}
            for item in state["lesson_cards"]
            if item.get("question")
        ]
        seen = {(item.get("question") or "").strip().lower() for item in cards}
        for lesson in FOUNDATION_LESSON_CARDS:
            if lesson["grade_id"] not in _allowed_grades(current_stage):
                continue
            question = lesson["question"].strip()
            key = question.lower()
            if key in seen:
                continue
            seen.add(key)
            cards.append(
                {
                    "id": _foundation_lesson_id(question),
                    "question": question,
                    "answer": lesson["answer"],
                    "subject": subject_of(lesson),
                    "subjects": parse_subjects(subject_of(lesson)),
                    "grade_id": lesson["grade_id"],
                    "kind": "starter",
                }
            )
        return cards


def tenant_ids_with_lessons() -> list[int | None]:
    """Tenant ids that have a saved language learner file."""
    keys: list[int | None] = []
    data_dir = runtime._DATA_DIR
    if not data_dir.is_dir():
        return keys
    for path in sorted(data_dir.glob("tenant-*.json")):
        try:
            tenant_key = int(path.stem.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        keys.append(None if tenant_key == 0 else tenant_key)
    return keys


def update_lesson_answer(
    tenant_id: int | None,
    lesson_id: str,
    answer: str,
    *details: object,
    **options: object,
) -> dict:
    if len(details) > 3:
        raise TypeError("update_lesson_answer takes at most six arguments")
    question = details[0] if len(details) > 0 else options.get("question")
    confidence = details[1] if len(details) > 1 else options.get("confidence")
    subject = details[2] if len(details) > 2 else options.get("subject")
    answer = answer.strip()
    if not answer:
        raise ValueError("An answer is required.")
    cleaned_question = str(question).strip() if question is not None else None
    if cleaned_question is not None and not cleaned_question:
        raise ValueError("A question is required.")
    topics = parse_subjects(str(subject)) if subject is not None else None
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        lesson = next((item for item in state["lesson_cards"] if item.get("id") == lesson_id), None)
        if not lesson:
            raise LookupError("Learner lesson not found.")
        lesson["answer"] = answer
        if cleaned_question is not None:
            lesson["question"] = cleaned_question
        if confidence is not None:
            lesson["confidence"] = _clamp_confidence(float(confidence))
        if topics is not None:
            lesson["subjects"] = topics
            lesson["subject"] = ", ".join(topics)
        lesson["updated_at"] = datetime.now(UTC).isoformat()
        runtime._save(tenant_id, state)
        return runtime.get_state(tenant_id)


def _clamp_confidence(value: float | None) -> float:
    if value is None:
        return settings.base_answer_confidence
    return max(0.0, min(settings.max_answer_confidence, round(float(value), 3)))


def find_lesson_by_answer(tenant_id: int | None, answer: str) -> dict | None:
    """The lesson that already stores this answer, if the learner has one."""
    target = _normalize_phrase(answer)
    if not target:
        return None
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        for lesson in state["lesson_cards"]:
            if _normalize_phrase(lesson.get("answer", "")) == target:
                return dict(lesson)
    return None


def match_lesson_by_words(question: str, tenant_id: int | None) -> dict | None:
    """The lesson that covers the question by its words.

    It is the fallback for a question that is asked in other words, so the
    assistant still finds what it was taught.
    """
    terms = {_stem(token) for token in _content_terms(question, _QUESTION_STOP_WORDS)}
    if not terms:
        return None
    cards = all_lesson_cards(tenant_id)
    # How often each word is used in a lesson. A rare word says much more
    # about the question than a word that is in every lesson.
    frequency: dict[str, int] = {}
    card_words: list[set[str]] = []
    for card in cards:
        words = {
            _stem(token) for token in _content_terms(card.get("question", ""), _QUESTION_STOP_WORDS)
        }
        card_words.append(words)
        for word in words:
            frequency[word] = frequency.get(word, 0) + 1

    def weight(word: str) -> float:
        return 1.0 / (1.0 + frequency.get(word, 0))

    best: tuple[float, dict] | None = None
    for card, words in zip(cards, card_words, strict=True):
        if not words:
            continue
        shared = terms & words
        if not shared:
            continue
        score = sum(weight(word) for word in shared) / max(
            sum(weight(word) for word in terms | words),
            1e-6,
        )
        if best is None or score > best[0]:
            best = (score, card)
    return best[1] if best and best[0] >= 0.3 else None


def find_known_answer(tenant_id: int | None, answer: str) -> dict | None:
    """The lesson that already stores this answer, taught or built in.

    It tells the trainer that a confirmed answer is already known, so it is
    not copied into the Trained answers a second time.
    """
    target = _normalize_phrase(answer)
    if not target:
        return None
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        for lesson in state["lesson_cards"]:
            if _normalize_phrase(lesson.get("answer", "")) == target:
                return {"kind": "trainer", "card": dict(lesson)}
    for lesson in FOUNDATION_LESSON_CARDS:
        if _normalize_phrase(lesson.get("answer", "")) == target:
            return {"kind": "starter", "card": dict(lesson)}
    return None


def bump_lesson_confidence(tenant_id: int | None, answer: str, question: str = "") -> dict | None:
    """Raise the confidence of the lesson that already holds this answer.

    A confirmed answer that the learner already knows is not copied anywhere.
    It only becomes more trusted.
    """
    target = _normalize_phrase(answer)
    if not target:
        return None
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        lesson = next(
            (
                item
                for item in state["lesson_cards"]
                if _normalize_phrase(item.get("answer", "")) == target
            ),
            None,
        )
        if lesson is None:
            return None
        current = float(lesson.get("confidence", settings.base_answer_confidence))
        lesson["confidence"] = round(
            min(settings.max_answer_confidence, current + settings.confidence_step), 3
        )
        lesson["confirmed_at"] = datetime.now(UTC).isoformat()
        lesson["confirmations"] = int(lesson.get("confirmations", 0)) + 1
        runtime._save(tenant_id, state)
        return dict(lesson)


def remove_lesson(tenant_id: int | None, lesson_id: str) -> dict:
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        remaining = [item for item in state["lesson_cards"] if item.get("id") != lesson_id]
        if len(remaining) == len(state["lesson_cards"]):
            raise LookupError("Learner lesson not found.")
        state["lesson_cards"] = remaining
        runtime._save(tenant_id, state)
        return runtime.get_state(tenant_id)
