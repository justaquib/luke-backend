"""The tenant's saved learner state and the small model built from it."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from uuid import uuid4

from app.config import settings
from app.services.language.curriculum import (
    _FOUNDATION_QUESTIONS,
    _GRADE_IDS,
    FOUNDATION_LESSON_CARDS,
    STAGES,
    _allowed_grades,
    _foundation_lessons_for_stage,
    _foundation_sentences_for_stage,
    _foundation_words_for_stage,
    _grammar_lessons,
    _tenant_words_for_stage,
)
from app.services.neural import engine_for
from app.services.neural.serving import ServingWord
from app.services.neural.tokens import is_a_word, tokenize

_DATA_DIR = Path(__file__).resolve().parents[3] / ".language_model"
_LOCK = threading.RLock()
_CACHE: dict[int, dict] = {}

LOGGER = logging.getLogger(__name__)


def data_dir() -> Path:
    """The folder the learner files and the trained models are kept in."""
    return _DATA_DIR


def parse_subjects(value: str | None) -> list[str]:
    """The topics of a lesson, one by one.

    A trainer can write "Language, Poems" and the lesson then belongs to both
    topics. The same words in a different case are the same topic.
    """
    if not value:
        return ["General"]
    topics: list[str] = []
    for part in str(value).replace("\n", ",").split(","):
        topic = " ".join(part.split())[:80]
        if not topic:
            continue
        if not any(topic.casefold() == existing.casefold() for existing in topics):
            topics.append(topic)
    return topics or ["General"]


def subject_of(card: dict) -> str:
    """The topics of a card as one readable string."""
    topics = card.get("subjects")
    if isinstance(topics, list) and topics:
        return ", ".join(str(topic) for topic in topics)
    return ", ".join(parse_subjects(card.get("subject")))


def _tenant_key(tenant_id: int | None) -> int:
    return tenant_id if tenant_id is not None else 0


def _path(tenant_id: int | None) -> Path:
    return _DATA_DIR / f"tenant-{_tenant_key(tenant_id)}.json"


def _empty_state(key: int) -> dict:
    return {
        "tenant_id": key,
        "lessons": [],
        "letters": "abcdefghijklmnopqrstuvwxyz",
        "word_cards": [],
        "lesson_cards": [],
        "passed_stages": [],
        "active_test": None,
        "test_history": [],
        "accepted_duplicates": [],
    }


def _stored_state(path: Path, key: int) -> dict | None:
    """The saved state of a tenant, or nothing when there is none to read."""
    if not path.is_file():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.warning("the learner state of tenant %s could not be read: %s", key, error)
        return None
    return stored if isinstance(stored, dict) else None


def _merge_state(state: dict, stored: dict) -> None:
    state["lessons"] = stored.get("lessons", [])
    for name in (
        "letters",
        "word_cards",
        "lesson_cards",
        "passed_stages",
        "active_test",
        "test_history",
        "accepted_duplicates",
    ):
        if name in stored:
            state[name] = stored[name]


def _repair_lessons(lesson_cards: list[dict]) -> bool:
    """Give every lesson an id and topics, and say whether anything changed."""
    changed = False
    for lesson in lesson_cards:
        if not lesson.get("id"):
            lesson["id"] = uuid4().hex
            changed = True
        topics = lesson.get("subjects")
        if not isinstance(topics, list) or not topics:
            # A lesson taught before topics were split keeps the subject it had.
            lesson["subjects"] = parse_subjects(lesson.get("subject"))
            lesson["subject"] = ", ".join(lesson["subjects"])
            changed = True
    return changed


def _read(tenant_id: int | None) -> dict:
    key = _tenant_key(tenant_id)
    if key in _CACHE:
        return _CACHE[key]
    state = _empty_state(key)
    stored = _stored_state(_path(tenant_id), key)
    if stored is not None:
        _merge_state(state, stored)
    _CACHE[key] = state
    if _repair_lessons(state["lesson_cards"]):
        _save(tenant_id, state)
    active_test = state.get("active_test")
    if active_test and active_test.get("stage") not in _GRADE_IDS:
        active_test["stage"] = "nursery"
    return state


def _save(tenant_id: int | None, state: dict) -> None:
    path = _path(tenant_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _model(state: dict, stage_id: str | None = None) -> ServingWord:
    """The neural word model, scoped to the words the current stage knows."""
    examples = _examples(state, stage_id)
    allowed = {token for sentence in examples for token in tokenize(sentence) if is_a_word(token)}
    return engine_for(state.get("tenant_id")).word_serving(allowed)


def _examples(state: dict, stage_id: str | None = None) -> list[str]:
    """Every sentence the word model of a stage is built from."""
    if stage_id is None:
        stage_id = next(
            (stage["id"] for stage in STAGES if stage["id"] not in state["passed_stages"]),
            "complete",
        )
    foundation_words = _foundation_words_for_stage(stage_id)
    foundation_lessons = _foundation_lessons_for_stage(stage_id)
    allowed_grades = _allowed_grades(stage_id)
    tenant_lessons = [
        lesson
        for lesson in state["lesson_cards"]
        if lesson.get("grade_id", "nursery") in allowed_grades
    ]
    word_cards = [*foundation_words, *_tenant_words_for_stage(state, stage_id)]
    # All the grammar lessons are taught to the writing model, so it knows how
    # a sentence is put together even before the learner reaches that grade.
    grammar_lessons = _grammar_lessons()
    lesson_sentences = [
        f"{item['question']} {item['answer']}"
        for item in [*grammar_lessons, *foundation_lessons, *tenant_lessons]
    ]
    # Every answer is also a sentence on its own, so the model learns how a
    # proper sentence is put together.
    answer_sentences = [
        item["answer"]
        for item in [*grammar_lessons, *foundation_lessons, *tenant_lessons]
        if item.get("answer")
    ]
    sentences = _foundation_sentences_for_stage(stage_id)
    return [
        " ".join(state["letters"]),
        *sentences,
        *answer_sentences,
        *(f"{item['word']} means {item['meaning']}." for item in word_cards),
        *(
            f"{item['word']}: {item['dictionary_meaning']}."
            for item in foundation_words
            if item.get("dictionary_meaning")
            and item["dictionary_meaning"].casefold() != item["meaning"].casefold()
        ),
        *(lesson["text"] for lesson in state["lessons"]),
        *lesson_sentences,
    ]


def learning_examples(tenant_id: int | None, stage_id: str | None = None) -> list[str]:
    """Every sentence the learner's word model is built from, for the trainer."""
    return _examples(_read(tenant_id), stage_id)


def _public_test(test: dict | None) -> dict | None:
    if not test:
        return None
    return {
        key: test.get(key)
        for key in (
            "stage",
            "kind",
            "questions_asked",
            "answers_passed",
            "needs_learning",
            "awaiting_score",
            "last_question",
            "last_answer",
            "last_source",
            "last_source_detail",
        )
    }


def _stage_counts(state: dict) -> dict[str, int]:
    return {
        stage["id"]: sum(
            1
            for lesson in state["lesson_cards"]
            if lesson.get("grade_id", "nursery") == stage["id"]
        )
        + sum(1 for lesson in FOUNDATION_LESSON_CARDS if lesson["grade_id"] == stage["id"])
        for stage in STAGES
    }


def _test_counts(state: dict) -> tuple[dict[str, int], dict[str, int]]:
    attempt_counts = {stage["id"]: 0 for stage in STAGES}
    pass_counts = {stage["id"]: 0 for stage in STAGES}
    for result in state["test_history"]:
        grade_id = result.get("stage")
        if grade_id in attempt_counts:
            attempt_counts[grade_id] += 1
            pass_counts[grade_id] += int(bool(result.get("passed")))
    return attempt_counts, pass_counts


def _state_lessons(state: dict, available_grades: set[str], available_lessons: list[dict]) -> list:
    return [
        *[
            {
                "kind": "starter"
                if (lesson.get("question") or "").strip().lower() in _FOUNDATION_QUESTIONS
                else "trainer",
                **lesson,
                "subject": subject_of(lesson),
                "subjects": parse_subjects(subject_of(lesson)),
                "confidence": float(lesson.get("confidence", settings.base_answer_confidence)),
            }
            for lesson in reversed(state["lesson_cards"])
            if lesson.get("grade_id", "nursery") in available_grades
        ],
        *available_lessons,
    ]


def get_state(tenant_id: int | None, prefix: str = "") -> dict:
    from app.services import vocabulary as vocabulary_service

    with _LOCK:
        state = _read(tenant_id)
        passed = set(state["passed_stages"])
        current_stage = next(
            (stage["id"] for stage in STAGES if stage["id"] not in passed), "complete"
        )
        model = _model(state, current_stage)
        available_grades = _allowed_grades(current_stage)
        available_sentences = _foundation_sentences_for_stage(current_stage)
        available_words = _foundation_words_for_stage(current_stage)
        available_lessons = _foundation_lessons_for_stage(current_stage)
        lessons = [{"text": text, "kind": "foundation"} for text in available_sentences] + [
            {"text": lesson["text"], "kind": "trainer", "created_at": lesson.get("created_at")}
            for lesson in reversed(state["lessons"])
        ]
        stage_counts = _stage_counts(state)
        attempt_counts, pass_counts = _test_counts(state)
        return {
            "model_name": _model_name(model),
            "description": _model_description(model),
            "backend": model.backend,
            "backend_reason": model.backend_reason,
            "stage": next(
                (item["title"] for item in STAGES if item["id"] == current_stage),
                "All stages passed",
            ),
            "current_stage_id": current_stage,
            "passed_stages": [stage["id"] for stage in STAGES if stage["id"] in passed],
            "stages": [
                {
                    **stage,
                    "lesson_count": stage_counts[stage["id"]],
                    "completed": stage["id"] in passed,
                    "attempt_count": attempt_counts[stage["id"]],
                    "pass_count": pass_counts[stage["id"]],
                }
                for stage in STAGES
            ],
            "letters": state["letters"],
            "word_cards": [*_tenant_words_for_stage(state, current_stage), *available_words],
            "lesson_cards": _state_lessons(state, available_grades, available_lessons),
            "active_test": _public_test(state.get("active_test")),
            "test_history": list(reversed(state["test_history"][-20:])),
            "foundation_lessons": len(available_sentences),
            "trainer_lessons": len(state["lessons"]),
            "sentence_count": model.sentence_count,
            "token_count": model.token_count,
            "vocabulary_size": len(vocabulary_service.canonical(tenant_id)),
            "word_vocabulary_size": len(model.vocabulary),
            "example_count": getattr(model, "example_count", model.sentence_count),
            "transition_count": model.transition_count,
            "next_words": model.predict_next(prefix) if prefix.strip() else [],
            "examples": lessons[:80],
            "example_total": len(lessons),
            "generated_example": model.generate_sentence(),
        }


def _model_name(model: ServingWord) -> str:
    return "Neural Word Model" if model.backend == "neural" else "No word model yet"


def _model_description(model: ServingWord) -> str:
    if model.backend == "neural":
        return (
            "A from-scratch transformer trained on this learner's own words. It predicts the "
            "next word from the words before it, and it was measured on held-out sentences."
        )
    return (
        "This learner's word model is trained from their own lessons and sentences. "
        f"{model.backend_reason.capitalize()}."
    )


def vocabulary_page(tenant_id: int | None, offset: int = 0, limit: int = 100) -> dict:
    """The words the answering model can write with, one page at a time.

    This is the transformer's vocabulary: the one live token list. A long
    list is not sent all at once, so it can be scrolled and the next words
    arrive when they are needed.
    """
    from app.services import vocabulary as vocabulary_service

    words = list(vocabulary_service.canonical(tenant_id).tokens)
    offset = max(0, offset)
    limit = min(200, max(1, limit))
    page = words[offset : offset + limit]
    return {
        "words": page,
        "total": len(words),
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(page) < len(words),
    }
