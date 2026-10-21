"""Stage questions, tests, and the dictionary words added by the trainer."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime

from app.services import dictionary
from app.services.language import runtime
from app.services.language.alphabet import _answer_alphabet_question
from app.services.language.basics import (
    _answer_counting_question,
    _answer_number_sequence_question,
    _answer_number_spelling_question,
    answer_assistant_age,
    answer_current_date_time,
)
from app.services.language.curriculum import (
    _EXAM_PASS_RATE,
    _GRADE_IDS,
    FOUNDATION_LESSON_CARDS,
    STAGES,
    _allowed_grades,
    _foundation_sentences_for_stage,
    _foundation_words_for_stage,
    _tenant_words_for_stage,
)
from app.services.language.matching import _answer_learned_lesson
from app.services.language.patterns import (
    _NEXT_WORD_REQUEST,
    _SENTENCE_REQUEST,
)
from app.services.language.teaching import _ensure_grade_allowed
from app.services.language.writing import _answer_from_sentence_examples


def _asked_word(question: str) -> str | None:
    q = question.strip().lower().rstrip("?.! ")
    patterns = (
        r"what\s+does\s+(?:the\s+word\s+)?([a-z][a-z'’-]*)(?:\s+mean)?",
        r"what\s+is\s+(?:the\s+meaning\s+of\s+)?([a-z][a-z'’-]*)",
        r"meaning\s+of\s+([a-z][a-z'’-]*)",
        r"define\s+([a-z][a-z'’-]*)",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, q)
        if match:
            return match.group(1).casefold()
    return None


def _answer_letter_question(question: str, alphabet: str) -> str | None:
    normalized = question.strip().lower().rstrip("?.! ")
    match = re.fullmatch(
        r"(?:what\s+)?(?:letter\s+)?comes\s+(after|before)\s+(?:the\s+)?(?:letter\s+)?([a-z])|"
        r"what\s+letter\s+comes\s+(after|before)\s+([a-z])",
        normalized,
    )
    if match:
        direction = match.group(1) or match.group(3)
        letter = match.group(2) or match.group(4)
        index = alphabet.find(letter)
        target = index + (1 if direction == "after" else -1)
        if index >= 0 and 0 <= target < len(alphabet):
            return f"The letter {direction} {letter.upper()} is {alphabet[target].upper()}."
    if dictionary.is_dictionary_question(question) and re.search(
        r"\b(?:start|begin|first\s+letter)\b", question, re.I
    ):
        return dictionary.answer_dictionary_question(question)
    return None


def _basic_stage_answer(question: str) -> tuple[str, str] | None:
    age_answer = answer_assistant_age(question)
    if age_answer:
        return age_answer, "assistant_profile"
    current_date_time = answer_current_date_time(question)
    if current_date_time:
        return current_date_time, "learner"
    for answer in (
        _answer_number_spelling_question(question),
        _answer_counting_question(question),
        _answer_number_sequence_question(question),
        _answer_alphabet_question(question),
    ):
        if answer:
            return answer, "learner"
    return None


def _answer_letters_stage(question: str, state: dict) -> tuple[str, str]:
    answer = _answer_letter_question(question, state["letters"])
    if answer:
        return answer, "learner"
    return "I haven't learned an answer to that in the Letters stage yet.", "not_learned"


def _word_answer(question: str, state: dict, stage: str) -> str | None:
    word = _asked_word(question)
    if not word:
        return None
    for item in [
        *_tenant_words_for_stage(state, stage),
        *_foundation_words_for_stage(stage),
    ]:
        if item["word"].casefold() == word:
            return f"{item['word'].capitalize()} means {item['meaning']}."
    return None


def _answer_words_stage(question: str, state: dict) -> tuple[str, str]:
    answer = _word_answer(question, state, "nursery")
    if answer:
        return answer, "learner"
    return (
        "I haven't learned that word in the Words stage yet. Look it up in the offline "
        "dictionary and teach it to me.",
        "not_learned",
    )


def _sentence_pattern_answer(question: str, state: dict) -> str | None:
    next_match = _NEXT_WORD_REQUEST.search(question.strip())
    if next_match:
        prefix = (next_match.group(1) or next_match.group(2) or "").strip(" \t\r\n'\".,?!")
        options = runtime._model(state, "nursery").predict_next(prefix)
        if options:
            return f"From my sentence lessons, after “{prefix}” I predict “{options[0]['word']}”."
    sentence_match = _SENTENCE_REQUEST.search(question.strip())
    if sentence_match:
        requested = re.sub(r"[^A-Za-z'’-].*$", "", sentence_match.group(1)).strip(" '’-")
        if requested:
            answer = runtime._model(state, "nursery").generate_sentence(seed=requested)
            if answer:
                return answer
    return _answer_from_sentence_examples(
        question,
        [
            *(item["text"] for item in state["lessons"]),
            *_foundation_sentences_for_stage("nursery"),
        ],
    )


def _answer_sentences_stage(question: str, state: dict) -> tuple[str, str]:
    answer = _sentence_pattern_answer(question, state)
    if answer:
        return answer, "learner"
    return "I haven't learned a sentence pattern that answers that yet.", "not_learned"


def _grade_sentence_answer(question: str, state: dict, stage: str) -> tuple[str, str] | None:
    sentence_match = _SENTENCE_REQUEST.search(question.strip())
    if sentence_match:
        requested = re.sub(r"[^A-Za-z'’-].*$", "", sentence_match.group(1)).strip(" '’-")
        answer = (
            runtime._model(state, stage).generate_sentence(seed=requested) if requested else None
        )
        if answer:
            return answer, "sentence_patterns"
    next_match = _NEXT_WORD_REQUEST.search(question.strip())
    if next_match:
        prefix = (next_match.group(1) or next_match.group(2) or "").strip(" \t\r\n'\".,?!")
        options = runtime._model(state, stage).predict_next(prefix)
        if options:
            return (
                f"From my grade lessons, after “{prefix}” I predict “{options[0]['word']}”.",
                "sentence_patterns",
            )
    return None


def _answer_grade_stage(question: str, state: dict, stage: str) -> tuple[str, str]:
    answer = _word_answer(question, state, stage)
    if answer:
        return answer, "learner"
    generated = _grade_sentence_answer(question, state, stage)
    if generated:
        return generated
    allowed_grades = _allowed_grades(stage)
    grade_lessons = [
        item
        for item in [*state["lesson_cards"], *FOUNDATION_LESSON_CARDS]
        if item.get("grade_id", "nursery") in allowed_grades
    ]
    answer = _answer_learned_lesson(question, grade_lessons)
    if answer:
        return answer, "learner"
    return "I haven't learned an answer to that grade's lessons yet.", "not_learned"


def _answer_stage_question(question: str, state: dict, stage: str) -> tuple[str, str]:
    basic = _basic_stage_answer(question)
    if basic:
        return basic
    if stage == "letters":
        return _answer_letters_stage(question, state)
    if stage == "words":
        return _answer_words_stage(question, state)
    if stage == "sentences":
        return _answer_sentences_stage(question, state)
    if stage in _GRADE_IDS:
        return _answer_grade_stage(question, state, stage)
    answer = _answer_learned_lesson(question, [*state["lesson_cards"], *FOUNDATION_LESSON_CARDS])
    if answer:
        return answer, "learner"
    return "I haven't learned an answer to the Nursery lessons yet.", "not_learned"


def start_test(tenant_id: int | None, stage_id: str | None = None, kind: str = "trainer") -> dict:
    """Start a trainer-led test, a mock, or the final exam."""
    if kind not in {"trainer", "mock", "exam"}:
        raise ValueError("Choose a test type of trainer, mock or exam.")
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        next_stage = next(
            (item["id"] for item in STAGES if item["id"] not in state["passed_stages"]), "complete"
        )
        stage = stage_id or next_stage
        if stage == "complete":
            return {"state": runtime.get_state(tenant_id), "message": "All grade tests are passed."}
        if stage not in _GRADE_IDS:
            raise ValueError("Choose a valid grade for this test.")
        order = [item["id"] for item in STAGES]
        if next_stage != "complete" and order.index(stage) > order.index(next_stage):
            raise ValueError("Pass the previous grade before testing this grade.")
        if kind == "exam" and stage in state["passed_stages"]:
            raise ValueError("This grade has already been passed.")
        state["active_test"] = {
            "stage": stage,
            "kind": kind,
            "questions_asked": 0,
            "answers_passed": 0,
            "needs_learning": 0,
            "awaiting_score": False,
            "last_question": "",
            "last_answer": "",
            "last_source": "",
        }
        runtime._save(tenant_id, state)
        return {"state": runtime.get_state(tenant_id), "kind": kind}


def ask_test_question(
    tenant_id: int | None,
    question: str,
    source_resolver: Callable[[str], tuple[str, str, str] | None] | None = None,
) -> dict:
    question = question.strip()
    if not question:
        raise ValueError("Enter a question for the learner.")
    if len(question) > 1000:
        raise ValueError("Keep a test question under 1,000 characters.")
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        test = state.get("active_test")
        if not test:
            raise ValueError("Start a trainer-led test before asking a question.")
        if test["awaiting_score"]:
            raise ValueError("Score the learner's last answer before asking another question.")
        answer = answer_assistant_age(question)
        source, source_detail = ("assistant_profile", "Assistant birthday") if answer else ("", "")
        if not answer:
            answer = answer_current_date_time(question)
            if answer:
                source, source_detail = "learner", "Date & Time lessons"
        if not answer and source_resolver:
            resolved = source_resolver(question)
            if resolved:
                answer, source, source_detail = resolved
        if not answer:
            answer, source = _answer_stage_question(question, state, test["stage"])
            source_detail = (
                "Learner curriculum" if source in ("learner", "sentence_patterns") else ""
            )
        test["questions_asked"] += 1
        test["last_question"] = question
        test["last_answer"] = answer
        test["last_source"] = source
        test["last_source_detail"] = source_detail
        test["awaiting_score"] = True
        runtime._save(tenant_id, state)
        return {
            "answer": answer,
            "source": source,
            "source_detail": source_detail,
            "state": runtime.get_state(tenant_id),
        }


def score_test_answer(tenant_id: int | None, passed: bool) -> dict:
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        test = state.get("active_test")
        if not test or not test["awaiting_score"]:
            raise ValueError("Ask a test question before scoring an answer.")
        if passed:
            test["answers_passed"] += 1
        else:
            test["needs_learning"] += 1
        test["awaiting_score"] = False
        runtime._save(tenant_id, state)
        return {"state": runtime.get_state(tenant_id)}


def finish_test(tenant_id: int | None, passed: bool) -> dict:
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        test = state.get("active_test")
        if not test:
            raise ValueError("Start a trainer-led test before scoring the stage.")
        if test["awaiting_score"]:
            raise ValueError("Score the learner's last answer first.")
        if test["questions_asked"] == 0:
            raise ValueError("Ask at least one question before scoring this stage.")
        kind = test.get("kind", "trainer")
        asked = max(test["questions_asked"], 1)
        pass_rate = test["answers_passed"] / asked
        # A mock test is practice only, and a final exam needs a strong score.
        if kind == "mock":
            passed = False
        elif kind == "exam":
            passed = passed and pass_rate >= _EXAM_PASS_RATE
        if passed and test["stage"] not in state["passed_stages"]:
            state["passed_stages"].append(test["stage"])
        result = {
            "stage": test["stage"],
            "kind": kind,
            "questions_asked": test["questions_asked"],
            "answers_passed": test["answers_passed"],
            "needs_learning": test["needs_learning"],
            "pass_rate": round(pass_rate, 2),
            "passed": passed,
            "created_at": datetime.now(UTC).isoformat(),
        }
        state["test_history"].append(result)
        state["test_history"] = state["test_history"][-200:]
        state["active_test"] = None
        runtime._save(tenant_id, state)
        return {"result": result, "state": runtime.get_state(tenant_id)}


def add_dictionary_word(
    tenant_id: int | None, word: str, meaning: str, stage: str = "nursery"
) -> dict:
    word, meaning = word.strip().lower(), meaning.strip()
    if not word or not meaning:
        raise ValueError("Choose a dictionary word with a definition first.")
    if stage not in _GRADE_IDS:
        raise ValueError("Choose a curriculum grade for this dictionary word.")
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        _ensure_grade_allowed(state, stage)
        now = datetime.now(UTC).isoformat()
        existing = next(
            (item for item in state["word_cards"] if item["word"].casefold() == word.casefold()),
            None,
        )
        if existing:
            existing["meaning"] = meaning
            existing["grade_id"] = stage
            existing["created_at"] = now
        else:
            state["word_cards"].append(
                {"word": word, "meaning": meaning, "grade_id": stage, "created_at": now}
            )
            state["word_cards"] = state["word_cards"][-1000:]
        runtime._save(tenant_id, state)
        return runtime.get_state(tenant_id)
