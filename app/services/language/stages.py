"""Stage questions, tests, and the dictionary words added by the trainer."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from app.services.language import answering, runtime
from app.services.language.curriculum import (
    _EXAM_PASS_RATE,
    _GRADE_IDS,
    STAGES,
)
from app.services.language.teaching import _ensure_grade_allowed

TestResolver = Callable[[str], tuple[str, str, str] | None]


def start_test(tenant_id: int | None, stage_id: str | None = None, kind: str = "trainer") -> dict:
    """Start a trainer-led test, a mock, or the final exam."""
    if kind not in {"trainer", "mock", "exam"}:
        raise ValueError("Choose a test type of trainer, mock or exam.")
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        next_stage = answering.current_stage(state)
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


def _answer_for_test(
    question: str, state: dict, stage: str, tenant_id: int | None, resolver: TestResolver | None
) -> tuple[str, str, str]:
    """The same answer the learner gives elsewhere, with the source beside it."""
    if resolver is not None:
        resolved = resolver(question)
        if resolved:
            return resolved
    taught = answering.answer_from_state(question, state, stage, tenant_id)
    if taught is not None:
        return taught.answer, taught.source, taught.detail
    return answering.NOT_LEARNED, "not_learned", answering.source_detail("not_learned")


def ask_test_question(
    tenant_id: int | None,
    question: str,
    source_resolver: TestResolver | None = None,
) -> dict:
    """Ask the learner a question during a grade test."""
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
        answer, source, source_detail = _answer_for_test(
            question, state, test["stage"], tenant_id, source_resolver
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


def _decide_result(kind: str, passed: bool, pass_rate: float) -> bool:
    """Whether this test advances the grade.

    A mock is practice and never advances. A final exam is decided by its
    measured pass rate, not by who is watching. A trainer-led test keeps the
    trainer's own decision.
    """
    if kind == "mock":
        return False
    if kind == "exam":
        return pass_rate >= _EXAM_PASS_RATE
    return passed


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
        passed = _decide_result(kind, passed, pass_rate)
        if passed and test["stage"] not in state["passed_stages"]:
            state["passed_stages"].append(test["stage"])
        result = {
            "stage": test["stage"],
            "kind": kind,
            "questions_asked": test["questions_asked"],
            "answers_passed": test["answers_passed"],
            "needs_learning": test["needs_learning"],
            "pass_rate": round(pass_rate, 2),
            "required_pass_rate": _EXAM_PASS_RATE if kind == "exam" else None,
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
