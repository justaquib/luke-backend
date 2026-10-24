"""The trainer's verdicts on answers, and the corrections they carry.

A correction is only counted once Luke answers the corrected question right
afterwards, so the number says what the model kept, not what it was told.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.services import brain_paths, vector_store
from app.services.brain_rungs import answer_matches

LOGGER = logging.getLogger(__name__)

_LOCK = threading.RLock()


@dataclass(frozen=True)
class Feedback:
    """One trainer verdict on one answer."""

    question: str
    answer: str
    correct: bool
    correction: str = ""
    learned: bool | None = None


def feedback_path(tenant_id: int | None) -> Path:
    return brain_paths.brain_dir() / f"feedback-{brain_paths.tenant_key(tenant_id)}.json"


def load_feedback(tenant_id: int | None) -> list[dict[str, Any]]:
    path = feedback_path(tenant_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_feedback(tenant_id: int | None, entries: list[dict[str, Any]]) -> None:
    path = feedback_path(tenant_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries[-500:], ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as error:
        LOGGER.warning("The feedback could not be saved: %s", error)


def record_feedback(tenant_id: int | None, feedback: Feedback) -> dict[str, Any]:
    """Remember whether an answer was right, and what the right answer was."""
    with _LOCK:
        entries = load_feedback(tenant_id)
        entries.append(
            {
                "question": feedback.question.strip()[:600],
                "answer": feedback.answer.strip()[:1200],
                "correct": bool(feedback.correct),
                "correction": feedback.correction.strip()[:1200],
                "learned": feedback.learned,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        save_feedback(tenant_id, entries)
    return {
        "entries": len(entries),
        "correct": sum(1 for entry in entries if entry.get("correct")),
        "learned_corrections": learned_corrections(tenant_id),
    }


def learned_corrections(tenant_id: int | None) -> int:
    """How many corrections Luke has answered correctly after being taught."""
    return sum(1 for entry in load_feedback(tenant_id) if entry.get("learned"))


def mark_correction_learned(tenant_id: int | None, question: str, correction: str) -> bool:
    """Mark the correction as kept, when the assistant now answers it right.

    The entry that was kept is the last one for this question and this
    correction, so a correction that is taught twice is not counted twice.
    """
    with _LOCK:
        entries = load_feedback(tenant_id)
        for entry in reversed(entries):
            if entry.get("question") == question and entry.get("correction") == correction:
                entry["learned"] = True
                save_feedback(tenant_id, entries)
                return True
    return False


def verify_correction(tenant_id: int | None, question: str, correction: str) -> dict[str, Any]:
    """Ask the corrected question again and check that Luke keeps the correction."""
    answer = ""
    try:
        hits = vector_store.search_faq(question, n_results=1, tenant_id=tenant_id)
        if hits and hits[0].get("distance") is not None:
            answer = hits[0].get("answer", "")
    except (OSError, RuntimeError, ValueError) as error:
        LOGGER.warning("The correction could not be checked: %s", error)
        answer = ""
    return {
        "learned": answer_matches(answer, correction),
        "answer": answer,
    }
