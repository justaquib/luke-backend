"""Send a new answer that disagrees with an approved one to the trainer.

The same question can be answered twice with different words. When the new
answer says materially less or more than the approved one, the trainer decides
which answer stands; nothing is overwritten silently and the entry is kept in
the inbox with both answers shown.
"""

from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import Faq, PendingQuestion
from app.services.pending_questions import record_pending

# An answer this close in wording to the approved one is a rephrasing, not a
# contradiction. The measured value is written in docs/07-model-measurements.md.
SAME_ANSWER_SIMILARITY = 0.8
_WORD = re.compile(r"[a-z0-9']+")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.casefold()))


def same_question(left: str, right: str) -> bool:
    """The same words mean the same question, in any order or punctuation."""
    left_words = _words(left)
    return bool(left_words) and left_words == _words(right)


def answer_similarity(left: str, right: str) -> float:
    """How much of the wording two answers share, from 0 to 1."""
    left_words = _words(left)
    right_words = _words(right)
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def conflicting_faq(db: Session, tenant_id: int | None, question: str, answer: str) -> Faq | None:
    """The approved answer to the same question whose wording disagrees."""
    conditions = [Faq.status == "approved"]
    if tenant_id is not None:
        conditions.append(Faq.tenant_id == tenant_id)
    for faq in db.execute(select(Faq).where(*conditions)).scalars():
        if same_question(faq.question, question) and (
            answer_similarity(faq.answer, answer) < SAME_ANSWER_SIMILARITY
        ):
            return faq
    return None


def note_for(approved: Faq, answer: str) -> str:
    """The inbox note: the two answers side by side, in the trainer's words."""
    return (
        f'New answer: "{answer.strip()}" — approved answer: "{approved.answer.strip()}". '
        "Choose the answer that stands."
    )


def flag_conflict(
    db: Session, tenant_id: int | None, question: str, answer: str
) -> PendingQuestion | None:
    """Put a contradictory answer in the trainer inbox; none for a rephrasing."""
    approved = conflicting_faq(db, tenant_id, question, answer)
    if approved is None:
        return None
    return record_pending(db, question, tenant_id, note_for(approved, answer))
