"""The lessons that repeat each other, and the groups the trainer can act on."""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.services.language import runtime
from app.services.language.curriculum import _DUPLICATE_QUESTION_THRESHOLD
from app.services.language.helpers import (
    _normalize_phrase,
    _word_by_word_similarity,
)

SAME_QUESTION = "same_question"
SAME_ANSWER = "same_answer"
SIMILAR = "similar"


def _card_view(card: dict) -> dict[str, Any]:
    """One lesson, in the shape the learner page shows it."""
    from app.services.language.runtime import parse_subjects, subject_of

    subject = subject_of(card)
    return {
        "id": card.get("id"),
        "question": card.get("question", ""),
        "answer": card.get("answer", ""),
        "subject": subject,
        "subjects": parse_subjects(subject),
        "grade_id": card.get("grade_id", "nursery"),
        "confidence": float(card.get("confidence", settings.base_answer_confidence)),
        "created_at": card.get("created_at"),
    }


def group_key(kind: str, cards: list[dict]) -> str:
    """The name of a group, so the trainer can say "these two are fine"."""
    if not cards:
        return ""
    if kind == SAME_QUESTION:
        return f"question:{_normalize_phrase(cards[0].get('question', ''))}"
    if kind == SAME_ANSWER:
        return f"answer:{_normalize_phrase(cards[0].get('answer', ''))}"
    questions = sorted(_normalize_phrase(card.get("question", "")) for card in cards)
    return f"pair:{questions[0]}|{questions[-1]}"


def accept_duplicate(tenant_id: int | None, key: str) -> dict:
    """Keep every lesson of a group: the trainer says the repeat is on purpose.

    The group is remembered so it is not reported again.
    """
    with runtime._LOCK:
        state = runtime._read(tenant_id)
        accepted = [str(item) for item in state.get("accepted_duplicates", [])]
        if key and key not in accepted:
            accepted.append(key)
        state["accepted_duplicates"] = accepted[-200:]
        runtime._save(tenant_id, state)
        return runtime.get_state(tenant_id)


def _taught_cards(tenant_id: int | None, stage: str) -> list[dict]:
    """The lessons the trainer taught, in the stage they were taught for."""
    with runtime._LOCK:
        state = runtime._read(tenant_id)
    cards = [item for item in state["lesson_cards"] if item.get("question")]
    if stage:
        cards = [card for card in cards if card.get("grade_id", "nursery") == stage]
    return cards


def _same_question_groups(cards: list[dict], used: set[str]) -> list[dict]:
    """The lessons that ask the very same question."""
    grouped: dict[str, list[dict]] = {}
    for card in cards:
        grouped.setdefault(_normalize_phrase(card.get("question", "")), []).append(card)
    groups = []
    for group in grouped.values():
        if len(group) < 2:
            continue
        used.update(str(card.get("id")) for card in group)
        groups.append(
            {
                "kind": SAME_QUESTION,
                "label": "Same question",
                "detail": "These ask the same thing.",
                "cards": [_card_view(card) for card in group],
            }
        )
    return groups


def _same_answer_groups(cards: list[dict], used: set[str]) -> list[dict]:
    """The lessons that give the very same answer."""
    grouped: dict[str, list[dict]] = {}
    for card in cards:
        if str(card.get("id")) in used:
            continue
        grouped.setdefault(_normalize_phrase(card.get("answer", "")), []).append(card)
    groups = []
    for group in grouped.values():
        if len(group) < 2:
            continue
        used.update(str(card.get("id")) for card in group)
        groups.append(
            {
                "kind": SAME_ANSWER,
                "label": "Same answer",
                "detail": "These give the same answer.",
                "cards": [_card_view(card) for card in group],
            }
        )
    return groups


def _similar_groups(cards: list[dict], used: set[str]) -> list[dict]:
    """The lessons that ask the same thing in other words."""
    groups = []
    for card in cards:
        if str(card.get("id")) in used:
            continue
        best: tuple[float, dict] | None = None
        for other in cards:
            if other is card or str(other.get("id")) in used:
                continue
            score = _word_by_word_similarity(card.get("question", ""), other.get("question", ""))
            if score >= _DUPLICATE_QUESTION_THRESHOLD and (best is None or score > best[0]):
                best = (score, other)
        if best is None:
            continue
        score, other = best
        used.update([str(card.get("id")), str(other.get("id"))])
        groups.append(
            {
                "kind": SIMILAR,
                "label": "Asked in other words",
                "detail": f"These look {round(score * 100)}% the same.",
                "cards": [_card_view(card), _card_view(other)],
            }
        )
    return groups


def duplicate_groups(tenant_id: int | None, stage: str = "") -> list[dict]:
    """The lessons that repeat each other, grouped by what repeats.

    A group holds the lessons that ask the same question, give the same
    answer, or ask the same thing in other words. The trainer decides which
    one to keep, which one to improve, and which one to remove. The groups
    the trainer has already accepted are not reported again.
    """
    cards = _taught_cards(tenant_id, stage)
    with runtime._LOCK:
        accepted = {str(item) for item in runtime._read(tenant_id).get("accepted_duplicates", [])}
    used: set[str] = set()
    groups = _same_question_groups(cards, used)
    groups += _same_answer_groups(cards, used)
    groups += _similar_groups(cards, used)
    for group in groups:
        group["key"] = group_key(group["kind"], group["cards"])
    groups = [group for group in groups if group["key"] not in accepted]
    groups.sort(key=lambda group: group["kind"])
    return groups
