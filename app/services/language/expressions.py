"""The expressions the trainer teaches for opening, closing, pausing and resuming.

A greeting, a goodbye, a pause and a resume belong to a conversation, not to a
lesson. The trainer writes the phrase a person uses and the reply Luke gives
back, so the assistant answers that phrase exactly and the training material
carries it. A taught expression is checked before the built-in small talk, and
it never replaces a taught lesson: an expression only matches the whole
message, so a question that happens to start with the same words still reaches
its lesson.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services import brain_paths

OPEN = "open"
CLOSE = "close"
PAUSE = "pause"
RESUME = "resume"

CATEGORIES = (OPEN, CLOSE, PAUSE, RESUME)

CATEGORY_LABELS = {
    OPEN: "Opening",
    CLOSE: "Closing",
    PAUSE: "Pausing",
    RESUME: "Resuming",
}

CATEGORY_DESCRIPTIONS = {
    OPEN: "How Luke answers when someone starts a conversation.",
    CLOSE: "How Luke answers when someone ends a conversation.",
    PAUSE: "How Luke answers when someone steps away for a moment.",
    RESUME: "How Luke answers when someone comes back to the conversation.",
}

MAX_ITEMS = 120
MAX_PHRASE_CHARS = 80
MAX_REPLY_CHARS = 240


class ExpressionError(ValueError):
    """The expression cannot be saved, with a reason the trainer can act on."""


@dataclass(frozen=True)
class Expression:
    """One phrase the trainer taught and the reply Luke gives it."""

    id: str
    category: str
    phrase: str
    reply: str

    @property
    def key(self) -> str:
        """The phrase as it is compared, without case or punctuation."""
        return normalize(self.phrase)

    def as_dict(self) -> dict[str, str]:
        """The expression as the API reports it."""
        return {
            "id": self.id,
            "category": self.category,
            "label": CATEGORY_LABELS[self.category],
            "phrase": self.phrase,
            "reply": self.reply,
        }


def normalize(text: str) -> str:
    """The words of a turn, without case, punctuation or extra spacing."""
    lowered = str(text or "").lower().replace("’", "").replace("'", "")
    cleaned = re.sub(r"[^a-z0-9 ]", " ", lowered)
    return " ".join(cleaned.split())


def store_path(tenant_id: int | None) -> Path:
    """Where a workspace's expressions are kept."""
    return brain_paths.brain_dir() / f"expressions-{brain_paths.tenant_key(tenant_id)}.json"


def load(tenant_id: int | None) -> list[Expression]:
    """The taught expressions, oldest first, or nothing when none were saved."""
    try:
        stored = json.loads(store_path(tenant_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = stored.get("items") if isinstance(stored, dict) else None
    if not isinstance(rows, list):
        return []
    items: list[Expression] = []
    for row in rows:
        expression = _read_row(row)
        if expression is not None:
            items.append(expression)
    return items


def add(tenant_id: int | None, category: str, phrase: str, reply: str) -> Expression:
    """Teach one expression, refusing a phrase that is already taught."""
    items = load(tenant_id)
    expression = _build(None, category, phrase, reply)
    _refuse_duplicate(items, expression, None)
    if len(items) >= MAX_ITEMS:
        raise ExpressionError(f"A workspace can teach at most {MAX_ITEMS} expressions.")
    _save(tenant_id, [*items, expression])
    return expression


def update(
    tenant_id: int | None,
    expression_id: str,
    category: str,
    phrase: str,
    reply: str,
) -> Expression:
    """Change a taught expression in place."""
    items = load(tenant_id)
    current = _find(items, expression_id)
    expression = _build(expression_id, category, phrase, reply)
    _refuse_duplicate(items, expression, expression_id)
    _save(tenant_id, [expression if item.id == current.id else item for item in items])
    return expression


def remove(tenant_id: int | None, expression_id: str) -> None:
    """Forget one taught expression."""
    items = load(tenant_id)
    _find(items, expression_id)
    _save(tenant_id, [item for item in items if item.id != expression_id])


def reply_for(tenant_id: int | None, question: str) -> str | None:
    """The taught reply when the whole message is one taught phrase."""
    asked = normalize(question)
    if not asked:
        return None
    for expression in load(tenant_id):
        if expression.key == asked:
            return expression.reply
    return None


def pairs(tenant_id: int | None) -> list[tuple[str, str]]:
    """The taught expressions as training pairs."""
    return [(item.phrase, item.reply) for item in load(tenant_id)]


def style_lines(tenant_id: int | None) -> list[str]:
    """The example replies the prompt gives the model, two per category."""
    items = load(tenant_id)
    lines: list[str] = []
    for category in CATEGORIES:
        chosen = [item for item in items if item.category == category][:2]
        if not chosen:
            continue
        examples = "; ".join(f'"{item.phrase}" → "{item.reply}"' for item in chosen)
        lines.append(f"{CATEGORY_LABELS[category]}: {examples}")
    return lines


def view(tenant_id: int | None) -> dict[str, Any]:
    """The expressions and the categories the dashboard reads."""
    items = load(tenant_id)
    return {
        "items": [item.as_dict() for item in items],
        "categories": [
            {
                "id": category,
                "label": CATEGORY_LABELS[category],
                "description": CATEGORY_DESCRIPTIONS[category],
                "count": sum(1 for item in items if item.category == category),
            }
            for category in CATEGORIES
        ],
        "limits": {"items": MAX_ITEMS, "phrase": MAX_PHRASE_CHARS, "reply": MAX_REPLY_CHARS},
    }


def _read_row(row: Any) -> Expression | None:
    """One stored row, or nothing when it cannot be read."""
    if not isinstance(row, dict):
        return None
    try:
        return _build(
            str(row.get("id") or "") or None,
            str(row.get("category") or ""),
            str(row.get("phrase") or ""),
            str(row.get("reply") or ""),
        )
    except ExpressionError:
        return None


def _build(
    expression_id: str | None,
    category: str,
    phrase: str,
    reply: str,
) -> Expression:
    """A checked expression, with a new id when it does not have one."""
    return Expression(
        id=expression_id or uuid.uuid4().hex[:12],
        category=_clean_category(category),
        phrase=_clean_text(phrase, "phrase", MAX_PHRASE_CHARS),
        reply=_clean_text(reply, "reply", MAX_REPLY_CHARS),
    )


def _clean_category(value: str) -> str:
    """One of the four conversation categories."""
    category = str(value or "").strip().lower()
    if category not in CATEGORIES:
        raise ExpressionError("Choose opening, closing, pausing or resuming.")
    return category


def _clean_text(value: str, label: str, limit: int) -> str:
    """A phrase or reply, with tidy spacing and no unusable input."""
    text = " ".join(str(value or "").split())
    if not text:
        raise ExpressionError(f"Write the {label} before saving.")
    if len(text) > limit:
        raise ExpressionError(f"The {label} can be at most {limit} characters.")
    return text


def _find(items: list[Expression], expression_id: str) -> Expression:
    """One taught expression by id."""
    for item in items:
        if item.id == expression_id:
            return item
    raise ExpressionError("That expression is no longer taught.")


def _refuse_duplicate(
    items: list[Expression],
    expression: Expression,
    editing: str | None,
) -> None:
    """A phrase is taught once, in one category."""
    for item in items:
        if item.id != editing and item.key == expression.key:
            raise ExpressionError(
                f'"{item.phrase}" is already taught as {CATEGORY_LABELS[item.category].lower()}.'
            )


def _save(tenant_id: int | None, items: list[Expression]) -> None:
    """Write every expression atomically."""
    path = store_path(tenant_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    payload = {
        "items": [
            {
                "id": item.id,
                "category": item.category,
                "phrase": item.phrase,
                "reply": item.reply,
            }
            for item in items
        ]
    }
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
