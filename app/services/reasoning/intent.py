"""The kind of question that was asked, learned instead of written down.

The assistant used to read a question with a table of patterns. The table is
still here, but it is only the answer for a brain that has not been trained
yet: once the intent head is trained, the kind of the question is read from
the meaning of the question, the same way the emotion head and the clarify
head read theirs.

The head is trained from the tenant's own lessons. The kind of a lesson comes
from the instruction data, where a lesson is a poem, a list of things or a
fact. The other kinds come from the shape of the question, and the shape only
names the training data: it never answers a question.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.services.heads import LinearHead

# The kinds of question the assistant can be asked.
KINDS: tuple[str, ...] = (
    "poem",
    "list",
    "fact",
    "calculation",
    "definition",
    "letter",
    "counting",
    "spelling",
    "time",
    "lookup",
)

# The kind that is used when nothing was trained and nothing matched.
DEFAULT_KIND = "lookup"

# The shape of a question, used to name the training data and to answer
# before the head is trained.
_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "calculation",
        re.compile(
            r"(\d+\s*[-+*/x×÷]\s*\d+)"
            r"|(\b(?:plus|minus|times|divided|multiplied|square root|percent)\b)"
        ),
    ),
    (
        "definition",
        re.compile(r"\bwhat (?:is|are|does|do)\b|\bdefine\b|\bmeaning of\b", re.IGNORECASE),
    ),
    ("letter", re.compile(r"\bletter\b|\balphabet\b", re.IGNORECASE)),
    ("counting", re.compile(r"\bhow many\b|\bcount\b", re.IGNORECASE)),
    ("spelling", re.compile(r"\bspell\b|\bwritten in words\b", re.IGNORECASE)),
    ("time", re.compile(r"\bwhat time\b|\bwhat day\b|\bwhat is the date\b", re.IGNORECASE)),
)

# The least sure the head may be and still be believed.
MIN_CONFIDENCE = 0.45


def shape_kind(question: str) -> str:
    """The kind of a question, read from its shape."""
    for name, pattern in _SHAPES:
        if pattern.search(question or ""):
            return name
    return DEFAULT_KIND


def _lesson_kind(card: dict[str, Any]) -> str:
    """The kind of a lesson: a poem, a list of things, or a fact."""
    from app.services.language.instructions import lesson_kind

    kind = lesson_kind(card)
    if kind in {"poem", "list"}:
        return kind
    shape = shape_kind(str(card.get("question") or ""))
    return shape if shape != DEFAULT_KIND else "fact"


def labelled_questions(cards: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """The questions of the lessons, each with the kind of question it is."""
    labelled: list[tuple[str, str]] = []
    for card in cards:
        question = str(card.get("question") or "").strip()
        if not question:
            continue
        labelled.append((question, _lesson_kind(card)))
    return labelled


def train(tenant_id: int | None, cards: list[dict[str, Any]]) -> int:
    """Teach the intent head from the tenant's own lessons."""
    from app.services import embeddings
    from app.services.brain_paths import brain_path

    labelled = labelled_questions(cards)
    if len(labelled) < 4:
        return 0
    try:
        features = embeddings.embed_texts([question for question, _ in labelled])
    except Exception:
        return 0
    if len(features) != len(labelled):
        return 0
    # The head only knows the kinds the tenant's own material has.
    labels = sorted({kind for _, kind in labelled})
    if len(labels) < 2:
        return 0
    head = LinearHead(labels)
    targets = [labels.index(kind) for _, kind in labelled]
    if not head.fit(features, targets):
        return 0
    head.save(brain_path(tenant_id, "intent"))
    return len(labelled)


def classify(tenant_id: int | None, question: str) -> tuple[str, str]:
    """The kind of the question, and where the answer came from.

    The trained head reads the meaning of the question. The shape of the
    question is only used when there is no head to ask yet.
    """
    from app.services import embeddings
    from app.services.brain_paths import brain_path

    head = LinearHead.load(brain_path(tenant_id, "intent"))
    if head is not None:
        try:
            features = embeddings.embed_texts([question])
        except Exception:
            features = []
        if features:
            label, confidence, _ = head.predict(features)
            if confidence >= MIN_CONFIDENCE:
                return label, "trained"
    return shape_kind(question), "shape"


def path_for(tenant_id: int | None) -> Path:
    """Where the intent head of this tenant is kept."""
    from app.services.brain_paths import brain_path

    return brain_path(tenant_id, "intent")
