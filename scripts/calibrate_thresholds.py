"""Find the FAQ distance thresholds the saved feedback supports.

Every feedback row is a question the user actually asked and a rating of the
answer they got. A row is a positive when the user liked a FAQ answer and that
FAQ is still the nearest one: the match was right at that distance. A row is a
negative when the nearest FAQ was not the answer the user wanted: either the
served FAQ answer was disliked, or another source answered and was liked.

Distances are cosine distances computed with the same embedding the index
uses, so they are the numbers the rebuilt collections report. The script writes
nothing; it prints the split and the counts behind it. When too few labelled
rows exist it says so instead of choosing a number.

Run with:
    .venv/bin/python scripts/calibrate_thresholds.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MIN_POSITIVES = 3
_MIN_NEGATIVES = 3


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if not left_norm or not right_norm:
        return 1.0
    return 1.0 - dot / (left_norm * right_norm)


def _faqs() -> list[dict[str, Any]]:
    from sqlalchemy import select

    from app.db.database import Faq, SessionLocal

    with SessionLocal() as db:
        return [
            {"question": row.question, "answer": row.answer, "faq_id": row.id}
            for row in db.execute(
                select(Faq).where(Faq.status == "approved").order_by(Faq.id)
            ).scalars()
        ]


def _feedback() -> list[dict[str, Any]]:
    from sqlalchemy import select

    from app.db.database import Feedback, SessionLocal

    with SessionLocal() as db:
        return [
            {
                "question": row.question,
                "answer": row.answer,
                "source": row.source,
                "rating": row.rating,
            }
            for row in db.execute(select(Feedback).order_by(Feedback.id)).scalars()
        ]


def _nearest(
    vector: list[float], faqs: list[dict[str, Any]], vectors: list[list[float]]
) -> tuple[int, float]:
    distances = [_cosine(vector, faq_vector) for faq_vector in vectors]
    index = min(range(len(distances)), key=lambda position: distances[position])
    return index, round(distances[index], 4)


def _labelled() -> dict[str, Any]:
    from app.services.embeddings import embed_texts

    faqs = _faqs()
    feedback = _feedback()
    if not faqs or not feedback:
        return {"faqs": len(faqs), "feedback": len(feedback), "positives": [], "negatives": []}
    faq_vectors = embed_texts([faq["question"] for faq in faqs])
    question_vectors = embed_texts([row["question"] for row in feedback])
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for row, vector in zip(feedback, question_vectors, strict=True):
        index, distance = _nearest(vector, faqs, faq_vectors)
        nearest = faqs[index]
        served = _normal(row["answer"]) == _normal(nearest["answer"])
        if row["rating"] == "like" and row["source"] in {"faq", "intent"} and served:
            positives.append(
                {"question": row["question"], "distance": distance, "why": "liked FAQ answer"}
            )
        elif row["rating"] == "dislike" and served:
            negatives.append(
                {"question": row["question"], "distance": distance, "why": "disliked FAQ answer"}
            )
        elif (
            row["rating"] == "like"
            and row["source"] not in {"faq", "intent"}
            and _normal(row["question"]) != _normal(nearest["question"])
        ):
            # The nearest FAQ is not the question that was asked, so this row
            # says the FAQ answer was rightly not used. An identical question
            # answered elsewhere is a fact about when it was approved, not a
            # distance label, so it is left out.
            negatives.append(
                {
                    "question": row["question"],
                    "distance": distance,
                    "why": f"answered by {row['source']} and liked",
                }
            )
    return {
        "faqs": len(faqs),
        "feedback": len(feedback),
        "positives": positives,
        "negatives": negatives,
    }


def _best_split(
    positives: list[float], negatives: list[float]
) -> tuple[float | None, int, float | None]:
    """The threshold that misclassifies the fewest rows, and its margin."""
    candidates = sorted({*positives, *negatives})
    best: tuple[int, float, float | None] | None = None
    for threshold in candidates:
        correct = sum(1 for value in positives if value <= threshold)
        correct += sum(1 for value in negatives if value > threshold)
        above = [value for value in negatives if value > threshold]
        margin = (min(above) - threshold) if above else threshold
        key = (correct, margin)
        if best is None or key > (best[0], best[1]):
            best = (correct, margin, threshold)
    if best is None:
        return None, 0, None
    threshold = best[2]
    accepted = [value for value in positives if value <= threshold]
    rejected = [value for value in negatives if value > threshold]
    if accepted and rejected:
        suggestion = round((max(accepted) + min(rejected)) / 2, 3)
    elif accepted:
        suggestion = round(max(accepted) + 0.01, 3)
    elif rejected:
        suggestion = round(min(rejected) - 0.01, 3)
    else:
        suggestion = threshold
    return suggestion, best[0], best[1]


def _exact(positives: list[float], negatives: list[float]) -> dict[str, Any]:
    wanted = [row["distance"] for row in positives]
    unwanted = [row["distance"] for row in negatives]
    suggested, correct, margin = _best_split(wanted, unwanted)
    enough = len(wanted) >= _MIN_POSITIVES and len(unwanted) >= _MIN_NEGATIVES
    return {
        "suggested": suggested,
        "accuracy": round(correct / max(len(wanted) + len(unwanted), 1), 4),
        "margin": None if margin is None else round(margin, 4),
        "enough": enough,
        "positives": len(wanted),
        "negatives": len(unwanted),
        "keep": None if suggested is None or not enough else suggested,
    }


def _loose(positives: list[float], negatives: list[float], exact: dict[str, Any]) -> dict[str, Any]:
    disliked = [row["distance"] for row in negatives if row["why"] == "disliked FAQ answer"]
    ceiling = min(disliked) if disliked else None
    floor = max(positives) if positives else None
    if ceiling is not None and floor is not None and floor < ceiling:
        suggested = round((floor + ceiling) / 2, 3)
    elif ceiling is not None:
        suggested = round(ceiling - 0.01, 3)
    elif floor is not None:
        suggested = round(floor + 0.05, 3)
    else:
        suggested = None
    if suggested is not None and exact.get("keep") is not None:
        suggested = max(suggested, round(float(exact["keep"]) + 0.05, 3))
    return {
        "suggested": suggested,
        "floor": None if floor is None else round(floor, 4),
        "ceiling": None if ceiling is None else round(ceiling, 4),
        "disliked": len(disliked),
        "enough": ceiling is not None and bool(positives),
        "keep": suggested if ceiling is not None and positives else None,
    }


def _report() -> dict[str, Any]:
    from app.config import settings

    data = _labelled()
    positives = data["positives"]
    negatives = data["negatives"]
    exact = _exact(positives, negatives)
    loose = _loose([row["distance"] for row in positives], negatives, exact)
    return {
        "schema": "thresholds/v1",
        "taken_at": datetime.now(UTC).isoformat(),
        "feedback": {"faqs": data["faqs"], "rows": data["feedback"]},
        "current": {
            "faq_match_threshold": settings.faq_match_threshold,
            "intent_match_threshold": settings.intent_match_threshold,
        },
        "positives": positives,
        "negatives": negatives,
        "faq_match": exact,
        "intent_match": loose,
    }


def main() -> int:
    print(json.dumps(_report(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
