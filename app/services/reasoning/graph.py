"""The shape of what the tenant has learned: subjects, lessons and answers."""

from __future__ import annotations

from typing import Any


def knowledge_graph(tenant_id: int | None, subject: str = "", limit: int = 90) -> dict[str, Any]:
    """The shape of what the tenant has learned: subjects, lessons, answers, intents."""
    from app.services import language_logic

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    try:
        cards = language_logic.all_lesson_cards(tenant_id)
    except Exception:
        cards = []

    wanted = (subject or "").strip().lower()
    if wanted:
        cards = [
            card for card in cards if (card.get("subject") or "General").strip().lower() == wanted
        ]

    by_subject: dict[str, int] = {}
    for card in cards:
        subject_name = (card.get("subject") or "General").strip() or "General"
        by_subject[subject_name] = by_subject.get(subject_name, 0) + 1

    for subject_name, count in sorted(by_subject.items(), key=lambda item: -item[1]):
        nodes.append(
            {
                "id": f"subject:{subject_name}",
                "label": subject_name,
                "type": "subject",
                "count": count,
            }
        )

    for card in cards[: max(0, limit)]:
        subject_name = (card.get("subject") or "General").strip() or "General"
        node_id = f"lesson:{card.get('id')}"
        nodes.append(
            {
                "id": node_id,
                "label": card.get("question", ""),
                "type": "lesson",
                "grade": card.get("grade_id", "nursery"),
            }
        )
        edges.append({"source": node_id, "target": f"subject:{subject_name}", "label": "taught in"})

    return {
        "nodes": nodes,
        "edges": edges,
        "subjects": sorted(by_subject, key=lambda item: -by_subject[item]),
        "total": len(cards),
    }
