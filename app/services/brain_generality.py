"""How far the brain can answer beyond what it was taught.

A frontier model answers questions it has never seen. This one is measured on
the same probes, so the page can say where the local model really stops.
"""

from __future__ import annotations

from typing import Any

from app.services.brain_rungs import GENERALITY_PROBES


def generality_score(tenant_id: int | None) -> dict[str, Any]:
    """How far the brain can answer beyond what it was taught."""
    from app.services import language_logic

    answered = 0
    probes: list[dict[str, Any]] = []
    for probe in GENERALITY_PROBES:
        try:
            result = language_logic.preview_answer(probe, tenant_id)
        except (OSError, ValueError, RuntimeError):
            result = {"answer": "", "source": "not_learned"}
        source = result.get("source", "not_learned")
        # A dictionary lookup is a lookup, not general reasoning.
        general = source not in {"not_learned", "dictionary"}
        answered += 1 if general else 0
        probes.append({"question": probe, "answered": general, "source": source})
    score = answered / max(len(GENERALITY_PROBES), 1)
    return {"score": round(score, 2), "cap": _cap(score), "probes": probes}


def _cap(score: float) -> int:
    """The level the generality score allows."""
    if score >= 0.75:
        return 6
    if score >= 0.5:
        return 5
    if score >= 0.25:
        return 4
    return 3
