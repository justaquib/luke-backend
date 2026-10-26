"""The state of the brain, and the level its measurements reach.

The state reads what is trained; the level reads what the model can do. The
displayed level never moves on its own: the measured level can run ahead of
it, and the trainer approves the promotion.
"""

from __future__ import annotations

import logging
from typing import Any

from app.services import (
    brain_generality,
    brain_material,
    brain_paths,
    brain_rung_results,
    model_settings,
    training_progress,
)
from app.services.brain_feedback import load_feedback
from app.services.brain_heads import (
    answer_confidence,
    feel,
    should_ask_back,
)
from app.services.brain_levels import (
    BENCHMARK_NOTE,
    BENCHMARK_TITLE,
    LUKE_LEVELS,
    SEED_LEVEL,
    approved_level,
    level_entry,
    level_order,
    level_title,
    level_value,
    next_level_entry,
    training_volume_level,
)
from app.services.brain_roadmap import ROADMAP
from app.services.brain_training import is_training
from app.services.heads import LinearHead
from app.services.neural import engine_for

LOGGER = logging.getLogger(__name__)

__all__ = [
    "answer_confidence",
    "feel",
    "knowledge_of",
    "model_level",
    "should_ask_back",
    "state",
]


def knowledge_of(tenant_id: int | None) -> dict[str, Any]:
    """What the school has taught: lessons, grades and the current one."""
    knowledge: dict[str, Any] = {"lessons": 0, "passed_grades": 0, "items": 0}
    try:
        from app.services import language_logic

        knowledge["lessons"] = len(language_logic.all_lesson_cards(tenant_id))
        model_state = language_logic.get_state(tenant_id)
        knowledge["passed_grades"] = len(model_state.get("passed_stages", []))
        knowledge["items"] = knowledge["lessons"]
        knowledge["stage"] = model_state.get("current_stage_id")
    except (OSError, ValueError, RuntimeError) as error:
        LOGGER.warning("The school state could not be read: %s", error)
    return knowledge


def _measured_level(rungs: list[dict[str, Any]], score: float) -> str:
    """The level the measurements qualify, before any approval.

    The rungs are taken in order from the seed, so a gap stops the climb,
    and the top rung answers a question it was never taught: the stretch a
    model reaches when its own words hold up without being taught.
    """
    ladder_ids = [str(entry["level"]) for entry in LUKE_LEVELS]
    by_id = {str(rung["rung"]): rung for rung in rungs}
    measured = SEED_LEVEL
    for level_id in ladder_ids[1:]:
        rung = by_id.get(level_id)
        if rung is None or not rung["probes"] or rung["rate"] < 0.6:
            break
        measured = level_id
    if score >= 0.5:
        measured = ladder_ids[-1]
    return measured


def _measure(tenant_id: int | None, force: bool) -> dict[str, Any]:
    """Everything the level reads, measured once."""
    brain_state = state(tenant_id)
    knowledge = knowledge_of(tenant_id)
    capabilities = brain_material.model_capabilities(tenant_id, brain_state, knowledge)
    counted = brain_material.coverage(tenant_id)
    knowledge["items"] = counted.get("chunks", 0)
    counted["lessons"] = knowledge.get("lessons", 0)
    counted["passed_grades"] = knowledge.get("passed_grades", 0)
    counted["current_stage"] = knowledge.get("stage", "nursery")
    rungs = brain_rung_results.rung_results(
        tenant_id, force=force, brain_state=brain_state, knowledge=knowledge
    )
    generality = brain_generality.generality_score(tenant_id)
    return {
        "brain_state": brain_state,
        "knowledge": knowledge,
        "capabilities": capabilities,
        "generality": generality,
        "coverage": counted,
        "rungs": rungs,
        "measured": _measured_level(rungs, generality["score"]),
    }


def _ladder(model_name: str) -> list[dict[str, Any]]:
    """The whole road to 1.0.0, with the open levels and the locked ones."""
    return [{**entry, "title": level_title(str(entry["level"]), model_name)} for entry in ROADMAP]


def _next_requirement(
    level: str,
    measured: str,
    following: dict[str, Any] | None,
    next_detail: str,
) -> str:
    """What stands between the displayed level and the next one."""
    if level_order(measured) > level_order(level):
        return f"Approve {measured} to make it the displayed level."
    if following is None:
        return "It has reached the highest level."
    if not following.get("open"):
        return f"{following['level']} is locked. No test has been written for it yet."
    return next_detail or f"Pass {following['level']} to move up."


def _level_view(
    tenant_id: int | None,
    measured: str,
    measured_data: dict[str, Any],
) -> dict[str, Any]:
    """The level the page shows, with what stands in the way of the next one."""
    capabilities = measured_data["capabilities"]
    counted = measured_data["coverage"]
    knowledge = measured_data["knowledge"]
    generality = measured_data["generality"]
    rungs = measured_data["rungs"]
    settings = model_settings.read(tenant_id)
    model_name = settings.name
    level = approved_level(tenant_id)
    current = level_entry(level, model_name)
    following = next_level_entry(
        measured if level_order(measured) > level_order(level) else level,
        model_name,
        ROADMAP,
    )
    steps = brain_material.next_steps(
        measured_data["brain_state"], knowledge, rungs, counted, model_name
    )
    next_detail = next(
        (step["detail"] for step in steps if step["title"].startswith(f"Pass the {model_name}")),
        steps[0]["detail"] if steps else "",
    )
    top_level = str(ROADMAP[-1]["level"])
    return {
        "level": level,
        "measured_level": measured,
        "approval_ready": level_order(measured) > level_order(level),
        "level_value": round(level_value(level), 4),
        "name": current["name"],
        "title": current["title"],
        "description": current["description"],
        "reference": current["reference"],
        "next_level": following["title"] if following else None,
        "next_requirement": _next_requirement(level, measured, following, next_detail),
        "capabilities": capabilities,
        "max_level": top_level,
        "ladder_size": len(ROADMAP),
        "levels": _ladder(model_name),
        "training_volume": training_volume_level(knowledge.get("items", 0)),
        "rungs": rungs,
        "next_steps": steps,
        "coverage": counted,
        "generality": generality["score"],
        "generality_cap": generality["cap"],
        "generality_probes": generality["probes"],
        "frontier_level": 6.0,
        "frontier_title": BENCHMARK_TITLE,
        "frontier_note": BENCHMARK_NOTE,
        "ladder_share": round(level_value(level) / max(level_value(top_level), 1e-9), 4),
        "model_name": model_name,
        "model_start_level": settings.start_level,
    }


def model_level(tenant_id: int | None, force: bool = False) -> dict[str, Any]:
    """The level the brain has reached, measured by what it can do.

    A local model is held to the same scale as a frontier one, and it is
    capped by how far it can answer on its own: a retriever cannot reach the
    levels that need real reasoning, however much it was taught. What it
    measures as and what is displayed are two different numbers: the level
    only moves when the trainer approves the promotion.
    """
    measured_data = _measure(tenant_id, force)
    return _level_view(tenant_id, measured_data["measured"], measured_data)


def state(tenant_id: int | None) -> dict[str, Any]:
    """What the brain currently knows about itself.

    The vocabulary here is the word model's: it is the one live token list,
    so the numbers and the ids always mean the same words.
    """
    from app.services import vocabulary as vocabulary_service

    feedback = load_feedback(tenant_id)
    emotion = LinearHead.load(brain_paths.brain_path(tenant_id, "emotion"))
    clarify = LinearHead.load(brain_paths.brain_path(tenant_id, "clarify"))
    correct = LinearHead.load(brain_paths.brain_path(tenant_id, "correct"))
    intent = LinearHead.load(brain_paths.brain_path(tenant_id, "intent"))
    canonical = vocabulary_service.canonical(tenant_id)
    engine = engine_for(tenant_id).state()
    words = engine["words"]
    return {
        "trained": bool(words["trained"]),
        "vocabulary": len(canonical),
        "words": words,
        "words_ready": bool(words["trained"]),
        "engine": engine,
        "backend": words["backend"],
        "backend_reason": words["backend_reason"],
        "training": is_training(tenant_id),
        "progress": training_progress.read(tenant_id),
        "emotions": emotion.labels if emotion else [],
        "emotion_ready": bool(emotion and emotion.fitted),
        "clarify_ready": bool(clarify and clarify.fitted),
        "confidence_ready": bool(correct and correct.fitted),
        "intent_ready": bool(intent and intent.fitted),
        "intents": intent.labels if intent else [],
        "feedback_entries": len(feedback),
        "correct_answers": sum(1 for entry in feedback if entry.get("correct")),
        "wrong_answers": sum(1 for entry in feedback if not entry.get("correct")),
        "corrections": sum(1 for entry in feedback if entry.get("correction")),
        "learned_corrections": sum(1 for entry in feedback if entry.get("learned")),
    }
