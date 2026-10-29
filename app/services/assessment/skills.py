"""The eight measured skills, their profiles, and the shape a bank item fits.

A skill is a profile the dashboard can report on. An item kind is how the
answer is decided: the deterministic kinds are scored in code, and a rubric is
scored only by the trainer, so a rubric skill stays unassessed until a review
exists.

A profile is built from outcomes, never from a model's opinion of itself. Too
little evidence is ``unassessed`` with a null score, which is not the same as
zero: a skill nobody has measured must not read as a skill the engine failed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

EVAL_ITEMS_PER_SKILL = 30
TRAIN_ITEMS_PER_SKILL = 15
MIN_STAGE = 1
MAX_STAGE = 3

PASS_MARK = 0.8
STAGE_MIN_ITEMS = 3
CONFIDENCE_HIGH = 20
CONFIDENCE_MEDIUM = 10
STRENGTH_MARK = 0.9
WEAKNESS_MARK = 0.5

SKILL_TITLES: dict[str, str] = {
    "grammar": "Language comprehension and grammar",
    "conversation": "Conversation and instruction following",
    "reading": "Reading comprehension and summarization",
    "creative_writing": "Creative writing, stories and poetry",
    "mathematics": "Mathematics and logical reasoning",
    "programming": "Programming and debugging",
    "general_knowledge": "General knowledge",
    "science": "Scientific and technical understanding",
}
SKILL_IDS: tuple[str, ...] = tuple(SKILL_TITLES)

OPTION = "option"
SPAN = "span"
NUMBER = "number"
LITERAL = "literal"
RULES = "rules"
RUBRIC = "rubric"

DETERMINISTIC_KINDS = frozenset({OPTION, SPAN, NUMBER, LITERAL, RULES})
REVIEW_KINDS = frozenset({RUBRIC})
EXPECTED_KINDS = DETERMINISTIC_KINDS | REVIEW_KINDS


@dataclass(frozen=True)
class Outcome:
    """One item's result as the profile reads it: a score or none yet."""

    skill: str
    stage: int
    topic: str
    kind: str
    score: float | None


@dataclass(frozen=True)
class Profile:
    """What one skill looks like after a run, in words a trainer can check."""

    skill: str
    title: str
    status: str
    score: float | None
    stage: int | None
    confidence: str
    verdict: str
    delta: float | None
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    untested: tuple[str, ...]
    waiting: int
    items_scored: int
    next_step: str


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4)


def _by_topic(outcomes: Sequence[Outcome]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for outcome in outcomes:
        if outcome.score is not None:
            grouped.setdefault(outcome.topic, []).append(outcome.score)
    return {topic: _mean(scores) for topic, scores in grouped.items()}


def _stage_cleared(outcomes: Sequence[Outcome]) -> int | None:
    """The highest stage whose items were answered at the pass mark."""
    cleared: list[int] = []
    for stage in range(MIN_STAGE, MAX_STAGE + 1):
        scores = [o.score for o in outcomes if o.stage == stage and o.score is not None]
        if len(scores) >= STAGE_MIN_ITEMS and _mean(scores) >= PASS_MARK:
            cleared.append(stage)
    return max(cleared) if cleared else None


def _confidence(scored: int) -> str:
    if scored >= CONFIDENCE_HIGH:
        return "high"
    if scored >= CONFIDENCE_MEDIUM:
        return "medium"
    if scored > 0:
        return "low"
    return "none"


def _verdict(score: float | None) -> str:
    if score is None:
        return "not measured yet"
    return "benchmark passed" if score >= PASS_MARK else "benchmark not yet passed"


def _untested(
    outcomes: Sequence[Outcome], bank_items: Iterable[Mapping[str, Any]]
) -> tuple[str, ...]:
    covered = {outcome.topic for outcome in outcomes}
    expected = {str(item.get("topic") or "") for item in bank_items}
    return tuple(sorted(topic for topic in expected if topic and topic not in covered))


def _next_step(
    *,
    score: float | None,
    stage: int | None,
    waiting: int,
    weaknesses: tuple[str, ...],
    attempted: int,
) -> str:
    if score is None:
        if waiting:
            return "A trainer needs to review the waiting items before this skill can be scored."
        return "Run the held-out items in unaided mode to get a first score."
    if stage is not None and stage < MAX_STAGE:
        return f"Move to stage {stage + 1}."
    if stage is not None:
        return f"Keep practising at stage {MAX_STAGE} to hold the score."
    if weaknesses:
        return f"Practise {weaknesses[0]} at stage {attempted}."
    return f"Repeat the stage {attempted} items; {score:.0%} passed."


def build_profile(
    skill: str,
    outcomes: Sequence[Outcome],
    *,
    previous: float | None = None,
    bank_items: Iterable[Mapping[str, Any]] | None = None,
) -> Profile:
    """Summarise one skill's outcomes; an empty or unreviewed run stays unassessed."""
    if bank_items is None:
        from app.services.assessment import bank

        bank_items = bank.items(skill, "eval")
    scored = [outcome for outcome in outcomes if outcome.score is not None]
    waiting = sum(
        1 for outcome in outcomes if outcome.score is None and outcome.kind in REVIEW_KINDS
    )
    topics = _by_topic(scored)
    score = _mean([outcome.score for outcome in scored]) if scored else None
    stage = _stage_cleared(scored) if scored else None
    strengths = tuple(sorted(topic for topic, value in topics.items() if value >= STRENGTH_MARK))
    weaknesses = tuple(sorted(topic for topic, value in topics.items() if value < WEAKNESS_MARK))
    attempted = min((outcome.stage for outcome in scored), default=MIN_STAGE)
    return Profile(
        skill=skill,
        title=SKILL_TITLES.get(skill, skill),
        status="assessed" if scored else "unassessed",
        score=score,
        stage=stage,
        confidence=_confidence(len(scored)),
        verdict=_verdict(score),
        delta=round(score - previous, 4) if score is not None and previous is not None else None,
        strengths=strengths,
        weaknesses=weaknesses,
        untested=_untested(outcomes, bank_items),
        waiting=waiting,
        items_scored=len(scored),
        next_step=_next_step(
            score=score,
            stage=stage,
            waiting=waiting,
            weaknesses=weaknesses,
            attempted=attempted,
        ),
    )
