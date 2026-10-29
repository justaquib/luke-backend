"""The eight skill cards, the history they compare against, and the review.

A profile never turns an unreviewed rubric into a zero: it counts the item as
waiting and stays unassessed until a trainer scores it. Everything here reads
the stored run, so the page and the review write back to the same rows.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select

from app.db import database
from app.db.database import AssessmentResult, AssessmentRun
from app.services.assessment import bank, skills


def owned(db: Any, run_id: int, tenant_id: int | None) -> AssessmentRun:
    """The run, or a 404 when this workspace does not own it."""
    run = db.get(AssessmentRun, run_id)
    if run is None or (tenant_id is not None and run.tenant_id != tenant_id):
        raise HTTPException(404, "Assessment run not found")
    return run


def owns(run_id: int, tenant_id: int | None) -> None:
    """Raise 404 unless this workspace owns the run."""
    with database.SessionLocal() as db:
        owned(db, run_id, tenant_id)


def latest(mode: str, tenant_id: int | None) -> AssessmentRun | None:
    """The newest run of one mode, whatever state it is in."""
    statement = (
        select(AssessmentRun)
        .where(AssessmentRun.mode == mode)
        .order_by(AssessmentRun.id.desc())
        .limit(1)
    )
    if tenant_id is not None:
        statement = statement.where(AssessmentRun.tenant_id == tenant_id)
    with database.SessionLocal() as db:
        return db.execute(statement).scalars().first()


def outcome(row: AssessmentResult) -> skills.Outcome:
    """One stored result as the profile reads it."""
    return skills.Outcome(
        skill=row.skill,
        stage=int(row.stage or 0),
        topic=row.topic,
        kind=row.kind,
        score=row.score,
    )


def result_dict(row: AssessmentResult) -> dict[str, Any]:
    """One stored result in the shape the review page reads."""
    return {
        "id": row.id,
        "run_id": row.run_id,
        "item_id": row.item_id,
        "skill": row.skill,
        "split": row.split,
        "kind": row.kind,
        "stage": int(row.stage or 0),
        "topic": row.topic,
        "response": row.response,
        "method": row.method,
        "score": row.score,
        "correct": row.correct,
        "latency_ms": row.latency_ms,
        "reviewed_by": row.reviewed_by,
        "reviewed_at": _iso(row.reviewed_at),
        "review_note": row.review_note,
        **_item_facts(row),
    }


def _item_facts(row: AssessmentResult) -> dict[str, Any]:
    """What the trainer judges a rubric answer against, from the bank item."""
    item = bank.item(row.item_id) or {}
    expected = item.get("expected")
    expected = expected if isinstance(expected, dict) else {}
    criteria = expected.get("criteria")
    reference = expected.get("reference")
    prompt = item.get("prompt")
    return {
        "prompt": prompt if isinstance(prompt, str) else None,
        "criteria": [str(line) for line in criteria] if isinstance(criteria, list) else [],
        "reference": reference if isinstance(reference, str) else None,
    }


def previous_score(skill: str, mode: str, before_run: int) -> float | None:
    """The last completed run's score for one skill, for the delta."""
    statement = (
        select(AssessmentRun)
        .where(
            AssessmentRun.mode == mode,
            AssessmentRun.status == "done",
            AssessmentRun.id < before_run,
        )
        .order_by(AssessmentRun.id.desc())
    )
    with database.SessionLocal() as db:
        for run in db.execute(statement).scalars():
            rows = db.execute(
                select(AssessmentResult).where(
                    AssessmentResult.run_id == run.id,
                    AssessmentResult.skill == skill,
                    AssessmentResult.score.is_not(None),
                )
            ).scalars()
            scores = [float(row.score) for row in rows]
            if scores:
                return round(sum(scores) / len(scores), 4)
    return None


def sheet(run_id: int | None, mode: str, tenant_id: int | None) -> dict[str, Any]:
    """The eight cards for a run, or for the latest run of a mode, or unassessed."""
    from app.services.assessment import runner

    if mode not in runner.MODES:
        raise HTTPException(400, f"unknown assessment mode '{mode}'")
    if run_id is not None:
        with database.SessionLocal() as db:
            run = owned(db, run_id, tenant_id)
    else:
        run = latest(mode, tenant_id)
    grouped: dict[str, list[skills.Outcome]] = {}
    if run is not None:
        for row in runner.results(run.id, tenant_id):
            grouped.setdefault(row.skill, []).append(outcome(row))
    cards = []
    for skill in skills.SKILL_IDS:
        previous = previous_score(skill, run.mode, run.id) if run is not None else None
        profile = skills.build_profile(
            skill,
            grouped.get(skill, []),
            previous=previous,
            bank_items=bank.items(skill, "eval"),
        )
        cards.append(asdict(profile))
    return {"run": runner.as_dict(run) if run is not None else None, "profiles": cards}


def review(
    result_id: int,
    tenant_id: int | None,
    score: float,
    note: str,
    reviewer: str,
) -> dict[str, Any]:
    """Store the trainer's score for one rubric answer; nothing else is reviewable."""
    with database.SessionLocal() as db:
        row = db.get(AssessmentResult, result_id)
        if row is None:
            raise HTTPException(404, "Assessment result not found")
        owned(db, row.run_id, tenant_id)
        if row.kind != skills.RUBRIC:
            raise HTTPException(400, "Only a rubric item is scored by a trainer")
        row.score = round(float(score), 4)
        row.correct = row.score >= skills.PASS_MARK
        row.reviewed_by = reviewer
        row.reviewed_at = datetime.utcnow()
        row.review_note = note.strip() or None
        db.commit()
        db.refresh(row)
        return result_dict(row)


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment is not None else None
