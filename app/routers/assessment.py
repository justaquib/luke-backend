"""Assessment runs, the eight skill cards and the trainer's rubric review.

A run is started once and followed from its own event stream; a profile always
names the run it came from, so an unaided score is never shown as assisted.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.security import get_bearer_user
from app.core.workspace import current_workspace
from app.db.database import User
from app.schemas import (
    AssessmentProfilesOut,
    AssessmentResultOut,
    AssessmentReviewIn,
    AssessmentRunOut,
    AssessmentStart,
    AssessmentStarted,
)
from app.services import streaming
from app.services.assessment import profiles, runner

router = APIRouter(tags=["assessment"])


@router.get("/assessment/profiles", response_model=AssessmentProfilesOut)
def assessment_profiles(
    mode: str = "unaided",
    run_id: int | None = None,
    user: User = Depends(get_bearer_user),
) -> dict:
    """The eight cards of one run, or of the newest run of a mode."""
    return profiles.sheet(run_id, mode, current_workspace())


@router.post("/assessment/runs", status_code=202, response_model=AssessmentStarted)
def start_assessment(req: AssessmentStart, user: User = Depends(get_bearer_user)) -> dict:
    """Queue a run; the page follows it from the returned id."""
    tenant_id = current_workspace()
    run_id = runner.start(tenant_id, req.mode)
    return {"run_id": run_id, "run": runner.snapshot(run_id, tenant_id)}


@router.get("/assessment/runs", response_model=list[AssessmentRunOut])
def list_assessment_runs(user: User = Depends(get_bearer_user)) -> list[dict]:
    return runner.recent(current_workspace())


@router.get("/assessment/runs/{run_id}", response_model=AssessmentRunOut)
def get_assessment_run(run_id: int, user: User = Depends(get_bearer_user)) -> dict:
    return runner.snapshot(run_id, current_workspace())


@router.get("/assessment/runs/{run_id}/events")
def assessment_run_events(run_id: int, user: User = Depends(get_bearer_user)) -> StreamingResponse:
    """The run as server-sent events until it reaches a terminal state."""
    tenant_id = current_workspace()
    profiles.owns(run_id, tenant_id)
    return StreamingResponse(
        streaming.frames(runner.events(run_id, tenant_id)),
        media_type="text/event-stream",
        headers=streaming.SSE_HEADERS,
    )


@router.post("/assessment/runs/{run_id}/cancel", response_model=AssessmentRunOut)
def cancel_assessment_run(run_id: int, user: User = Depends(get_bearer_user)) -> dict:
    return runner.cancel(run_id, current_workspace())


@router.get("/assessment/runs/{run_id}/results", response_model=list[AssessmentResultOut])
def assessment_results(run_id: int, user: User = Depends(get_bearer_user)) -> list[dict]:
    rows = runner.results(run_id, current_workspace())
    return [profiles.result_dict(row) for row in rows]


@router.post("/assessment/results/{result_id}/review", response_model=AssessmentResultOut)
def review_assessment_result(
    result_id: int,
    req: AssessmentReviewIn,
    user: User = Depends(get_bearer_user),
) -> dict:
    """Score one rubric answer; deterministic results are never overwritten."""
    return profiles.review(
        result_id,
        current_workspace(),
        req.score,
        req.note,
        user.email,
    )
