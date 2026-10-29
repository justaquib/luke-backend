"""The assessment run, the eight skill cards and the trainer's review.

Every score is either measured or explicitly absent: a rubric answer waits for
a trainer, and its card stays unassessed until that review exists.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AssessmentStart(BaseModel):
    """The two ways to measure: with the corpus or without it."""

    mode: Literal["unaided", "assisted"] = "unaided"


class AssessmentBankOut(BaseModel):
    version: int
    salt: str
    eval_items: int


class AssessmentRunOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: int
    mode: str
    status: str
    step: str
    progress: float
    done: int
    total: int
    model_id: str
    model_artifact: str
    tokenizer_id: str
    generation: dict[str, Any] = Field(default_factory=dict)
    bank: AssessmentBankOut
    error: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class AssessmentStarted(BaseModel):
    run_id: int
    run: AssessmentRunOut


class AssessmentProfileOut(BaseModel):
    skill: str
    title: str
    status: str
    score: float | None
    stage: int | None
    confidence: str
    verdict: str
    delta: float | None
    strengths: list[str]
    weaknesses: list[str]
    untested: list[str]
    waiting: int
    items_scored: int
    next_step: str


class AssessmentProfilesOut(BaseModel):
    run: AssessmentRunOut | None = None
    profiles: list[AssessmentProfileOut]


class AssessmentResultOut(BaseModel):
    id: int
    run_id: int
    item_id: str
    skill: str
    split: str
    kind: str
    stage: int
    topic: str
    response: str | None = None
    method: str
    score: float | None = None
    correct: bool | None = None
    latency_ms: float | None = None
    prompt: str | None = None
    criteria: list[str] = Field(default_factory=list)
    reference: str | None = None
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    review_note: str | None = None


class AssessmentReviewIn(BaseModel):
    """The trainer's score for one rubric answer, on the same 0-1 scale."""

    score: float = Field(ge=0.0, le=1.0)
    note: str = ""
