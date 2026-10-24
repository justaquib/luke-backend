"""The parenting module schemas: the coaching chat of the parent and the coach."""

from datetime import datetime

from pydantic import BaseModel


class CoachingAnswerOut(BaseModel):
    """The coach's answer, and how sure the coach is of it."""

    text: str
    confidence: float
    grounded: bool
    source: str
    grammar_notes: list[str] = []


class CoachingScaffoldOut(BaseModel):
    """The ways the parent can coach the coach better."""

    suggestions: list[str] = []
    retry_context: str = ""
    failure: str = ""


class CoachingTurnOut(BaseModel):
    """The coach's answer to one turn, or the correction it still needs."""

    status: str
    answer: CoachingAnswerOut | None = None
    scaffold: CoachingScaffoldOut | None = None


class CoachingSessionOut(BaseModel):
    """One coaching session between the parent and the coach."""

    id: str
    topic: str
    child_age_band: str = ""
    status: str = "open"
    turns: int = 0
    corrections_used: int = 0
    scaffold_failures: int = 0
    created_at: datetime
    updated_at: datetime


class ParentingSessionCreate(BaseModel):
    """The start of a coaching session."""

    topic: str = "General"
    child_age_band: str = ""


class ParentingTurnBody(BaseModel):
    """What the parent said to the coach."""

    message: str


class ParentingCorrection(BaseModel):
    """The better way the parent, or the trainer, coached the coach."""

    correction: str
