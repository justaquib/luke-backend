"""The parenting module schemas: the coaching chat of the parent and the coach."""

from datetime import datetime

from pydantic import BaseModel

from app.schemas.understanding import QuestionUnderstandingOut


class CoachingAnswerOut(BaseModel):
    """The coach's answer, and how sure the coach is of it."""

    text: str
    confidence: float
    grounded: bool
    source: str


class CoachingTurnOut(BaseModel):
    """The coach's answer to one turn."""

    status: str
    answer: CoachingAnswerOut | None = None
    understanding: QuestionUnderstandingOut | None = None


class CoachingSessionOut(BaseModel):
    """One coaching session between the parent and the coach."""

    id: str
    topic: str
    child_age_band: str = ""
    status: str = "open"
    turns: int = 0
    created_at: datetime
    updated_at: datetime


class ParentingSessionCreate(BaseModel):
    """The start of a coaching session."""

    topic: str = "General"
    child_age_band: str = ""


class ParentingTurnBody(BaseModel):
    """What the parent said to the coach."""

    message: str
