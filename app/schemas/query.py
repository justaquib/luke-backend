"""The chat question and the answer it receives.

The request may carry the recent conversation turns and the two switches that
decide how the answer is produced: `retrieval` reads the saved material and the
document passages, `compose` asks Luke to write from numbered passages instead
of returning one passage. The answer reports which of those produced it, so a
written answer is never shown as a retrieved one.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.understanding import QuestionUnderstandingOut


class QueryRequest(BaseModel):
    """One chat turn: the question, the recent turns and the two switches."""

    question: str
    n_results: int = 4
    apps: dict | None = None
    location: dict | None = None
    model: str | None = None
    history: list[dict] = Field(default_factory=list)
    retrieval: bool = True
    compose: bool = False
    # The reader turned web evidence on for this workspace. The server policy
    # still has to allow it, and only allowlisted pages ever answer.
    web: bool = False


class QueryHit(BaseModel):
    """One matched passage or saved question."""

    text: str
    metadata: dict
    distance: float | None = None


class CitationOut(BaseModel):
    """One numbered piece of evidence a composed answer cites."""

    index: int
    source: str
    label: str
    excerpt: str


class QueryResponse(BaseModel):
    """The answer, how it was produced and what it was produced from."""

    model_config = ConfigDict(protected_namespaces=())

    question: str
    source: str
    answer: str
    hits: list[QueryHit]
    entities: list[dict]
    pending_id: int | None = None
    intent: dict | None = None
    app: dict | None = None
    disabled_external_sources: list[str] | None = None
    understanding: QuestionUnderstandingOut | None = None
    model: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    mode: str = "retrieved"
    citations: list[CitationOut] = Field(default_factory=list)
    tokens_per_second: float | None = None
    distance: float | None = None
    confidence: float | None = None
