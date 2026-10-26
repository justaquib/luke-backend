from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr

from app.schemas.parenting import (
    CoachingAnswerOut as CoachingAnswerOut,
)
from app.schemas.parenting import (
    CoachingSessionOut as CoachingSessionOut,
)
from app.schemas.parenting import (
    CoachingTurnOut as CoachingTurnOut,
)
from app.schemas.parenting import (
    ParentingSessionCreate as ParentingSessionCreate,
)
from app.schemas.parenting import (
    ParentingTurnBody as ParentingTurnBody,
)
from app.schemas.understanding import (
    ConnectionOut as ConnectionOut,
)
from app.schemas.understanding import (
    QuestionUnderstandingOut as QuestionUnderstandingOut,
)
from app.schemas.understanding import (
    WordReadingOut as WordReadingOut,
)

RULE_KINDS = {"instruction", "own_words", "short"}


class IngestRequest(BaseModel):
    source: str
    text: str


class IngestResponse(BaseModel):
    source: str
    chunks: int
    start_id: int | None = None


class QueryRequest(BaseModel):
    question: str
    n_results: int = 4
    apps: dict | None = None
    location: dict | None = None


class QueryHit(BaseModel):
    text: str
    metadata: dict
    distance: float | None = None


class QueryResponse(BaseModel):
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


class StatsResponse(BaseModel):
    chunks: int
    faqs: int
    pending: int


class FaqCreate(BaseModel):
    question: str
    answer: str


class FaqUpdate(BaseModel):
    answer: str


class FaqStatusUpdate(BaseModel):
    status: str


class FaqOut(BaseModel):
    id: int
    question: str
    answer: str
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PendingOut(BaseModel):
    id: int
    question: str
    occurrences: int
    resolved: bool
    created_at: datetime
    last_seen_at: datetime

    class Config:
        from_attributes = True


class PendingLog(BaseModel):
    question: str


class IntentCreate(BaseModel):
    name: str
    response_template: str
    faq_id: int | None = None


class IntentOut(BaseModel):
    id: int
    name: str
    response_template: str
    faq_id: int | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class IntentUtterance(BaseModel):
    text: str


class MergePreview(BaseModel):
    canonical_id: int
    canonical_question: str
    answer: str
    matches: list["MergeMatch"]


class MergeMatch(BaseModel):
    pending_id: int
    question: str
    occurrences: int
    distance: float


class MergeRequest(BaseModel):
    pending_ids: list[int]
    answer: str


class MergeResult(BaseModel):
    canonical_faq_id: int
    created: int
    resolved: int


class FaqSuggestion(BaseModel):
    faq_id: int
    question: str
    answer: str
    status: str
    distance: float


class PendingSuggestion(BaseModel):
    pending_id: int
    question: str
    occurrences: int
    distance: float


class IntentSuggestions(BaseModel):
    query: str
    faqs: list[FaqSuggestion]
    pending: list[PendingSuggestion]


class FaqMergePreview(BaseModel):
    canonical_id: int
    canonical_question: str
    canonical_answer: str
    matches: list[FaqSuggestion]


class FaqMergeRequest(BaseModel):
    answer: str | None = None
    target_id: int | None = None
    faq_ids: list[int]


class FaqMergeResult(BaseModel):
    canonical_faq_id: int
    absorbed: int
    recreated: int


class DocumentInfo(BaseModel):
    source: str
    chunks: int
    created_at: datetime | None = None


class DocumentDeleteResponse(BaseModel):
    source: str
    deleted_chunks: int
    status: str


class FeedbackCreate(BaseModel):
    message_id: int | None = None
    question: str
    answer: str
    source: str
    rating: str


class FeedbackOut(BaseModel):
    id: int
    message_id: int | None = None
    question: str
    answer: str
    source: str
    rating: str
    created_at: datetime | None = None

    class Config:
        from_attributes = True


class GoogleAuthRequest(BaseModel):
    id_token: str
    invite_token: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 -- OAuth token type label, not a secret


class UserOut(BaseModel):
    id: int
    google_id: str | None
    email: str
    name: str
    picture: str | None = None
    tenant_id: int | None = None
    role: str = "user"
    is_systemadmin: bool = False
    created_at: datetime

    class Config:
        from_attributes = True


class TenantOut(BaseModel):
    id: int
    name: str
    slug: str
    created_at: datetime

    class Config:
        from_attributes = True


class InviteCreate(BaseModel):
    email: EmailStr
    role: str = "user"


class InviteOut(BaseModel):
    id: int
    tenant_id: int
    email: str
    role: str
    token: str
    expires_at: datetime
    used_at: datetime | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class SystemAdminLogin(BaseModel):
    pin: str


class AcceptInviteRequest(BaseModel):
    id_token: str


MergePreview.model_rebuild()


class RuleCreate(BaseModel):
    text: str
    kind: Literal["instruction", "own_words", "short"] = "instruction"


class LessonBody(BaseModel):
    """A lesson the trainer wants to teach."""

    lesson: str
    stage: str = "nursery"
    subject: str = "General"
    skip_duplicates: bool = False
    kind: Literal["qa", "story"] = "qa"
    title: str = ""


class CorrectionBody(BaseModel):
    """The right answer to a question the assistant answered wrongly."""

    question: str
    answer: str = ""
    correction: str


class ModelSettingsBody(BaseModel):
    """The name the model answers to and the version it starts from."""

    name: str = ""
    start_level: str = "0.0.0"


class DuplicateAcceptBody(BaseModel):
    """The group of repeating lessons the trainer wants to keep as they are."""

    key: str


class LessonUpdateBody(BaseModel):
    """An edit to a lesson the learner already holds."""

    answer: str
    question: str | None = None
    confidence: float | None = None
    subject: str | None = None


class PreviewBody(BaseModel):
    """A question the trainer asks the learner."""

    question: str
    skip_trainer: bool = False


class TestStartBody(BaseModel):
    """The kind of grade test to start."""

    stage: str | None = None
    kind: str = "trainer"


class TestQuestionBody(BaseModel):
    """A question asked during a grade test."""

    question: str


class TestScoreBody(BaseModel):
    """Whether the learner's answer was good enough."""

    passed: bool


class DictionaryLearnBody(BaseModel):
    """A dictionary word and the meaning to teach."""

    word: str
    meaning: str
    stage: str = "nursery"


class SentenceBody(BaseModel):
    """The words to build a sentence from."""

    words: list[str] = []
    word: str | None = None
