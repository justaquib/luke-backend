from typing import Literal

from pydantic import BaseModel, EmailStr

RULE_KINDS = {"instruction", "own_words", "short"}
from typing import Optional
from datetime import datetime


class IngestRequest(BaseModel):
    source: str
    text: str


class IngestResponse(BaseModel):
    source: str
    chunks: int
    start_id: Optional[int] = None


class QueryRequest(BaseModel):
    question: str
    n_results: int = 4
    apps: Optional[dict] = None
    location: Optional[dict] = None


class QueryHit(BaseModel):
    text: str
    metadata: dict
    distance: Optional[float] = None


class QueryResponse(BaseModel):
    question: str
    source: str
    answer: str
    hits: list[QueryHit]
    entities: list[dict]
    pending_id: Optional[int] = None
    intent: Optional[dict] = None
    app: Optional[dict] = None
    disabled_external_sources: Optional[list[str]] = None


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
    faq_id: Optional[int] = None


class IntentOut(BaseModel):
    id: int
    name: str
    response_template: str
    faq_id: Optional[int]
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
    answer: Optional[str] = None
    target_id: Optional[int] = None
    faq_ids: list[int]


class FaqMergeResult(BaseModel):
    canonical_faq_id: int
    absorbed: int
    recreated: int


class DocumentInfo(BaseModel):
    source: str
    chunks: int
    created_at: Optional[datetime] = None


class DocumentDeleteResponse(BaseModel):
    source: str
    deleted_chunks: int
    status: str


class FeedbackCreate(BaseModel):
    message_id: Optional[int] = None
    question: str
    answer: str
    source: str
    rating: str


class FeedbackOut(BaseModel):
    id: int
    message_id: Optional[int] = None
    question: str
    answer: str
    source: str
    rating: str
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class GoogleAuthRequest(BaseModel):
    id_token: str
    invite_token: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 -- OAuth token type label, not a secret


class UserOut(BaseModel):
    id: int
    google_id: Optional[str]
    email: str
    name: str
    picture: Optional[str] = None
    tenant_id: Optional[int] = None
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
    used_at: Optional[datetime] = None
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
