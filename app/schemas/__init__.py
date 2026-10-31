from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.schemas.accounts import (
    AcceptInviteRequest,
    GoogleAuthRequest,
    InviteCreate,
    InviteOut,
    SystemAdminLogin,
    TokenResponse,
    UserOut,
)
from app.schemas.assessment import (
    AssessmentBankOut,
    AssessmentProfileOut,
    AssessmentProfilesOut,
    AssessmentResultOut,
    AssessmentReviewIn,
    AssessmentRunOut,
    AssessmentStart,
    AssessmentStarted,
)
from app.schemas.models import HuggingFaceTrainBody, PredictBody
from app.schemas.parenting import (
    CoachingAnswerOut,
    CoachingSessionOut,
    CoachingTurnOut,
    ParentingSessionCreate,
    ParentingTurnBody,
)
from app.schemas.query import CitationOut, QueryHit, QueryRequest, QueryResponse
from app.schemas.tools import (
    AppAllowlistBody,
    AppAllowlistEntry,
    AppOut,
    ToolExecuteRequest,
    ToolExecuteResponse,
    ToolOut,
    ToolPermissionOut,
    ToolPermissionsBody,
)
from app.schemas.understanding import (
    ConnectionOut,
    QuestionUnderstandingOut,
    WordReadingOut,
)
from app.schemas.voice import (
    SetModuleRequest,
    SttResponse,
    TtsRequest,
    VoiceInfoOut,
    VoiceModuleEntryOut,
    VoiceStatusOut,
)
from app.schemas.web import WebSettingsBody, WebSourceBody

__all__ = [
    "RULE_KINDS",
    "AcceptInviteRequest",
    "AppAllowlistBody",
    "AppAllowlistEntry",
    "AppOut",
    "AssessmentBankOut",
    "AssessmentProfileOut",
    "AssessmentProfilesOut",
    "AssessmentResultOut",
    "AssessmentReviewIn",
    "AssessmentRunOut",
    "AssessmentStart",
    "AssessmentStarted",
    "CitationOut",
    "CoachingAnswerOut",
    "CoachingSessionOut",
    "CoachingTurnOut",
    "ConnectionOut",
    "GoogleAuthRequest",
    "HuggingFaceTrainBody",
    "InviteCreate",
    "InviteOut",
    "ParentingSessionCreate",
    "ParentingTurnBody",
    "PredictBody",
    "QueryHit",
    "QueryRequest",
    "QueryResponse",
    "QuestionUnderstandingOut",
    "SetModuleRequest",
    "SttResponse",
    "SystemAdminLogin",
    "TokenResponse",
    "ToolExecuteRequest",
    "ToolExecuteResponse",
    "ToolOut",
    "ToolPermissionOut",
    "ToolPermissionsBody",
    "TtsRequest",
    "UserOut",
    "VoiceInfoOut",
    "VoiceModuleEntryOut",
    "VoiceStatusOut",
    "WebSettingsBody",
    "WebSourceBody",
    "WordReadingOut",
]

RULE_KINDS = {"instruction", "own_words", "short"}


class IngestRequest(BaseModel):
    source: str
    text: str


class IngestResponse(BaseModel):
    source: str
    chunks: int
    start_id: int | None = None
    duplicates: dict[str, int] | None = None


class IngestJobOut(BaseModel):
    id: int
    source: str
    status: str
    step: str
    progress: float
    done: int
    total: int
    chunks: int
    duplicates: dict[str, int]
    error: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


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
    note: str | None = None
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
