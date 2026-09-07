from pydantic import BaseModel
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


MergePreview.model_rebuild()