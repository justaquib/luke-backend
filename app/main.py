"""FastAPI application factory and compatibility exports."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.core.files import convert_for_preview, extract_text_from_file
from app.core.pdf_fields import (
    _extract_pdf_field_values,
    answer_from_document,
    faq_field_subjects_compatible,
)
from app.core.security import (
    create_access_token,
    ensure_systemadmin,
    get_bearer_user,
    hash_pin,
)
from app.core.startup import on_startup
from app.schemas import QueryRequest, QueryResponse
from app.routers import (
    auth,
    documents,
    faq,
    intents,
    language_brain,
    language_model,
    query as query_router,
    system,
)
from app.services import apps, nlp, vector_store
from app.services.documents import resync_approved_faqs
from app.services.query import (
    _default_response,
    _record_pending,
    answer_question,
    confirm_existing_answer,
    faq_confidence_rank,
    propose_answer,
)


def create_app() -> FastAPI:
    """Build the configured FastAPI application."""
    application = FastAPI(title="Assistance RAG API", version="0.2.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_event_handler("startup", on_startup)
    for router in (
        system.router,
        auth.router,
        documents.router,
        query_router.router,
        faq.router,
        intents.router,
        language_model.router,
        language_brain.router,
    ):
        application.include_router(router)
    return application


app = create_app()


def query(
    req: QueryRequest, authorization: str = "", db: Session | None = None
) -> QueryResponse:
    """Compatibility wrapper used by callers that import from app.main."""
    return answer_question(req, authorization, db, _get_bearer_user, _default_response)


_hash_pin = hash_pin
_ensure_systemadmin = ensure_systemadmin
_create_access_token = create_access_token
_get_bearer_user = get_bearer_user
_extract_text_from_file = extract_text_from_file
_convert_for_preview = convert_for_preview
_resync_approved_faqs_to_chroma = resync_approved_faqs
_extract_pdf_field_values = _extract_pdf_field_values
_answer_from_document = answer_from_document
_faq_field_subjects_compatible = faq_field_subjects_compatible
_confirm_existing_answer = confirm_existing_answer
_propose_answer = propose_answer
_faq_confidence_rank = faq_confidence_rank