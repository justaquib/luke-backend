"""FastAPI application factory and compatibility exports."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.core.files import convert_for_preview
from app.core.pdf_fields import (
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
from app.routers import (
    assessment,
    auth,
    documents,
    expressions,
    faq,
    foundation,
    huggingface,
    intents,
    language_brain,
    language_model,
    parenting,
    system,
    tools,
    voice,
    web,
)
from app.routers import (
    query as query_router,
)
from app.schemas import QueryRequest, QueryResponse
from app.services import ingest_jobs
from app.services.assessment import runner as assessment_runner
from app.services.documents import resync_approved_faqs
from app.services.query import (
    _default_response,
    answer_question,
    confirm_existing_answer,
    faq_confidence_rank,
)


def create_app() -> FastAPI:
    """Build the configured FastAPI application."""
    application = FastAPI(title="Luke Lab API", version="0.2.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_list,
        # The dev server picks whichever local port is free, so any loopback
        # origin is allowed; the API itself only listens on the loopback host.
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_event_handler("startup", on_startup)
    application.add_event_handler("shutdown", ingest_jobs.shutdown)
    application.add_event_handler("shutdown", assessment_runner.shutdown)
    for router in (
        system.router,
        auth.router,
        assessment.router,
        documents.router,
        query_router.router,
        faq.router,
        intents.router,
        language_model.router,
        expressions.router,
        foundation.router,
        huggingface.router,
        language_brain.router,
        parenting.router,
        tools.router,
        voice.router,
        web.router,
    ):
        application.include_router(router)
    return application


app = create_app()


def query(req: QueryRequest, authorization: str = "", db: Session | None = None) -> QueryResponse:
    """Compatibility wrapper used by callers that import from app.main."""
    return answer_question(req, authorization, db, _get_bearer_user, _default_response)


_hash_pin = hash_pin
_ensure_systemadmin = ensure_systemadmin
_create_access_token = create_access_token
_get_bearer_user = get_bearer_user
_convert_for_preview = convert_for_preview
_resync_approved_faqs_to_chroma = resync_approved_faqs
_answer_from_document = answer_from_document
_faq_field_subjects_compatible = faq_field_subjects_compatible
_confirm_existing_answer = confirm_existing_answer
_faq_confidence_rank = faq_confidence_rank
