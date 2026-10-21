"""Question answering flow for the chat endpoint."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import Faq, PendingQuestion, User
from app.schemas import (
    QueryHit,
    QueryRequest,
    QueryResponse,
)
from app.services import apps, brain, language_logic, nlp, vector_store
from app.services.transformer.settings import WritingSettings

logger = logging.getLogger(__name__)
UserLookup = Callable[[str, Session], User]
DefaultResponse = Callable[..., QueryResponse]


def _record_pending(db: Session, question: str, tenant_id: int | None = None) -> PendingQuestion:
    """Log an unanswered question, counting repeat occurrences."""
    normalized = question.strip()
    existing = db.execute(
        select(PendingQuestion).where(
            PendingQuestion.question == normalized,
            PendingQuestion.resolved.is_(False),
        )
    ).scalar_one_or_none()
    if existing is not None:
        existing.occurrences += 1
        existing.last_seen_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()
        db.refresh(existing)
        vector_store.index_pending(existing.id, existing.question, tenant_id)
        return existing
    pending = PendingQuestion(question=normalized, tenant_id=tenant_id)
    db.add(pending)
    db.commit()
    db.refresh(pending)
    vector_store.index_pending(pending.id, pending.question, tenant_id)
    return pending


def _default_response(
    db: Session,
    req: QueryRequest,
    entities: list,
    disabled_external_sources: list[str] | None,
    tenant_id: int | None = None,
) -> QueryResponse:
    """Return the trainer-notified response for an unanswered question."""
    pending = _record_pending(db, req.question, tenant_id)
    return QueryResponse(
        question=req.question,
        source="pending",
        answer="I don't have an answer for that yet. A trainer has been notified.",
        hits=[],
        entities=entities,
        pending_id=pending.id,
        disabled_external_sources=disabled_external_sources,
    )


def _disabled_sources(detected: list[str], app_settings: dict) -> list[str]:
    disabled: list[str] = []
    if "weather" in detected and not app_settings.get("weather"):
        disabled.append("weather")
    if "wiki" in detected and not app_settings.get("wiki"):
        disabled.append("wiki")
    return disabled


def _faq_hit(req: QueryRequest, entities: list, tenant_id: int) -> QueryResponse | None:
    hits = vector_store.search_faq(req.question, n_results=1, tenant_id=tenant_id)
    if not hits:
        return None
    best = hits[0]
    distance = best["distance"]
    if distance is not None and distance <= settings.faq_match_threshold:
        return QueryResponse(
            question=req.question,
            source="faq",
            answer=best["answer"],
            hits=[
                QueryHit(
                    text=best["question"],
                    metadata={"source": "faq", "faq_id": best["faq_id"]},
                    distance=distance,
                )
            ],
            entities=entities,
        )
    return None


def _loose_intent(req: QueryRequest, tenant_id: int) -> dict | None:
    hits = vector_store.search_faq_loose(req.question, n_results=3, tenant_id=tenant_id)
    if not hits:
        return None
    candidate = hits[0]
    distance = candidate.get("distance")
    if (
        distance is None
        or not settings.faq_match_threshold < distance <= settings.intent_match_threshold
    ):
        return None
    return {
        "matched_question": candidate.get("question"),
        "matched_faq_id": candidate.get("faq_id"),
        "distance": distance,
        "answer": candidate.get("answer"),
    }


def _intent_response(req: QueryRequest, entities: list, intent: dict) -> QueryResponse:
    return QueryResponse(
        question=req.question,
        source="intent",
        answer=intent["answer"],
        hits=[
            QueryHit(
                text=intent["matched_question"],
                metadata={
                    "source": "faq",
                    "faq_id": intent["matched_faq_id"],
                    "via": "intent",
                },
                distance=intent["distance"],
            )
        ],
        entities=entities,
        intent=intent,
    )


def _weather_response(
    req: QueryRequest, entities: list, app_settings: dict, location: dict, detected: list[str]
) -> QueryResponse | None:
    if not (app_settings.get("weather") and "weather" in detected):
        return None
    result = apps.weather_for(req.question, lat=location.get("lat"), lon=location.get("lon"))
    if not result:
        return None
    return QueryResponse(
        question=req.question,
        source="weather",
        answer=result["answer"],
        hits=[],
        entities=entities,
        app=result["payload"],
    )


def _wiki_response(
    req: QueryRequest, entities: list, app_settings: dict, detected: list[str]
) -> QueryResponse | None:
    if not (app_settings.get("wiki") and "wiki" in detected):
        return None
    result = apps.wiki_summary(req.question)
    if not result:
        return None
    return QueryResponse(
        question=req.question,
        source="wiki",
        answer=result["answer"],
        hits=[
            QueryHit(
                text=result["payload"].get("title", req.question),
                metadata={"source": "wiki", "url": result["payload"].get("url")},
                distance=None,
            )
        ],
        entities=entities,
        app=result["payload"],
    )


def _docs_response(req: QueryRequest, entities: list, tenant_id: int) -> QueryResponse | None:
    hits = vector_store.query(req.question, n_results=req.n_results, tenant_id=tenant_id)
    if not hits:
        return None
    if hits[0].get("distance") is not None and hits[0]["distance"] >= 1:
        return None
    return QueryResponse(
        question=req.question,
        source="docs",
        answer=hits[0]["text"],
        hits=[QueryHit(**hit) for hit in hits],
        entities=entities,
    )


def _request_response(req: QueryRequest, entities: list, tenant_id: int) -> QueryResponse | None:
    """Answer a request for something to be written, from the taught material.

    A request is not a question, so it is not looked up in the trained
    answers. The lesson about the topic is found instead, and the trained
    model writes the answer from it.
    """
    from app.services.language import requests as request_service
    from app.services.language.lessons import all_lesson_cards

    try:
        cards = all_lesson_cards(tenant_id)
    except (OSError, ValueError):
        return None
    result = request_service.answer_request(
        req.question, tenant_id, cards, writer=brain.load_writer(tenant_id)
    )
    if result is None:
        return None
    return QueryResponse(
        question=req.question,
        source="written",
        answer=result["answer"],
        hits=[],
        entities=entities,
    )


def _disabled_response(
    req: QueryRequest, entities: list, disabled: list[str]
) -> QueryResponse | None:
    if not disabled:
        return None
    names: list[str] = []
    if "weather" in disabled:
        names.append("Weather")
    if "wiki" in disabled:
        names.append("Wikipedia")
    joined = " and ".join(names)
    return QueryResponse(
        question=req.question,
        source="pending",
        answer=(
            "I don't have a trained answer for that yet. "
            f"Enable {joined} in Settings to get live answers from your location or the web."
        ),
        hits=[],
        entities=entities,
        disabled_external_sources=disabled,
    )


def answer_question(
    req: QueryRequest,
    authorization: str,
    db: Session,
    lookup_user: UserLookup,
    default_response: DefaultResponse,
) -> QueryResponse:
    """Answer a chat question using FAQs, documents, intents, or integrations."""
    user = lookup_user(authorization, db)
    if user.is_systemadmin:
        raise HTTPException(403, "System admin cannot query")
    if not user.tenant_id:
        raise HTTPException(403, "User is not assigned to a tenant")
    return _answer_for_tenant(req, db, user.tenant_id, default_response)


def _answer_for_tenant(
    req: QueryRequest, db: Session, tenant_id: int, default_response: DefaultResponse
) -> QueryResponse:
    entities = nlp.extract_entities(req.question)
    app_settings = req.apps or {}
    location = req.location or {}
    detected = apps.detect_intents(req.question)
    disabled = _disabled_sources(detected, app_settings)
    # The sources are tried in order, and the first one that answers wins.
    for find_answer in (
        lambda: _faq_hit(req, entities, tenant_id),
        lambda: _loose_intent_response(req, entities, tenant_id),
        lambda: _request_response(req, entities, tenant_id),
        lambda: _weather_response(req, entities, app_settings, location, detected),
        lambda: _wiki_response(req, entities, app_settings, detected),
        lambda: _docs_response(req, entities, tenant_id),
    ):
        response = find_answer()
        if response is not None:
            response.disabled_external_sources = disabled or None
            return response
    return (
        _disabled_response(req, entities, disabled)
        or default_response(db, req, entities, disabled or None, tenant_id)
    )


def _loose_intent_response(
    req: QueryRequest, entities: list, tenant_id: int
) -> QueryResponse | None:
    """The answer of a saved intent that covers the question."""
    intent = _loose_intent(req, tenant_id)
    if intent is None:
        return None
    return _intent_response(req, entities, intent)


def propose_answer(tenant_id: int | None, question: str) -> str | None:
    """Ask the local writer for a possible answer to an unseen question."""
    writer = brain.load_writer(tenant_id)
    if writer is None:
        return None
    try:
        settings = WritingSettings(max_words=24)
        proposal = str(writer.generate(question, settings=settings)).strip()
    except (RuntimeError, ValueError, TypeError) as error:
        logger.warning("Writer proposal failed: %s", error)
        return None
    if len(proposal.split()) < 3:
        return None
    return proposal


def faq_confidence_rank(hit: dict) -> float:
    """A lower rank means a more trustworthy FAQ hit."""
    distance = float(hit.get("distance", 1.0))
    confidence = float(hit.get("confidence", 0.0))
    return distance - confidence * 0.1


def confirm_existing_answer(tenant_id: int | None, question: str, answer: str, db: Session) -> dict:
    """Keep a trainer confirmation without duplicating a known answer."""
    cleaned_answer = answer.strip()
    lesson = language_logic.find_lesson_by_answer(tenant_id, cleaned_answer)
    if lesson is not None:
        card = language_logic.bump_lesson_confidence(tenant_id, cleaned_answer, question)
        return {
            "target": "lesson",
            "faq_id": None,
            "lesson_id": lesson.get("id"),
            "confidence": (card or lesson).get("confidence"),
        }
    faq = _find_matching_faq(db, tenant_id, question, cleaned_answer)
    if faq is not None:
        faq.confidence = min(
            settings.max_answer_confidence, float(faq.confidence) + settings.confidence_step
        )
        db.commit()
        db.refresh(faq)
        vector_store.upsert_faq(faq.id, faq.question, faq.answer, faq.status, tenant_id)
        return {"target": "trained", "faq_id": faq.id, "confidence": faq.confidence}
    faq = Faq(
        question=question.strip(),
        answer=cleaned_answer,
        status="pending",
        tenant_id=tenant_id,
        confidence=settings.base_answer_confidence,
    )
    db.add(faq)
    db.commit()
    db.refresh(faq)
    vector_store.upsert_faq(faq.id, faq.question, faq.answer, faq.status, tenant_id)
    return {"target": "trained", "faq_id": faq.id, "confidence": faq.confidence}


def _find_matching_faq(
    db: Session, tenant_id: int | None, question: str, answer: str
) -> Faq | None:
    stmt = select(Faq).where(
        func.lower(Faq.question) == question.strip().lower(),
        func.lower(Faq.answer) == answer.lower(),
    )
    if tenant_id is None:
        stmt = stmt.where(Faq.tenant_id.is_(None))
    else:
        stmt = stmt.where(Faq.tenant_id == tenant_id)
    return db.execute(stmt).scalars().first()
