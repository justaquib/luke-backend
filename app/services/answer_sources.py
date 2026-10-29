"""One source of the answer flow, and the dots it connected.

Each finder reads one place an answer can come from and returns what it found
with the connection that says why. The order they are tried in belongs to the
answer flow; what each one measured belongs here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import settings
from app.schemas import QueryHit, QueryRequest, QueryResponse
from app.services import vector_store
from app.services.connections import Connection
from app.services.foundation import answers as foundation_answers
from app.services.foundation.identity import FoundationError as FoundationFailure
from app.services.language import answering
from app.services.retrieval import similarity_from_distance
from app.services.web import step as web_evidence

logger = logging.getLogger(__name__)

# A match beyond the exact threshold is only a guess, so the answer says so
# until a trainer confirms it.
NOT_SURE_NOTE = "I am not sure about the answer to “{0}”. A trainer can check it."

RETRIEVAL_OFF = "Retrieval is off for this question."


@dataclass(frozen=True)
class SourceStep:
    """What one source found, and the dots connected on the way."""

    source: str
    response: QueryResponse | None
    connections: tuple[Connection, ...] = ()


def _retrieval_off(source: str) -> SourceStep:
    """The step that never reads saved material, so a run stays unaided."""
    return SourceStep(source, None, (Connection(source, "skipped", RETRIEVAL_OFF),))


def _flag_missing_confidence(answer: str, matched_question: str) -> str:
    """The loosely matched answer, with a word that it is a guess."""
    subject = matched_question.strip() or "this question"
    return f"{answer.rstrip('. ')}. {NOT_SURE_NOTE.format(subject)}"


def _confidence(distance: float | None, collection: str) -> float | None:
    """The similarity the vector search gives the match, in its real space."""
    return similarity_from_distance(distance, vector_store.collection_space(collection))


def _faq_step(req: QueryRequest, entities: list, tenant_id: int | None) -> SourceStep:
    if not req.retrieval:
        return _retrieval_off("faq")
    hits = vector_store.search_faq(req.question, n_results=1, tenant_id=tenant_id)
    if not hits:
        return SourceStep(
            "faq", None, (Connection("faq", "miss", "No saved answer to compare with."),)
        )
    best = hits[0]
    distance = best.get("distance")
    matched = str(best.get("question") or "")
    if distance is None:
        return SourceStep(
            "faq",
            None,
            (Connection("faq", "miss", "The closest saved answer has no distance.", matched),),
        )
    if distance <= settings.faq_match_threshold:
        return SourceStep(
            "faq",
            QueryResponse(
                question=req.question,
                source="faq",
                answer=best["answer"],
                hits=[
                    QueryHit(
                        text=matched,
                        metadata={"source": "faq", "faq_id": best["faq_id"]},
                        distance=distance,
                    )
                ],
                entities=entities,
                distance=distance,
                confidence=_confidence(distance, settings.chroma_faq_collection),
            ),
            (
                Connection(
                    "faq",
                    "match",
                    f"Closest saved answer was {float(distance):.3f} away, inside the "
                    f"{settings.faq_match_threshold:.3f} exact-match limit.",
                    matched,
                ),
            ),
        )
    return SourceStep(
        "faq",
        None,
        (
            Connection(
                "faq",
                "miss",
                f"Closest saved answer was {float(distance):.3f} away, beyond the "
                f"{settings.faq_match_threshold:.3f} exact-match limit.",
                matched,
            ),
        ),
    )


def _loose_candidate(req: QueryRequest, tenant_id: int | None) -> dict | None:
    hits = vector_store.search_faq_loose(req.question, n_results=3, tenant_id=tenant_id)
    return hits[0] if hits else None


def _intent_response(req: QueryRequest, entities: list, intent: dict) -> QueryResponse:
    matched = str(intent["matched_question"])
    flagged = _flag_missing_confidence(str(intent["answer"]), matched)
    return QueryResponse(
        question=req.question,
        source="intent",
        answer=flagged,
        hits=[
            QueryHit(
                text=matched,
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
        distance=intent["distance"],
        confidence=_confidence(intent["distance"], settings.chroma_intents_collection),
    )


def _loose_intent_step(req: QueryRequest, entities: list, tenant_id: int | None) -> SourceStep:
    if not req.retrieval:
        return _retrieval_off("intent")
    candidate = _loose_candidate(req, tenant_id)
    if candidate is None:
        return SourceStep(
            "intent",
            None,
            (Connection("intent", "miss", "No saved answer was close enough to compare."),),
        )
    distance = candidate.get("distance")
    matched = str(candidate.get("question") or "")
    if (
        distance is None
        or not settings.faq_match_threshold < distance <= settings.intent_match_threshold
    ):
        detail = (
            "The closest saved answer has no distance."
            if distance is None
            else f"Closest saved answer was {float(distance):.3f} away, beyond the "
            f"{settings.intent_match_threshold:.3f} loose limit."
        )
        return SourceStep("intent", None, (Connection("intent", "miss", detail, matched),))
    intent = {
        "matched_question": candidate.get("question"),
        "matched_faq_id": candidate.get("faq_id"),
        "distance": distance,
        "answer": candidate.get("answer"),
    }
    detail = (
        f"Closest saved answer was {float(distance):.3f} away: past the exact limit, "
        f"inside the {settings.intent_match_threshold:.3f} loose limit. "
        "The answer says it is a guess."
    )
    return SourceStep(
        "intent",
        _intent_response(req, entities, intent),
        (Connection("intent", "match", detail, matched),),
    )


def _taught_step(req: QueryRequest, entities: list, tenant_id: int | None) -> SourceStep:
    if not req.retrieval:
        return _retrieval_off("taught")
    try:
        taught = answering.answer_from_curriculum(req.question, tenant_id)
    except (OSError, ValueError, RuntimeError) as error:
        logger.warning("The taught answer failed: %s", error)
        return SourceStep(
            "taught",
            None,
            (Connection("taught", "fail", "The taught material could not be read."),),
        )
    if taught is None:
        return SourceStep(
            "taught",
            None,
            (
                Connection(
                    "taught",
                    "miss",
                    "No lesson, word card, dictionary entry or sentence example "
                    "covers the question.",
                ),
            ),
        )
    hits = (
        [QueryHit(text=taught.matched_question, metadata={"source": taught.source}, distance=None)]
        if taught.matched_question
        else []
    )
    connections = taught.evidence or (
        Connection(
            taught.source,
            "match",
            f"{taught.detail} answers this.",
            taught.matched_question,
        ),
    )
    return SourceStep(
        "taught",
        QueryResponse(
            question=req.question,
            source=taught.source,
            answer=taught.answer,
            hits=hits,
            entities=entities,
            confidence=taught.confidence,
        ),
        connections,
    )


def _docs_step(req: QueryRequest, entities: list, tenant_id: int | None) -> SourceStep:
    if not req.retrieval:
        return _retrieval_off("docs")
    hits = vector_store.query(req.question, n_results=req.n_results, tenant_id=tenant_id)
    if not hits:
        return SourceStep(
            "docs", None, (Connection("docs", "miss", "No document passage was found."),)
        )
    distance = hits[0].get("distance")
    if distance is not None and distance >= 1:
        return SourceStep(
            "docs",
            None,
            (
                Connection(
                    "docs",
                    "miss",
                    f"The closest passage was {float(distance):.3f} away; only passages "
                    "closer than 1.000 are used.",
                ),
            ),
        )
    detail = (
        "The closest passage has no distance."
        if distance is None
        else f"The closest passage was {float(distance):.3f} away, inside the 1.000 limit."
    )
    return SourceStep(
        "docs",
        QueryResponse(
            question=req.question,
            source="docs",
            answer=hits[0]["text"],
            hits=[QueryHit(**hit) for hit in hits],
            entities=entities,
            distance=distance,
            confidence=_confidence(distance, settings.chroma_collection),
        ),
        (Connection("docs", "match", detail),),
    )


def _compose_step(
    req: QueryRequest,
    entities: list,
    tenant_id: int | None,
    model_name: str,
) -> SourceStep:
    """Luke writes from numbered passages, and the verifier decides if it may."""
    if not req.retrieval:
        return _retrieval_off("foundation")
    if not foundation_answers.installed():
        return SourceStep(
            "foundation",
            None,
            (foundation_answers.missing_connection(),),
        )
    hits = vector_store.query(req.question, n_results=req.n_results, tenant_id=tenant_id)
    usable = [hit for hit in hits if hit.get("distance") is None or float(hit["distance"]) < 1]
    if not usable:
        return SourceStep(
            "foundation",
            None,
            (Connection("foundation", "miss", "No passage was close enough to compose from."),),
        )
    try:
        composed = foundation_answers.compose(req, tenant_id, entities, model_name, usable)
    except FoundationFailure as error:
        return SourceStep("foundation", None, (Connection("foundation", "fail", str(error)),))
    if composed is None:
        return SourceStep(
            "foundation",
            None,
            (Connection("foundation", "miss", "The passages could not be numbered."),),
        )
    response, connection = composed
    return SourceStep("foundation", response, (connection,))


def _web_step(req: QueryRequest, entities: list, tenant_id: int | None, db: Session) -> SourceStep:
    """A trusted page, quoted with its address and the date it was fetched."""
    response, connections = web_evidence.web_answer(req, entities, tenant_id, db)
    return SourceStep("web", response, connections)
