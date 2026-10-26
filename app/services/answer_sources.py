"""One source of the answer flow, and the dots it connected.

Each finder reads one place an answer can come from and returns what it found
with the connection that says why. The order they are tried in belongs to the
answer flow; what each one measured belongs here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import settings
from app.schemas import QueryHit, QueryRequest, QueryResponse
from app.services import apps, vector_store
from app.services.connections import Connection
from app.services.language import answering

logger = logging.getLogger(__name__)

# A match beyond the exact threshold is only a guess, so the answer says so
# until a trainer confirms it.
NOT_SURE_NOTE = "I am not sure about the answer to “{0}”. A trainer can check it."


@dataclass(frozen=True)
class SourceStep:
    """What one source found, and the dots connected on the way."""

    source: str
    response: QueryResponse | None
    connections: tuple[Connection, ...] = ()


def _flag_missing_confidence(answer: str, matched_question: str) -> str:
    """The loosely matched answer, with a word that it is a guess."""
    subject = matched_question.strip() or "this question"
    return f"{answer.rstrip('. ')}. {NOT_SURE_NOTE.format(subject)}"


def _confidence(distance: float | None) -> float | None:
    """Cosine similarity, the chance the vector search gives the match."""
    if distance is None:
        return None
    return round(max(0.0, min(1.0, 1.0 - float(distance))), 4)


def _faq_step(req: QueryRequest, entities: list, tenant_id: int | None) -> SourceStep:
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
                confidence=_confidence(distance),
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
        confidence=_confidence(intent["distance"]),
    )


def _loose_intent_step(req: QueryRequest, entities: list, tenant_id: int | None) -> SourceStep:
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


def _weather_step(
    req: QueryRequest,
    entities: list,
    app_settings: dict,
    location: dict,
    detected: list[str],
) -> SourceStep:
    if "weather" not in detected:
        return SourceStep("weather", None)
    if not app_settings.get("weather"):
        return SourceStep(
            "weather",
            None,
            (Connection("weather", "skipped", "Weather is not enabled in Settings."),),
        )
    result = apps.weather_for(req.question, lat=location.get("lat"), lon=location.get("lon"))
    if not result:
        return SourceStep(
            "weather",
            None,
            (Connection("weather", "miss", "No live weather answer was found."),),
        )
    return SourceStep(
        "weather",
        QueryResponse(
            question=req.question,
            source="weather",
            answer=result["answer"],
            hits=[],
            entities=entities,
            app=result["payload"],
        ),
        (
            Connection(
                "weather",
                "match",
                "A live weather answer was found for the place asked about.",
            ),
        ),
    )


def _wiki_step(
    req: QueryRequest, entities: list, app_settings: dict, detected: list[str]
) -> SourceStep:
    if "wiki" not in detected:
        return SourceStep("wiki", None)
    if not app_settings.get("wiki"):
        return SourceStep(
            "wiki",
            None,
            (Connection("wiki", "skipped", "Wikipedia is not enabled in Settings."),),
        )
    result = apps.wiki_summary(req.question)
    if not result:
        return SourceStep(
            "wiki",
            None,
            (Connection("wiki", "miss", "No matching Wikipedia page was found."),),
        )
    title = result["payload"].get("title", req.question)
    return SourceStep(
        "wiki",
        QueryResponse(
            question=req.question,
            source="wiki",
            answer=result["answer"],
            hits=[
                QueryHit(
                    text=title,
                    metadata={"source": "wiki", "url": result["payload"].get("url")},
                    distance=None,
                )
            ],
            entities=entities,
            app=result["payload"],
        ),
        (Connection("wiki", "match", f"Wikipedia has a page for “{title}”.", str(title)),),
    )


def _docs_step(req: QueryRequest, entities: list, tenant_id: int | None) -> SourceStep:
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
            confidence=_confidence(distance),
        ),
        (Connection("docs", "match", detail),),
    )


def _disabled_sources(detected: list[str], app_settings: dict) -> list[str]:
    disabled: list[str] = []
    if "weather" in detected and not app_settings.get("weather"):
        disabled.append("weather")
    if "wiki" in detected and not app_settings.get("wiki"):
        disabled.append("wiki")
    return disabled


def _disabled_response(req: QueryRequest, entities: list, disabled: list[str]) -> QueryResponse:
    names = ["Weather" if item == "weather" else "Wikipedia" for item in disabled]
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
