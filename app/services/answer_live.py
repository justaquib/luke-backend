"""The live sources an answer may quote: weather and Wikipedia.

A live source is asked only when the question names it, and it answers with
the measurement that was made: the place, the page or the reason it was not
used. Neither source reads the saved material, so the retrieval switch has no
say over them.
"""

from __future__ import annotations

from app.schemas import QueryHit, QueryRequest, QueryResponse
from app.services import apps
from app.services.answer_sources import SourceStep
from app.services.connections import Connection


def weather_step(
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


def wiki_step(
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
