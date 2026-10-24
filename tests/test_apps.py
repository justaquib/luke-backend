from types import SimpleNamespace
from typing import Any

from app import main as api
from app.schemas import QueryRequest, QueryResponse
from app.services import apps
from app.services import query as query_service


def test_weather_question_is_not_also_classified_as_wikipedia():
    assert apps.detect_intents("What is current weather in kolkata?") == ["weather"]


def test_wiki_topic_removes_leading_article():
    assert apps._extract_topic("What is the capital of France?") == "capital of France"


def test_wiki_answers_father_of_computer_question(monkeypatch):
    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "title": "Charles Babbage",
                "extract": "Charles Babbage was an English polymath.",
                "content_urls": {
                    "desktop": {"page": "https://en.wikipedia.org/wiki/Charles_Babbage"}
                },
            }

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "Client":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def get(self, url: str, **kwargs: Any) -> Response:
            assert url.endswith("/Charles_Babbage")
            return Response()

    monkeypatch.setattr(apps.httpx, "Client", Client)

    result = apps.wiki_summary("Who is the father of computer?")

    assert result is not None
    assert result["payload"]["title"] == "Charles Babbage"
    assert result["answer"] == (
        "Charles Babbage is often regarded as the father of the computer "
        "for his design of the Analytical Engine."
    )


def test_where_question_is_routed_to_wikipedia_and_extracts_topic():
    question = "Where is Eiffel Tower."

    assert apps.detect_intents(question) == ["wiki"]
    assert apps._extract_topic(question) == "Eiffel Tower"


def test_wiki_summary_returns_none_when_search_has_no_results(monkeypatch):
    class Response:
        def __init__(self, status_code: int, data: Any) -> None:
            self.status_code = status_code
            self._data = data

        def raise_for_status(self) -> None:
            pass

        def json(self) -> Any:
            return self._data

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "Client":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def get(self, url: str, **kwargs: Any) -> Response:
            if "/api/rest_v1/" in url:
                return Response(404, {})
            return Response(200, ["capital of France", [], [], []])

    monkeypatch.setattr(apps.httpx, "Client", Client)

    assert apps.wiki_summary("What is the capital of France?") is None


def test_weather_integration_precedes_irrelevant_document_matches(monkeypatch):
    monkeypatch.setattr(
        api,
        "_get_bearer_user",
        lambda authorization, db: SimpleNamespace(is_systemadmin=False, tenant_id=1),
    )
    monkeypatch.setattr(query_service.nlp, "extract_entities", lambda question: [])
    monkeypatch.setattr(query_service.vector_store, "search_faq", lambda *args, **kwargs: [])
    monkeypatch.setattr(query_service.vector_store, "search_faq_loose", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        query_service.vector_store,
        "query",
        lambda *args, **kwargs: [{"distance": 1.0}],
    )
    monkeypatch.setattr(
        api,
        "_default_response",
        lambda db, req, ents, disabled, tenant_id=None: QueryResponse(
            question=req.question,
            source="pending",
            answer="pending",
            hits=[],
            entities=ents,
        ),
    )
    monkeypatch.setattr(
        query_service.apps,
        "weather_for",
        lambda question, lat=None, lon=None: {
            "answer": "Currently clear sky, 30°C.",
            "payload": {"temperature": 30},
        },
    )

    response = api.query(
        QueryRequest(
            question="What is current weather in kolkata?",
            apps={"weather": True},
        ),
        authorization="Bearer token",
        db=object(),
    )

    assert response.source == "weather"
    assert response.answer == "Currently clear sky, 30°C."


def test_wiki_integration_precedes_document_matches(monkeypatch):
    monkeypatch.setattr(
        api,
        "_get_bearer_user",
        lambda authorization, db: SimpleNamespace(is_systemadmin=False, tenant_id=1),
    )
    monkeypatch.setattr(query_service.nlp, "extract_entities", lambda question: [])
    monkeypatch.setattr(query_service.vector_store, "search_faq", lambda *args, **kwargs: [])
    monkeypatch.setattr(query_service.vector_store, "search_faq_loose", lambda *args, **kwargs: [])
    document_search = SimpleNamespace(called=False)

    def query_documents(*args: Any, **kwargs: Any) -> list[dict[str, float]]:
        document_search.called = True
        return [{"distance": 0.1}]

    monkeypatch.setattr(query_service.vector_store, "query", query_documents)
    monkeypatch.setattr(
        query_service.apps,
        "wiki_summary",
        lambda question: {
            "answer": "Paris is the capital of France.",
            "payload": {
                "title": "Capital of France",
                "url": "https://en.wikipedia.org/wiki/Capital_of_France",
            },
        },
    )

    response = api.query(
        QueryRequest(
            question="What is the capital of France?",
            apps={"wiki": True},
        ),
        authorization="Bearer token",
        db=object(),
    )

    assert response.source == "wiki"
    assert response.answer == "Paris is the capital of France."
    assert not document_search.called
