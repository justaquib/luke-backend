"""Tests for the shared answer flow: the stages, the details and the parity."""

from app.schemas import QueryRequest, QueryResponse
from app.services import answer_flow, answer_sources
from app.services.connections import Connection
from app.services.language import answering


def _pending(db, req, entities, disabled, tenant_id=None):
    return QueryResponse(
        question=req.question,
        source="pending",
        answer=answering.NOT_LEARNED,
        hits=[],
        entities=entities,
        disabled_external_sources=disabled,
    )


def _events(monkeypatch, question, taught=None):
    monkeypatch.setattr(answer_sources.vector_store, "search_faq", lambda *args, **kwargs: [])
    monkeypatch.setattr(answer_sources.vector_store, "search_faq_loose", lambda *args, **kwargs: [])
    monkeypatch.setattr(answer_sources.vector_store, "query", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        answer_sources.answering, "answer_from_curriculum", lambda question, tenant: taught
    )
    return list(
        answer_flow.iter_answer_events(QueryRequest(question=question), None, None, _pending)
    )


def _payload(events, name):
    return next(data for event, data in events if event == name)


def _connections(events):
    return [data for event, data in events if event == "connecting"]


def test_the_stages_are_yielded_in_order_and_the_answer_is_last(monkeypatch):
    taught = answering.TaughtAnswer(
        answer="A drum is something you hit with a stick.",
        source="lesson_cards",
        matched_question="What is a drum?",
        evidence=(Connection("lesson_cards", "match", "Lesson “What is a drum?” covers it."),),
    )

    events = _events(monkeypatch, "What is a drum?", taught)

    assert [event for event, _ in events] == [
        "reading",
        "thinking",
        "connecting",
        "connecting",
        "connecting",
        "answer",
        "done",
    ]
    assert [item["source"] for item in _connections(events)] == ["faq", "intent", "lesson_cards"]


def test_the_answer_carries_the_reading_the_plan_and_the_connections(monkeypatch):
    taught = answering.TaughtAnswer(
        answer="A drum is something you hit with a stick.",
        source="lesson_cards",
        matched_question="What is a drum?",
    )

    events = _events(monkeypatch, "What is a drum?", taught)
    payload = _payload(events, "answer")
    understanding = payload["understanding"]

    assert payload["answer"] == "A drum is something you hit with a stick."
    assert understanding["question_type"] == "definition"
    assert understanding["key_words"] == ["drum"]
    assert understanding["plan"] == [
        "Saved answers (FAQ)",
        "Taught lessons and the offline dictionary",
        "Wikipedia",
        "Your documents",
    ]
    assert understanding["words"]
    assert understanding["verified"] is True
    assert understanding["verification"]
    assert understanding["grammar_notes"] == []


def test_the_json_route_and_the_stream_read_the_same_answer(monkeypatch):
    taught = answering.TaughtAnswer(
        answer="a drum is something you hit with a stick",
        source="lesson_cards",
        matched_question="What is a drum?",
    )
    events = _events(monkeypatch, "What is a drum?", taught)

    response = answer_flow.answer_for_tenant(
        QueryRequest(question="What is a drum?"), None, None, _pending
    )

    assert response.model_dump(mode="json") == _payload(events, "answer")
    assert response.answer == "A drum is something you hit with a stick."


def test_an_exact_saved_answer_stops_the_flow_before_the_lessons(monkeypatch):
    hit = {
        "question": "What is a drum?",
        "answer": "a drum is taught",
        "distance": 0.25,
        "faq_id": 7,
    }
    monkeypatch.setattr(answer_sources.vector_store, "search_faq", lambda *args, **kwargs: [hit])
    monkeypatch.setattr(
        answer_sources.answering,
        "answer_from_curriculum",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("the lessons ran")),
    )

    events = list(
        answer_flow.iter_answer_events(
            QueryRequest(question="What is a drum?"), None, None, _pending
        )
    )
    connections = _connections(events)
    payload = _payload(events, "answer")

    assert len(connections) == 1
    assert connections[0]["source"] == "faq"
    assert connections[0]["outcome"] == "match"
    assert "0.250" in connections[0]["detail"]
    assert payload["source"] == "faq"
    assert payload["answer"] == "A drum is taught."


def test_an_unknown_question_ends_with_a_trainer_notification(monkeypatch):
    events = _events(monkeypatch, "Describe a flurble.", None)
    connections = _connections(events)
    payload = _payload(events, "answer")

    assert connections[-1]["source"] == "pending"
    assert connections[-1]["outcome"] == "miss"
    assert payload["source"] == "pending"
    assert payload["answer"] == answering.NOT_LEARNED
    assert payload["understanding"]["verified"] is False
    assert "trainer" in payload["understanding"]["verification"]


def test_a_disabled_weather_answer_says_so_instead_of_calling_out(monkeypatch):
    monkeypatch.setattr(
        answer_flow.apps,
        "weather_for",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("weather was called")),
    )

    events = _events(monkeypatch, "What is the weather in Paris?", None)
    connections = _connections(events)
    payload = _payload(events, "answer")

    weather = next(item for item in connections if item["source"] == "weather")
    pending = next(item for item in connections if item["source"] == "pending")
    assert weather["outcome"] == "skipped"
    assert "not enabled" in weather["detail"]
    assert pending["outcome"] == "skipped"
    assert "disabled" in pending["detail"]
    assert payload["source"] == "pending"
    assert "Enable Weather in Settings" in payload["answer"]
    assert payload["disabled_external_sources"] == ["weather"]


def test_the_word_model_and_the_document_search_are_never_invented(monkeypatch):
    events = _events(monkeypatch, "What is a drum?", None)

    assert [item["source"] for item in _connections(events)] == [
        "faq",
        "intent",
        "taught",
        "wiki",
        "docs",
        "foundation",
        "pending",
    ]
    assert all(item["detail"] for item in _connections(events))
