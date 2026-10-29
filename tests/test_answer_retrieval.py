"""An unaided question reads no saved material and asks the corpus nothing."""

from __future__ import annotations

from typing import Any

from app.schemas import QueryRequest, QueryResponse
from app.services import answer_flow, answer_sources
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


def _forbidden(name: str) -> Any:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"{name} was called with retrieval off")

    return fail


def test_an_unaided_question_never_reaches_a_retrieval_step(monkeypatch):
    for name in ("search_faq", "search_faq_loose", "query", "search_lessons"):
        monkeypatch.setattr(answer_sources.vector_store, name, _forbidden(name))
    monkeypatch.setattr(
        answer_sources.answering,
        "answer_from_curriculum",
        _forbidden("answer_from_curriculum"),
    )
    monkeypatch.setattr(answer_flow.foundation_answers, "installed", lambda: False)
    req = QueryRequest(question="What is a drum?", retrieval=False, compose=True)

    response = answer_flow.answer_for_tenant(req, None, None, _pending)

    assert response.source == "pending"
    assert response.understanding is not None
    skipped = {
        item.source for item in response.understanding.connections if item.outcome == "skipped"
    }
    assert {"faq", "intent", "taught", "foundation", "docs"} <= skipped


def test_an_assisted_question_still_reads_the_saved_answers(monkeypatch):
    calls: list[str] = []

    def record(*args: Any, **kwargs: Any) -> list[Any]:
        calls.append("faq")
        return []

    monkeypatch.setattr(answer_sources.vector_store, "search_faq", record)
    monkeypatch.setattr(answer_sources.vector_store, "search_faq_loose", lambda *a, **k: [])
    monkeypatch.setattr(answer_sources.vector_store, "query", lambda *a, **k: [])
    monkeypatch.setattr(answer_sources.answering, "answer_from_curriculum", lambda *a, **k: None)
    monkeypatch.setattr(answer_flow.foundation_answers, "installed", lambda: False)

    answer_flow.answer_for_tenant(QueryRequest(question="What is a drum?"), None, None, _pending)

    assert calls == ["faq"]
