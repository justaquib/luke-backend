"""Tests for the note on an answer that came from a loose match.

A match beyond the exact threshold is a guess, so the answer says what it
matched and who can settle it instead of speaking with false confidence.
"""

from app.schemas import QueryRequest
from app.services import answer_sources

QUESTION = "what flies on the sea?"
MATCH = {
    "question": "What is a kite?",
    "answer": "A kite flies on the wind.",
    "distance": 0.6,
    "faq_id": 1,
}


def test_a_loose_match_says_it_is_not_sure(monkeypatch):
    monkeypatch.setattr(
        answer_sources.vector_store,
        "search_faq_loose",
        lambda question, n_results=3, tenant_id=None: [MATCH],
    )

    step = answer_sources._loose_intent_step(QueryRequest(question=QUESTION), [], None)

    assert step.response is not None
    assert step.response.source == "intent"
    assert "A kite flies on the wind." in step.response.answer
    assert "I am not sure about the answer to “What is a kite?”" in step.response.answer


def test_without_a_loose_match_nothing_is_answered(monkeypatch):
    monkeypatch.setattr(
        answer_sources.vector_store,
        "search_faq_loose",
        lambda *args, **kwargs: [],
    )

    step = answer_sources._loose_intent_step(QueryRequest(question=QUESTION), [], None)

    assert step.response is None
