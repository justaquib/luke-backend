"""The rain regression: a taught answer is served whole, in every wording.

The regression was seen with a poem on rain. A saved poem that merely looked
close was taken as an exact answer, and the taught poem was never reached; the
partial wordings were answered by something other than the lesson as well. A
taught answer must come back whole and grammatical, and a question nothing was
taught about must be written and labelled as written.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.schemas import QueryRequest, QueryResponse
from app.services import answer_flow, answer_sources
from app.services.foundation import answers as foundation_answers
from app.services.language import answering, runtime, teaching

POEM = "Rain, rain, go away.\nCome again another day.\nLittle children want to play."
POEM_LESSON = f"Q: Poem: Rain Rain Go Away\nA: {POEM}"
FACT_LESSON = "Q: What is rain?\nA: Rain is water falling from the clouds."

# The nearest saved answer to a rain wording, measured on the rebuilt index.
# It is the Twinkle poem, and it was served as an exact answer before the
# thresholds were calibrated for the cosine space.
NEAR_SAVED_DISTANCE = 0.5339


@pytest.fixture()
def learner(tmp_path, monkeypatch) -> None:
    """A learner file of its own, with the rain material already taught."""
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    runtime._CACHE.clear()
    teaching.teach(None, POEM_LESSON, "nursery", "Language, Poems")
    teaching.teach(None, FACT_LESSON, "nursery", "Science")
    yield
    runtime._CACHE.clear()


def _answer(question: str) -> answering.TaughtAnswer:
    taught = answering.answer_from_curriculum(question, None)
    assert taught is not None, f"nothing taught answered {question!r}"
    return taught


def _assert_whole_poem(taught: answering.TaughtAnswer) -> None:
    assert taught.answer == POEM
    assert taught.answer[0].isupper(), "the answer starts as a sentence"
    assert taught.answer.endswith("."), "the answer ends as a sentence"
    assert taught.answer.count("go away") == 1, "the poem is served once, not looped"


@pytest.mark.parametrize(
    "question",
    [
        "Write a short poem about the rain.",
        "Tell me a poem on rain.",
        "rain rain go away",
    ],
    ids=["exact", "paraphrase", "partial-phrase"],
)
def test_the_taught_poem_is_served_whole(learner, question):
    _assert_whole_poem(_answer(question))


@pytest.mark.parametrize(
    "question",
    ["What is rain?", "Tell me what rain is.", "rain?"],
    ids=["exact", "paraphrase", "partial-phrase"],
)
def test_the_taught_fact_is_a_full_sentence(learner, question):
    taught = _answer(question)

    assert taught.answer == "Rain is water falling from the clouds."
    assert taught.answer[0].isupper()
    assert taught.answer.endswith(".")


def test_the_measured_saved_answer_stays_a_guess():
    assert settings.faq_match_threshold < NEAR_SAVED_DISTANCE
    assert settings.intent_match_threshold < NEAR_SAVED_DISTANCE


def test_a_near_saved_poem_does_not_replace_the_taught_poem(learner, monkeypatch):
    near = {
        "question": "twinkle twinkle poem",
        "answer": "Twinkle, twinkle, little star.",
        "distance": NEAR_SAVED_DISTANCE,
        "faq_id": 3,
    }
    monkeypatch.setattr(answer_sources.vector_store, "search_faq", lambda *a, **k: [near])
    monkeypatch.setattr(answer_sources.vector_store, "search_faq_loose", lambda *a, **k: [])
    monkeypatch.setattr(answer_sources.vector_store, "query", lambda *a, **k: [])

    events = list(
        answer_flow.iter_answer_events(
            QueryRequest(question="Write a short poem about the rain."),
            None,
            None,
            _pending,
        )
    )
    answer = next(data for event, data in events if event == "answer")

    assert answer["answer"] == POEM
    assert answer["source"] in {"taught", "lesson_cards"}
    faq = next(item for event, item in events if event == "connecting" and item["source"] == "faq")
    assert faq["outcome"] == "miss"
    assert f"{settings.faq_match_threshold:.3f}" in faq["detail"]


def test_a_question_with_no_lesson_is_written_and_labelled(learner, monkeypatch):
    monkeypatch.setattr(answer_sources.vector_store, "search_faq", lambda *a, **k: [])
    monkeypatch.setattr(answer_sources.vector_store, "search_faq_loose", lambda *a, **k: [])
    monkeypatch.setattr(answer_sources.vector_store, "query", lambda *a, **k: [])
    monkeypatch.setattr(foundation_answers, "installed", lambda: True)
    monkeypatch.setattr(foundation_answers, "stream", _written_stream)

    events = list(
        answer_flow.iter_answer_events(
            QueryRequest(question="Describe a volcano."), None, None, _pending
        )
    )
    answer = next(data for event, data in events if event == "answer")

    assert answer["source"] == "generated"
    assert answer["mode"] == "generated"
    assert answer["answer"] == "A volcano is a mountain that opens downward to a pool of magma."
    assert answer["understanding"]["verified"] is False
    assert "base checkpoint" in answer["understanding"]["verification"]


def _pending(db, req, entities, disabled, tenant_id=None) -> QueryResponse:
    return QueryResponse(
        question=req.question,
        source="pending",
        answer=answering.NOT_LEARNED,
        hits=[],
        entities=entities,
    )


def _written_stream(req, tenant_id, entities, model_name, cancel=None):
    yield "writing", {"model": model_name, "mode": "generated"}
    yield (
        "answer",
        QueryResponse(
            question=req.question,
            source="generated",
            answer="A volcano is a mountain that opens downward to a pool of magma.",
            hits=[],
            entities=entities,
            mode="generated",
        ),
    )
