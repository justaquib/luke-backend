"""Tests for the training data that shows the model where its material stops.

The training holds requests about things that were never taught with the
refusal to answer them, and the held-out requests are never the ones the
training saw, so the refusal is measured honestly.
"""

from app.services.language.negatives import (
    MAX_NEGATIVE_EXAMPLES,
    REFUSAL,
    build_held_out_negatives,
    build_negatives,
    unknown_topics,
)

QUESTION = "Where does elephant number {0} sleep?"
ANSWER = "Elephant {0} sleeps in the shade."


def _elephant_cards() -> list[dict]:
    return [
        {"question": QUESTION.format(index), "answer": ANSWER.format(index)}
        for index in range(1, 7)
    ]


def _volcano_cards() -> list[dict]:
    return [{"question": "What is a volcano?", "answer": "A volcano is a mountain that erupts."}]


def test_an_unknown_topic_is_asked_in_every_wording():
    negatives = dict(build_negatives(_elephant_cards()))

    assert negatives.get("Write a poem about the accordion.") == REFUSAL
    assert negatives.get("Tell me everything about the accordion.") == REFUSAL
    assert negatives.get("Describe the accordion to me.") == REFUSAL


def test_a_topic_the_lessons_mention_is_not_unknown():
    assert "volcano" not in unknown_topics(_volcano_cards())

    negatives = dict(build_negatives(_volcano_cards()))
    assert not any("volcano" in request for request in negatives)


def test_the_negatives_stop_at_the_cap():
    negatives = build_negatives([])

    assert len(negatives) == MAX_NEGATIVE_EXAMPLES
    assert len({request for request, _ in negatives}) == MAX_NEGATIVE_EXAMPLES
    assert all(response == REFUSAL for _, response in negatives)


def test_the_training_set_holds_the_refusal_for_an_unknown_topic(monkeypatch):
    from app.services.language import training_materials as materials_service

    monkeypatch.setattr(materials_service, "_paraphrased", lambda tenant_id, answers: [])
    materials = [
        materials_service.Material("lesson", QUESTION.format(index), ANSWER.format(index))
        for index in range(1, 7)
    ]

    training = materials_service.training_set(materials)

    refusals = dict(training.instructions)
    assert refusals.get("Write a poem about the accordion.") == REFUSAL
    assert any(response != REFUSAL for _, response in training.instructions)


def test_the_held_out_refusals_are_never_trained():
    cards = _elephant_cards()
    trained = {request for request, _ in build_negatives(cards)}

    held = build_held_out_negatives(cards, 20)

    assert held
    assert all(request not in trained for request, _ in held)
