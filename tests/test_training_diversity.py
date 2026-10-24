"""Tests for the training data that shows the model where its material stops.

The training holds requests about things that were never taught with the
refusal to answer them. The adversarial score is the refusal the product
gives the user, and the writer's own refusal is counted beside it.
"""

from app.services.language.evaluation import _refusals
from app.services.language.negatives import (
    MAX_NEGATIVE_EXAMPLES,
    REFUSAL,
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


class _StubWriter:
    """A writer that always writes the same answer."""

    def __init__(self, says: str) -> None:
        self.says = says

    def respond(self, request: str, context: str = "") -> str:
        return self.says


def test_the_material_boundary_refuses_what_was_never_taught():
    result = _refusals(
        None,
        _elephant_cards(),
        _StubWriter("Elephants sleep in the shade."),
        [("Write a poem about the accordion.", REFUSAL)],
    )

    assert result["passed"] == 1
    assert result["total"] == 1
    assert result["score"] == 1.0
    assert result["model"] == 0
    assert result["probes"][0]["correct"] is True


def test_the_writer_that_says_no_is_counted_beside_the_boundary():
    result = _refusals(
        None,
        _elephant_cards(),
        _StubWriter(REFUSAL),
        [("Write a poem about the accordion.", REFUSAL)],
    )

    assert result["passed"] == 1
    assert result["model"] == 1


def test_an_answer_about_taught_material_is_not_a_refusal():
    result = _refusals(
        None,
        _volcano_cards(),
        _StubWriter("Elephants sleep in the shade."),
        [("Tell me everything about the volcano.", REFUSAL)],
    )

    assert result["passed"] == 0
    assert result["probes"][0]["answered"]
