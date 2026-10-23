"""Tests for the training data that shows the model where its material stops.

The training holds requests about things that were never taught with the
refusal to answer them, and the adversarial probes are scored by how often
the model refuses instead of inventing.
"""

from app.services.language.evaluation import _refusals
from app.services.language.topics import topic_index
from app.services.language.training_materials import REFUSAL, _negative_examples

QUESTION = "Where does elephant number {0} sleep?"
ANSWER = "Elephant {0} sleeps in the shade."


def _elephant_cards() -> list[dict]:
    return [
        {"question": QUESTION.format(index), "answer": ANSWER.format(index)}
        for index in range(1, 7)
    ]


def test_a_subordinate_many_lessons_share_is_not_a_taught_topic():
    negatives = dict(_negative_examples(_elephant_cards(), topic_index(_elephant_cards())))

    assert negatives.get("Write a poem about the elephant.") == REFUSAL


def test_the_training_set_holds_the_refusal_for_the_untouched_word(monkeypatch):
    from app.services.language import training_materials as materials_service

    monkeypatch.setattr(materials_service, "_paraphrased", lambda tenant_id, answers: [])
    materials = [
        materials_service.Material("lesson", QUESTION.format(index), ANSWER.format(index))
        for index in range(1, 7)
    ]

    training = materials_service.training_set(materials)

    refusals = dict(training.instructions)
    assert refusals.get("Write a poem about the elephant.") == REFUSAL
    assert any(response != REFUSAL for _, response in training.instructions)


class _StubWriter:
    """A writer that always writes the same answer."""

    def __init__(self, says: str) -> None:
        self.says = says

    def respond(self, request: str, context: str = "") -> str:
        return self.says


def test_every_refused_request_counts_as_refused():
    requests = [
        ("Write a poem about the elephant.", REFUSAL),
        ("Write a poem about a thing that is not the rain.", REFUSAL),
    ]

    result = _refusals(_StubWriter(REFUSAL), requests)

    assert result["passed"] == 2
    assert result["total"] == 2
    assert result["score"] == 1.0


def test_an_invented_answer_is_not_scored_as_a_refusal():
    result = _refusals(
        _StubWriter("Elephants sleep in the shade."),
        [("Write a poem about the elephant.", REFUSAL)],
    )

    assert result["passed"] == 0
    assert result["score"] == 0.0
