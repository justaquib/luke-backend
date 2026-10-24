"""Tests for answering a request from the taught material.

The request path serves "write a poem about the rain" from the lesson the
trainer taught. These tests cover the writer being trusted, the writer being
refused, and the taught answer that is given instead.
"""

from app.services.language import requests
from app.services.neural.critique import Critique

POEM_CARD = {
    "id": "poem-1",
    "question": "Poem: Rain Rain Go Away",
    "answer": "Rain, rain, go away,\nCome again another day.\nLittle children want to play.",
    "subject": "Language, Poems",
    "subjects": ["Language", "Poems"],
    "grade_id": "lkg",
}


class FakeWriter:
    """A writer that answers with the words it is given."""

    def __init__(self, answer: str = "Rain, rain, go away.") -> None:
        self.answer = answer
        self.asked = []
        self.last_confidence = 0.9

    def respond(self, instruction, context="", max_words=64, reading=None):
        self.asked.append((instruction, context))
        return self.answer


# -- answering the request -----------------------------------------------


def test_a_request_is_answered_from_the_taught_material():
    writer = FakeWriter("Rain, rain, go away,\nCome again another day.")

    result = requests.answer_request(
        "Write a short poem about the rain.", None, [POEM_CARD], writer=writer
    )

    assert result is not None
    assert result["answer"] == "Rain, rain, go away,\nCome again another day."
    assert result["source"] == "written"


def test_the_taught_answer_is_used_when_the_model_says_something_else():
    writer = FakeWriter("Purple elephants fly sideways at noon.")

    result = requests.answer_request(
        "Write a short poem about the rain.", None, [POEM_CARD], writer=writer
    )

    assert result is not None
    assert result["source"] == "taught"
    assert result["answer"] == POEM_CARD["answer"]


def test_the_taught_answer_is_used_when_the_model_rejects_its_own_answer():
    class UnhappyWriter(FakeWriter):
        def __init__(self) -> None:
            super().__init__(POEM_CARD["answer"])
            self.last_critique = Critique(
                problems=["The sentence has no verb."],
                notes=[],
                confidence=0.9,
                grounded=True,
            )

    result = requests.answer_request(
        "Write a short poem about the rain.", None, [POEM_CARD], writer=UnhappyWriter()
    )

    assert result is not None
    assert result["source"] == "taught"


def test_a_word_that_only_sits_inside_another_word_is_not_grounded():
    class InsideWordWriter(FakeWriter):
        def __init__(self) -> None:
            super().__init__("A dragon is an insect.")

    dragonfly_card = {
        "id": "fact-dragonfly",
        "question": "What is a dragonfly?",
        "answer": "A dragonfly is an insect with wings.",
        "subject": "Nature",
        "subjects": ["Nature"],
        "grade_id": "nursery",
    }

    result = requests.answer_request(
        "Tell me about the dragonfly.", None, [dragonfly_card], writer=InsideWordWriter()
    )

    assert result is not None
    assert result["source"] == "taught"
    assert result["answer"] == dragonfly_card["answer"]


def test_the_taught_answer_is_used_when_the_model_repeats_itself():
    """Looping words are not an answer: the taught poem is given instead."""
    writer = FakeWriter("Rain, go away, go away, go away, go away.")

    result = requests.answer_request(
        "Write a short poem about the rain.", None, [POEM_CARD], writer=writer
    )

    assert result is not None
    assert result["source"] == "taught"
    assert result["answer"] == POEM_CARD["answer"]


def test_a_poem_that_says_its_line_again_is_still_the_answer_of_the_model():
    """The same line twice is a refrain, so the written poem is given away."""
    writer = FakeWriter(
        "Rain, rain, go away,"
        "\nCome again another day."
        "\nLittle children want to play,"
        "\nRain, rain, go away."
    )

    result = requests.answer_request(
        "Write a short poem about the rain.", None, [POEM_CARD], writer=writer
    )

    assert result is not None
    assert result["source"] == "written"


def test_a_request_about_something_never_taught_is_refused():
    writer = FakeWriter("Rain, rain, go away.")

    result = requests.answer_request(
        "Write a short poem about volcanoes.", None, [POEM_CARD], writer=writer
    )

    assert result is None


def test_a_request_is_answered_from_the_taught_answer_when_there_is_no_model():
    result = requests.answer_request(
        "Write a short poem about the rain.", None, [POEM_CARD], writer=None
    )

    assert result is not None
    assert result["source"] == "taught"


def test_the_material_given_to_the_model_holds_the_lesson():
    writer = FakeWriter()

    requests.answer_request("Write a short poem about the rain.", None, [POEM_CARD], writer=writer)

    instruction, context = writer.asked[0]
    assert "poem about the rain" in instruction.lower()
    assert "rain" in context.lower()
    assert "go away" in context.lower()
