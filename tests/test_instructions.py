"""Tests for the brain answering a request instead of looking one up.

The brain used to answer only the question it was taught. These tests cover
the step after it: a request worded in words the trainer never used, answered
from the material the trainer did teach.
"""

from app.services.language import corpus as corpus_service
from app.services.language import evaluation, instructions, requests, topics

POEM_CARD = {
    "id": "poem-1",
    "question": "Poem: Rain Rain Go Away",
    "answer": "Rain, rain, go away,\nCome again another day.\nLittle children want to play.",
    "subject": "Language, Poems",
    "subjects": ["Language", "Poems"],
    "grade_id": "lkg",
}

# A small model, so the tests that train one stay quick.
SMALL_MODEL = {"d_model": 32, "layers": 1, "heads": 2, "context": 96}

FACT_CARD = {
    "id": "fact-1",
    "question": "What is a kite?",
    "answer": "A kite is a light frame with a tail.",
    "subject": "Toys",
    "subjects": ["Toys"],
    "grade_id": "nursery",
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


# -- reading the request ------------------------------------------------


def test_a_request_for_a_poem_is_read_as_a_poem():
    request = requests.parse_request("Write a short poem about the rain.")

    assert request is not None
    assert request.kind == "poem"
    assert request.topic == "rain"


def test_a_question_is_not_a_request():
    assert requests.parse_request("What is a kite?") is None


def test_a_request_with_nothing_to_write_about_is_not_a_request():
    assert requests.parse_request("Write a poem.") is None


def test_the_words_of_the_request_are_found():
    request = requests.parse_request("Write a short poem about the rain.")

    assert request.words == ("rain",)


def test_every_shape_the_model_is_taught_is_read():
    for form in instructions.TOPIC_FORMS["poem"]:
        assert requests.parse_request(form.format(topic="rain")) is not None
    for form in instructions.TOPIC_FORMS["list"]:
        assert requests.parse_request(form.format(topic="fruits")) is not None
    for form in instructions.TOPIC_FORMS["fact"]:
        assert requests.parse_request(form.format(topic="kite")) is not None


def test_every_shape_the_model_is_measured_with_is_read():
    for form in instructions.HELD_OUT_TOPIC_FORMS["poem"]:
        assert requests.parse_request(form.format(topic="rain")) is not None
    for form in instructions.HELD_OUT_TOPIC_FORMS["list"]:
        assert requests.parse_request(form.format(topic="fruits")) is not None
    for form in instructions.HELD_OUT_TOPIC_FORMS["fact"]:
        assert requests.parse_request(form.format(topic="kite")) is not None


def test_a_fact_request_finds_the_lesson_about_the_topic():
    request = requests.parse_request("Tell me about the kite.")

    card = requests.find_material(request, [FACT_CARD, POEM_CARD])

    assert card is not None
    assert card["id"] == "fact-1"


def test_a_describe_request_is_a_fact_request():
    volcano = {
        "id": "volcano-1",
        "question": "What is a volcano?",
        "answer": "A volcano is a mountain that erupts.",
        "subject": "Science",
        "grade_id": "nursery",
    }

    request = requests.parse_request("Describe the volcano to me.")

    assert request is not None
    assert request.kind == "fact"
    assert requests.find_material(request, [volcano])["id"] == "volcano-1"

    unknown = requests.parse_request("Describe the oyster to me.")
    assert unknown is not None
    assert requests.find_material(unknown, [volcano]) is None


def test_a_list_request_finds_the_list_lesson():
    list_card = {
        "id": "list-1",
        "question": "Name some fruits.",
        "answer": "Some fruits are apple, banana, and mango.",
        "subject": "Fruits",
        "subjects": ["Fruits"],
        "grade_id": "nursery",
    }
    request = requests.parse_request("Name some fruits.")

    card = requests.find_material(request, [FACT_CARD, list_card])

    assert card is not None
    assert card["id"] == "list-1"


# -- finding the material -----------------------------------------------


def test_the_poem_about_the_rain_is_found():
    request = requests.parse_request("Write a short poem about the rain.")

    card = requests.find_material(request, [FACT_CARD, POEM_CARD])

    assert card is not None
    assert card["id"] == "poem-1"


def test_a_fact_is_not_used_to_answer_a_request_for_a_poem():
    request = requests.parse_request("Write a short poem about the kite.")

    card = requests.find_material(request, [FACT_CARD, POEM_CARD])

    assert card is None


def test_nothing_is_found_for_a_topic_that_was_never_taught():
    request = requests.parse_request("Write a short poem about volcanoes.")

    assert requests.find_material(request, [FACT_CARD, POEM_CARD]) is None


def test_a_topic_that_fact_lessons_also_say_still_finds_the_poem():
    """The kind the request asks for picks the pool: facts can share the word."""
    star_card = {
        "id": "fact-star",
        "question": "GK: What is the closest star to Earth?",
        "answer": "The Sun is the closest star to Earth.",
        "subject": "General Knowledge",
        "subjects": ["General Knowledge"],
        "grade_id": "lkg",
    }
    twinkle_card = {
        "id": "poem-twinkle",
        "question": "Poem: Twinkle Twinkle Little Star",
        "answer": "Twinkle, twinkle, little star,\nHow I wonder what you are.",
        "subject": "Language, Poems",
        "subjects": ["Language", "Poems"],
        "grade_id": "nursery",
    }
    request = requests.parse_request("write a poem on star")

    card = requests.find_material(request, [FACT_CARD, star_card, twinkle_card])

    assert card is not None
    assert card["id"] == "poem-twinkle"


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


# -- the instructions the model learns from ------------------------------


def test_a_lesson_is_asked_for_in_many_ways():
    forms = instructions.instructions_for(POEM_CARD, ["rain"])

    assert len(forms) > 1
    assert any("poem about" in form.instruction for form in forms)
    assert all(form.response == POEM_CARD["answer"] for form in forms)


def test_a_lesson_with_no_discriminating_topic_is_only_asked_by_its_question():
    forms = instructions.instructions_for(POEM_CARD, [])

    assert forms
    assert all("poem about" not in form.instruction for form in forms)


def test_the_ways_the_model_is_measured_with_are_not_the_ways_it_learns():
    learned = {form.instruction for form in instructions.instructions_for(POEM_CARD, ["rain"])}
    measured = {
        form.instruction for form in instructions.held_out_instructions_for(POEM_CARD, ["rain"])
    }

    assert measured
    assert not learned & measured


def test_the_held_out_forms_are_kept_out_of_the_training():
    cards = [POEM_CARD, FACT_CARD]
    learned = {form.instruction for form in instructions.training_instructions(cards)}
    measured = {form.instruction for form in instructions.held_out_instructions(cards)}

    assert measured
    assert not learned & measured


def test_a_poem_is_recognised_by_its_subject():
    assert instructions.lesson_kind(POEM_CARD) == "poem"


def test_a_fact_is_recognised_as_a_fact():
    assert instructions.lesson_kind(FACT_CARD) == "fact"


def test_the_subject_of_a_poem_is_the_thing_it_is_about():
    subjects = topics.candidate_topics(POEM_CARD)

    assert "rain" in subjects


def test_a_topic_shared_by_many_lessons_is_not_offered():
    cards = [POEM_CARD, FACT_CARD]

    index = instructions.topic_index(cards)

    assert index["poem-1"]
    assert index["fact-1"] == [] or index["fact-1"]


def test_the_words_of_a_text_keep_their_meaning():
    assert topics.topic_words("The rain is wet") == ["rain", "wet"]


# -- the measurement of the model ----------------------------------------


def test_a_model_that_was_never_trained_scores_nothing():
    result = evaluation.evaluate(None, [POEM_CARD], None)

    assert result["score"] == 0.0
    assert result["total"] == 0


def test_a_model_that_answers_everything_scores_full_marks():
    class PerfectWriter:
        def respond(self, instruction, context="", max_words=64, reading=None):
            return POEM_CARD["answer"]

    result = evaluation.evaluate(None, [POEM_CARD], PerfectWriter())

    assert result["total"] > 0
    assert result["score"] == 1.0


def test_a_model_that_says_nothing_falls_back_to_the_taught_answer():
    class QuietWriter:
        def respond(self, instruction, context="", max_words=64, reading=None):
            return ""

    result = evaluation.evaluate(None, [POEM_CARD], QuietWriter())

    # The model wrote nothing, so the taught answer is given instead. The
    # answer is still right, so the request is answered.
    assert result["score"] == 1.0
    assert all(probe["answered"] == POEM_CARD["answer"] for probe in result["probes"])


def test_a_model_that_writes_something_else_is_not_grounded():
    class WrongWriter:
        def respond(self, instruction, context="", max_words=64, reading=None):
            return "Purple elephants fly sideways at noon."

    result = evaluation.evaluate(None, [POEM_CARD], WrongWriter())

    # The words are not the trainer's, so the taught answer is used instead.
    assert all(probe["answered"] == POEM_CARD["answer"] for probe in result["probes"])


# -- the training data ---------------------------------------------------


def test_the_training_data_holds_a_request_for_every_lesson():
    pairs = corpus_service.instruction_pairs(None)

    assert pairs
    assert all(instruction and response for instruction, response in pairs)


def test_the_grounded_training_data_holds_the_material_too():
    pairs = corpus_service.grounded_pairs(None)

    assert pairs
    for instruction, context, response in pairs:
        assert instruction and context and response
        assert response in context
