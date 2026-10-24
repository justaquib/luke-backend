"""Tests for the transformer that answers a request from the material it was given.

The model is trained from scratch on the tenant's own lessons, and it is
measured on the requests it was never worded for.
"""

from app.services import brain, neural
from app.services.language import lessons, requests, runtime, teaching
from app.services.language import training_materials as materials_service
from app.services.neural.settings import TrainingSettings

POEM_CARD = {
    "id": "poem-1",
    "question": "Poem: Rain Rain Go Away",
    "answer": "Rain, rain, go away,\nCome again another day.\nLittle children want to play.",
    "subject": "Language, Poems",
    "subjects": ["Language", "Poems"],
    "grade_id": "lkg",
}

FACT_CARD = {
    "id": "fact-1",
    "question": "What is a kite?",
    "answer": "A kite is a light frame with a tail.",
    "subject": "Toys",
    "subjects": ["Toys"],
    "grade_id": "nursery",
}

# A small model, so the tests that train one stay quick.
SMALL_MODEL = {"d_model": 32, "layers": 1, "heads": 2, "context": 96}


def _trained_writer():
    """A small model trained on a handful of lessons."""
    writer = neural.AnswerTask()
    writer.config = {**writer.config, "d_model": 32, "layers": 1, "heads": 2, "context": 64}
    sentences = [
        "Rain, rain, go away.",
        "Come again another day.",
        "Little children want to play.",
        "A kite is a light frame with a tail.",
    ]
    instructions_ = [
        ("Write a short poem about the rain.", POEM_CARD["answer"]),
        ("What is a kite?", FACT_CARD["answer"]),
    ]
    contexts = [
        (
            "Write a short poem about the rain.",
            f"{POEM_CARD['question']}. {POEM_CARD['answer']}",
            POEM_CARD["answer"],
        )
    ]
    writer.fit(
        neural.AnswerCorpus(sentences, instructions_, contexts),
        settings=TrainingSettings(steps=200),
    )
    return writer


def test_a_model_that_was_never_trained_says_nothing():
    writer = neural.AnswerTask()

    assert writer.respond("Write a short poem about the rain.") is None


def test_a_trained_model_answers_a_request_it_was_taught():
    writer = _trained_writer()

    answer = writer.respond(
        "Write a short poem about the rain.",
        context=f"{POEM_CARD['question']}. {POEM_CARD['answer']}",
    )

    assert answer is not None
    assert "rain" in answer.lower()


def test_the_answer_keeps_the_lines_of_a_poem():
    writer = _trained_writer()

    answer = writer.respond(
        "Write a short poem about the rain.",
        context=f"{POEM_CARD['question']}. {POEM_CARD['answer']}",
    )

    assert answer is not None
    assert "\n" in answer


def test_a_poem_is_written_for_a_request_it_was_never_worded_for(tmp_path, monkeypatch):
    """The whole path: a poem is taught, then asked for in new words."""
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    runtime._CACHE.clear()
    teaching.teach(
        None,
        "Q: Poem: Rain Rain Go Away\n"
        "A: Rain, rain, go away,\nCome again another day,\nLittle children want to play.",
        "nursery",
        "Poems",
    )
    cards = lessons.all_lesson_cards(None)
    training = materials_service.training_set(materials_service.collect(None))
    report = brain.train(None, training, config=SMALL_MODEL, steps=400)
    writer = brain.load_writer(None)

    assert report["writer"]["trained"] is True
    for question in (
        "Write a short poem about the rain.",
        "Write a little poem about the rain.",
        "Compose a poem about rain.",
    ):
        result = requests.answer_request(question, None, cards, writer=writer)

        assert result is not None, f"the poem was not written for {question!r}"
        assert "rain" in result["answer"].lower()
    runtime._CACHE.clear()


def test_a_poem_about_something_never_taught_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    runtime._CACHE.clear()
    teaching.teach(
        None,
        "Q: Poem: Rain Rain Go Away\nA: Rain, rain, go away,\nCome again another day.",
        "nursery",
        "Poems",
    )
    cards = lessons.all_lesson_cards(None)
    training = materials_service.training_set(materials_service.collect(None))
    brain.train(None, training, config=SMALL_MODEL, steps=400)
    writer = brain.load_writer(None)

    result = requests.answer_request(
        "Write a short poem about volcanoes.", None, cards, writer=writer
    )

    assert result is None
    runtime._CACHE.clear()
