"""Tests that the language services write proper sentences instead of looping.

The word model that writes them is tested in ``test_word_model``; these tests
keep the language services honest about the sentences they build and repair.
"""

from app.services import language_logic


def test_a_word_that_was_a_lesson_answer_becomes_a_sentence(monkeypatch, tmp_path):
    monkeypatch.setattr(language_logic, "_DATA_DIR", tmp_path)
    language_logic._CACHE.clear()
    language_logic.teach(
        None,
        "Q: Which is the largest country by area?\nA: Russia.",
        "nursery",
        "Geography",
    )

    sentence = language_logic.make_sentence_from_words(None, ["Russia"])

    assert sentence == "Russia is the largest country by area."


def test_a_lesson_question_becomes_a_statement_for_its_answer():
    statement = language_logic._statement_from_lesson(
        "What is the opposite of hot?",
        "Cold!",
    )

    assert statement == "Cold is the opposite of hot."


def test_a_question_that_cannot_be_a_statement_is_left_alone():
    assert language_logic._statement_from_lesson("Where is the school?", "Main road.") is None
    assert (
        language_logic._statement_from_lesson("Who wrote this book?", "A long name here.") is None
    )


def test_a_which_question_about_a_subject_becomes_a_statement():
    statement = language_logic._statement_from_lesson(
        "Which planet has the most moons?",
        "Saturn (146 confirmed moons).",
    )

    assert statement == "Saturn has the most moons."


def test_a_which_question_about_a_famous_feature_becomes_a_statement():
    statement = language_logic._statement_from_lesson(
        "Which planet is famous for its rings?", "Saturn."
    )

    assert statement == "Saturn is famous for its rings."


def test_a_name_request_becomes_a_statement():
    statement = language_logic._statement_from_lesson("Name the largest planet", "Jupiter.")

    assert statement == "Jupiter is the largest planet."


def test_a_long_answer_is_not_turned_into_a_statement():
    assert (
        language_logic._statement_from_lesson(
            "Name some shapes",
            "Some shapes are circle, triangle, square, rectangle, oval, diamond, "
            "star, heart, pentagon, hexagon.",
        )
        is None
    )


def test_a_word_from_a_list_becomes_a_sentence_of_its_own(monkeypatch, tmp_path):
    monkeypatch.setattr(language_logic, "_DATA_DIR", tmp_path)
    language_logic._CACHE.clear()
    language_logic.teach(
        None,
        "Q: Name some tools.\nA: Some tools are hammer, spanner, chisel.",
        "nursery",
        "Tools",
    )

    sentence = language_logic.make_sentence_from_words(None, ["chisel"])

    assert sentence == "Chisel is a tool."


def test_the_shortest_taught_sentence_is_used(monkeypatch, tmp_path):
    monkeypatch.setattr(language_logic, "_DATA_DIR", tmp_path)
    language_logic._CACHE.clear()
    language_logic.teach(
        None,
        "Q: What is a spanner?\nA: The spanner is a long metal tool.",
        "nursery",
        "Tools",
    )
    language_logic.teach(
        None,
        "Q: Name some tools.\nA: Some tools are hammer, spanner, chisel, file.",
        "nursery",
        "Tools",
    )

    sentence = language_logic.make_sentence_from_words(None, ["spanner"])

    assert sentence == "The spanner is a long metal tool."


def test_a_name_request_keeps_its_article():
    statement = language_logic._statement_from_lesson(
        "Name a fruit that is sour and yellow.", "Lemon!"
    )

    assert statement == "Lemon is a fruit that is sour and yellow."


def test_a_word_is_found_inside_a_word_with_an_ending():
    assert language_logic._token_matches("bee's", "bee") is True
    assert language_logic._token_matches("bee", "bee") is True
    assert language_logic._token_matches("beech", "bee") is False


def test_the_plural_is_written_when_a_possessive_is_typed():
    assert language_logic._repair_sentence("Bee's make honey.") == "Bees make honey."


def test_a_real_possessive_is_left_alone():
    assert language_logic._repair_sentence("The dog's tail wags.") == "The dog's tail wags."
    assert (
        language_logic._repair_sentence("My mother's house is big.") == "My mother's house is big."
    )


def test_a_sentence_that_begins_with_the_word_is_used(monkeypatch, tmp_path):
    monkeypatch.setattr(language_logic, "_DATA_DIR", tmp_path)
    language_logic._CACHE.clear()
    language_logic.teach(
        None,
        "Q: What is a liquid?\nA: A liquid is a state of matter that has a fixed "
        "volume but no fixed shape.",
        "nursery",
        "Science",
    )
    language_logic.teach(
        None,
        "Q: What happens to ice when it melts?\nA: It changes from a solid to a liquid (water).",
        "nursery",
        "Science",
    )

    sentence = language_logic.make_sentence_from_words(None, ["liquid"])

    assert sentence == "A liquid is a state of matter that has a fixed volume but no fixed shape."


def test_a_short_answer_is_not_mistaken_for_a_sentence(monkeypatch, tmp_path):
    monkeypatch.setattr(language_logic, "_DATA_DIR", tmp_path)
    language_logic._CACHE.clear()
    language_logic.teach(
        None,
        "Q: Name a yellow fruit.\nA: Banana / Mango!",
        "nursery",
        "Fruits",
    )

    sentence = language_logic.make_sentence_from_words(None, ["banana"])

    assert sentence != "Banana / Mango!"


def test_a_missing_article_is_put_back():
    assert language_logic._repair_sentence("The cow is animal.") == "The cow is an animal."


def test_an_adjective_is_left_without_an_article():
    assert language_logic._repair_sentence("The night sky is black.") == "The night sky is black."
    assert language_logic._repair_sentence("The apple is red.") == "The apple is red."


def test_a_list_is_not_cut_into_its_first_item(monkeypatch, tmp_path):
    monkeypatch.setattr(language_logic, "_DATA_DIR", tmp_path)
    language_logic._CACHE.clear()
    answer = "Some animals are ant, bear, cat, cow, deer, dog, elephant, fish."

    written = language_logic.paraphrase_answer(None, answer)

    assert written is None


def test_the_number_that_was_asked_for_is_given():
    answer = "Some animals are ant, bear, cat, cow, deer, dog, elephant, fish."

    assert (
        language_logic.apply_requested_count("name 3 animals", answer)
        == "Some animals are ant, bear, cat."
    )
    assert language_logic.apply_requested_count("name some animals", answer) == answer


def test_a_list_is_never_put_into_other_words():
    answer = "Some animals are ant, bear, cat, cow, deer, dog."

    assert language_logic.paraphrase_answer(None, answer) is None


def test_a_number_after_any_number_is_worked_out():
    assert (
        language_logic._answer_number_sequence_question("what comes after 20")
        == "The number after 20 is 21."
    )
    assert (
        language_logic._answer_number_sequence_question("what comes before 20")
        == "The number before 20 is 19."
    )
    assert language_logic._answer_number_sequence_question("what comes after b") is None


def test_the_count_is_honoured_wherever_it_is_asked():
    answer = "Some fruits are apple, banana, blueberry, cherry, grape, guava."

    assert (
        language_logic.apply_requested_count("write 5 fruits name", answer)
        == "Some fruits are apple, banana, blueberry, cherry, grape."
    )
    assert (
        language_logic.apply_requested_count("name five fruits", answer)
        == "Some fruits are apple, banana, blueberry, cherry, grape."
    )
    assert language_logic.apply_requested_count("how many fruits", answer) == answer
