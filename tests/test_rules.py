"""Tests for the rules the trainer keeps and how they change the answer."""

from app.schemas import RULE_KINDS, RuleCreate
from app.services import brain, language_logic


def _use_temp_state(monkeypatch, tmp_path):
    monkeypatch.setattr(language_logic, "_DATA_DIR", tmp_path)
    language_logic._CACHE.clear()


def test_the_answer_is_written_in_the_models_own_words(monkeypatch, tmp_path):
    _use_temp_state(monkeypatch, tmp_path)
    # The wording is chosen by the count of the words when there is no
    # transformer trained yet.
    monkeypatch.setattr(brain, "load_writer", lambda tenant_id: None)
    language_logic.teach(
        None,
        "Q: What is a cow?\nA: The cow is animal. It is a farm animal that gives milk.",
        "nursery",
        "Animals",
    )

    written = language_logic.paraphrase_answer(None, "The cow is animal. It is a farm animal that gives milk.")

    assert written == "The cow is a farm animal."


def test_the_facts_of_the_answer_are_kept(monkeypatch, tmp_path):
    _use_temp_state(monkeypatch, tmp_path)
    language_logic.teach(
        None,
        "Q: What is a cow?\nA: The cow is animal. It is a farm animal that gives milk.",
        "nursery",
        "Animals",
    )

    facts = language_logic._predicates_of(
        "The cow is animal. It is a farm animal that gives milk.",
        "The cow",
    )

    assert "is a farm animal" in facts
    assert "gives milk" in facts


def test_a_rule_kind_is_kept():
    rule = RuleCreate(text="Answer in your own words.", kind="own_words")

    assert rule.kind == "own_words"
    assert "own_words" in RULE_KINDS


def test_the_known_rule_kinds_are_the_three_the_assistant_understands():
    assert RULE_KINDS == {"instruction", "own_words", "short"}


def test_a_question_asked_in_other_words_still_finds_the_lesson(monkeypatch, tmp_path):
    _use_temp_state(monkeypatch, tmp_path)
    language_logic.teach(
        None,
        "Q: Which animal gives us milk?\nA: A cow!",
        "nursery",
        "Animals",
    )

    card = language_logic.match_lesson_by_words("Who give milk?", None)

    assert card is not None
    assert card["question"] == "Which animal gives us milk?"


def test_a_different_question_finds_its_own_lesson(monkeypatch, tmp_path):
    _use_temp_state(monkeypatch, tmp_path)
    language_logic.teach(
        None,
        "Q: Which animal gives us milk?\nA: A cow!",
        "nursery",
        "Animals",
    )
    language_logic.teach(
        None,
        "Q: What color is milk?\nA: White!",
        "nursery",
        "Animals",
    )

    card = language_logic.match_lesson_by_words("What color is milk?", None)

    assert card is not None
    assert card["question"] == "What color is milk?"


def test_the_words_are_stemmed_before_they_are_matched():
    assert language_logic._stem("gives") == "give"
    assert language_logic._stem("animals") == "animal"
    assert language_logic._stem("is") == "be"


def test_a_rare_word_decides_the_lesson(monkeypatch, tmp_path):
    _use_temp_state(monkeypatch, tmp_path)
    language_logic.teach(
        None,
        "Q: Which animal says \"meow\"?\nA: The cat!",
        "nursery",
        "Animals",
    )
    language_logic.teach(
        None,
        "Q: What sound does a dog make?\nA: Woof woof!",
        "nursery",
        "Animals",
    )
    language_logic.teach(
        None,
        "Q: What sound does a lion make?\nA: Roar!",
        "nursery",
        "Animals",
    )
    language_logic.teach(
        None,
        "Q: What colour is milk?\nA: White!",
        "nursery",
        "Animals",
    )
    language_logic.teach(
        None,
        "Q: How many days are in a week?\nA: There are seven days in a week.",
        "nursery",
        "Time",
    )
    for other in ("cow", "lion", "horse", "sheep"):
        language_logic.teach(
            None,
            f"Q: What sound does a {other} make?\nA: It makes its own sound.",
            "nursery",
            "Animals",
        )

    card = language_logic.match_lesson_by_words("Who make meow sound?", None)

    assert card is not None
    assert card["question"] == 'Which animal says "meow"?'
