"""Topics: a lesson can belong to more than one."""

import pytest

from app.services import language_logic


@pytest.fixture(autouse=True)
def isolated_learner(tmp_path, monkeypatch):
    """The tests write to a folder of their own, never to the real lessons."""
    monkeypatch.setattr(language_logic, "_DATA_DIR", tmp_path)
    language_logic._CACHE.clear()
    yield
    language_logic._CACHE.clear()


def _teach(text: str, subject: str) -> dict:
    language_logic._CACHE.clear()
    return language_logic.teach(None, text, "nursery", subject)


def test_a_topic_list_is_split_on_commas():
    assert language_logic.parse_subjects("Language,Poems") == ["Language", "Poems"]
    assert language_logic.parse_subjects(" Language , Poems , Language ") == ["Language", "Poems"]
    assert language_logic.parse_subjects("Poems") == ["Poems"]
    assert language_logic.parse_subjects("") == ["General"]


def test_a_lesson_can_belong_to_several_topics():
    result = _teach("Q: What is a poem?\nA: A poem is a short song.", "Language,Poems")

    taught = [card for card in result["lesson_cards"] if card.get("kind") == "trainer"]
    assert len(taught) == 1
    assert taught[0]["subjects"] == ["Language", "Poems"]
    assert taught[0]["subject"] == "Language, Poems"


def test_the_topics_of_a_lesson_can_be_changed():
    result = _teach("Q: What is a poem?\nA: A poem is a short song.", "Language")
    lesson = next(card for card in result["lesson_cards"] if card.get("kind") == "trainer")

    updated = language_logic.update_lesson_answer(
        None,
        lesson["id"],
        "A poem is a short song.",
        "What is a poem?",
        None,
        "Poems, Rhymes",
    )

    taught = next(card for card in updated["lesson_cards"] if card.get("id") == lesson["id"])
    assert taught["subjects"] == ["Poems", "Rhymes"]
    assert taught["subject"] == "Poems, Rhymes"


def test_a_lesson_keeps_its_topics_when_only_the_answer_changes():
    result = _teach("Q: What is a poem?\nA: A poem is a short song.", "Language,Poems")
    lesson = next(card for card in result["lesson_cards"] if card.get("kind") == "trainer")

    updated = language_logic.update_lesson_answer(None, lesson["id"], "A poem is words that rhyme.")

    taught = next(card for card in updated["lesson_cards"] if card.get("id") == lesson["id"])
    assert taught["subjects"] == ["Language", "Poems"]


def test_a_lesson_taught_before_topics_still_has_its_subject():
    state = language_logic._read(None)
    state["lesson_cards"] = [
        {
            "id": "old-1",
            "question": "What is a poem?",
            "answer": "A poem is a short song.",
            "grade_id": "nursery",
            "subject": "Poems",
        }
    ]
    language_logic._save(None, state)
    language_logic._CACHE.clear()

    result = language_logic.get_state(None)
    taught = next(card for card in result["lesson_cards"] if card.get("id") == "old-1")
    assert taught["subjects"] == ["Poems"]
