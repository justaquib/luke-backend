"""The conversation expressions the trainer teaches.

An expression is a phrase and the reply Luke gives it. The four categories are
opening, closing, pausing and resuming. A taught phrase is answered exactly and
joins the training material, but it only wins when the whole message is that
phrase, so a question that begins with the same words still reaches its lesson.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.security import get_bearer_user
from app.db.database import User
from app.routers import expressions as expression_routes
from app.services.language import answering, conversation, expressions, runtime
from app.services.language import training_materials as materials_service


@pytest.fixture()
def brain_dir(tmp_path, monkeypatch):
    """A learner folder of its own, so the tests do not share one."""
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    runtime._CACHE.clear()
    yield tmp_path
    runtime._CACHE.clear()


@pytest.fixture()
def client(brain_dir):
    app = FastAPI()
    app.include_router(expression_routes.router)
    app.dependency_overrides[get_bearer_user] = lambda: User(
        id=1,
        email="trainer@example.com",
        name="Trainer",
        is_systemadmin=False,
        tenant_id=7,
    )
    return TestClient(app)


def test_an_expression_is_taught_and_read_back(brain_dir) -> None:
    taught = expressions.add(7, expressions.OPEN, "  hey   Luke ", "Hey, good to see you.")

    assert taught.category == "open"
    assert taught.phrase == "hey Luke"
    assert expressions.load(7) == [taught]
    assert expressions.reply_for(7, "Hey, Luke!") == "Hey, good to see you."


def test_the_four_categories_are_counted_in_the_view(brain_dir) -> None:
    expressions.add(7, expressions.OPEN, "hey luke", "Hello there.")
    expressions.add(7, expressions.PAUSE, "hold on a second", "I will wait right here.")
    expressions.add(7, expressions.RESUME, "back again", "Welcome back.")

    view = expressions.view(7)

    assert [item["id"] for item in view["categories"]] == ["open", "close", "pause", "resume"]
    assert {item["id"]: item["count"] for item in view["categories"]} == {
        "open": 1,
        "close": 0,
        "pause": 1,
        "resume": 1,
    }
    assert view["items"][0]["label"] == "Opening"


def test_a_phrase_is_taught_only_once(brain_dir) -> None:
    expressions.add(7, expressions.OPEN, "hey luke", "Hello there.")

    with pytest.raises(expressions.ExpressionError):
        expressions.add(7, expressions.OPEN, "HEY, LUKE!", "Hello again.")
    with pytest.raises(expressions.ExpressionError):
        expressions.add(7, expressions.CLOSE, "hey luke", "Goodbye.")


def test_an_unusable_expression_is_refused(brain_dir) -> None:
    with pytest.raises(expressions.ExpressionError):
        expressions.add(7, "greeting", "hey", "Hello.")
    with pytest.raises(expressions.ExpressionError):
        expressions.add(7, expressions.OPEN, "   ", "Hello.")
    with pytest.raises(expressions.ExpressionError):
        expressions.add(7, expressions.OPEN, "hey", "  ")
    with pytest.raises(expressions.ExpressionError):
        expressions.add(7, expressions.OPEN, "x" * 81, "Hello.")
    with pytest.raises(expressions.ExpressionError):
        expressions.add(7, expressions.OPEN, "hey", "x" * 241)


def test_an_expression_is_edited_in_place(brain_dir) -> None:
    taught = expressions.add(7, expressions.OPEN, "hey luke", "Hello there.")

    saved = expressions.update(
        7, taught.id, expressions.RESUME, "hey luke again", "Good to see you."
    )

    assert saved.id == taught.id
    assert saved.category == "resume"
    assert expressions.reply_for(7, "hey luke") is None
    assert expressions.reply_for(7, "hey luke again") == "Good to see you."


def test_an_expression_can_be_forgotten(brain_dir) -> None:
    taught = expressions.add(7, expressions.CLOSE, "that is all for now", "Understood.")

    expressions.remove(7, taught.id)

    assert expressions.load(7) == []
    with pytest.raises(expressions.ExpressionError):
        expressions.remove(7, taught.id)


def test_only_the_whole_message_is_the_expression(brain_dir) -> None:
    expressions.add(7, expressions.OPEN, "hey luke", "Hello there.")

    assert expressions.reply_for(7, "hey luke") == "Hello there."
    assert expressions.reply_for(7, "hey luke, what is rain") is None
    assert expressions.reply_for(7, "do you hear me say hey luke") is None


def test_a_taught_expression_beats_the_built_in_greeting(brain_dir) -> None:
    expressions.add(7, expressions.OPEN, "hello", "Good to have you back.")

    answer = answering.answer_from_curriculum("Hello!", 7)

    assert answer is not None
    assert answer.source == "expressions"
    assert answer.answer == "Good to have you back."
    built_in = conversation.greeting_answer("Luke")
    assert conversation.answer_conversation("Hello!", "Luke") == built_in


def test_a_question_with_the_same_first_words_reaches_its_lesson(brain_dir) -> None:
    expressions.add(7, expressions.OPEN, "hello", "Good to have you back.")

    answer = answering.answer_from_curriculum("hello, what is your name", 7)

    assert answer is not None
    assert answer.source == "assistant_profile"


def test_taught_expressions_are_part_of_the_training(brain_dir) -> None:
    expressions.add(7, expressions.OPEN, "hey luke", "Good to see you.")
    expressions.add(7, expressions.OPEN, "hello", "Good to have you back.")

    materials = materials_service.collect(7)
    spoken = [item for item in materials if item.kind == materials_service.CONVERSATION]
    asked = [item.question for item in spoken]

    assert "hey luke" in asked
    assert asked.count("hello") == 1
    assert next(item for item in spoken if item.question == "hello").answer == (
        "Good to have you back."
    )
    training = materials_service.training_set(spoken, 7)
    assert ("hey luke", "Good to see you.") in training.instructions


def test_the_expression_routes_teach_edit_and_remove(client) -> None:
    listed = client.get("/language-model/expressions").json()
    assert listed["items"] == []
    assert len(listed["categories"]) == 4

    created = client.post(
        "/language-model/expressions",
        json={"category": "open", "phrase": "hey luke", "reply": "Hello there."},
    )
    assert created.status_code == 201
    expression_id = created.json()["id"]

    edited = client.put(
        f"/language-model/expressions/{expression_id}",
        json={"category": "close", "phrase": "bye luke", "reply": "See you soon."},
    )
    assert edited.status_code == 200
    assert edited.json()["category"] == "close"

    assert client.get("/language-model/expressions").json()["categories"][1]["count"] == 1
    removed = client.delete(f"/language-model/expressions/{expression_id}")
    assert removed.status_code == 200
    assert client.get("/language-model/expressions").json()["items"] == []


def test_the_expression_routes_report_bad_input(client) -> None:
    refused = client.post(
        "/language-model/expressions",
        json={"category": "open", "phrase": "hey luke", "reply": "Hello there."},
    )
    assert refused.status_code == 201

    duplicate = client.post(
        "/language-model/expressions",
        json={"category": "open", "phrase": "Hey Luke", "reply": "Again."},
    )
    assert duplicate.status_code == 400
    unknown = client.delete("/language-model/expressions/missing")
    assert unknown.status_code == 404
