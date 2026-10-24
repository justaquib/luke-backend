"""The model name, the starting version and the basic conversation.

A tenant names its model and chooses the version it starts from; the name
reaches the brain, the conversation and the training. The conversation answers
a hello and a name question without a lesson, and those pairs are part of the
material the model trains on.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.security import get_bearer_user
from app.db.database import User
from app.routers import language_brain
from app.services import brain_levels, brain_state, model_settings
from app.services.language import answering, conversation, runtime
from app.services.language import training_materials as materials_service


@pytest.fixture()
def brain_dir(tmp_path, monkeypatch):
    """A learner folder of its own, so the tests do not share one."""
    monkeypatch.setattr(runtime, "_DATA_DIR", tmp_path)
    runtime._CACHE.clear()
    yield tmp_path
    runtime._CACHE.clear()


def test_the_model_starts_as_luke_on_version_zero(brain_dir) -> None:
    settings = model_settings.read(7)

    assert settings.name == "Luke"
    assert settings.start_level == "0.0.0"
    view = model_settings.view(7)
    assert view["level"] == "0.0.0"
    assert view["levels"][0]["title"] == "Luke 0.0.0"
    assert view["defaults"] == {"name": "Luke", "start_level": "0.0.0"}


def test_the_name_and_the_starting_version_are_kept(brain_dir) -> None:
    saved = model_settings.save(7, "  Nova  ", "0.0.4")

    assert saved.name == "Nova"
    stored = json.loads((brain_dir / "model-7.json").read_text(encoding="utf-8"))
    assert stored == {"name": "Nova", "start_level": "0.0.4"}
    view = model_settings.view(7)
    assert view["name"] == "Nova"
    assert view["start_level"] == "0.0.4"
    assert view["level"] == "0.0.4"
    assert view["levels"][4]["title"] == "Nova 0.0.4"


def test_the_starting_version_is_the_floor_of_the_level(brain_dir) -> None:
    level_path = brain_levels.level_state_path(7)
    level_path.write_text(json.dumps({"approved_level": "0.0.2"}), encoding="utf-8")

    model_settings.save(7, "Nova", "0.0.4")

    assert brain_levels.approved_level(7) == "0.0.4"


def test_a_level_above_the_starting_version_is_kept(brain_dir) -> None:
    level_path = brain_levels.level_state_path(7)
    level_path.write_text(json.dumps({"approved_level": "0.0.6"}), encoding="utf-8")

    model_settings.save(7, "Nova", "0.0.4")

    assert brain_levels.approved_level(7) == "0.0.6"


def test_an_unusable_name_or_version_is_refused(brain_dir) -> None:
    with pytest.raises(ValueError):
        model_settings.save(7, "   ", "0.0.0")
    with pytest.raises(ValueError):
        model_settings.save(7, "x" * 41, "0.0.0")
    with pytest.raises(ValueError):
        model_settings.save(7, "Nova", "9.9.9")


def test_the_conversation_uses_the_configured_name() -> None:
    assert conversation.answer_conversation("Hello!", "Nova") == (
        "Hello! I am Nova. How can I help you today?"
    )
    assert conversation.answer_conversation("hi nova, how are you", "Nova") == (
        conversation.status_answer()
    )
    assert conversation.answer_conversation("what is your name", "Nova") == (
        "My name is Nova. I am a model trained on what you teach me."
    )
    assert conversation.answer_conversation("what is 2 + 2", "Nova") is None


def test_the_taught_answer_reaches_the_conversation(brain_dir) -> None:
    model_settings.save(None, "Nova", "0.0.0")

    answer = answering.answer_from_curriculum("hello", None)

    assert answer is not None
    assert answer.source == "assistant_profile"
    assert "Nova" in answer.answer


def test_the_conversation_is_part_of_the_training(brain_dir) -> None:
    model_settings.save(None, "Nova", "0.0.0")

    materials = materials_service.collect(None)
    spoken = [item for item in materials if item.kind == materials_service.CONVERSATION]

    assert spoken
    assert any("Nova" in item.answer for item in spoken)
    training = materials_service.training_set(spoken)
    assert any("hello" in instruction for instruction, _ in training.instructions)


def test_the_brain_page_takes_the_name_and_start_level(brain_dir, monkeypatch) -> None:
    model_settings.save(7, "Nova", "0.0.4")
    monkeypatch.setattr(
        brain_state,
        "_measure",
        lambda tenant_id, force: {
            "brain_state": {},
            "knowledge": {},
            "capabilities": [],
            "generality": {"score": 0.0, "cap": None, "probes": []},
            "instructions": {"score": 0.0, "probes": 0, "total": 0},
            "coverage": {},
            "rungs": [],
            "measured": "0.0.0",
        },
    )

    level = brain_state.model_level(7)

    assert level["model_name"] == "Nova"
    assert level["model_start_level"] == "0.0.4"
    assert level["level"] == "0.0.4"
    assert level["title"] == "Nova 0.0.4"


def test_the_settings_route_reads_and_saves(brain_dir) -> None:
    app = FastAPI()
    app.include_router(language_brain.router)
    app.dependency_overrides[get_bearer_user] = lambda: User(
        id=1,
        email="trainer@example.com",
        name="Trainer",
        is_systemadmin=False,
        tenant_id=7,
    )
    client = TestClient(app)

    assert client.get("/language-model/settings").json()["name"] == "Luke"
    saved = client.put("/language-model/settings", json={"name": "Nova", "start_level": "0.0.4"})
    assert saved.status_code == 200
    assert saved.json()["name"] == "Nova"
    assert client.get("/language-model/settings").json()["start_level"] == "0.0.4"
    refused = client.put("/language-model/settings", json={"name": "Nova", "start_level": "9.9.9"})
    assert refused.status_code == 400
