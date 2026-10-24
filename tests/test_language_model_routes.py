"""Tests for the language model routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(language_client: TestClient) -> TestClient:
    """The shared language route client under the name this module uses."""
    return language_client


def test_the_state_lists_the_grades_and_the_approved_answers(client: TestClient):
    response = client.get("/language-model/state")

    assert response.status_code == 200
    state = response.json()
    assert state["current_stage_id"] == "nursery"
    assert [stage["id"] for stage in state["stages"]][:2] == ["nursery", "lkg"]
    assert state["trainer_answers"] == []


def test_the_brain_reads_its_numbers_with_the_level_it_reached(client: TestClient):
    response = client.get("/language-model/brain")

    assert response.status_code == 200
    brain = response.json()
    assert "vocabulary" in brain, "the brain page reads the vocabulary from here"
    assert "emotions" in brain
    assert "model_level" in brain, "the level sits next to the numbers it explains"
    assert "rungs" in brain["model_level"]
    assert "generality_probes" in brain["model_level"]


def test_the_level_test_runs_the_rungs_again(client: TestClient):
    response = client.post("/language-model/brain/test")

    assert response.status_code == 200
    result = response.json()
    assert result["total"] == len(result["rungs"])
    assert result["passed"] <= result["total"]
    assert "model_level" in result["state"], "the level is measured again with the rungs"


def test_a_duplicate_lesson_is_kept_and_reported(client: TestClient):
    lesson = "Q: What is a kite?\nA: A kite is a light frame with a tail."

    first = client.post(
        "/language-model/lessons",
        json={"lesson": lesson, "stage": "nursery", "subject": "Toys"},
    )
    assert first.status_code == 200
    assert first.json()["added_lessons"] == 1

    second = client.post(
        "/language-model/lessons",
        json={"lesson": lesson, "stage": "nursery", "subject": "Toys"},
    )
    assert second.status_code == 200
    assert second.json()["added_lessons"] == 1, "a repeat is kept so the trainer can decide"

    duplicates = client.get("/language-model/lessons/duplicates")
    assert duplicates.status_code == 200
    report = duplicates.json()
    assert report["duplicate_count"] >= 1
    questions = [card["question"] for group in report["groups"] for card in group["cards"]]
    assert "What is a kite?" in questions


def test_a_group_can_be_accepted_and_is_not_reported_again(client: TestClient):
    lesson = "Q: What is a drum?\nA: A drum is something you hit with a stick."
    client.post(
        "/language-model/lessons",
        json={"lesson": lesson, "stage": "nursery", "subject": "Music"},
    )
    client.post(
        "/language-model/lessons",
        json={"lesson": lesson, "stage": "nursery", "subject": "Music"},
    )

    reported = client.get("/language-model/lessons/duplicates").json()
    assert reported["duplicate_count"] >= 1
    group = next(
        item
        for item in reported["groups"]
        if any("drum" in card["question"].lower() for card in item["cards"])
    )

    accepted = client.post(
        "/language-model/lessons/duplicates/accept",
        json={"key": group["key"]},
    )
    assert accepted.status_code == 200

    after = client.get("/language-model/lessons/duplicates").json()
    keys = [item["key"] for item in after["groups"]]
    assert group["key"] not in keys, "an accepted group is not reported again"


def test_training_the_brain_starts_it_and_says_so(client: TestClient, monkeypatch):
    started: list[object] = []

    def fake_train(tenant_id, materials, config=None, steps=3000):
        started.append(materials)
        return {"writer": {"trained": True}}

    class _FakeProcess:
        def __init__(self, target, args, daemon) -> None:
            self._target, self._args = target, args

        def start(self) -> None:
            self._target(*self._args)

    class _FakeContext:
        Process = _FakeProcess

    monkeypatch.setattr("app.services.brain.train", fake_train)
    monkeypatch.setattr("app.services.brain.mark_training", lambda tenant_id, value: None)
    monkeypatch.setattr(
        "app.routers.language_brain.multiprocessing.get_context", lambda name: _FakeContext()
    )

    response = client.post("/language-model/brain/train")

    assert response.status_code == 200
    assert response.json()["training"] is True
    assert started, "the training was started"


def test_a_question_is_traced_and_an_untrained_writer_is_named(client: TestClient, monkeypatch):
    monkeypatch.setattr("app.services.brain_trace.brain.load_writer", lambda tenant_id: None)

    response = client.post("/language-model/trace", json={"question": "who is your creator?"})

    assert response.status_code == 200
    trace = response.json()
    assert trace["trained"] is False
    assert trace["reason"]


def test_a_trace_needs_a_question(client: TestClient):
    response = client.post("/language-model/trace", json={"question": ""})

    assert response.status_code == 422


def test_the_store_reports_the_files_the_numbers_and_the_size(client: TestClient):
    response = client.get("/language-model/store")

    assert response.status_code == 200
    store = response.json()
    assert store["totals"]["files"] == 0
    assert store["totals"]["chunks"] == 0
    assert "vocabulary" in store["totals"]
    assert "parameters" in store["model"]
    assert store["disk"]["learner_bytes"] >= 0
    assert store["disk"]["vector_store_bytes"] >= 0


def test_a_taught_lesson_comes_back_in_the_state(client: TestClient):
    taught = client.post(
        "/language-model/lessons",
        json={
            "lesson": "Q: What is a drum?\nA: A drum is something you hit with a stick.",
            "stage": "nursery",
            "subject": "Music",
        },
    )

    assert taught.status_code == 200
    state = taught.json()
    assert state["added_lessons"] == 1
    questions = [lesson["question"] for lesson in state["lesson_cards"]]
    assert "What is a drum?" in questions


def test_the_duplicate_check_reports_the_same_lesson(client: TestClient):
    client.post(
        "/language-model/lessons",
        json={
            "lesson": "Q: What is a drum?\nA: A drum is something you hit with a stick.",
            "stage": "nursery",
            "subject": "Music",
        },
    )

    checked = client.post(
        "/language-model/lessons/check",
        json={
            "lesson": "Q: What is a drum?\nA: A drum is something you hit with a stick.",
            "stage": "nursery",
            "subject": "Music",
        },
    )

    assert checked.status_code == 200
    report = checked.json()
    assert report["exact"], "the same lesson should be reported as an exact duplicate"
    assert report["faqs"] == []
    assert report["intents"] == []


def test_the_preview_answers_from_the_learners_own_material(client: TestClient):
    client.post(
        "/language-model/lessons",
        json={
            "lesson": "Q: What is a drum?\nA: A drum is something you hit with a stick.",
            "stage": "nursery",
            "subject": "Music",
        },
    )

    preview = client.post("/language-model/preview", json={"question": "What is a drum?"})

    assert preview.status_code == 200
    answer = preview.json()
    assert "drum" in answer["answer"].lower()
    assert answer["reasoning"]["steps"]


def test_a_request_for_a_poem_is_answered_from_the_taught_poem(client: TestClient):
    client.post(
        "/language-model/lessons",
        json={
            "lesson": (
                "Q: Poem: Rain Rain Go Away\nA: Rain, rain, go away,\nCome again another day."
            ),
            "stage": "nursery",
            "subject": "Poems",
        },
    )

    preview = client.post(
        "/language-model/preview",
        json={"question": "Write a short poem about the rain."},
    )

    assert preview.status_code == 200
    answer = preview.json()
    assert "rain" in answer["answer"].lower()
    assert answer["source"] in {"taught", "written"}


def test_a_request_about_something_never_taught_is_not_answered(client: TestClient):
    client.post(
        "/language-model/lessons",
        json={
            "lesson": "Q: What is a drum?\nA: A drum is something you hit with a stick.",
            "stage": "nursery",
            "subject": "Music",
        },
    )

    preview = client.post(
        "/language-model/preview",
        json={"question": "Write a short poem about volcanoes."},
    )

    assert preview.status_code == 200
    assert preview.json()["source"] == "not_learned"


def test_a_lesson_can_be_edited_and_removed(client: TestClient):
    taught = client.post(
        "/language-model/lessons",
        json={
            "lesson": "Q: What is a drum?\nA: A drum is something you hit with a stick.",
            "stage": "nursery",
            "subject": "Music",
        },
    )
    lesson = next(
        item for item in taught.json()["lesson_cards"] if item["question"] == "What is a drum?"
    )

    edited = client.put(
        f"/language-model/lessons/{lesson['id']}",
        json={"answer": "A drum is a round instrument you hit.", "subject": "Music"},
    )
    assert edited.status_code == 200
    answers = [item["answer"] for item in edited.json()["lesson_cards"]]
    assert "A drum is a round instrument you hit." in answers

    removed = client.delete(f"/language-model/lessons/{lesson['id']}")
    assert removed.status_code == 200
    questions = [item["question"] for item in removed.json()["lesson_cards"]]
    assert "What is a drum?" not in questions


def test_an_unknown_lesson_is_not_found(client: TestClient):
    response = client.put("/language-model/lessons/missing", json={"answer": "Nothing"})

    assert response.status_code == 404
